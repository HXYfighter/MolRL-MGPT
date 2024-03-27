import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.tensorboard import SummaryWriter
import multiprocessing

# rdkit
from rdkit import Chem, DataStructs

# guacamol
import guacamol
from guacamol.scoring_function import ScoringFunction
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.utils.chemistry import canonicalize_list
from assess_goal_directed_generation import assess_goal_directed_generation

from model import GPT, GPTConfig
from vocabulary import read_vocabulary
from utils import set_seed, sample_SMILES, likelihood, to_tensor, calc_fingerprints

from time import time
class MARL_goal_directed_generator(GoalDirectedGenerator):

    def __init__(self, logger, configs):
        self.writer = logger
        self.model_type = configs.model_type
        self.task_id = configs.task_id
        self.num_agents = configs.num_agents
        self.prior_path = configs.prior_path
        self.voc = read_vocabulary(configs.vocab_path)
        self.batch_size = configs.batch_size
        self.n_steps = configs.n_steps
        self.learning_rate = configs.learning_rate
        self.sigma1 = configs.sigma1
        self.sigma2 = configs.sigma2
        # experience replay
        self.memory = pd.DataFrame(columns=["smiles", "scores", "seqs", "fps"])
        self.memory_size = configs.memory_size
        self.replay = configs.replay
        # penalize similarity
        self.sim_penalize = configs.sim_penalize
        self.sim_thres = configs.sim_thres
        

    def _memory_update(self, smiles, scores, seqs):
        scores = list(scores)
        seqs_list = [seqs[i, :].cpu().numpy() for i in range(len(smiles))]

        fps_memory = list(self.memory["fps"])

        mean_coef = 0
        for i in range(len(smiles)):
            if scores[i] < 0:
                continue
            # canonicalized SMILES and fingerprints
            fp, smiles_i = calc_fingerprints([smiles[i]])
            new_data = pd.DataFrame({"smiles": smiles_i[0], "scores": scores[i], "seqs": [seqs_list[i]], "fps": fp[0]})
            self.memory = pd.concat([self.memory, new_data], ignore_index=True, sort=False)

            # penalize similarity
            if self.sim_penalize and len(fps_memory) > 0:
                sims = [DataStructs.FingerprintSimilarity(fp[0], x) for x in fps_memory]
                if np.sum(np.array(sims) >= self.sim_thres) > 20:
                	scores[i] = 0

        self.memory = self.memory.drop_duplicates(subset=["smiles"])
        self.memory = self.memory.sort_values('scores', ascending=False)
        self.memory = self.memory.reset_index(drop=True)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory.head(self.memory_size)

        # experience replay
        if self.replay > 0:
            s = min(len(self.memory), self.replay)
            experience = self.memory.head(5 * self.replay).sample(s)
            experience = experience.reset_index(drop=True)
            smiles += list(experience["smiles"])
            scores += list(experience["scores"])
            for index in experience.index:
                seqs = torch.cat((seqs, torch.tensor(experience.loc[index, "seqs"], dtype=torch.long).view(1, -1).cuda()), dim=0)

        return smiles, np.array(scores), seqs


    def generate_optimized_molecules(self, scoring_function, number_molecules, starting_population=None):

        if self.model_type == "gpt":
            prior_config = GPTConfig(self.voc.__len__(), n_layer=8, n_head=8, n_embd=256, block_size=128)
            prior = GPT(prior_config).to("cuda")
            agents = []
            optimizers = []
            for i in range(self.num_agents):
                agents.append(GPT(prior_config).to("cuda"))
                optimizers.append(agents[i].configure_optimizers(weight_decay=0.1, 
                                                                learning_rate=self.learning_rate, 
                                                                betas=(0.9, 0.95)))
        
        scaler = torch.cuda.amp.GradScaler()
        prior.load_state_dict(torch.load(self.prior_path), strict=True)
        for param in prior.parameters():
            param.requires_grad = False
        prior.eval()
        for i in range(self.num_agents):
            agents[i].load_state_dict(torch.load(self.prior_path), strict=True)
            agents[i].eval()

        for step in tqdm(range(self.n_steps)):
            for i in range(self.num_agents):
                samples, seqs, _ = sample_SMILES(agents[i], self.voc, n_mols=self.batch_size)

                scores = scoring_function.score_list(samples)
                samples, scores, seqs = self._memory_update(samples, scores, seqs)
            
                prior_likelihood = likelihood(prior, seqs)
                agent_likelihood = likelihood(agents[i], seqs)
                loss = torch.pow(self.sigma1 * to_tensor(np.array(scores)) - (prior_likelihood - agent_likelihood), 2)
                for j in range(i):
                    agent_j_likelihood = likelihood(agents[j], seqs)
                    loss -= self.sigma2 * torch.pow(agent_j_likelihood - agent_likelihood, 2)
                loss = loss.mean()

                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()

            if self.task_id in list(range(3, 6)) + list(range(8, 20)):
                self.writer.add_scalar('top-1 score', np.max(np.array(self.memory["scores"])), step)
                self.writer.add_scalar('top-10 score', np.mean(np.array(self.memory["scores"][:10])), step)
                self.writer.add_scalar('top-100 score', np.mean(np.array(self.memory["scores"][:100])), step)
            elif self.task_id in list(range(0, 3)):
                self.writer.add_scalar('top-1 score', np.max(np.array(self.memory["scores"])), step)
            elif self.task_id == 6:
                self.writer.add_scalar('top-159 score', np.mean(np.array(self.memory["scores"][:159])), step)
            elif self.task_id == 7:
                self.writer.add_scalar('top-250 score', np.mean(np.array(self.memory["scores"][:250])), step)

            self.writer.add_scalar('mean score in memory', np.mean(np.array(self.memory["scores"])), step)

        samples_all = canonicalize_list(list(self.memory['smiles']))
        scores_all = scoring_function.score_list(samples_all)
        scored_molecules = zip(samples_all, scores_all)
        assert len(samples_all) == len(scores_all)
        sorted_scored_molecules = sorted(scored_molecules, key=lambda x: (x[1], hash(x[0])), reverse=True)
        top_scored_molecules = sorted_scored_molecules[:number_molecules]

        return [x[0] for x in top_scored_molecules]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="gpt")
    parser.add_argument('--task_id', type=int, default=15)
    parser.add_argument('--num_agents', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--sigma1', type=float, default=1000)
    parser.add_argument('--sigma2', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--replay', type=int, default=5)
    parser.add_argument('--sim_penalize', type=bool, default=False)
    parser.add_argument('--sim_thres', type=float, default=0.7)
    parser.add_argument('--prior_path', type=str, default="ckpt/your_pretrained_model.pt")
    parser.add_argument('--vocab_path', type=str, default="data/vocab.txt")
    parser.add_argument('--output_dir', type=str, default="guacamol_log/")
    args = parser.parse_args()
    print(args)

    set_seed(42)

    writer = SummaryWriter(args.output_dir + f"log_task{args.task_id}/{args.num_agents}_{args.model_type}/")
    if not os.path.exists(args.output_dir + f"results_task{args.task_id}"):
        os.makedirs(args.output_dir + f"results_task{args.task_id}")
    writer.add_text("configs", str(args))

    generator = MARL_goal_directed_generator(logger=writer, configs=args)
    assess_goal_directed_generation(generator, 
        json_output_file=args.output_dir + f"results_task{args.task_id}/{args.num_agents}_{args.model_type}.json", 
        task_id=args.task_id)

    writer.close()
    