import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from tdc import Evaluator
from tqdm import tqdm

import threading

from vocabulary import SMILESTokenizer, read_vocabulary

def randomize_smiles(smiles):
    # randomize SMILES for data augmentation
    mol = Chem.MolFromSmiles(smiles)
    ans = list(range(mol.GetNumAtoms()))
    if mol == None or ans == []:
        return smiles
    np.random.shuffle(ans)
    new_mol = Chem.RenumberAtoms(mol, ans)
    return Chem.MolToSmiles(new_mol, canonical=False)

# @torch.no_grad()
def likelihood(model, seqs):
    nll_loss = nn.NLLLoss(reduction="none")
    seqs = seqs.cuda()
    logits, _ = model(seqs[:, :-1])
    log_probs = logits.log_softmax(dim=2)
    return nll_loss(log_probs.transpose(1, 2), seqs[:, 1:]).sum(dim=1)


@torch.no_grad()
def sample_SMILES(model, voc, n_mols=100, block_size=100, temperature=1.0, top_k=10):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    nll_loss = nn.NLLLoss(reduction="none")
    codes = torch.zeros((n_mols, 1), dtype=torch.long).to("cuda")
    codes[:] = voc["^"]
    nlls = torch.zeros(n_mols).to("cuda")

    model.eval()
    for k in range(block_size - 1):
        logits, _ = model(codes)  
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, k=top_k)
        # apply softmax to convert to probabilities
        probs = logits.softmax(dim=-1)
        log_probs = logits.log_softmax(dim=1)
        # sample from the distribution
        code_i = torch.multinomial(probs, num_samples=1)
        # print(probs)
        # append to the sequence and continue
        codes = torch.cat((codes, code_i), dim=1)

        nlls += nll_loss(log_probs, code_i.view(-1))
        if code_i.sum() == 0:
            break

    # codes = codes
    smiles = []
    Tokenizer = SMILESTokenizer()
    for i in range(n_mols):
        tokens_i = voc.decode(np.array(codes[i, :].cpu()))
        smiles_i = Tokenizer.untokenize(tokens_i)
        smiles.append(smiles_i)

    return smiles, codes, nlls
    

def model_validity(model, vocab_path, n_mols=100, block_size=100):
    evaluator = Evaluator(name = 'Validity')
    voc = read_vocabulary(vocab_path)
    smiles, _, _ = sample_SMILES(model, voc=voc, n_mols=n_mols, block_size=block_size, top_k=10)
    return evaluator(smiles)


def calc_fingerprints(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, radius = 2, nBits = 2048) for x in mols]
    smiles_canonicalized = [Chem.MolToSmiles(x, isomericSmiles=False) for x in mols]
    return fps, smiles_canonicalized

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)