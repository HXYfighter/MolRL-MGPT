import torch
import numpy as np
from tqdm import tqdm

from utils import randomize_smiles

class Dataset(torch.utils.data.Dataset):
    # Custom PyTorch Dataset for SMILES

    def __init__(self, smiles_list, vocabulary, tokenizer, aug_prob=0, preprocess=False):
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._smiles_list = list(smiles_list)
        self._aug_prob = aug_prob
        
        if preprocess: # preprocess - remove the smiles with unknown tokens
            remove_list = []
            for s in tqdm(self._smiles_list):
                tokens = self._tokenizer.tokenize(s)
                encoded = self._vocabulary.encode(tokens)
                if encoded[0] == -1:
                    remove_list.append(s)
            for s in tqdm(remove_list):
                self._smiles_list.remove(s)

    def __getitem__(self, i):
        smi = self._smiles_list[i]

        p = np.random.uniform() # data augmentation
        if p < self._aug_prob:
            smi = randomize_smiles(smi)

        tokens = self._tokenizer.tokenize(smi)
        encoded = self._vocabulary.encode(tokens)
        return encoded[:-1], encoded[1:]

    def __len__(self):
        return len(self._smiles_list)

    @staticmethod
    def collate_fn(encoded_seqs):
        # Converts a list of encoded sequences into a padded tensor
        max_length = max([len(seq[0]) for seq in encoded_seqs])

        collated_arr_x = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
        collated_arr_y = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)
        for i, seq in enumerate(encoded_seqs):
            collated_arr_x[i, :len(seq[0])] = torch.as_tensor(seq[0], dtype=torch.long)
            collated_arr_y[i, :len(seq[1])] = torch.as_tensor(seq[1], dtype=torch.long)
        # collated_arr = torch.tensor(collated_arr.to, dtype=torch.long)
        return collated_arr_x, collated_arr_y