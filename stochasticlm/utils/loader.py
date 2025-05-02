import torch
import torch.nn as nn
from torch.utils.data import Dataset

class LatticeSeqDataset(Dataset):
    def __init__(self, rule_class, shape=(20,20), seq_length=32, num_samples=1000, device='cpu'):
        """
        rule_class: callable returning an object with __next__() that has .lattice tensor
        shape: tuple (H, W)
        seq_length: number of steps per sample
        num_samples: total samples in dataset
        device: 'cpu' or 'cuda'
        """
        self.rule_class  = rule_class
        self.shape       = shape
        self.seq_length  = seq_length
        self.num_samples = num_samples
        self.device      = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rule = self.rule_class(self.shape, device=self.device)
        frames = []
        for _ in range(self.seq_length + 1):
            sample = next(rule).lattice
            frames.append(sample)
        seq = torch.stack(frames, dim=0)
        seq = seq.long()
        inp, tgt = seq[:-1], seq[1:]
        return inp, tgt
