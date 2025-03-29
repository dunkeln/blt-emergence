from torch.utils.data import Dataset, DataLoader
import polars as pl

class LatticeDataset(Dataset):
    def __init__(self, model) -> None:
        self.df = pl.with_columns("past_states", "future_states")
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
