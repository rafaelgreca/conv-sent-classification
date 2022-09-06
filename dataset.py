import pandas as pd
from torch.utils.data import Dataset
from typing import Dict

class DatasetDL(Dataset):
    def __init__(self,
                 data: pd.DataFrame) -> None:
        self.df = data
    
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self,
                    idx: int) -> Dict:
        return {"X": self.df["X"][idx], "y": self.df["y"][idx]}