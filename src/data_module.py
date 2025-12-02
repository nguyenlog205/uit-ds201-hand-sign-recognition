from torch.utils.data import Dataset
from typing import List
import os
from pathlib import Path
import pandas as pd

DATA_DIR = "./data"

class CustomDataset(Dataset):
    def __init__(
        self,
        data_dir:str,
    ):
        """
        The dataset MUST FOLLOW the following structure:
        ```
            data_dir/
            ├─ raw_video/
            │   ├─ video_001.mp4
            │   ├─ video_002.mp4
            │   └─ ...
            └─ metadata/
                ├─ video_001.txt
                ├─ video_002.txt
                └─ ...
        ```

        Args:
            `data_dir` (str): The directory of the dataset.
        """
        super(CustomDataset, self).__init__()
        
        self.data_dir = Path(data_dir)
        self.video_dir = self.data_dir / "raw_video"
        self.metadata_dir = self.data_dir / "metadata"

        if not self.metadata_dir.exists():
            print(f"!!! CRITICAL ERROR !!!")
            print(f"Looking for metadata at: {self.metadata_dir}")
            print(f"But this folder does not exist. Check your DATA_DIR path.")
            self.metadata_paths = []
        else:
            self.metadata_paths = sorted(list(self.metadata_dir.glob("*.txt")))
        self.l2i = {}
        self.i2l = {}

dataset = CustomDataset(
    data_dir=DATA_DIR
)

print(dataset.metadata_paths)
print(dataset.l2i)
print(dataset.i2l)
print(dataset.video_dir)
