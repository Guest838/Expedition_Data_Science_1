import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import json  
from tools.object_extractor import generate_objects
class FilePathDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_test(filepath, seed):
    li, ae, spor = generate_objects(filepath)

    return (FilePathDataset(li),
            FilePathDataset(ae),
            FilePathDataset(spor))


