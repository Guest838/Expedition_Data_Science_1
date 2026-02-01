import pandas as pd
from sklearn.model_selection import train_test_split
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

def create_train_test(filepath, seed):
    li, ae, spor = generate_objects(filepath)
    train_files_li, test_files_li, = train_test_split(
        li,
        test_size=0.2,
        shuffle=True, random_state = seed
    )
    train_files_ae, test_files_ae, = train_test_split(
        ae,
        test_size=0.2,
        shuffle=True, random_state = seed
    )
    train_files_spor, test_files_spor, = train_test_split(
        spor,
        test_size=0.2,
        shuffle=True, random_state = seed
    )

    return (FilePathDataset(train_files_li), FilePathDataset(test_files_li), 
            FilePathDataset(train_files_ae), FilePathDataset(test_files_ae),
            FilePathDataset(train_files_spor), FilePathDataset(test_files_spor))



