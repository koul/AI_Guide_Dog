from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
# import ffmpeg
import os
import os.path as osp

class FrameDataset(Dataset):
    def __init__(self, x, y, transforms, base_path):
        self.transforms = transforms
        self.X = x
        self.y = y
        # self.seq_len = seq_len
        self.base_path = base_path
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        seq_filename = self.X[idx]
        try:
            frame = np.load(osp.join(self.base_path,seq_filename), allow_pickle=True)
            frame = (frame - frame.min())/(frame.max() - frame.min())
            frame = self.transforms(frame)
            
        except Exception as ex:
            print("Error occured while loading frame: ", ex)
            frame = torch.zeros((CHANNELS, HEIGHT, WIDTH))
        
        return frame, self.y[idx]