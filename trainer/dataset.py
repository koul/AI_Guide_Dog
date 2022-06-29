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
from utils import *

# For ConvLSTM model (or other sequence-based models requiring a sequences of frames as a single training example).
class VideoDataset(Dataset):
    def __init__(self, files, transforms, seq_len, frame_path):
        self.transforms = transforms
        self.files = files
        self.seq_len = seq_len
        self.frame_path = frame_path
        X = []
        y = []
        for f in files:
            df = pd.read_csv(f)
            df_processed = preprocess_labels(df) # assign 1 sec forward labels

            # Generate training sequences
            for i in range(len(df_processed)-self.seq_len):
                X.append(df_processed['frames'][i:i+seq_len].to_numpy())
                y.append(df_processed['labels'][i+seq_len-1])
        
        self.X = np.stack(X, axis = 0)
        self.y = np.array(y)
        
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        seq_filename = self.X[idx]
        video = torch.FloatTensor(self.seq_len, CHANNELS, HEIGHT, WIDTH)
        for e,filename in enumerate(seq_filename):
            try:
                frame = np.load(osp.join(self.frame_path,filename), allow_pickle=True)
                frame = (frame - frame.min())/(frame.max() - frame.min())
                frame = self.transforms(frame)

            except Exception as ex:
                print(ex)
                frame = torch.zeros((CHANNELS, HEIGHT, WIDTH))

            video[e,:,:,:] = frame
          
        return video, torch.LongTensor(self.y[idx])
        
# For CNN model
class FrameDataset(Dataset):
    def __init__(self, x, y, transforms, frame_path):
        self.transforms = transforms
        self.X = x
        self.y = y
        # self.seq_len = seq_len
        self.frame_path = frame_path
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        seq_filename = self.X[idx]
        try:
            frame = np.load(osp.join(self.frame_path,seq_filename), allow_pickle=True)
            frame = (frame - frame.min())/(frame.max() - frame.min())
            frame = self.transforms(frame)
            
        except Exception as ex:
            print("Error occured while loading frame: ", ex)
            frame = torch.zeros((CHANNELS, HEIGHT, WIDTH))
        
        return frame, self.y[idx]