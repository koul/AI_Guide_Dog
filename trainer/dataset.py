from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from config import *
# import ffmpeg
import os
import os.path as osp
from utils import *

# For ConvLSTM model (or other sequence-based models requiring a sequences of frames as a single training example).
class VideoDataset(Dataset):
    def __init__(self, df_videos, files, transforms, seq_len, config_dict=None):
        self.transforms = transforms
        self.files = files
        self.seq_len = seq_len
        self.df_videos = df_videos
        self.config = config_dict
        # self.frame_path = frame_path
        self.X_vid = []
        self.X_index = []
        y = []
        for f in files:

        #     df = pd.read_csv(f)
        #     df_processed = preprocess_labels(df) # assign 1 sec forward labels

        #     # Generate training sequences
            # for i in range(len(df_processed)-self.seq_len):
            #     X.append(df_processed['frames'][i:i+seq_len].to_numpy())
            #     y.append(df_processed['labels'][i+seq_len-1])
            for i in range(20):
                self.X_vid.append(f)
                self.X_index.append(i)
                y.append(2)

        # self.X = np.stack(X, axis = 0)
        # self.X = np.stack(X, axis = 0)
        self.y = np.array(y)
        
        
    def __len__(self):
        return len(self.y)
        # return 1
    
    def __getitem__(self, idx):
        # vid = self.df_videos['sample'][idx:idx+self.seq_len]
        # print(vid.shape)
        # seq_filename = self.X[idx]
        vid_file = self.X_vid[idx]
        vid_idx = self.X_index[idx]

        video = torch.FloatTensor(self.seq_len, self.config['data']['CHANNELS'], self.config['data']['HEIGHT'], self.config['data']['WIDTH'])
        
        # for e,filename in enumerate(seq_filename):
        for i in range(vid_idx, vid_idx+self.seq_len): 
            try:
                # frame = np.load(osp.join(self.frame_path,filename), allow_pickle=True)
                frame = self.df_videos[vid_file][i]
                frame = (frame - frame.min())/(frame.max() - frame.min())
                frame = self.transforms(frame)

            except Exception as ex:
                print(ex)
                frame = torch.zeros((self.config['data']['CHANNELS'], self.config['data']['HEIGHT'], self.config['data']['WIDTH']))
        
            video[i-vid_idx,:,:,:] = frame
          
        # return video
        # return video, torch.LongTensor(self.y[idx])
        return video, self.y[idx]
        
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