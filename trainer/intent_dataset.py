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
import pdb
from torch.utils.data import Sampler
from collections import Counter
import scipy.stats as ss
import random


class NewIntentVideoDataset(Dataset):
    def __init__(self, df_videos, df_sensor, files, transforms, seq_len, config_dict=None):
        self.transforms = transforms
        
        self.files = files # files sorted by names
        self.seq_len = seq_len
        self.df_videos = df_videos
        self.df_sensor = df_sensor #df_sensor['sample']['direction_label']['direction']
        self.config = config_dict

        self.items = []

        same_count = 0
        diff_count = 0
        turn_diff_count = 0
        for f in files:
            df_processed = convert_to_dataframe(self.df_sensor[f]['direction_label']['direction']) # assign 1 sec forward labels
            # pdb.set_trace()
            df_processed['labels'] = df_processed['labels'].apply(label_map)
            # Generate training sequences
            for i in range(len(df_processed)-self.seq_len):
                index_item = df_processed['frame_index'][i]
                y_list = list(df_processed['labels'][i: i+self.seq_len])
                
                if(not NewIntentVideoDataset.checkAllSame(y_list)):
                    for x in range(3, 10):
                        intent_list = [-1 for _ in range(self.seq_len)]
                        for inx in range(0, self.seq_len - x):
                            if(y_list[inx + x]!=2 and y_list[inx + x]!=y_list[inx] and ((inx+x+1)>= len(df_processed) or y_list[inx + x]==df_processed['labels'][i + inx + x + 1])):
                                if(y_list[inx + x] != y_list[inx + x-1]):
                                    intent_list[inx] = y_list[inx + x]
                                    turn_diff_count +=1 
                        self.items.append((y_list, f, index_item, intent_list))   
                        diff_count += 1
                else:
                    self.items.append((y_list, f, index_item, [-1 for _ in range(self.seq_len)]))
                    same_count += 1

        # print(self.items)
        print("Same Count", same_count)
        print("Diff Count", diff_count)
        print("Turn Diff Count", turn_diff_count)
        
        
   
    def checkAllSame(lst):
        return len(set(lst)) == 1
    
    
    def __len__(self):
        return len(self.items)
      
    
    def __getitem__(self, idx):
       
        labels = self.items[idx][0]
        intents = self.items[idx][3]
        vid_file = self.items[idx][1]
        vid_idx = self.items[idx][2]

        #+1 for intent channels
        video = torch.FloatTensor(self.seq_len, self.config['data']['CHANNELS']+1, self.config['data']['HEIGHT'], self.config['data']['WIDTH'])

        for it in range(self.seq_len): 
            try:
                # frame = np.load(osp.join(self.frame_path,filename), allow_pickle=True)
                frame = self.df_videos[vid_file][it + vid_idx]
                # frame = (frame - frame.min())/(frame.max() - frame.min())
                frame = self.transforms(frame)
                intent_tensor = torch.full((1, self.config['data']['HEIGHT'], self.config['data']['WIDTH']), intents[it])  # no context signal


            except Exception as ex:
                print("Error reading frame", ex)
                frame = torch.zeros((self.config['data']['CHANNELS'], self.config['data']['HEIGHT'], self.config['data']['WIDTH']))
                intent_tensor = torch.full((1, self.config['data']['HEIGHT'], self.config['data']['WIDTH']), -1)
        

            context_frame = torch.cat((frame, intent_tensor), dim = 0) #attach intent as last channel
            # print(context_frame.shape)
            video[it,:,:,:] = context_frame
            
        
        # return video
        # return video, torch.LongTensor(self.y[idx])
        return video, torch.LongTensor(labels)