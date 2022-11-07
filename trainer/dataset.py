from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp
from utils import *
import pdb
import scipy.stats as ss

from transformers import ViTForImageClassification, ViTFeatureExtractor

from sklearn import preprocessing
import itertools

# For ConvLSTM model (or other sequence-based models requiring a sequences of sensor data as a single training example).
class SensorDataset(Dataset):
    def __init__(self, df_videos, df_sensor, files, seq_len, sensor_attr_list, config_dict=None):

        self.files = files
        self.seq_len = seq_len
        self.sensor_feat_len = len(sensor_attr_list)
        self.df_videos = df_videos
        self.df_sensor = df_sensor  # df_sensor['sample']['direction_label']['direction']
        self.config = config_dict
        self.attr_list = sensor_attr_list
        self.X_vid = []
        self.X_index = []
        self.X_sensor_timestamp = []
        y = []

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        for f in files:
            df = convert_to_dataframe(self.df_sensor[f]['direction_label']['direction'])
            df_processed = preprocess_labels(df)  # assign 1 sec forward labels

            # Generate training sequences
            # for each time stamp in df_processed
            for i in range(len(df_processed) - self.seq_len):
                self.X_vid.append(f)
                self.X_index.append(df_processed['frame_index'][i])
                self.X_sensor_timestamp.append(df_processed['timestamp'][i])

                # Picking the label of the last element of the sequence
                y.append(label_map(df_processed['labels'][i + self.seq_len - 1]))

            
        self.y = np.array(y)

        # TODO: later apply normalization for the entire dataframe in SensorTransformer coz this sees only 8 instances of values
        # normalize feature
        #sensor = (sensor - sensor.min()) / (sensor.max() - sensor.min())

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        vid_file = self.X_vid[idx]
        vid_idx = self.X_index[idx]

        sensor = torch.FloatTensor(self.seq_len, self.sensor_feat_len)

        for i in range(vid_idx, vid_idx + self.seq_len):
            try:
                frame_sensor = torch.FloatTensor([self.df_sensor[vid_file][str(attr).split(':')[0]][str(attr).split(':')[1]][self.X_sensor_timestamp[i]] for attr in self.attr_list])
            except Exception as ex:
                print(ex)
                frame_sensor = torch.zeros(len(self.attr_list))

            sensor[i - vid_idx, :] = frame_sensor # tensor of seq_len * attr_list

        # TODO: later apply normalization for the entire dataframe in SensorTransformer coz this sees only 8 instances of values
        # normalize feature
        sensor = (sensor - sensor.min()) / (sensor.max() - sensor.min())

        #TODO: inspect what this sensor object contains
        return sensor, self.y[idx]


# For ConvLSTM model (or other sequence-based models requiring a sequences of frames as a single training example).
class VideoDataset(Dataset):
    def __init__(self, df_videos, df_sensor, files, transforms, seq_len, config_dict=None):
        self.transforms = transforms
        self.files = files
        self.seq_len = seq_len
        self.df_videos = df_videos
        self.df_sensor = df_sensor #df_sensor['sample']['direction_label']['direction']
        self.config = config_dict
        # self.frame_path = frame_path
        self.X_vid = []
        self.X_index = []
        y = []
        for f in files:
            df = convert_to_dataframe(self.df_sensor[f]['direction_label']['direction'])
            df_processed = preprocess_labels(df) # assign 1 sec forward labels
            # pdb.set_trace()

            # Generate training sequences
            for i in range(len(df_processed)-self.seq_len):
                self.X_vid.append(f)
                self.X_index.append(df_processed['frame_index'][i])

                # Picking the label of the last element of the sequence
                y.append(label_map(df_processed['labels'][i+self.seq_len-1]))
              
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
                # frame = (frame - frame.min())/(frame.max() - frame.min())
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

        
# For ConvLSTM model (or other sequence-based models requiring a sequences of frames as a single training example) along with high-level GPS information.
class IntentVideoDataset(Dataset):
    def __init__(self, df_videos, df_sensor, files, transforms, seq_len, config_dict=None, test=False):
        self.transforms = transforms
        self.files = files
        self.seq_len = seq_len
        self.df_videos = df_videos
        self.df_sensor = df_sensor #df_sensor['sample']['direction_label']['direction']
        self.config = config_dict
        # self.frame_path = frame_path
        self.X_vid = []
        self.X_index = []
        y = []

        # input sequence of seq_len frames at config_dict['transformer']['fps'] fps
        #label is 1-second future direction
        #we can have a gps signal where between 2 to 6 seconds (inclusive) before the actual turn (1-5 seconds before the label timestamp) with probabilities that are normally distributed.
        self.gps_range = (0, seq_len - self.config['transformer']['fps'])
        self.prob_gps = get_gps_probabilities(self.gps_range) # (0, 20-4 = 16)

        
        for f in files:
            df = convert_to_dataframe(self.df_sensor[f]['direction_label']['direction'])
            df_processed = preprocess_labels(df) # assign 1 sec forward labels
            # pdb.set_trace()

            # Generate training sequences
            for i in range(len(df_processed)-self.seq_len):
                self.X_vid.append(f)
                self.X_index.append(df_processed['frame_index'][i])

                # Picking the label of the last element of the sequence
                y.append(label_map(df_processed['labels'][i+self.seq_len-1]))
              
        # self.X = np.stack(X, axis = 0)
        self.y = np.array(y)
        self.test = test
        if(test):
            if(config_dict['data']['VAL_INTENT'] == ""):
                self.intent_positions = []
                for i in self.y:
                    if(i!=2):
                        self.intent_positions.append(np.random.choice(np.arange(self.gps_range[0], self.gps_range[1]), p = self.prob_gps)) #intent start position
                    else:
                        self.intent_positions.append(-1)
                
                np.save('validation_intent.npy', np.array(self.intent_positions))
            else:
                self.intent_positions = list(np.load(config_dict['data']['VAL_INTENT']))
   
    def __len__(self):
        return len(self.y)
        # return 1
    
    def __getitem__(self, idx):
        # vid = self.df_videos['sample'][idx:idx+self.seq_len]
        # print(vid.shape)
        # seq_filename = self.X[idx]
        vid_file = self.X_vid[idx]
        vid_idx = self.X_index[idx]

        #+1 for intent channels
        video = torch.FloatTensor(self.seq_len, self.config['data']['CHANNELS']+1, self.config['data']['HEIGHT'], self.config['data']['WIDTH'])

        if(self.test):
            intent = self.intent_positions[idx]
        else:
            if(self.y[idx] != 2): # it is not front label
                intent = np.random.choice(np.arange(self.gps_range[0], self.gps_range[1]), p = self.prob_gps) #intent start position
            else:
                intent = -1 # none (2) intent



        # for e,filename in enumerate(seq_filename):
        for i in range(vid_idx, vid_idx+self.seq_len): 
            try:
                # frame = np.load(osp.join(self.frame_path,filename), allow_pickle=True)
                frame = self.df_videos[vid_file][i]
                # frame = (frame - frame.min())/(frame.max() - frame.min())
                frame = self.transforms(frame)
                if(intent!=-1 and i>=intent and i<intent+self.config['transformer']['fps']):  #pass intent vector for just 1 second
                    intent_tensor = torch.full((1, self.config['data']['HEIGHT'], self.config['data']['WIDTH']), self.y[idx])
                else:
                    intent_tensor = torch.full((1, self.config['data']['HEIGHT'], self.config['data']['WIDTH']), 2)  # no context, just go straight/front


            except Exception as ex:
                print("Error reading frame", ex)
                frame = torch.zeros((self.config['data']['CHANNELS'], self.config['data']['HEIGHT'], self.config['data']['WIDTH']))
                intent_tensor = torch.full((1, self.config['data']['HEIGHT'], self.config['data']['WIDTH']), 2)
        
            context_frame = torch.cat((frame, intent_tensor), dim = 0) #attach intent as last channel
            video[i-vid_idx,:,:,:] = frame
          
        # return video
        # return video, torch.LongTensor(self.y[idx])
        return video, self.y[idx]

# this class return multi modal dataset with sensor + video fused together
# For ConvLSTM model (or other sequence-based models requiring a sequences of frames as a single training example).
class SensorVideoDataset(Dataset):
    def __init__(self, df_videos, df_sensor, files, attr_list, transforms, seq_len, dense_frame_len, sensor_attr_list, config_dict=None):

        self.transforms = transforms
        self.files = files
        self.seq_len = seq_len
        self.dense_frame_len = dense_frame_len
        self.sensor_feat_len = len(sensor_attr_list)
        self.df_videos = df_videos
        self.df_sensor = df_sensor  # df_sensor['sample']['direction_label']['direction']
        self.config = config_dict
        self.attr_list = sensor_attr_list
        # self.frame_path = frame_path
        self.X_vid = []
        self.X_index = []
        self.X_sensor_timestamp = []
        y = []

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        #self.vit_model = self.vit_model.to(self.device)
        self.vit_model.eval()
        self.vit_model.to(self.device)

        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')


        for f in files:
            df = convert_to_dataframe(self.df_sensor[f]['direction_label']['direction'])
            df_processed = preprocess_labels(df)  # assign 1 sec forward labels
            # pdb.set_trace()

            # Generate training sequences
            # for each time stamp in df_processed
            for i in range(len(df_processed) - self.seq_len):
                self.X_vid.append(f)
                self.X_index.append(df_processed['frame_index'][i])
                self.X_sensor_timestamp.append(df_processed['timestamp'][i])

                # Picking the label of the last element of the sequence
                y.append(label_map(df_processed['labels'][i + self.seq_len - 1]))

                

        # self.X = np.stack(X, axis = 0)
        self.y = np.array(y)

        #normalize sensor data set for multimodal fusion
        #create a new df with just the sensor values that are need with timestamp as index
        #normalize it using sklearn.preprocessing.minmaxscaler

        # TODO: later apply normalization for the entire dataframe in SensorTransformer coz this sees only 8 instances of values
        # normalize feature
        #sensor = (sensor - sensor.min()) / (sensor.max() - sensor.min())

    def __len__(self):
        return len(self.y)
        # return 1

    def __getitem__(self, idx):
        # vid = self.df_videos['sample'][idx:idx+self.seq_len]
        # print(vid.shape)
        # seq_filename = self.X[idx]
        vid_file = self.X_vid[idx]
        vid_idx = self.X_index[idx]

        sensor = torch.FloatTensor(self.seq_len, self.sensor_feat_len)

        for i in range(vid_idx, vid_idx + self.seq_len):
            try:
                frame_sensor = torch.FloatTensor([self.df_sensor[vid_file][str(attr).split(':')[0]][str(attr).split(':')[1]][self.X_sensor_timestamp[i]] for attr in self.attr_list])
            except Exception as ex:
                print(ex)
                frame_sensor = torch.zeros(len(self.attr_list))

            sensor[i - vid_idx, :] = frame_sensor # tensor of seq_len * attr_list


        video = torch.FloatTensor(self.seq_len, self.dense_frame_len)
        # 1000 is size od dense data representation, change it accordingly later on

        # TODO: later apply normalization for the entire dataframe in SensorTransformer coz this sees only 8 instances of values
        # normalize feature
        sensor = (sensor - sensor.min()) / (sensor.max() - sensor.min())

        #TODO: inspect what this sensor object contains

        # for e,filename in enumerate(seq_filename):
        for i in range(vid_idx, vid_idx + self.seq_len):
            try:
                # frame = np.load(osp.join(self.frame_path,filename), allow_pickle=True)
                frame = self.df_videos[vid_file][i]
                # frame = (frame - frame.min())/(frame.max() - frame.min())
                frame = self.transforms(frame)

                #convert this frame to 1*1000 after passing it through VIT

            except Exception as ex:
                print(ex)
                frame = torch.zeros(
                    (self.config['data']['CHANNELS'], self.config['data']['HEIGHT'], self.config['data']['WIDTH']))

            #print("Reached here ------")

            encoding = self.feature_extractor(images=frame, return_tensors="pt")

            pixel_values = encoding['pixel_values'].to(self.device)

            with torch.no_grad():
                outputs = self.vit_model(pixel_values)
                logits = outputs.logits

            #outputs = self.vit_model(frame.unsqueeze(0))

            #print("Reached after vit pass ------")

            video[i - vid_idx, :] = logits # here we instead want 8*1000

        #finally return sensor + video merged together along an axis [8*10003]

        # return video
        # return video, torch.LongTensor(self.y[idx])
        fused_data = torch.cat((video,sensor), 1) # size (seq_len, (dense_video_dim + sensor_dim))
        
        return fused_data, self.y[idx]
        # return (video, sensor), self.y[idx]