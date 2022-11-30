import os
import cv2
import pandas as pd
import random
import os.path as osp
import torch
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import scipy.stats as ss
import torch.nn as nn
import math

# df is pandas dataframe of the form: frame_path, direction, timestamp
# direction is the current direction at the timestamp of the frame.
# We use the following function to create label: direction 1 sec ahead 
def preprocess_labels(df):
    label_indices = df['timestamp'].searchsorted(df['timestamp']+1000)
    viable_indices = label_indices[label_indices!=len(df)]
    df_new = df[label_indices != len(df)].reset_index(drop = True)
    df_new['labels'] = df['directions'][viable_indices].reset_index(drop = True)
    return df_new


def convert_to_dataframe(d):
    df = pd.DataFrame.from_dict(d, orient ='index') 
    df.sort_index(inplace = True)
    df = df.reset_index(drop = False).reset_index(drop = False)
    df.columns = ['frame_index', 'timestamp', 'directions']
    return df

def make_tt_split(files, seed):
    random.Random(seed).shuffle(files)
    split_ratio = 0.25
    ts = int(len(files) * split_ratio)
    test_files = files[:ts]
    train_files = files[ts:]
    return train_files, test_files

def labelCount(label, n_classes):
    label_count = [0]*(n_classes)
    for lab in label:
        label_count[lab] += 1
    return label_count

def sampler_(dataset_labels, n_classes):
    dataset_counts = labelCount(dataset_labels, n_classes)
    print("Label counts before balancing: ", dataset_counts)
    num_samples = sum(dataset_counts) + n_classes
    class_weights = [num_samples/(i+1) for i in dataset_counts] #TODO: Add +1 to avoid division by zero
    weights = [class_weights[y] for y in dataset_labels]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    return sampler

def get_gps_probabilities(gps_range) :
    mid = (gps_range[0]+gps_range[1])/2
    x = np.arange(-mid, mid)
    xU, xL = x + 0.5, x - 0.5 
    prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    return prob


def save(config, model, index, acc, optim = False):
    save_path = os.path.join(config['global']['root_dir'],config['trainer']['model_save_path'], str(config['global']['iteration']))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if(optim):
        torch.save(model.state_dict(), save_path+'/{}_optimizer_params_epoch_{:08d}_acc_{}.pth'.format(config['trainer']['model']['name'], index, acc))
    else:
        torch.save(model.state_dict(), save_path+'/{}_model_params_epoch_{:08d}_acc_{}.pth'.format(config['trainer']['model']['name'], index, acc))

def label_map(lab):
    if(lab == 0):
        return 2
    elif(lab == -1):
        return 0
    else:
        return 1
    
def get_all_files_from_dir(directory, vids = False):
    file_paths = []
    print(directory)
    try:
        for root, dirs, files in os.walk(directory):
            # print(files)
            if(vids):
                file_paths += [os.path.join(root, x,x+".mp4") for x in dirs]
            else:
                file_paths += [os.path.join(root, x) for x in files]
        return sorted(file_paths)
    except Exception as e:
        print(e)

def dcr_helper(actual, predictions, wandb = False):
    cm = confusion_matrix(actual, predictions)
    print(cm)
    print('\nClassification Report\n')
    print(classification_report(actual, predictions))
    return cm

def display_classification_report(train_actual, train_predictions, val_actual, val_predictions):
    print('\nTaining set stats\n')
    dcr_helper(train_actual, train_predictions)

    print('\nValidation set stats\n')
    dcr_helper(val_actual, val_predictions)
    return precision_recall_fscore_support(val_actual, val_predictions)

def display_test_classification_report(test_actual, test_predictions):
    print('\nTest set stats\n')
    cm = dcr_helper(test_actual, test_predictions)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_actual, test_predictions)
    return cm, test_precision, test_recall, test_f1
    
def prep_video_test(filename):
    X = []
    y = []
    
    df = pd.read_csv(filename)
    X.append(df['frames'])
    y.append(df['labels'])   
    
    X = pd.concat(X)    # print(X.head())
    X.reset_index(drop=True,inplace=True)
    X = X.to_numpy()

    
    y = pd.concat(y)
    y.reset_index(drop=True,inplace=True)
    y = y.to_numpy()
    
    return X, y, df

class ConvertEmbedding(nn.Module):
    def __init__(self, inp_size,model_dim):
        super(ConvertEmbedding, self).__init__()
        self.linear_layer = nn.Linear(inp_size, model_dim)
        self.model_dim = model_dim

    def forward(self, x):
        return self.linear_layer(x) * math.sqrt(self.model_dim)

