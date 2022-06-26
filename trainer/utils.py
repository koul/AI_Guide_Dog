import os
import cv2
import pandas as pd
from config import *
from random import random
import os.path as osp
import torch

# df is pandas dataframe of the form: frame_path, direction, timestamp
# direction is the current direction at the timestamp of the frame.
# We use the following function to create label: direction 1 sec ahead 
def preprocess_labels(df):
    label_indices = df['timestamp'].searchsorted(df['timestamp']+1000)
    viable_indices = label_indices[label_indices!=len(df)]
    df_new = df[label_indices != len(df)].reset_index(drop = True)
    df_new['labels'] = df['directions'][viable_indices].reset_index(drop = True)
    return df_new
    
def make_tt_split(data_folder):
    X = []
    y = []
    files = []
    for filename in os.listdir(data_folder):
        if(filename[-3:]=="csv"):
            files.append(osp.join(data_folder,filename))
    
    random.shuffle(files)
    
    ts = int(len(files) * 0.25)
    test_files = files[:ts]
    train_files = files[ts:]
    print("Test files ",test_files)
    return train_files, test_files

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


def save(config, model, index, acc, optim = False):
    save_path = os.path.join(config['global']['root_dir'],config['trainer']['model_save_path'], str(config['global']['iteration']))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if(optim):
        torch.save(model.state_dict(), save_path+'/{}_optimizer_params_epoch_{:08d}_acc_{}.pth'.format(config['trainer']['model']['name'], index, acc))
    else:
        torch.save(model.state_dict(), save_path+'/attempt_8_frames_resnet18_new_data_diff_split'+'/{}_model_params_epoch_{:08d}_acc_{}.pth'.format(config['trainer']['model']['name'], index, acc))


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