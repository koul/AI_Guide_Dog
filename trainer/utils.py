import os
import cv2
import pandas as pd
from config import *

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


def save(path, model, index, optim = False):
    if not os.path.exists(MODELS_PATHS+path):
        os.mkdir(MODELS_PATHS+path)
    if(optim):
        torch.save(model.state_dict(), MODELS_PATHS+path+'/optimizer_params_{:08d}.pth'.format(index))
    else:
        torch.save(model.state_dict(), MODELS_PATHS+'/attempt_8_frames_resnet18_new_data_diff_split'+'/model_params_{:08d}.pth'.format(index))


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