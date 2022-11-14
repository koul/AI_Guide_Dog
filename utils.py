import os
import cv2
import pandas as pd
# from config import *
import random
import os.path as osp
import torch
from torch.utils.data import WeightedRandomSampler
# import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import scipy.stats as ss

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
    ts = int(len(files) * 0.25)
    test_files = files[:ts]
    train_files = files[ts:]
    print("Test files ",test_files)
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
    class_weights = [num_samples/(i+1) for i in dataset_counts] #Adding +1 to avoid division by zero

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
    # cm_df = pd.DataFrame(cm, index = ['0','1','2'], columns = ['0','1','2'])
    # print(cm_df)
    print('\nClassification Report\n')
    print(classification_report(actual, predictions))

def display_classification_report(train_actual, train_predictions, val_actual, val_predictions):
    print('\nTaining set stats\n')
    dcr_helper(train_actual, train_predictions)

    print('\nValidation set stats\n')
    dcr_helper(val_actual, val_predictions)
    return precision_recall_fscore_support(val_actual, val_predictions)


def display_test_classification_report(test_actual, test_predictions):
    print('\nTest set stats\n')
    dcr_helper(test_actual, test_predictions)
    return precision_recall_fscore_support(test_actual, test_predictions)

# Function for processing the videos and labels to get labels at the frame level
# def process_video(video_file, labels):
#     video_filename = video_file.split('/')[-1].split('.')[0]
#     vidcap = cv2.VideoCapture(video_file)

#     ctr = 0
#     video_frames = []
#     video_labels = []
#     frame_ts = []
    
#     # read each frame
#     hasFrames,image = vidcap.read()

#     while (hasFrames):
        
#         # Process the frame and save it to the processed_frames folder
#         save_file_name = video_filename + "_" + str(ctr) + ".npy"
#         image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
#         np.save(osp.join(config_dict['data']['processed_frames'], save_file_name), image)  

#         # Get label corresponding to tht timestamp of the frame in the video
#         label_ts = vidcap.get(cv2.CAP_PROP_POS_MSEC)
#         label_ts = label_ts - (label_ts%100) # adjusting timestamp acc to the 100ms intervals created by the transformer
#         frame_ts.append(label_ts)

#         if(label_ts not in labels.keys()):
#             print(label_ts)
#             hasFrames,image = vidcap.read()
#             continue

#         label = labels[label_ts]
#         video_labels.append(label_map(label))
#         video_frames.append(save_file_name)
#         hasFrames,image = vidcap.read()
#         ctr += 1
        
#     df = pd.DataFrame({'frames': video_frames, 'directions': video_labels, 'timestamp': frame_ts})

#     # save this data frame as a csv into the 
#     df.to_csv(osp.join(config_dict['data']['processed_csvs'],video_filename+".csv"), index=None)

#     print("After processing:")
#     print("Number of frames labelled: ", ctr)
    
# # Main entry point for processing raw video file and the npz file labels
# def preprocess():
#     # load the npz label file from transformer
#     f = np.load(LABEL_FILE, allow_pickle = True)
#     print(f.keys())
#     # for each single video
#     for video_file in get_all_files_from_dir(VID_PATH):
#         video_filename = video_file.split('/')[-1].split('.')[0]
#         print(video_filename)

#         # Check if it has already been processed, i.e. a processed csv exists in the processed folder for this video
#         if(video_filename+".csv" not in os.listdir(DATA_SAVE_PATH)):
#             labels = f[video_filename]['Sensor']['direction_label']['direction']
#             process_video(video_file, labels)
#             print("Finished processing ", video_file)

# # FPS processing raw video file
# def process_videos(vid_path = VID_PATH_OG):
#     fp = get_all_files_from_dir(vid_path, vids=True)
#     print(fp)
#     # for each single video
#     for fl in fp:
#         video_filename = fl.split('/')[-1]
#         # Check if it has already been processed
#         if(video_filename not in os.listdir(VID_PATH)):
#             # change fps
#             ffmpeg.input(fl).filter('fps', fps=10, round='up').output(VID_PATH+video_filename).run() 

# Transformer code ends here
            
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