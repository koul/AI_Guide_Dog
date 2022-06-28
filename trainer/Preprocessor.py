#unused file

import os
import os.path as osp
import cv2
from utils import *
from config import *
import numpy as np
import ffmpeg
from preprocessing import *

# Wrapper class for pre-processing functions
class Preprocessor:

    def process_video(video_file, labels):
        video_filename = video_file.split('/')[-1].split('.')[0]
        vidcap = cv2.VideoCapture(video_file)

        ctr = 0
        video_frames = []
        video_labels = []
        
        hasFrames,image = vidcap.read()

        while (hasFrames):
            save_file_name = video_filename + "_" + str(ctr) + ".npy"
            np.save(osp.join(PROCESSED_PATH, save_file_name), image)  
            label_ts = vidcap.get(cv2.CAP_PROP_POS_MSEC) + 1000 #take 1 sec ahead labels 
            label_ts = label_ts - (label_ts%100)
            if(label_ts not in labels.keys()):
                print(label_ts)
                hasFrames,image = vidcap.read()
                continue
            label = labels[label_ts]
            video_labels.append(label_map(label))
            video_frames.append(save_file_name)
            hasFrames,image = vidcap.read()
            ctr += 1
            
        df = pd.DataFrame({'frames': video_frames, 'labels': video_labels})
        df.to_csv(osp.join(DATA_SAVE_PATH,video_filename+".csv"), index=None)

        print("After processing:")
        print("Number of frames labelled: ", ctr)
        
    def preprocess(video_ids):
        f = np.load(LABEL_FILE, allow_pickle = True)
        print(f.keys())
        for video_filename in video_ids:
            # video_filename = video_file.split('/')[-1].split('.')[0]
            print(video_filename)
            if(video_filename+".csv" not in os.listdir(DATA_SAVE_PATH)):
                labels = f[video_filename]['Sensor']['direction_label']['direction']
                process_video(osp.join(VID_PATH,video_filename+".mp4"), labels)
                print("Finished processing ", video_filename+".mp4")
            
    def process_raw_videos(video_ids):
        fp = get_all_files_from_dir(VID_PATH_OG, vids=True)
        fp = [f for f in fp if f.split('/')[-1].split('.')[0] in video_ids]
        print(fp)
        for fl in fp:
            video_filename = fl.split('/')[-1]
            if(video_filename not in os.listdir(VID_PATH)):
                ffmpeg.input(fl).filter('fps', fps=10, round='up').output(VID_PATH+video_filename).run() 


    def preprocessing_pipeline(video_ids):
        process_raw_videos(video_ids)
        preprocess(video_ids)