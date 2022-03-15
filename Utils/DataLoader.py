import numpy as np
import cv2
import os

import VideoLoader.VideoLoader as vl
import SensorLoader.SensorLoader as sl

class DataTransformer(object):
    def __init__(self, fps=10):
        self.fps = fps
        self.videoTransformer = vl.VideoTransformer(fps)
        self.sensorTransformer = sl.SensorTransformer(fps)
    
    def transform(self, dir_path, ref_time=-1, secs_in_past = -1, secs_in_future=-1):
        video_file = os.path.join(dir_path, dir_path.split('/')[-1]+'.mp4')
        video_frames = self.videoTransformer.transform(video_file, ref_time, secs_in_past, secs_in_future)
        sensor_file = os.path.join(dir_path, dir_path.split('/')[-1]+'.csv')
        sensor_frames = self.sensorTransformer.transform(sensor_file, ref_time, secs_in_past, secs_in_future)
        print(video_frames.shape)
        print(sensor_frames.keys())


    def scrape_all_data(self, path):
        pass

dataTransformer = DataTransformer()
dataTransformer.transform("../data/sample")

