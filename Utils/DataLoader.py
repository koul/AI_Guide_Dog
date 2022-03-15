import numpy as np
import cv2
import os

import VideoLoader.VideoLoader as vl
import SensorLoader.SensorLoader as sl

class VideoTransformer(object):
    def __init__(self, fps=10):
        self.fps = fps
        self.videoTransformer = vl.VideoTransformer(fps)
        self.sensorTransformer = sl.SensorTransformer(fps)
    
    def transform(self, filename, ref_time=-1, secs_in_past = -1, secs_in_future=-1):
        video_frames = self.videoTransformer.transform(filename, ref_time, secs_in_past, secs_in_future)
        sensor_frames = self.sensorTransformer.transform(filename, ref_time)


    def scrape_all_data(self, path):
        pass
