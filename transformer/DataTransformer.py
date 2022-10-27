import numpy as np
import cv2
import os
import yaml


from transformer.VideoTransformer import VideoTransformer
from transformer.SensorTransformer import SensorTransformer


class DataTransformer(object):
    def __init__(self, fps=10, resolution = [512,512], channels = 3):
        self.fps = fps
        self.videoTransformer = VideoTransformer(fps, resolution=resolution, channels = channels)
        self.sensorTransformer = SensorTransformer(fps)

    def transform(self, dir_path, ref_time=-1, secs_in_past = -1, secs_in_future=-1):
        video_file = os.path.join(dir_path, dir_path.split('/')[-1]+'.mp4')
        video_frames = self.videoTransformer.transform(video_file, ref_time, secs_in_past, secs_in_future)
        sensor_file = os.path.join(dir_path, dir_path.split('/')[-1]+'.csv')
        sensor_frames = self.sensorTransformer.transform(sensor_file, ref_time, secs_in_past, secs_in_future)
        print(video_frames.shape)
        print(sensor_frames.keys())


    def scrape_all_data(self, path):
        video_data = self.videoTransformer.scrape_all_data(path)
        sensor_data = self.sensorTransformer.scrape_all_data(path)
        # result_dict = {}
        # for key in video_data.keys():
        #     result_dict[key] = {'Video': video_data[key], 'Sensor': sensor_data[key]}
        # return result_dict
        return video_data, sensor_data