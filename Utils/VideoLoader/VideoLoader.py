import numpy as np
import cv2
import os

class VideoTransformer(object):
    def __init__(self, fps = 10, resolution = '1080p'):
        self.fps = fps
        self.resolution = resolution

    def _getVideo(self, filename):
        cap = cv2.VideoCapture(filename)
        fps = int(cap.get(5))
        if fps < self.fps:
            print(f"Video captured at lower frame rate than the requested fps. Video FPS - {fps}. Requested FPS - {self.fps}")
        return cap
    
    # def _resize(self, image):
    #     resized = cv2.resize(image, (1560, 1170), interpolation = cv2.INTER_AREA)
    #     return resized

    def _getFrames(self, videoCapture, fps, ref_time=-1, secs_in_past = -1, secs_in_future=-1):
        
        count = 1
        if ref_time == -1:
            end_time = 10000000
            start_time = 0
        else:
            start_time = (ref_time - secs_in_past) * 1000
            end_time = (ref_time + secs_in_future) * 1000
        
        
        success = True
        interval = 1000.0/fps

        frames = []
        
        while success and start_time + count<end_time:
            videoCapture.set(cv2.CAP_PROP_POS_MSEC,(start_time + count*interval))
            count = count + 1
            success,image = videoCapture.read()
            
            imageInNumpy = np.array(image)
            
            if imageInNumpy.shape != (1560, 1170, 3):
                continue
            
            frames.append(imageInNumpy)
            
        if not success:
            # throw an exception
            pass
        
        return frames

    def _convertToNumpy(self, frames):
        return np.array(frames)

    def transform(self, filename, ref_time=-1, secs_in_past = -1, secs_in_future=-1):
        video = self._getVideo(filename)
        frames = self._getFrames(video, self.fps, ref_time, secs_in_past, secs_in_future)
        numpyFrames = self._convertToNumpy(frames)
        return numpyFrames

    def scrape_all_data(self, path):
        directories = [f for f in os.listdir(path)]
        for directory in directories:
            dir_path = os.path.join(path, directory)
            video_files = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
            result_dict = {}
            for video_file in video_files:
                output = self.transform(dir_path + '/' + video_file)
                name = video_file.split('.')[0]
                result_dict[name] = output
        return result_dict

videoTransformer = VideoTransformer()
