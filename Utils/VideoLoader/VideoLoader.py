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

    def _getFrames(self, videoCapture, fps):
        count = 1
        success = True
        interval = 1000.0/fps

        frames = []
        
        while success:
            videoCapture.set(cv2.CAP_PROP_POS_MSEC,(count*interval))
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

    def transform(self, filename):
        video = self._getVideo(filename)
        frames = self._getFrames(video, self.fps)
        numpyFrames = self._convertToNumpy(frames)
        return numpyFrames

    def scrape_all_data(self, path):
        video_files = [f for f in os.listdir(path) if f.endswith('.mp4')]
        result_dict = {}
        for video_file in video_files:
            output = self.transform(path + '/' + video_file)
            result_dict[video_file] = output
        return result_dict

videoTransformer = VideoTransformer()
files_frames_dict = videoTransformer.scrape_all_data('../../data')
