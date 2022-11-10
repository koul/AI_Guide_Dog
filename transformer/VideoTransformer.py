import numpy as np
import os
from multiprocessing import Pool
import ffmpeg

class VideoTransformer(object):
    def __init__(self, fps = 10, resolution = [512,512], channels = 3):
        self.fps = fps
        self.resolution = resolution
        self.channels =channels

    def _getVideo(self, filename):
        video_stream = ffmpeg.input(filename)
        return video_stream

    def _getFrames(self, videoCapture, fps, ref_time=-1, secs_in_past = -1, secs_in_future=-1):
        
        if ref_time != -1 and secs_in_past != -1 and secs_in_future != -1:
            videoCapture = videoCapture.trim(start = ref_time - secs_in_past, duration = secs_in_past + secs_in_future).filter('setpts', 'PTS-STARTPTS')
        
        videoCapture = videoCapture.filter('fps', fps = fps, round = 'up').filter('scale', w = self.resolution[0], h = self.resolution[1])
        return videoCapture

    def _convertToNumpy(self, frames):
        pix_format = 'rgb24'
        if self.channels == 1:
            pix_format = 'gray'
        elif self.channels == 3:
            pix_format = 'rgb24'
        else:
            print("Incorrect number of channels")
        out, err = (
            frames
            .output('pipe:', format='rawvideo', pix_fmt=pix_format)
            .run(capture_stdout=True)
        )
        print("Error: ", err)
        video = np.frombuffer(out, np.uint8).reshape([-1, self.resolution[0], self.resolution[1], self.channels])[1:,:,:,:]
        return video

    def transform(self, filename, ref_time=-1, secs_in_past = -1, secs_in_future=-1):
        video = self._getVideo(filename)
        frames = self._getFrames(video, self.fps, ref_time, secs_in_past, secs_in_future)
        numpyFrames = self._convertToNumpy(frames)
        return numpyFrames

    def scrape_all_data(self, path):
        directories = [f for f in os.listdir(path)]
        file_list = []
        for directory in directories:
            if directory.startswith("."):  # to ignore the temp files that get created like ".DS_Store"
                continue
            dir_path = os.path.join(path, directory)
            video_files = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
            print(video_files)
            if(len(video_files) == 0):
                continue
            video_file = video_files[0]
            file_list.append(dir_path + '/' + video_file)

        pool = Pool(os.cpu_count())
        results = pool.map(self.transform, file_list)

        result_dict = {}
        for idx, file in enumerate(file_list):
            output = results[idx]
            name = file_list[idx].split('/')[-1].split('.')[0]
            result_dict[name] = output
            # import cv2
            # cv2.imwrite('myImage.png', output[0])

        return result_dict
