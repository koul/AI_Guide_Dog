from multiprocessing import Pool

import numpy as np
import cv2
import os


class VideoTransformer(object):
    def __init__(self, fps=10, resolution=[512, 512]):
        self.fps = fps
        self.resolution = resolution

    def _getVideo(self, filename):
        cap = cv2.VideoCapture(filename)
        cap.set(3, self.resolution[0])
        cap.set(4, self.resolution[1])
        fps = int(cap.get(5))
        if fps < self.fps:
            print(
                f"Video captured at lower frame rate than the requested fps. Video FPS - {fps}. Requested FPS - {self.fps}")
        return cap

    # def _resize(self, image):
    #     resized = cv2.resize(image, (1560, 1170), interpolation = cv2.INTER_AREA)
    #     return resized

    def _getFrames(self, videoCapture, fps, ref_time=-1, secs_in_past=-1, secs_in_future=-1):

        count = 1

        if ref_time == -1:
            start_time = 0
            frame_rate = videoCapture.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
            frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
            end_time = (frame_count / frame_rate) * 1000
        else:
            start_time = (ref_time - secs_in_past) * 1000
            end_time = (ref_time + secs_in_future) * 1000

        success = True
        interval = 1000.0 / fps

        frames = []

        while success and start_time + count < end_time:
            videoCapture.set(cv2.CAP_PROP_POS_MSEC, (start_time + count * interval))
            count = count + 1
            success, image = videoCapture.read()
            if success:
                image = cv2.resize(
                    image,
                    (self.resolution[0], self.resolution[1]),
                    interpolation=cv2.INTER_CUBIC
                )

            imageInNumpy = np.array(image)

            if imageInNumpy.shape != (self.resolution[0], self.resolution[1], 3):
                print("Error processing frame: Shape error: ", imageInNumpy.shape)
                continue

            frames.append(imageInNumpy)

        return frames

    def _convertToNumpy(self, frames):
        return np.array(frames)

    def transform(self, filename, ref_time=-1, secs_in_past=-1, secs_in_future=-1):
        video = self._getVideo(filename)
        frames = self._getFrames(video, self.fps, ref_time, secs_in_past, secs_in_future)
        numpyFrames = self._convertToNumpy(frames)
        return numpyFrames

    def scrape_all_data(self, path):
        directories = [f for f in os.listdir(path)]
        file_list = []
        for directory in directories:
            dir_path = os.path.join(path, directory)
            video_files = [f for f in os.listdir(dir_path) if f.endswith('.mp4')]
            video_file = video_files[0]
            file_list.append(dir_path + '/' + video_file)

        pool = Pool(os.cpu_count())
        results = pool.map(self.transform, file_list)

        result_dict = {}
        for idx, file in enumerate(file_list):
            output = results[idx]
            name = file_list[idx].split('/')[-1].split('.')[0]
            result_dict[name] = output

        return result_dict
