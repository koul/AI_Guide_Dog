import pandas as pd
import numpy as np
import os


#import labeler.Labeler as L
from labeler.Labeler import Labeler

'''
To work:
- When all sensor values are -1, provide default values to get for the entire duration
- Handle the case where fps > frames available in dtaframe
'''

class SensorTransformer(object):
    def __init__(self, fps=10):
        self.fps = fps
        self.columns = ['Timestamp (ms)', 'Triggering Sensor', 'Latitude', 'Longitude',
                   'Altitude', 'Heading (degrees)', 'Heading accuracy (degrees)',
                   'Relative Altitude (m)', 'Pressure (kPa)', 'Speed (m/s)',
                   'Speed Accuracy (m/s)', 'Floor', 'Acceleration - X axis',
                   'Acceleration - Y axis', 'Acceleration - Z axis', 'Gyrometer - X axis',
                   'Gyrometer - Y axis', 'Gyrometer - Z axis', 'Magnetometer - X axis',
                   'Magnetometer - Y axis', 'Magnetometer - Z axis',
                   'User acceleration - X axis', 'User acceleration - Y axis',
                   'User acceleration - Z axis', 'Attitude Pitch', 'Attitude Roll',
                   'Attitude Yaw', 'Gravity - X axis', 'Gravity - Y axis',
                   'Gravity - Z axis', 'Rotation Rate - X axis', 'Rotation Rate - Y axis',
                   'Rotation Rate - Z axis', 'Activity Type', 'Number of Steps',
                   'Distance', 'Floors Ascended', 'Floors Descended', 'Current Pace',
                   'Current Cadence', 'Average Active Pace']
        self.sensor_types = ["Gyrometer", "Magnetometer", "Accelerometer", "Device Motion", "GPS",
                        "Altimeter", "Activity Manager"]
        self.senor_to_metric_dict = {
            "Magnetometer": ['Magnetometer - X axis', 'Magnetometer - Y axis', 'Magnetometer - Z axis'],
            "Gyrometer": ['Gyrometer - X axis', 'Gyrometer - Y axis', 'Gyrometer - Z axis'],
            "Accelerometer": ['Acceleration - X axis', 'Acceleration - Y axis', 'Acceleration - Z axis',
                              'User acceleration - X axis', 'User acceleration - Y axis', 'User acceleration - Z axis'],
            "Altimeter": ['Altitude', 'Relative Altitude (m)', 'Pressure (kPa)', 'Attitude Pitch', 'Attitude Roll',
                          'Attitude Yaw'],
            "Device Motion": ['Speed (m/s)', 'Speed Accuracy (m/s)', 'Floor', 'Gravity - X axis', 'Gravity - Y axis',
                              'Gravity - Z axis', 'Rotation Rate - X axis', 'Rotation Rate - Y axis',
                              'Rotation Rate - Z axis'],
            "GPS": ['Latitude', 'Longitude', 'Heading (degrees)', 'Heading accuracy (degrees)'],
            "Activity Manager": ['Activity Type', 'Number of Steps', 'Distance', 'Floors Ascended', 'Floors Descended',
                                 'Current Pace', 'Current Cadence', 'Average Active Pace']}

        self.labeler = Labeler()

    def transform(self, filename, ref_time=-1, secs_in_past = -1, secs_in_future=-1):
        df = self.labeler.add_direction(filename)
        df.fillna(method='bfill', inplace=True)  # backfill the data to account for missing values
        df.dropna(axis=1, how='all', inplace=True)  # drop a metric if it has no sensor information

        # currently, takes in from start to endtime
        return self.get_sensor_data(df, ref_time=-1, secs_in_past=-1, secs_in_future=-1, sampling_rate=self.fps,
                                    sensor_list=self.sensor_types)

        #return self.get_sensor_data(df, ref_time=2, secs_in_past=1, secs_in_future=1, sampling_rate=self.fps,
        #                       sensor_list=self.sensor_types)


    def add_missing_timestamp_to_df(self, df, ref_time, secs_in_past, secs_in_future, fps):

        # set the index to timestamp if it has not been set already
        if 'Timestamp (ms)' in df.columns:
            df.set_index('Timestamp (ms)', inplace=True)

        # get missing indices
        last_csv_time_stamp = df.index.values.tolist()[::-1][0]

        # when all these 3 are -1, we consider the entire video [default scenario]
        if(ref_time==-1 and secs_in_past==-1 and secs_in_future==-1):
            start_time = 0
            end_time = last_csv_time_stamp
        else:
            # to handle the corner case where start_time can go negative
            start_time = max((ref_time - secs_in_past) * 1000, 0)
            # to handle the corner case where we don't cross the boundary
            end_time = min((ref_time + secs_in_future) * 1000, last_csv_time_stamp)
            total_time = (ref_time + 2 * secs_in_future) * 1000
        sample_duration = (int)(1000 / fps)  # fps is the sampling rate
        # end_time + 1 since we want end_time to be inclusive
        timestamp_of_interest = np.arange(start_time + sample_duration, end_time + 1, sample_duration)
        timestamp_available = df.index.values.tolist()
        indices_to_add = [x for x in timestamp_of_interest if x not in timestamp_available]

        # add the rows for the missing indices + backfill data
        line = pd.DataFrame({}, index=indices_to_add)
        df = df.append(line, ignore_index=False)
        df.sort_index(inplace=True)
        # backfill the data to account for missing values in newly added indices
        df.fillna(method='bfill', inplace=True)
        return timestamp_of_interest, df

    # get samples for the most recent time
    def get_sensor_data(self, df, ref_time, secs_in_past, secs_in_future, sampling_rate, sensor_list, metric_spec="all",
                        sensor_metric_spec=[]):
        # without any restriction for sensor type
        # sensor_metric_spec cab be used if one needs only specific matrics for a sensor
        timestamp_of_interest, df = self.add_missing_timestamp_to_df(df, ref_time, secs_in_past, secs_in_future,
                                                                sampling_rate)
        result_dict = {}
        for sensor in sensor_list:
            columns_to_retrieve = [x for x in self.senor_to_metric_dict[sensor] if x in df.columns]
            # execute this if you don't need timestamp value
            #result_dict[sensor] = df[(df.index.isin(timestamp_of_interest))].drop(['Triggering Sensor'], axis=1)[
            #    columns_to_retrieve].to_dict('list')
            # execute this if you need timestamp value as well
            result_dict[sensor] = df[(df.index.isin(timestamp_of_interest))].drop(['Triggering Sensor'], axis=1)[columns_to_retrieve].to_dict()
        result_dict["direction_label"] = df[(df.index.isin(timestamp_of_interest))].drop(['Triggering Sensor'], axis=1)[
            ["direction"]].to_dict()
        return result_dict


    def scrape_all_data(self, path):
        directories = [f for f in os.listdir(path)]
        result_dict = {}
        for directory in directories:
            if directory.startswith("."):  # to ignore the temp files that get created like ".DS_Store"
                continue
            dir_path = os.path.join(path, directory)
            sensor_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
            for sensor_file in sensor_files:
                output = self.transform(dir_path + '/' + sensor_file)
                name = sensor_file.split('.')[0]
                result_dict[name] = output
        return result_dict
