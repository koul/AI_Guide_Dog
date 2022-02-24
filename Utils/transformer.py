import pandas as pd
import numpy as np
import pickle

'''
columns_from_csv
columns = ['Timestamp (ms)', 'Triggering Sensor', 'Latitude', 'Longitude',
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
'''

def slice_timestamp(array, start_time, end_time):
  start_pos = 0
  end_pos = len(array)
  for idx,ele in enumerate(array):
    if ele >= start_time:
      start_pos = idx
      break
  for idx,ele in enumerate(array[::-1]):
    if ele <= end_time:
      end_pos = len(array) - idx -1 #since we're traversing in reverse
      break
  return array[start_pos:end_pos+1]

def get_timestamp_for_sampling_rate(sliced_timestamp_list, sampling_rate, total_time, start_time, end_time):
  #need to finish implementing this with capturing the most recent timestamp for each sampling window later
  return sliced_timestamp_list[::-1][:int(sampling_rate*total_time/1000)]

def get_data_for_sensor(df, sensor, count_per_sec, ref_time, secs_in_future):
    start_time = (ref_time - secs_in_future) * 1000
    end_time = (ref_time + secs_in_future) * 1000
    total_time = (ref_time + 2 * secs_in_future) * 1000
    sampling_rate = count_per_sec  # samples count per second
    sliced_timestamp_list = slice_timestamp(np.array(df['Timestamp (ms)'].values), start_time, end_time)
    # print(sliced_timestamp_list)
    timestamp_after_sampling = get_timestamp_for_sampling_rate(sliced_timestamp_list, sampling_rate, total_time,
                                               start_time, end_time)
    result = df[(df['Triggering Sensor'] == sensor) & (df['Timestamp (ms)'].isin(timestamp_after_sampling))].drop(
        ['Triggering Sensor', 'Timestamp (ms)'], axis=1)
    final_dict = {}
    for col in result.columns.values:
        final_dict[col] = np.array(result[col])
    return final_dict

#get samples for the most recent time
def get_sensor_data(df, ref_time, secs_in_future, sensor_list, sampling_rate, metric_spec = "all",sensor_metric_spec = []):
  #sensor_metric_spec -- can specify exact sensor metrics needed for each Sensor, is a nested list
  #without any restriction for sensor type
  sensor_dict = {}
  for sensor,count_per_sec in zip(sensor_list,sampling_rate):
    #get all the values for that sensor type
    sensor_dict[sensor] = get_data_for_sensor(df, sensor, count_per_sec, ref_time, secs_in_future)
  return sensor_dict

def SensorDataTransformer(file_path):
    df = pd.read_csv(file_path)
    df.fillna(method='bfill', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)  # drop a metric (i.e, column) if it has no sensor information
    sensor_types = ["Gyrometer", "Magnetometer", "Accelerometer", "Device Motion", "GPS",
                    "Altimeter", "Activity Manager"]
    sensor_info = get_sensor_data(df, ref_time=1, secs_in_future=1, sensor_list=sensor_types,
                    sampling_rate=[5] * len(sensor_types))
    return sensor_info

if __name__ == "__main__":
    file_path = "2022-02-17T20_32_33.869Z.csv"
    sensor_info = SensorDataTransformer(file_path)
    print(sensor_info)
    f = open("sample_sensor_info.txt", "wb+")
    f.write(pickle.dumps(sensor_info))
    f.close()
