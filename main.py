import transformer.DataTransformer as DataTransformer
import yaml
import numpy as np
from utils import *
from torchvision import transforms
from trainer.Trainer import Trainer
import pickle
import pdb
import warnings
warnings.filterwarnings("ignore")
import torch
import wandb

'''
Input: a path to folder of subfolders. Each subfolder will have a CSV and MP4 file
OUTPUT: N/A - data is dumped to a folder

For data:
https://www.dropbox.com/sh/o8orrxczmthtja6/AABCl_5tqbHt-DJoc1RnnjVDa?dl=0
https://www.dropbox.com/sh/fbo4dr3wlpob3px/AADKhrnCyaGWCSDb6XoVOBMna?dl=0
'''

def save_data(video_data, sensor_data, filename):
    # np.savez(filename, **data)
    # np.savez(filename+'_sensor', **sensor_data)
    save_dir = filename.split('/')
    save_dir = "/".join(save_dir[:-1])
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    np.savez(filename+'_video', **video_data)
    with open(filename+'_sensor.pickle', 'wb') as handle:
        pickle.dump(sensor_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def load_data(filename):
    return np.load(filename, allow_pickle=True)

def transform(data_file_path, fps, data_save_file, resolution, channels):
    dataTransformer = DataTransformer.DataTransformer(fps, resolution, channels)
    video_data, sensor_data = dataTransformer.scrape_all_data(data_file_path)
    # print(video_data.keys())
    # exit()
    save_data(video_data, sensor_data, data_save_file)

def train(config_dict, 
            train_transforms, 
            val_transforms, 
            train_files, 
            val_files, 
            df_videos, 
            df_sensor, 
            test_videos,
            test_sensor):

    # Start training
    trainer = Trainer(config_dict, 
                    train_transforms, 
                    val_transforms, 
                    train_files, 
                    val_files, 
                    df_videos, 
                    df_sensor, 
                    test_videos,
                    test_sensor)
    trainer.save(0, -1)
    
    epochs = config_dict['trainer']['epochs']
    for epoch in range(epochs):
        train_actual, train_predictions = trainer.train(epoch)
        acc, val_actual, val_predictions = trainer.validate()
        display_classification_report(train_actual, train_predictions, val_actual, val_predictions)
        trainer.save(acc, epoch)
    return trainer

'''
After calling transform, train model on the dumped data
in the folders
'''
def load_config():
    with open("config.yaml", "r") as configfile:
        config_dict = yaml.load(configfile, Loader=yaml.FullLoader)
    return config_dict

'''
TODO: full pipeline
'''
if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')

    config_dict = load_config()
    transformer_config = config_dict['transformer']
    data_config = config_dict['data']
    benchmark_enabled = transformer_config['enable_benchmark_test']

    # avoid running transform if .nz has already been generated
    if (config_dict['global']['enable_preprocessing'] == True):
        transform(transformer_config['path'], transformer_config['fps'],
                  transformer_config['data_save_file'],
                  [data_config['HEIGHT'], data_config['WIDTH']],
                  data_config['CHANNELS'])
        if (benchmark_enabled == True): transform(
            transformer_config['test_path'], transformer_config['fps'],
            transformer_config['test_save_file'], [data_config['HEIGHT'], data_config['WIDTH']],
            data_config['CHANNELS'])

    df_videos = dict(np.load(transformer_config['data_save_file'] + '_video.npz', allow_pickle=True))
    # print(df_videos.keys())

    # need video and sensor data separately
    with open(transformer_config['data_save_file'] + '_sensor.pickle', 'rb') as handle:
        df_sensor = pickle.load(handle)
        # tmp = df_sensor['2022-07-12T16-34-07']
        # for key, dic in tmp.items():
        #     for k, v in dic.items():
        #         print("- \"" + ":".join([key,k]) + '\"')

    # Data transformations
    # train_transforms = [ttf.ToTensor(), transforms.Resize((HEIGHT, WIDTH)), transforms.ColorJitter(), transforms.RandomRotation(10), transforms.GaussianBlur(3)]
    # train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((data_config['HEIGHT'], data_config['WIDTH']))])
    train_transforms = transforms.Compose([transforms.ToTensor()])
    val_transforms = transforms.Compose([transforms.ToTensor()])

    # following functions returns a list of file paths (relative paths to video csvs) for train and test sets
    train_files, val_files = make_tt_split(list(df_videos.keys()),config_dict['global']['seed'])
    print("Train Files:", train_files)
    print("Val Files:", val_files)

    test_videos = None
    test_sensor = None
    if (benchmark_enabled == True):
        test_videos = dict(np.load(transformer_config['test_save_file'] + '_video.npz', allow_pickle=True))
        with open(transformer_config['test_save_file'] + '_sensor.pickle', 'rb') as handle:
            test_sensor = pickle.load(handle)

    num_hid_layer_l = [2, 3]
    num_att_head_l = [2, 3, 6]

    for num_hid_layer in num_hid_layer_l:
        config_dict['trainer']['model']['layer_num'] = num_hid_layer
        trainer = train(config_dict, 
                train_transforms, 
                val_transforms, 
                train_files, 
                val_files, 
                df_videos, 
                df_sensor, 
                test_videos,
                test_sensor)
        wandb.finish()

    # performs final benchmarking after training
    if (benchmark_enabled == True):
        acc, test_actual, test_predictions = trainer.test()
        display_test_classification_report(test_actual, test_predictions)
