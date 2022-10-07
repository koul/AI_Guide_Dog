
import transformer.DataTransformer as DataTransformer
import yaml
import numpy as np
from utils import *
from torchvision import transforms
from trainer.Trainer import Trainer
import pickle
import pdb
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

def transform(data_file_path, fps, data_save_file, resolution):
    dataTransformer = DataTransformer.DataTransformer(fps, resolution)
    video_data, sensor_data = dataTransformer.scrape_all_data(data_file_path)
    save_data(video_data, sensor_data, data_save_file)


'''
After calling transform, train model on the dumped data
in the folders
'''
#def train_model():


def load_config():
    with open("config.yaml", "r") as configfile:
        config_dict = yaml.load(configfile, Loader=yaml.FullLoader)
    # print(config_dict)
    return config_dict


'''
TODO: full pipeline
'''
if __name__ == "__main__":
    config_dict = load_config()

    #avoid running transform if .nz has already been generated
    if(config_dict['global']['enable_preprocessing'] == True):
        transform(config_dict['transformer']['path'], config_dict['transformer']['fps'], config_dict['transformer']['data_save_file'],[config_dict['data']['HEIGHT'],config_dict['data']['WIDTH']])
    
    df_videos = dict(np.load(config_dict['transformer']['data_save_file']+'_video.npz', allow_pickle=True))
    print(df_videos.keys())

    
    # need video and sensor data separately
    with open(config_dict['transformer']['data_save_file']+'_sensor.pickle', 'rb') as handle:
        df_sensor = pickle.load(handle)
    
    # pdb.set_trace()
    # print(df_sensor['sample']['direction_label']['direction'])
    
    # Training setup begins
    # train_transforms = [ttf.ToTensor(), transforms.Resize((HEIGHT, WIDTH)), transforms.ColorJitter(), transforms.RandomRotation(10), transforms.GaussianBlur(3)]
    # train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((config_dict['data']['HEIGHT'], config_dict['data']['WIDTH']))])

    train_transforms = transforms.Compose([transforms.ToTensor()])

    val_transforms = transforms.Compose([transforms.ToTensor()])

    # following functions returns a list of file paths (relative paths to video csvs) for train and test sets
    train_files, test_files = make_tt_split(list(df_videos.keys()))
    
    print(train_files)
    print(test_files)
    
    trainer = Trainer(config_dict, train_transforms, val_transforms, train_files, test_files, df_videos, df_sensor)
    trainer.save(0, -1)
    
    epochs = config_dict['trainer']['epochs']
    for epoch in range(epochs):
        train_actual, train_predicitons = trainer.train(epoch)
        acc, val_actual, val_predictions = trainer.validate()
        display_classification_report(train_actual, train_predictions, val_actual, val_predictions)
        trainer.save(acc, epoch)
