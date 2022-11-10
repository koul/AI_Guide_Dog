
import transformer.DataTransformer as DataTransformer
from vidaug import augmentors as va
import yaml
import numpy as np
from utils import *
from torchvision import transforms
from trainer.Trainer import Trainer
import pickle
import pdb
import warnings
warnings.filterwarnings("ignore")
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


'''
After calling transform, train model on the dumped data
in the folders
'''
#def train_model():


def load_config():
    with open("AI_Guide_Dog/config.yaml", "r") as configfile:
        config_dict = yaml.load(configfile, Loader=yaml.FullLoader)
    # print(config_dict)
    return config_dict


'''
TODO: full pipeline
'''
if __name__ == "__main__":
    config_dict = load_config()


    # avoid running transform if .nz has already been generated
    if (config_dict['global']['enable_preprocessing'] == True):
        transform(config_dict['transformer']['path'], config_dict['transformer']['fps'],
                  config_dict['transformer']['data_save_file'],
                  [config_dict['data']['HEIGHT'], config_dict['data']['WIDTH']],
                  config_dict['data']['CHANNELS'])
        if (config_dict['transformer']['enable_benchmark_test'] == True): transform(
            config_dict['transformer']['test_path'], config_dict['transformer']['fps'],
            config_dict['transformer']['test_save_file'], [config_dict['data']['HEIGHT'], config_dict['data']['WIDTH']],
            config_dict['data']['CHANNELS'])

    df_videos = dict(np.load(config_dict['transformer']['data_save_file'] + '_video.npz', allow_pickle=True))
    print(df_videos.keys())

    if (config_dict['transformer']['enable_benchmark_test'] == True):
        test_videos = dict(np.load(config_dict['transformer']['test_save_file'] + '_video.npz', allow_pickle=True))
    with open(config_dict['transformer']['test_save_file'] + '_sensor.pickle', 'rb') as handle:
        test_sensor = pickle.load(handle)

    # need video and sensor data separately
    with open(config_dict['transformer']['data_save_file'] + '_sensor.pickle', 'rb') as handle:
        df_sensor = pickle.load(handle)
    
    # pdb.set_trace()

    # Training setup begins
    # train_transforms = [ttf.ToTensor(), transforms.Resize((HEIGHT, WIDTH)), transforms.ColorJitter(), transforms.RandomRotation(10), transforms.GaussianBlur(3)]
    # train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((config_dict['data']['HEIGHT'], config_dict['data']['WIDTH']))])

    # https://github.com/okankop/vidaug
    # train_transforms = transforms.Compose([transforms.ToTensor()])
    transformations = [va.RandomTranslate(x=15, y=15), # Translates by 5 pixels
                # va.Multiply(0.6), # Makes video less bright
                # va.Multiply(1.4), # Makes video brighter
                # va.Pepper(95), # Makes 5% of video black pixels in each frame
                # va.Salt(95), # Makes 5% of video white pixels in each frame
                # va.Superpixel(p_replace=0.1, n_segments=50), # Make group of pixel become superpixel
    ]
    # transformations = [va.RandomTranslate(x=5, y=5), # Translates by 5 pixels
    # transformations = [va.Multiply(0.6), # Makes video less bright
                # va.Multiply(1.4), # Makes video brighter
                # va.Superpixel(p_replace=0.1, n_segments=50), # Make group of pixel become superpixel
    # ]
    sometimes = lambda aug: va.Sometimes(0.3, aug) # Used to apply augmentor with 30% probability
    train_transforms = sometimes(va.SomeOf(transformations, 1, True)) # Picks 3 transformations 30% of the time)

    val_transforms = transforms.Compose([transforms.ToTensor()])

    # following functions returns a list of file paths (relative paths to video csvs) for train and test sets

    # if(config_dict['data']['TEST_FILES'] is not None):
    #     test_files = config_dict['data']['TEST_FILES']
    #     test_files = [t.strip() for t in test_files.split(',')]
    #     train_files = []
    #     for f in list(df_videos.keys()):
    #         if (f not in test_files):
    #             train_files.append(f)
    # else:
    train_files, val_files = make_tt_split(list(df_videos.keys()),config_dict['global']['seed'])
    
    print("Train Files:", train_files)
    print("Val Files:", val_files)
    
    trainer = Trainer(config_dict, train_transforms, val_transforms, train_files, val_files, df_videos, df_sensor, test_videos,test_sensor)
    trainer.save(0, -1)
    
    epochs = config_dict['trainer']['epochs']

    for epoch in range(epochs):
        # print("TRAIN")
        train_actual, train_predictions = trainer.train(epoch)
        # print("VALIDATE")
        acc, val_actual, val_predictions = trainer.validate()
        display_classification_report(train_actual, train_predictions, val_actual, val_predictions)
        trainer.save(acc, epoch)

    # performs final benchmarking after training
    # if (config_dict['transformer']['enable_benchmark_test'] == True):
    #     acc, test_actual, test_predictions = trainer.test()
    #     # print(f'test_actual length: {len(test_actual)}, test_predictions length: {len(test_predictions)}')
    #     display_test_classification_report(test_actual, test_predictions)
