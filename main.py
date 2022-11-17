
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
    with open("config.yaml", "r") as configfile:
        config_dict = yaml.load(configfile, Loader=yaml.FullLoader)
    # print(config_dict)
    return config_dict


'''
TODO: full pipeline
'''
if __name__ == "__main__":
    config_dict = load_config()

    if(config_dict['global']['enable_wandb']):
        import wandb
        wandb.init(name=config_dict['global']['iteration'], 
          project="AIGD", 
          notes=config_dict['global']['description']) 
        #  , config=config_dict)
    else:
        wandb = None

    print(config_dict)
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
    else:
        test_videos = None
        test_sensor = None

    # need video and sensor data separately
    with open(config_dict['transformer']['data_save_file'] + '_sensor.pickle', 'rb') as handle:
        df_sensor = pickle.load(handle)
    
    # pdb.set_trace()
    # print(df_sensor['sample']['direction_label']['direction'])
    
    # Training setup begins
    # train_transforms = [ttf.ToTensor(), transforms.Resize((HEIGHT, WIDTH)), transforms.ColorJitter(), transforms.RandomRotation(10), transforms.GaussianBlur(3)]
    # train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((config_dict['data']['HEIGHT'], config_dict['data']['WIDTH']))])

    train_transforms = transforms.Compose([transforms.ToTensor()])

    val_transforms = transforms.Compose([transforms.ToTensor()])

    # following functions returns a list of file paths (relative paths to video csvs) for train and val sets
    train_files, val_files = make_tt_split(list(df_videos.keys()),config_dict['global']['seed'])
    
    print("Train Files:", train_files)
    print("Val Files:", val_files)
    
    trainer = Trainer(config_dict, train_transforms, val_transforms, train_files, val_files, df_videos, df_sensor, test_videos,test_sensor, wandb=wandb)
    
    trainer.save(0, -1)
    
    epochs = config_dict['trainer']['epochs']
    
    for epoch in range(epochs):
        train_actual, train_predictions = trainer.train(epoch)
        acc, val_actual, val_predictions = trainer.validate()
        val_precision, val_recall, val_f1, _ = display_classification_report(train_actual, train_predictions, val_actual, val_predictions)
        
        if(config_dict['global']['enable_wandb']):
            wandb.log({"Val Precision 0": val_precision[0], "Val Precision 1": val_precision[1], "Val Precision 2": val_precision[2]})
            wandb.log({"Val Recall 0": val_recall[0], "Val Recall 1": val_recall[1], "Val Recall 2": val_recall[2]})
            wandb.log({"Val F1 0": val_f1[0], "Val F1 1": val_f1[1], "Val F1 2": val_f1[2]})

        trainer.save(acc, epoch)

    print("Completed Training!!")
    # performs final benchmarking after training
    if (config_dict['transformer']['enable_benchmark_test'] == True):
        print("Starting benchmark testing!!")
        acc, test_actual, test_predictions = trainer.test()
        test_precision, test_recall, test_f1, _ = display_test_classification_report(test_actual, test_predictions)
        if(config_dict['global']['enable_wandb']):
            wandb.log({"Test Precision 0": test_precision[0], "Test Precision 1": test_precision[1], "Test Precision 2": test_precision[2]})
            wandb.log({"Test Recall 0": test_recall[0], "Test Recall 1": test_recall[1], "Test Recall 2": test_recall[2]})
            wandb.log({"Test F1 0": test_f1[0], "Test F1 1": test_f1[1], "Test F1 2": test_f1[2]})

    
    print("Done!")
