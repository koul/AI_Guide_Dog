
from distutils.command import config
import transformer.DataTransformer as DataTransformer
import yaml
import numpy as np
from utils import *
from torchvision import transforms
from trainer.Trainer import Trainer, TrainerPredRNN
# import pickle
import pickle5 as pickle
import pdb

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

    ## DEBUG
    # train_files = train_files[:1]
    # test_files = test_files[:1]

    print(train_files)
    print(test_files)

    # trainer = Trainer(config_dict, train_transforms, val_transforms, train_files, test_files, df_videos, df_sensor)
    # if config_dict['trainer']['wandb']:
    wandb.init(project='AI_Guide_Dog')

    trainer = TrainerPredRNN(config_dict, train_transforms, val_transforms, train_files, test_files, df_videos, df_sensor)
    trainer.save(0, -1)

    epochs = config_dict['trainer']['epochs']
    for epoch in range(epochs):
        train_acc, train_actual, train_predictions = trainer.train(epoch)
        val_acc, val_actual, val_predictions = trainer.validate()
        display_classification_report(train_actual, train_predictions, val_actual, val_predictions)
        # actual_left_pred_left, actual_left_pred_right, actual_left_pred_front, actual_right_pred_left, actual_right_pred_right, actual_right_pred_front, actual_front_pred_left, actual_front_pred_right, actual_front_pred_front = get_confusion_matrix(val_actual, val_predictions)

        train_report = get_classification_report(train_actual, train_predictions)
        val_report = get_classification_report(val_actual, val_predictions)

        trainer.save(val_acc, epoch)

        wandb.log({
            'train_acc': train_acc,
            'val_acc': val_acc,
            'lr': trainer.optimizer.param_groups[0]['lr'],

            "train_left_precision": train_report['left']['precision'],
            "train_left_recall": train_report['left']['recall'],
            "train_left_f1": train_report['left']['f1-score'],

            "train_right_precision": train_report['right']['precision'],
            "train_right_recall": train_report['right']['recall'],
            "train_right_f1": train_report['right']['f1-score'],

            "train_front_precision": train_report['front']['precision'],
            "train_front_recall": train_report['front']['recall'],
            "train_front_f1": train_report['front']['f1-score'],

            "val_left_precision": val_report['left']['precision'],
            "val_left_recall": val_report['left']['recall'],
            "val_left_f1": val_report['left']['f1-score'],

            "val_right_precision": val_report['right']['precision'],
            "val_right_recall": val_report['right']['recall'],
            "val_right_f1": val_report['right']['f1-score'],

            "val_front_precision": val_report['front']['precision'],
            "val_front_recall": val_report['front']['recall'],
            "val_front_f1": val_report['front']['f1-score'],
        })