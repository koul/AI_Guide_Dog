import transformer.DataTransformer as DataTransformer
import yaml
import numpy as np
from trainer.utils import *
from torchvision import transforms
from Trainer import Trainer
'''
Input: a path to folder of subfolders. Each subfolder will have a CSV and MP4 file
OUTPUT: N/A - data is dumped to a folder

For data:
https://www.dropbox.com/sh/o8orrxczmthtja6/AABCl_5tqbHt-DJoc1RnnjVDa?dl=0
https://www.dropbox.com/sh/fbo4dr3wlpob3px/AADKhrnCyaGWCSDb6XoVOBMna?dl=0
'''

def save_data(data, filename):
    np.savez(filename, data)

def load_data(filename):
    return np.load(filename, allow_pickle=True)

def transform(data_file_path, fps):
    dataTransformer = DataTransformer.DataTransformer(fps)
    result = dataTransformer.scrape_all_data(data_file_path)
    save_data(result, 'data.npz')


'''
After calling transform, train model on the dumped data
in the folders
'''
#def train_model():


def load_config():
    with open("config.yaml", "r") as configfile:
        config_dict = yaml.load(configfile, Loader=yaml.FullLoader)
    return config_dict


'''
TODO: full pipeline
'''
if __name__ == "__main__":
    config_dict = load_config()[0]

    #avoid running transform if .nz has already been generated
    if(config_dict['trainer']['enable_preprocessing'] == True):
        #The following function is expected to:
        #  1. Pick up the extract the video and sensor data
        #  2. Transform video according to the required fps
        #  3. For each frame in a given video do:
        #   a. resize the frame according to resolution specified in config.yaml  
        #   b. save the frames to a data.processed_frames
        #   c. get the frame label and timestamp
        #  4. Create a csv - video_filename.csv containing {frame_name, label(rename as direction), timestamp}
        #  5. Store csv to data.processed_csvs
        transform(config_dict['transformer']['path'], config_dict['transformer']['fps'])
    

    # Training setup begins

    # train_transforms = [ttf.ToTensor(), transforms.Resize((HEIGHT, WIDTH)), transforms.ColorJitter(), transforms.RandomRotation(10), transforms.GaussianBlur(3)]
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((config_dict['data']['HEIGHT'], config_dict['data']['WIDTH']))])

    val_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((config_dict['data']['HEIGHT'], config_dict['data']['WIDTH']))])

    # following functions returns a list of file paths (relative paths to video csvs) for train and test sets
    train_files, test_files = make_tt_split(config_dict['data']['processed_csvs'])

    trainer = Trainer(config_dict, train_transforms, val_transforms, train_files, test_files)

    epochs = config_dict['trainer']['epochs']
    for epoch in range(epochs):
        trainer.train()
        acc = trainer.validate()
        trainer.save(acc)
