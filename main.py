import transformer.DataTransformer as DataTransformer
import yaml
import numpy as np

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
    config_dict = load_config()
    #avoid running transform if .nz has already been generated
    transform(config_dict[0]['transformer']['path'], config_dict[0]['transformer']['fps'])
    #train_model()
    # loaded = load_data('temp.npy')