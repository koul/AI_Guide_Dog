import os
import cv2
import pandas as pd
# from config import *
import random
import os.path as osp
import torch
from torch.utils.data import WeightedRandomSampler
# import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import scipy.stats as ss
import torch.nn as nn
import math

# df is pandas dataframe of the form: frame_path, direction, timestamp
# direction is the current direction at the timestamp of the frame.
# We use the following function to create label: direction 1 sec ahead 
def preprocess_labels(df):
    label_indices = df['timestamp'].searchsorted(df['timestamp']+1000)
    viable_indices = label_indices[label_indices!=len(df)]
    df_new = df[label_indices != len(df)].reset_index(drop = True)
    df_new['labels'] = df['directions'][viable_indices].reset_index(drop = True)
    return df_new


def convert_to_dataframe(d):
    df = pd.DataFrame.from_dict(d, orient ='index') 
    df.sort_index(inplace = True)
    df = df.reset_index(drop = False).reset_index(drop = False)
    df.columns = ['frame_index', 'timestamp', 'directions']
    return df

def make_tt_split(files, seed):
    random.Random(seed).shuffle(files)
    split_ratio = 0.4
    ts = int(len(files) * split_ratio)
    test_files = files[:ts]
    train_files = files[ts:]
    print("Test files ",test_files)
    return train_files, test_files

def labelCount(label, n_classes):
    label_count = [0]*(n_classes)
    for lab in label:
        label_count[lab] += 1
    return label_count

def sampler_(dataset_labels, n_classes):
    dataset_counts = labelCount(dataset_labels, n_classes)
    print("Label counts before balancing: ", dataset_counts)
    num_samples = sum(dataset_counts)
    class_weights = [num_samples/i for i in dataset_counts]
    weights = [class_weights[y] for y in dataset_labels]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    return sampler

def get_gps_probabilities(gps_range) :
    mid = (gps_range[0]+gps_range[1])/2
    x = np.arange(-mid, mid)
    xU, xL = x + 0.5, x - 0.5 
    prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
    prob = prob / prob.sum() # normalize the probabilities so their sum is 1
    return prob


def save(config, model, index, acc, optim = False):
    save_path = os.path.join(config['global']['root_dir'],config['trainer']['model_save_path'], str(config['global']['iteration']))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if(optim):
        torch.save(model.state_dict(), save_path+'/{}_optimizer_params_epoch_{:08d}_acc_{}.pth'.format(config['trainer']['model']['name'], index, acc))
    else:
        torch.save(model.state_dict(), save_path+'/{}_model_params_epoch_{:08d}_acc_{}.pth'.format(config['trainer']['model']['name'], index, acc))

def label_map(lab):
    if(lab == 0):
        return 2
    elif(lab == -1):
        return 0
    else:
        return 1
    
def get_all_files_from_dir(directory, vids = False):
    file_paths = []
    print(directory)
    try:
        for root, dirs, files in os.walk(directory):
            # print(files)
            if(vids):
                file_paths += [os.path.join(root, x,x+".mp4") for x in dirs]
            else:
                file_paths += [os.path.join(root, x) for x in files]
        return sorted(file_paths)
    except Exception as e:
        print(e)

def dcr_helper(actual, predictions):
    cm = confusion_matrix(actual, predictions)
    print(cm)
    # cm_df = pd.DataFrame(cm, index = ['0','1','2'], columns = ['0','1','2'])
    # print(cm_df)
    print('\nClassification Report\n')
    print(classification_report(actual, predictions))

def display_classification_report(train_actual, train_predictions, val_actual, val_predictions):
    print('\nTaining set stats\n')
    dcr_helper(train_actual, train_predictions)

    print('\nValidation set stats\n')
    dcr_helper(val_actual, val_predictions)

def display_test_classification_report(test_actual, test_predictions):
    print('\nTest set stats\n')
    dcr_helper(test_actual, test_predictions)

# Function for processing the videos and labels to get labels at the frame level
# def process_video(video_file, labels):
#     video_filename = video_file.split('/')[-1].split('.')[0]
#     vidcap = cv2.VideoCapture(video_file)

#     ctr = 0
#     video_frames = []
#     video_labels = []
#     frame_ts = []
    
#     # read each frame
#     hasFrames,image = vidcap.read()

#     while (hasFrames):
        
#         # Process the frame and save it to the processed_frames folder
#         save_file_name = video_filename + "_" + str(ctr) + ".npy"
#         image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
#         np.save(osp.join(config_dict['data']['processed_frames'], save_file_name), image)  

#         # Get label corresponding to tht timestamp of the frame in the video
#         label_ts = vidcap.get(cv2.CAP_PROP_POS_MSEC)
#         label_ts = label_ts - (label_ts%100) # adjusting timestamp acc to the 100ms intervals created by the transformer
#         frame_ts.append(label_ts)

#         if(label_ts not in labels.keys()):
#             print(label_ts)
#             hasFrames,image = vidcap.read()
#             continue

#         label = labels[label_ts]
#         video_labels.append(label_map(label))
#         video_frames.append(save_file_name)
#         hasFrames,image = vidcap.read()
#         ctr += 1
        
#     df = pd.DataFrame({'frames': video_frames, 'directions': video_labels, 'timestamp': frame_ts})

#     # save this data frame as a csv into the 
#     df.to_csv(osp.join(config_dict['data']['processed_csvs'],video_filename+".csv"), index=None)

#     print("After processing:")
#     print("Number of frames labelled: ", ctr)
    
# # Main entry point for processing raw video file and the npz file labels
# def preprocess():
#     # load the npz label file from transformer
#     f = np.load(LABEL_FILE, allow_pickle = True)
#     print(f.keys())
#     # for each single video
#     for video_file in get_all_files_from_dir(VID_PATH):
#         video_filename = video_file.split('/')[-1].split('.')[0]
#         print(video_filename)

#         # Check if it has already been processed, i.e. a processed csv exists in the processed folder for this video
#         if(video_filename+".csv" not in os.listdir(DATA_SAVE_PATH)):
#             labels = f[video_filename]['Sensor']['direction_label']['direction']
#             process_video(video_file, labels)
#             print("Finished processing ", video_file)

# # FPS processing raw video file
# def process_videos(vid_path = VID_PATH_OG):
#     fp = get_all_files_from_dir(vid_path, vids=True)
#     print(fp)
#     # for each single video
#     for fl in fp:
#         video_filename = fl.split('/')[-1]
#         # Check if it has already been processed
#         if(video_filename not in os.listdir(VID_PATH)):
#             # change fps
#             ffmpeg.input(fl).filter('fps', fps=10, round='up').output(VID_PATH+video_filename).run() 

# Transformer code ends here
            
def prep_video_test(filename):
    X = []
    y = []
    
    df = pd.read_csv(filename)
    X.append(df['frames'])
    y.append(df['labels'])   
    
    X = pd.concat(X)    # print(X.head())
    X.reset_index(drop=True,inplace=True)
    X = X.to_numpy()

    
    y = pd.concat(y)
    y.reset_index(drop=True,inplace=True)
    y = y.to_numpy()
    
    return X, y, df

class ConvertEmbedding(nn.Module):
    def __init__(self, inp_size,d_model):
        super(ConvertEmbedding, self).__init__()
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

from typing import Dict, List, Optional, Union, Tuple, BinaryIO
import os
import sys
import json
import tempfile
import copy
from tqdm.auto import tqdm
from functools import partial
from urllib.parse import urlparse
from pathlib import Path
import requests
from hashlib import sha256
from filelock import FileLock
# import importlib_metadata
import torch
import torch.nn as nn
from torch import Tensor

# __version__ = "4.0.0"
# _torch_version = importlib_metadata.version("torch")

hf_cache_home = os.path.expanduser(os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface")))
default_cache_path = os.path.join(hf_cache_home, "transformers")
PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", PYTORCH_TRANSFORMERS_CACHE)

PRESET_MIRROR_DICT = {
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models",
    "bfsu": "https://mirrors.bfsu.edu.cn/hugging-face-models",
}
HUGGINGFACE_CO_PREFIX = "https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"


def is_torch_available():
  return True


def is_tf_available():
  return False


def is_remote_url(url_or_filename):
  parsed = urlparse(url_or_filename)
  return parsed.scheme in ("http", "https")


def http_get(url: str, temp_file: BinaryIO, proxies=None, resume_size=0, headers: Optional[Dict[str, str]] = None):
  headers = copy.deepcopy(headers)
  if resume_size > 0:
    headers["Range"] = "bytes=%d-" % (resume_size,)
  r = requests.get(url, stream=True, proxies=proxies, headers=headers)
  r.raise_for_status()
  content_length = r.headers.get("Content-Length")
  total = resume_size + int(content_length) if content_length is not None else None
  progress = tqdm(
    unit="B",
    unit_scale=True,
    total=total,
    initial=resume_size,
    desc="Downloading",
    disable=False,
  )
  for chunk in r.iter_content(chunk_size=1024):
    if chunk:  # filter out keep-alive new chunks
      progress.update(len(chunk))
      temp_file.write(chunk)
  progress.close()


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
  url_bytes = url.encode("utf-8")
  filename = sha256(url_bytes).hexdigest()

  if etag:
    etag_bytes = etag.encode("utf-8")
    filename += "." + sha256(etag_bytes).hexdigest()

  if url.endswith(".h5"):
    filename += ".h5"

  return filename


def hf_bucket_url(
  model_id: str, filename: str, subfolder: Optional[str] = None, revision: Optional[str] = None, mirror=None
) -> str:
  if subfolder is not None:
    filename = f"{subfolder}/{filename}"

  if mirror:
    endpoint = PRESET_MIRROR_DICT.get(mirror, mirror)
    legacy_format = "/" not in model_id
    if legacy_format:
      return f"{endpoint}/{model_id}-{filename}"
    else:
      return f"{endpoint}/{model_id}/{filename}"

  if revision is None:
    revision = "main"
  return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
  ua = "transformers/{}; python/{}".format(__version__, sys.version.split()[0])
  if is_torch_available():
    ua += f"; torch/{_torch_version}"
  if is_tf_available():
    ua += f"; tensorflow/{_tf_version}"
  if isinstance(user_agent, dict):
    ua += "; " + "; ".join("{}/{}".format(k, v) for k, v in user_agent.items())
  elif isinstance(user_agent, str):
    ua += "; " + user_agent
  return ua


def get_from_cache(
  url: str,
  cache_dir=None,
  force_download=False,
  proxies=None,
  etag_timeout=10,
  resume_download=False,
  user_agent: Union[Dict, str, None] = None,
  use_auth_token: Union[bool, str, None] = None,
  local_files_only=False,
) -> Optional[str]:
  if cache_dir is None:
    cache_dir = TRANSFORMERS_CACHE
  if isinstance(cache_dir, Path):
    cache_dir = str(cache_dir)

  os.makedirs(cache_dir, exist_ok=True)

  headers = {"user-agent": http_user_agent(user_agent)}
  if isinstance(use_auth_token, str):
    headers["authorization"] = "Bearer {}".format(use_auth_token)
  elif use_auth_token:
    token = HfFolder.get_token()
    if token is None:
      raise EnvironmentError("You specified use_auth_token=True, but a huggingface token was not found.")
    headers["authorization"] = "Bearer {}".format(token)

  url_to_download = url
  etag = None
  if not local_files_only:
    try:
      r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=etag_timeout)
      r.raise_for_status()
      etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
      # We favor a custom header indicating the etag of the linked resource, and
      # we fallback to the regular etag header.
      # If we don't have any of those, raise an error.
      if etag is None:
        raise OSError(
          "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
        )
      # In case of a redirect,
      # save an extra redirect on the request.get call,
      # and ensure we download the exact atomic version even if it changed
      # between the HEAD and the GET (unlikely, but hey).
      if 300 <= r.status_code <= 399:
        url_to_download = r.headers["Location"]
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
      # etag is already None
      pass

  filename = url_to_filename(url, etag)

  # get cache path to put the file
  cache_path = os.path.join(cache_dir, filename)

  # etag is None == we don't have a connection or we passed local_files_only.
  # try to get the last downloaded one
  if etag is None:
    if os.path.exists(cache_path):
      return cache_path
    else:
      matching_files = [
        file
        for file in fnmatch.filter(os.listdir(cache_dir), filename.split(".")[0] + ".*")
        if not file.endswith(".json") and not file.endswith(".lock")
      ]
      if len(matching_files) > 0:
        return os.path.join(cache_dir, matching_files[-1])
      else:
        # If files cannot be found and local_files_only=True,
        # the models might've been found if local_files_only=False
        # Notify the user about that
        if local_files_only:
          raise FileNotFoundError(
            "Cannot find the requested files in the cached path and outgoing traffic has been"
            " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
            " to False."
          )
        else:
          raise ValueError(
            "Connection error, and we cannot find the requested files in the cached path."
            " Please try again or make sure your Internet connection is on."
          )

  # From now on, etag is not None.
  if os.path.exists(cache_path) and not force_download:
    return cache_path

  # Prevent parallel downloads of the same file with a lock.
  lock_path = cache_path + ".lock"
  with FileLock(lock_path):

    # If the download just completed while the lock was activated.
    if os.path.exists(cache_path) and not force_download:
      # Even if returning early like here, the lock will be released.
      return cache_path

    if resume_download:
      incomplete_path = cache_path + ".incomplete"

      @contextmanager
      def _resumable_file_manager() -> "io.BufferedWriter":
        with open(incomplete_path, "ab") as f:
          yield f

      temp_file_manager = _resumable_file_manager
      if os.path.exists(incomplete_path):
        resume_size = os.stat(incomplete_path).st_size
      else:
        resume_size = 0
    else:
      temp_file_manager = partial(tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False)
      resume_size = 0

    # Download to temporary file, then copy to cache dir once finished.
    # Otherwise you get corrupt cache entries if the download gets interrupted.
    with temp_file_manager() as temp_file:
      http_get(url_to_download, temp_file, proxies=proxies, resume_size=resume_size, headers=headers)

    os.replace(temp_file.name, cache_path)

    meta = {"url": url, "etag": etag}
    meta_path = cache_path + ".json"
    with open(meta_path, "w") as meta_file:
      json.dump(meta, meta_file)

  return cache_path


def cached_path(
  url_or_filename,
  cache_dir=None,
  force_download=False,
  proxies=None,
  resume_download=False,
  user_agent: Union[Dict, str, None] = None,
  extract_compressed_file=False,
  force_extract=False,
  use_auth_token: Union[bool, str, None] = None,
  local_files_only=False,
) -> Optional[str]:
  if cache_dir is None:
    cache_dir = TRANSFORMERS_CACHE
  if isinstance(url_or_filename, Path):
    url_or_filename = str(url_or_filename)
  if isinstance(cache_dir, Path):
    cache_dir = str(cache_dir)

  if is_remote_url(url_or_filename):
    # URL, so get it from the cache (downloading if necessary)
    output_path = get_from_cache(
      url_or_filename,
      cache_dir=cache_dir,
      force_download=force_download,
      proxies=proxies,
      resume_download=resume_download,
      user_agent=user_agent,
      use_auth_token=use_auth_token,
      local_files_only=local_files_only,
    )
  elif os.path.exists(url_or_filename):
    # File, and it exists.
    output_path = url_or_filename
  elif urlparse(url_or_filename).scheme == "":
    # File, but it doesn't exist.
    raise EnvironmentError("file {} not found".format(url_or_filename))
  else:
    # Something unknown
    raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

  if extract_compressed_file:
    if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
      return output_path

    # Path where we extract compressed archives
    # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
    output_dir, output_file = os.path.split(output_path)
    output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
    output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

    if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
      return output_path_extracted

    # Prevent parallel extractions
    lock_path = output_path + ".lock"
    with FileLock(lock_path):
      shutil.rmtree(output_path_extracted, ignore_errors=True)
      os.makedirs(output_path_extracted)
      if is_zipfile(output_path):
        with ZipFile(output_path, "r") as zip_file:
          zip_file.extractall(output_path_extracted)
          zip_file.close()
      elif tarfile.is_tarfile(output_path):
        tar_file = tarfile.open(output_path)
        tar_file.extractall(output_path_extracted)
        tar_file.close()
      else:
        raise EnvironmentError("Archive format of {} could not be identified".format(output_path))

    return output_path_extracted

  return output_path


def get_parameter_dtype(parameter: Union[nn.Module]):
  try:
    return next(parameter.parameters()).dtype
  except StopIteration:
    # For nn.DataParallel compatibility in PyTorch 1.5

    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
      tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
      return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    first_tuple = next(gen)
    return first_tuple[1].dtype


def get_extended_attention_mask(attention_mask):
  # attention_mask [batch_size, seq_length]
  assert attention_mask.dim() == 2
  # [batch_size, 1, 1, seq_length] for multi-head attention
  extended_attention_mask = attention_mask[:, None, None, :]
  extended_attention_mask = extended_attention_mask # fp16 compatibility
  extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
  return extended_attention_mask