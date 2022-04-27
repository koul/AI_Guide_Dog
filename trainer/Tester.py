from asyncio.windows_events import NULL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from models import *
from utils import *
from preprocessing import *
from dataset import *

# import ffmpeg
import os
import os.path as osp
import torchvision.models as models

from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt

class Tester:
    def __init__(self, model_path,device):
        self.model_path = model_path
        self.device = device
        self.val_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((HEIGHT, WIDTH))])
        self.set_up()
    
    def set_up(self):
        self.model = ResNet18()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(self.device)
        

    def validate(self,val_loader,val_dataset):
        self.model.eval()
        val_num_correct = 0
        predictions = []
        
        for i, (vx, vy) in enumerate(val_loader):
        
            vx = vx.float().to(self.device)
            vy = vy.to(self.device)

            with torch.no_grad():
                outputs = self.model(vx)
                del vx

            preds = torch.argmax(outputs, axis=1)
            predictions.append(preds.cpu().detach().numpy())
            val_num_correct += int((preds == vy).sum())
            del outputs
        
            # val_num_correct += int((torch.argmax(outputs, axis=1) == vy).sum())
            # del outputs
            # break
        
        # print(predictions)
        predictions = np.concatenate(predictions)
        acc = 100 * val_num_correct / (len(val_dataset))
        print("Validation: {:.04f}%".format(acc))
        return predictions, acc

    def validation_pipeline(self, video_ids):
        preprocessing_pipeline(video_ids)
        for full in video_ids:
            # fn =  full.split('/')[-1].split('.')[0]
            X, y, df = prep_video_test(osp.join(DATA_SAVE_PATH,full+".csv"))
            test_dataset = FrameDataset(X, y, transforms=self.val_transforms, base_path = PROCESSED_PATH)
            val_args = dict(shuffle=False, batch_size=BATCH, num_workers=2, pin_memory=True, drop_last=False) if cuda else dict(shuffle=False, batch_size=BATCH, drop_last=False)
            test_loader = DataLoader(test_dataset, **val_args)
            print(len(test_dataset))
            predictions, acc = self.validate(test_loader, test_dataset)
            df['predictions'] = predictions
            print(df.head())
            df.to_csv("predictions_{}_acc={}.csv".format(full,acc), index=None)
            print("Precdcitions written to file: ", "predictions_{}_acc={}.csv".format(full,acc))


        
            