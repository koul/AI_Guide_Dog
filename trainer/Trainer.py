import torch
from trainer.dataset import SensorVideoDataset,VideoDataset,IntentVideoDataset, SensorDataset
from torch.utils.data import DataLoader
from trainer.models import *
from tqdm import tqdm
from utils import *
import wandb


from AI_Guide_Dog.trainer.dataset import SensorDataset


class Trainer:
    # initialize a new trainer
    def __init__(self, config_dict, train_transforms, val_transforms, train_files, val_files, df_videos, df_sensor,
                 test_videos = None, test_sensor = None):
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("\nCurrent device is: ", self.device, " \n")

        self.config = config_dict
        
        self.seq_len = config_dict['data']['SEQUENCE_LENGTH']
        self.epochs = config_dict['trainer']['epochs']

        self.model_name = config_dict['trainer']['model']['name']
        self.data_type = config_dict['trainer']['data_type']
        model_config = config_dict['trainer']['model']


        if (config_dict['global']['enable_intent']):
            self.train_dataset = IntentVideoDataset(df_videos, df_sensor, train_files, transforms=train_transforms,
                                                    seq_len=self.seq_len, config_dict=self.config)
            self.val_dataset = IntentVideoDataset(df_videos, df_sensor, val_files, transforms=val_transforms,
                                                  seq_len=self.seq_len, config_dict=self.config, test='validation')
        else: # PS: multiple and enable_intent are complimentary
            if self.data_type == "multimodal":  # If multi_modal training
                sensor_attr_list = model_config['sensor_attr_list']
                dense_frame_len = model_config['dense_frame_input_dim']
                self.train_dataset = SensorVideoDataset(df_videos, df_sensor, train_files, self.model_name,
                                                        transforms=train_transforms,
                                                        seq_len=self.seq_len, dense_frame_len=dense_frame_len,
                                                        sensor_attr_list=sensor_attr_list, config_dict=self.config)
                self.val_dataset = SensorVideoDataset(df_videos, df_sensor, val_files, self.model_name,
                                                      transforms=val_transforms,
                                                      seq_len=self.seq_len, dense_frame_len=dense_frame_len,
                                                      sensor_attr_list=sensor_attr_list, config_dict=self.config)
            elif self.data_type == 'sensor':
                sensor_attr_list = model_config['sensor_attr_list']
                self.train_dataset = SensorDataset(df_videos, df_sensor, train_files, self.seq_len,
                                                   sensor_attr_list=sensor_attr_list, config_dict=self.config)
                self.val_dataset = SensorDataset(df_videos, df_sensor, val_files, self.seq_len,
                                                 sensor_attr_list=sensor_attr_list, config_dict=self.config)
                self.test_dataset = SensorDataset(df_videos, df_sensor, val_files, self.seq_len,
                                                 sensor_attr_list=sensor_attr_list, config_dict=self.config)

            else:
                self.train_dataset = VideoDataset(df_videos, df_sensor, train_files, transforms=train_transforms,
                                                  seq_len=self.seq_len, config_dict=self.config)
                self.val_dataset = VideoDataset(df_videos, df_sensor, val_files, transforms=val_transforms,
                                                seq_len=self.seq_len, config_dict=self.config)
                if config_dict['transformer']['enable_benchmark_test'] and test_videos is not None:
                    self.test_dataset = VideoDataset(test_videos, test_sensor, list(test_videos.keys()), transforms=val_transforms,
                                                    seq_len=self.seq_len, config_dict=self.config)



        sampler = sampler_(self.train_dataset.y, config_dict['trainer']['num_classes'])
        train_args = dict(batch_size=config_dict['trainer']['BATCH'], sampler=sampler, num_workers=2, pin_memory=True,
                          drop_last=False) if self.cuda else dict(batch_size=config_dict['trainer']['BATCH'],
                                                                  sampler=sampler, drop_last=False)

        self.train_loader = DataLoader(self.train_dataset, **train_args)

        val_args = dict(shuffle=False, batch_size=config_dict['trainer']['BATCH'], num_workers=2, pin_memory=True, drop_last=False) if self.cuda else dict(shuffle=False, batch_size=config_dict['trainer']['BATCH'], drop_last=False)
        self.val_loader = DataLoader(self.val_dataset, **val_args)
       
        self.epochs = config_dict['trainer']['epochs']   
        self.hidden_dim =  model_config['sensor_hidden_dim']   
        self.layer_num =  model_config['layer_num']   




        #please note that channel need to be 3 for LSTM_Multimodal

        if self.model_name == "ConvLSTM":

            hidden_dim = [int(k.strip()) for k in config_dict['trainer']['model']['convlstm_hidden'].split(',')]

            channels = config_dict['data']['CHANNELS']
            if (config_dict['global']['enable_intent']):
                channels = channels + 1

            self.model = ConvLSTMModel(channels, hidden_dim, (3,3),
                                       config_dict['trainer']['model']['num_conv_lstm_layers'], config_dict['data']['HEIGHT'],
                                       config_dict['data']['WIDTH'], True)

        elif self.model_name == "LSTM_Multimodal":
            self.model = LSTMModel(input_dim = config_dict['trainer']['model']['dense_frame_input_dim'] + len(config_dict['trainer']['model']['sensor_attr_list']),
                                   layer_dim = config_dict['trainer']['model']['num_lstm_layers'], hidden_dim = config_dict['trainer']['model']['lstm_hidden'],
                                   num_classes = 3)
            elif self.model_name == "bert":
            self.model = Bert(self.device,
                              github_id=config_dict['wandb']['github_id'],
                              num_attr=len(model_config['sensor_attr_list']),
                              hidden_dim=self.hidden_dim,
                              data_type=self.data_type,
                              layer_num=self.layer_num
                              )
        else:
            print("Error parsing model name, please reverify model details in config.yaml")



        if config_dict['transformer']['enable_benchmark_test'] and test_videos is not None:
            if (config_dict['global']['enable_intent']):
                self.test_dataset = IntentVideoDataset(test_videos, test_sensor, list(test_videos.keys()),
                                                       transforms=val_transforms, seq_len=self.seq_len,
                                                       config_dict=self.config, test='benchmark_test')
            elif self.data_type == "multimodal":
                sensor_attr_list = model_config['sensor_attr_list']
                dense_frame_len = model_config['dense_frame_input_dim']
                self.test_dataset = SensorVideoDataset(test_videos, test_sensor, list(test_videos.keys()),
                                                 transforms=val_transforms,
                                                 seq_len=self.seq_len, config_dict=self.config)
            elif self.data_type == 'sensor':
                sensor_attr_list = model_config['sensor_attr_list']
                self.train_dataset = SensorDataset(df_videos, df_sensor, train_files, self.seq_len,
                                                   sensor_attr_list=sensor_attr_list,
                                                   seq_len=self.seq_len, config_dict=self.config)

            else:
                self.train_dataset = VideoDataset(test_videos, test_sensor, list(test_videos.keys()),
                                                  transforms=val_transforms,
                                                  seq_len=self.seq_len, config_dict=self.config)

            test_args = dict(shuffle=False, batch_size=config_dict['trainer']['BATCH'], num_workers=2, pin_memory=True,
                             drop_last=False) if self.cuda else dict(shuffle=False,
                                                                     batch_size=config_dict['trainer']['BATCH'],
                                                                     drop_last=False)
            self.test_loader = DataLoader(self.test_dataset, **test_args)



        if(config_dict['trainer']['model']['pretrained_path'] != ""):
            self.model.load_state_dict(torch.load(config_dict['trainer']['model']['pretained_path']))


        
        self.model = self.model.to(self.device)

        #Assigning more weight to left and right turns in loss calculation
        weights = [2.0,2.0,1.0]
        class_weights = torch.FloatTensor(weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=lamda, momentum=0.9)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config_dict['trainer']['lr'], weight_decay=config_dict['trainer']['lambda'])
        
        if(model_config['optimizer_path'] != ""):
            self.optimizer.load_state_dict(torch.load(model_config['optimizer_path']))

        # for g in optimizer.param_groups:
        #     g['lr'] = lr
        #     g['weight_decay']= lamda
            
        self.scaler = torch.cuda.amp.GradScaler()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(len(self.train_loader) * self.epochs))

        # print(self.model)


    def train(self, epoch):
        batch_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

        num_correct = 0
        total_loss = 0
        actual = []
        predictions = []
        
        for i, (x, y) in enumerate(self.train_loader):
        
            self.model.train()
            self.optimizer.zero_grad()
            
            x = x.float().to(self.device)
            y = y.to(self.device)
            
            
            with torch.cuda.amp.autocast():

                outputs = self.model(x)
                del x
                loss = self.criterion(outputs, y.long())
            pred_class = torch.argmax(outputs, axis=1)

            actual.extend(y.detach().cpu())
            predictions.extend(pred_class.detach().cpu())


            num_correct += int((pred_class == y).sum())
            del outputs
            total_loss += float(loss)
            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / ((i + 1) * self.config['trainer']['BATCH'])),
                loss="{:.04f}".format(float(total_loss / (i + 1))),
                num_correct=num_correct,
                lr="{:.04f}".format(float(self.optimizer.param_groups[0]['lr'])))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer) 
            self.scaler.update()

            self.scheduler.step()
            batch_bar.update() # Update tqdm bar

        batch_bar.close()
        acc = 100 * num_correct / (len(self.train_dataset))
        loss = total_loss / len(self.train_loader)
        print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
            epoch + 1,
            self.epochs,
            acc,
            float(loss),
            float(self.optimizer.param_groups[0]['lr'])))
        
        wandb.log({"loss": loss, "train_acc": acc, "lr": self.optimizer.param_groups[0]['lr'] })

        return actual, predictions

    
    def validate(self):
        self.model.eval()
        val_num_correct = 0

        actual = []
        predictions = []

        
        for i, (vx, vy) in tqdm(enumerate(self.val_loader)):
        
            vx = vx.to(self.device)
            vy = vy.to(self.device)

            with torch.no_grad():
                outputs = self.model(vx)
                del vx

            pred_class = torch.argmax(outputs, axis=1)

            actual.extend(vy.detach().cpu())
            predictions.extend(pred_class.detach().cpu())

            val_num_correct += int((pred_class == vy).sum())
            del outputs
         
            
        acc = 100 * val_num_correct / (len(self.val_dataset))
        print("Validation: {:.04f}%".format(acc))
        wandb.log({"val_acc": acc})
        return acc, actual, predictions


    # runs benchmark test at the end (after train and validation)
    def test(self):
        self.model.eval()
        val_num_correct = 0

        actual = []
        predictions = []

        for i, (vx, vy) in tqdm(enumerate(self.test_loader)):
            vx = vx.to(self.device)
            vy = vy.to(self.device)

            with torch.no_grad():
                outputs = self.model(vx)
                del vx

            pred_class = torch.argmax(outputs, axis=1)

            actual.extend(vy.detach().cpu())
            predictions.extend(pred_class.detach().cpu())

            val_num_correct += int((pred_class == vy).sum())
            del outputs

        acc = 100 * val_num_correct / (len(self.val_dataset))
        print("Benchmark test: {:.04f}%".format(acc))

        return acc, actual, predictions


    def save(self, acc, epoch):
        save(self.config, self.model, epoch, acc, optim = False)
        save(self.config, self.optimizer, epoch, acc, optim = True)
        