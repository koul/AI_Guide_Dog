import torch
from trainer.dataset import VideoDataset
from torch.utils.data import DataLoader
from trainer.models import *
from tqdm import tqdm
from utils import *

class Trainer:
    # initialize a new trainer
    def __init__(self, config_dict, train_transforms, val_transforms, train_files, test_files, df_videos, df_sensor):    
        self.cuda = torch.cuda.is_available()
        # print(self.cuda)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.config = config_dict
        self.seq_len = config_dict['data']['SEQUENCE_LENGTH']
        self.epochs = config_dict['trainer']['epochs']

        self.enable_qat = config_dict['trainer']['enable_qat']
        
        self.train_dataset = VideoDataset(df_videos, df_sensor, train_files, transforms=train_transforms, seq_len = self.seq_len, config_dict=self.config)
        
        # a,x = self.train_dataset.__getitem__(0)
        # b,y = self.train_dataset.__getitem__(1)
        # print(a.shape)
        # print(b.shape)
        # # print(self.train_dataset.__getitem__(0))
        # # print(self.train_dataset.__getitem__(1))
        # print(a[1,:,:,:] == b[0,:,:,:])
        # print(x)
        # print(y)
      
        sampler = sampler_(self.train_dataset.y, config_dict['trainer']['num_classes'])
        
        train_args = dict(batch_size=config_dict['trainer']['BATCH'], sampler = sampler, num_workers=2, pin_memory=True, drop_last=False) if self.cuda else dict(batch_size=config_dict['trainer']['BATCH'], sampler = sampler, drop_last=False)

        self.train_loader = DataLoader(self.train_dataset, **train_args)


        self.val_dataset = VideoDataset(df_videos, df_sensor, test_files, transforms=val_transforms, seq_len = self.seq_len, config_dict=self.config)

        val_args = dict(shuffle=False, batch_size=config_dict['trainer']['BATCH'], num_workers=2, pin_memory=True, drop_last=False) if self.cuda else dict(shuffle=False, batch_size=config_dict['trainer']['BATCH'], drop_last=False)

        self.val_loader = DataLoader(self.val_dataset, **val_args)
       
        self.epochs = config_dict['trainer']['epochs']

        self.model = ConvLSTMModel(config_dict['data']['CHANNELS'], config_dict['trainer']['model']['convlstm_hidden'],(3,3),config_dict['trainer']['model']['num_conv_lstm_layers'], config_dict['data']['HEIGHT'],config_dict['data']['WIDTH'], config_dict['trainer']['enable_qat'],True)

        if(config_dict['trainer']['model']['pretrained_path'] != ""):
            self.model.load_state_dict(torch.load(config_dict['trainer']['model']['pretained_path']))
        
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=lamda, momentum=0.9)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config_dict['trainer']['lr'], weight_decay=config_dict['trainer']['lambda'])
        
        if(config_dict['trainer']['model']['optimizer_path'] != ""):
            self.optimizer.load_state_dict(torch.load('./models/attempt3_1sec_prior/optimizer_params_00000000.pth'))

        # for g in optimizer.param_groups:
        #     g['lr'] = lr
        #     g['weight_decay']= lamda
            
        self.scaler = torch.cuda.amp.GradScaler()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(len(self.train_loader) * self.epochs))

        print(self.model)


    def train(self, epoch):
        batch_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

        num_correct = 0
        total_loss = 0

        self.model.train()

        if self.enable_qat:
            # TODO: try different global config parameters
            self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(self.model, inplace = True)
        
        for i, (x, y) in enumerate(self.train_loader):

            self.optimizer.zero_grad()

            x = x.float().to(self.device)
            y = y.to(self.device)
            
            with torch.cuda.amp.autocast():
                outputs = self.model(x)
                del x
                loss = self.criterion(outputs.view(-1, self.config['trainer']['num_classes']), y.long().view(-1))

            num_correct += int((torch.argmax(outputs, axis=1) == y).sum())
            del outputs
            total_loss += float(loss)

            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / ((i + 1) * self.config['trainer']['BATCH'] * self.seq_len)),
                loss="{:.04f}".format(float(total_loss / (i + 1))),
                num_correct=num_correct,
                lr="{:.04f}".format(float(self.optimizer.param_groups[0]['lr'])))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer) 
            self.scaler.update()

            self.scheduler.step()
            batch_bar.update() # Update tqdm bar
            

        batch_bar.close()
        acc = 100 * num_correct / (len(self.train_dataset) * self.seq_len)
        print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
            epoch + 1,
            self.epochs,
            acc,
            float(total_loss / len(self.train_loader)),
            float(self.optimizer.param_groups[0]['lr'])))

    
    def validate(self):

        self.model.eval()
        if self.enable_qat:
            self.model_int8 = torch.quantization.convert(self.model)
        val_num_correct = 0
        
        for i, (vx, vy) in tqdm(enumerate(self.val_loader)):
        
            vx = vx.to(self.device)
            vy = vy.to(self.device)

            with torch.no_grad():
                if self.enable_qat:
                    outputs = self.model_int8(vx)
                else:
                    outputs = self.model(vx)
                del vx

            val_num_correct += int((torch.argmax(outputs, axis=1) == vy).sum())
            del outputs

        print("Len of Val Dataset:", len(self.val_dataset))
        print("Seq Len:", self.seq_len)

        acc = 100 * val_num_correct / (len(self.val_dataset) * self.seq_len)
        print("Validation: {:.04f}%".format(acc))
        return acc

    def save(self, acc, epoch):
        save(self.config, self.model, epoch, acc, optim = False)
        save(self.config, self.optimizer, epoch, acc, optim = True)
        