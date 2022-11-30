import torch
from trainer.dataset import VideoDataset,IntentVideoDataset
from torch.utils.data import DataLoader
from trainer.models import *
from tqdm import tqdm
from trainer.predrnn import PredRnnModel
from utils import *
from trainer.loss import get_focal_loss, get_inverse_class_freq

import wandb

class Trainer():
    # initialize a new trainer
    def __init__(self, config_dict, train_transforms, val_transforms, train_files, val_files, df_videos, df_sensor,
                 test_videos = None, test_sensor = None, wandb = None):
        self.cuda = torch.cuda.is_available()
        print(self.cuda)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.config = config_dict
        self.seq_len = config_dict['data']['SEQUENCE_LENGTH']
        self.epochs = config_dict['trainer']['epochs']

        if(config_dict['global']['enable_intent']):
            self.train_dataset = IntentVideoDataset(df_videos, df_sensor, train_files, transforms=train_transforms, seq_len = self.seq_len, config_dict=self.config, train=True)
            self.val_dataset = IntentVideoDataset(df_videos, df_sensor, val_files, transforms=val_transforms, seq_len = self.seq_len, config_dict=self.config, test='validation')
        else:
            self.train_dataset = VideoDataset(df_videos, df_sensor, train_files, transforms=train_transforms, seq_len = self.seq_len, config_dict=self.config, train=True)
            self.val_dataset = VideoDataset(df_videos, df_sensor, val_files, transforms=val_transforms, seq_len = self.seq_len, config_dict=self.config)

        sampler = sampler_(self.train_dataset.y, config_dict['trainer']['num_classes'])
        train_args = dict(batch_size=config_dict['trainer']['BATCH'], sampler = sampler, num_workers=2, pin_memory=True, drop_last=False) if self.cuda else dict(batch_size=config_dict['trainer']['BATCH'], sampler = sampler, drop_last=False)
        self.train_loader = DataLoader(self.train_dataset, **train_args)

        val_args = dict(shuffle=False, batch_size=config_dict['trainer']['BATCH'], num_workers=2, pin_memory=True, drop_last=False) if self.cuda else dict(shuffle=False, batch_size=config_dict['trainer']['BATCH'], drop_last=False)
        self.val_loader = DataLoader(self.val_dataset, **val_args)

        # TODO: Add test loader with sampler - try with replacement to True and False
        if config_dict['transformer']['enable_benchmark_test'] and test_videos is not None:
            if(config_dict['global']['enable_intent']):
                self.test_dataset = IntentVideoDataset(test_videos, test_sensor, list(test_videos.keys()), transforms=val_transforms, seq_len = self.seq_len, config_dict=self.config, test= 'benchmark_test')
            else:
                self.test_dataset = VideoDataset(test_videos, test_sensor, list(test_videos.keys()), transforms=val_transforms,
                                            seq_len=self.seq_len, config_dict=self.config)

            test_args = dict(shuffle=False, batch_size=config_dict['trainer']['BATCH'], num_workers=2, pin_memory=True,
                            drop_last=False) if self.cuda else dict(shuffle=False,
                                                                    batch_size=config_dict['trainer']['BATCH'],
                                                                    drop_last=False)
            self.test_loader = DataLoader(self.test_dataset, **test_args)



        self.epochs = config_dict['trainer']['epochs']
        hidden_dim = [int(k.strip()) for k in config_dict['trainer']['model']['convlstm_hidden'].split(',')]

        channels = config_dict['data']['CHANNELS']
        if(config_dict['global']['enable_intent']):
            channels = channels + 1

        self.model = ConvLSTMModel(channels, hidden_dim,(3,3),config_dict['trainer']['model']['num_conv_lstm_layers'], config_dict['data']['HEIGHT'],config_dict['data']['WIDTH'],True)

        if(config_dict['trainer']['model']['pretrained_path'] != ""):
            self.model.load_state_dict(torch.load(config_dict['trainer']['model']['pretained_path']))

        self.model = self.model.to(self.device)

        #Assigning more weight to left and right turns in loss calculation
        weights = [2.0,2.0,1.0]
        class_weights = torch.FloatTensor(weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=lamda, momentum=0.9)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config_dict['trainer']['lr'], weight_decay=config_dict['trainer']['lambda'])

        if(config_dict['trainer']['model']['optimizer_path'] != ""):
            self.optimizer.load_state_dict(torch.load(config_dict['trainer']['model']['optimizer_path']))

        # for g in optimizer.param_groups:
        #     g['lr'] = lr
        #     g['weight_decay']= lamda

        self.scaler = torch.cuda.amp.GradScaler()
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(len(self.train_loader) * self.epochs))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config_dict['trainer']['max_lr'], steps_per_epoch=len(self.train_loader), epochs=self.epochs)

        print(self.model)


    def train(self, epoch):
        batch_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

        num_correct = 0.0
        total_loss = 0.0
        y_cnt = 0.0
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
            total_loss += (float(loss)*len(y))
            y_cnt += len(y)

            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * float(num_correct) / y_cnt),
                loss="{:.04f}".format(float(total_loss) / y_cnt),
                num_correct=num_correct,
                lr="{:.04f}".format(float(self.optimizer.param_groups[0]['lr'])))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()
            batch_bar.update() # Update tqdm bar


        batch_bar.close()
        total_loss = float(total_loss) / len(self.train_dataset)
        acc = 100 * float(num_correct) / (len(self.train_dataset))
        print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
            epoch + 1,
            self.epochs,
            acc,
            float(total_loss),
            float(self.optimizer.param_groups[0]['lr'])))

        if(self.wandb is not None):
            self.wandb.log({"Train Loss": total_loss, "Train Accuracy": acc, "Learning Rate": float(self.optimizer.param_groups[0]['lr'])})


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



        acc = 100 * float(val_num_correct) / (len(self.val_dataset))
        print("Validation: {:.04f}%".format(acc))

        return acc, actual, predictions


    # runs benchmark test at the end (after train and validation)
    def test(self):
        self.model.eval()
        test_num_correct = 0

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

            test_num_correct += int((pred_class == vy).sum())
            del outputs

        acc = 100 * test_num_correct / (len(self.test_dataset))
        print("Benchmark test: {:.04f}%".format(acc))

        return acc, actual, predictions


    def save(self, acc, epoch):
        save(self.config, self.model, epoch, acc, optim = False)
        save(self.config, self.optimizer, epoch, acc, optim = True)


class TrainerPredRNN():
    # initialize a new trainer

    def __init__(self, config_dict, train_transforms, val_transforms, train_files, val_files, df_videos, df_sensor,
                 test_videos = None, test_sensor = None):
        self.cuda = torch.cuda.is_available()
        print(self.cuda)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.config = config_dict
        self.seq_len = config_dict['data']['SEQUENCE_LENGTH']
        self.epochs = config_dict['trainer']['epochs']

        if(config_dict['global']['enable_intent']):
            self.train_dataset = IntentVideoDataset(df_videos, df_sensor, train_files, transforms=train_transforms, seq_len = self.seq_len, config_dict=self.config)
            self.val_dataset = IntentVideoDataset(df_videos, df_sensor, val_files, transforms=val_transforms, seq_len = self.seq_len, config_dict=self.config, test='validation')
        else:
            self.train_dataset = VideoDataset(df_videos, df_sensor, train_files, transforms=train_transforms, seq_len = self.seq_len, config_dict=self.config)
            self.val_dataset = VideoDataset(df_videos, df_sensor, val_files, transforms=val_transforms, seq_len = self.seq_len, config_dict=self.config)

        # sampler = sampler_(self.train_dataset.y, config_dict['trainer']['num_classes'])
        # train_args = dict(batch_size=config_dict['trainer']['BATCH'], sampler = sampler, num_workers=2, pin_memory=True, drop_last=False) if self.cuda else dict(batch_size=config_dict['trainer']['BATCH'], sampler = sampler, drop_last=False)
        train_args = dict(batch_size=config_dict['trainer']['BATCH'], num_workers=2, pin_memory=True, drop_last=False) if self.cuda else dict(batch_size=config_dict['trainer']['BATCH'], drop_last=False)
        self.train_loader = DataLoader(self.train_dataset, **train_args)

        # self.val_dataset = VideoDataset(df_videos, df_sensor, val_files, transforms=val_transforms, seq_len = self.seq_len, config_dict=self.config)


        val_args = dict(shuffle=False, batch_size=config_dict['trainer']['BATCH'], num_workers=2, pin_memory=True, drop_last=False) if self.cuda else dict(shuffle=False, batch_size=config_dict['trainer']['BATCH'], drop_last=False)
        self.val_loader = DataLoader(self.val_dataset, **val_args)

        # TODO: Add test loader with sampler - try with replacement to True and False
        if config_dict['transformer']['enable_benchmark_test'] and test_videos is not None:
            if(config_dict['global']['enable_intent']):
                self.test_dataset = IntentVideoDataset(test_videos, test_sensor, list(test_videos.keys()), transforms=val_transforms, seq_len = self.seq_len, config_dict=self.config, test= 'benchmark_test')
            else:
                self.test_dataset = VideoDataset(test_videos, test_sensor, list(test_videos.keys()), transforms=val_transforms,
                                            seq_len=self.seq_len, config_dict=self.config)

            test_args = dict(shuffle=False, batch_size=config_dict['trainer']['BATCH'], num_workers=2, pin_memory=True,
                            drop_last=False) if self.cuda else dict(shuffle=False,
                                                                    batch_size=config_dict['trainer']['BATCH'],
                                                                    drop_last=False)
            self.test_loader = DataLoader(self.test_dataset, **test_args)

        channels = config_dict['data']['CHANNELS']
        if(config_dict['global']['enable_intent']):
            channels = channels + 1

        self.model = PredRnnModel(channels, config_dict['trainer']['model']['convlstm_hidden'],(3,3),config_dict['trainer']['model']['num_conv_lstm_layers'], config_dict['data']['HEIGHT'],config_dict['data']['WIDTH'],True, conv_dropout=config_dict['trainer']['model']['conv_dropout'])

        if(config_dict['trainer']['model']['pretrained_path'] != ""):
            self.model.load_state_dict(torch.load(config_dict['trainer']['model']['pretrained_path']))
            print('Loaded Model state')

        self.model = self.model.to(self.device)

        # weights = [2.0,2.0,1.0]
        # class_weights = torch.FloatTensor(weights).to(self.device)
        # self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Focal Loss #
        alpha_weights = get_inverse_class_freq(self.train_dataset.y, config_dict['trainer']['num_classes'])
        self.criterion = get_focal_loss(alpha=alpha_weights, gamma=config_dict['trainer']['model']['focal_loss_gamma'], device=self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config_dict['trainer']['lr'], weight_decay=config_dict['trainer']['lambda'])

        if(config_dict['trainer']['model']['optimizer_path'] != ""):
            self.optimizer.load_state_dict(torch.load(config_dict['trainer']['model']['optimizer_path']))
            print('Loaded Optimizer state')
            if config_dict['trainer']['model']['lr_after_load'] != "":
                for g in self.optimizer.param_groups:
                    g['lr'] = config_dict['trainer']['model']['lr_after_load']

                print(f"LR set to {config_dict['trainer']['model']['lr_after_load']}")

        # for g in optimizer.param_groups:
        #     g['lr'] = lr
        #     g['weight_decay']= lamda

        self.scaler = torch.cuda.amp.GradScaler()
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(len(self.train_loader) * self.epochs))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min',
                                                                    factor=0.75,
                                                                    patience=1)

        if(config_dict['trainer']['model']['scheduler_path'] != ""):
            self.scheduler.load_state_dict(torch.load(config_dict['trainer']['model']['scheduler_path']))
            print('Loaded Scheduler state')

        self.decouple_beta = config_dict['trainer']['model']['decouple_beta']

        print(self.model)


    def train(self, epoch):
        batch_bar = tqdm(total=len(self.train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

        num_correct = 0
        total_loss = 0
        total_ce_loss = 0
        total_decouple_loss = 0
        actual = []
        predictions = []

        for i, (x, y) in enumerate(self.train_loader):

            self.model.train()
            self.optimizer.zero_grad()

            x = x.float().to(self.device)
            y = y.to(self.device)

            with torch.cuda.amp.autocast():
                outputs, decouple_loss = self.model(x)
                del x
                CE_loss = self.criterion(outputs, y.long())
                decouple_loss = self.decouple_beta * decouple_loss # Add Decouple loss to objective to decouple C and M
                loss = CE_loss + decouple_loss

            pred_class = torch.argmax(outputs, axis=1)

            y_cpu = y.cpu()
            pred_class_cpu = pred_class.cpu()
            actual.extend(y_cpu)
            predictions.extend(pred_class_cpu)

            num_correct += int((pred_class_cpu == y_cpu).sum())
            del outputs
            total_loss += float(loss)
            total_ce_loss += float(CE_loss)
            total_decouple_loss += float(decouple_loss)
            # print(f'Batch ce_loss {CE_loss.item() / len(y)}')
            # print(f'Batch decouple_loss {decouple_loss.item() / len(y)}')

            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / ((i + 1) * self.config['trainer']['BATCH'])),
                loss="{:.04f}".format(float(total_loss / (i + 1))),
                num_correct=num_correct,
                lr="{:.04f}".format(float(self.optimizer.param_groups[0]['lr'])))

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # self.scheduler.step()
            batch_bar.update() # Update tqdm bar


        batch_bar.close()
        acc = 100 * num_correct / (len(self.train_dataset))
        print("Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
            epoch + 1,
            self.epochs,
            acc,
            float(total_loss / len(self.train_loader)),
            float(total_ce_loss / len(self.train_loader)),
            float(total_decouple_loss / len(self.train_loader)),
            float(self.optimizer.param_groups[0]['lr'])))

        return acc, actual, predictions


    def validate(self):
        self.model.eval()
        val_num_correct = 0
        actual = []
        predictions = []
        total_loss = 0.
        total_ce_loss = 0.
        total_decouple_loss = 0.

        for i, (vx, vy) in tqdm(enumerate(self.val_loader)):

            vx = vx.to(self.device)
            vy = vy.to(self.device)

            with torch.no_grad():
                outputs, decouple_loss = self.model(vx)
                del vx
                CE_loss = self.criterion(outputs, vy.long())
                decouple_loss = self.decouple_beta * decouple_loss # Add Decouple loss to objective to decouple C and M
                loss = CE_loss + decouple_loss

            pred_class = torch.argmax(outputs, axis=1)

            vy_cpu = vy.cpu()
            pred_class_cpu = pred_class.cpu()

            actual.extend(vy_cpu)
            predictions.extend(pred_class_cpu)

            total_loss += float(loss)
            total_ce_loss += float(CE_loss)
            total_decouple_loss += float(decouple_loss)
            val_num_correct += int((pred_class_cpu == vy_cpu).sum())
            del outputs

        acc = 100 * val_num_correct / (len(self.val_dataset))
        avg_val_loss = float(total_loss / len(self.val_dataset))

        print("Validation Acc {:.04f}%, Validation Loss {:.04f}, Learning Rate {:.04f}".format(
            acc,
            float(total_loss / len(self.val_dataset)),
            float(self.optimizer.param_groups[0]['lr'])))

        self.scheduler.step(total_loss)

        return avg_val_loss, acc, actual, predictions

    # runs benchmark test at the end (after train and validation)
    def test(self):
        self.model.eval()
        test_num_correct = 0

        actual = []
        predictions = []

        for i, (vx, vy) in tqdm(enumerate(self.test_loader)):
            vx = vx.to(self.device)
            vy = vy.to(self.device)

            with torch.no_grad():
                outputs, _ = self.model(vx)
                del vx

            pred_class = torch.argmax(outputs, axis=1)

            actual.extend(vy.cpu())
            predictions.extend(pred_class.cpu())

            test_num_correct += int((pred_class == vy).sum())
            del outputs

        acc = 100 * test_num_correct / (len(self.test_dataset))
        print("Benchmark test: {:.04f}%".format(acc))

        return acc, actual, predictions

    def save(self, acc, epoch):
        save(self.config, self.model, epoch, acc, save_type='model')
        save(self.config, self.optimizer, epoch, acc, save_type='optim')
        save(self.config, self.scheduler, epoch, acc, save_type='scheduler')
