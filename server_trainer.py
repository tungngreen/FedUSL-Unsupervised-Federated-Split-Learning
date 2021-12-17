import os
import torch
import time
import datetime
import copy

import torch.nn as nn
from torch import random
from dataset import CIFAR10
from model import resnet20
from server_config import config

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

class server_trainer():
    def __init__(
        self,
        config
    ) -> None:
        self.num_clients = config.num_clients
        self.num_active_clients = config.num_active_clients
        self.run_idx = config.run_idx
        self.client_avg = config.client_avg
        self.server_avg = config.server_avg
        
        self.seed = config.seed
        self.test_size = config.test_size
        self.generated_label = config.generated_label

        self.is_training = config.is_training
        self.is_testing = config.is_testing
        self.resume = config.resume

        self.data_dir = config.data_dir
        self.output_dir = config.output_dir
        self.ckpt_save_dir = config.ckpt_save_dir
        self.ckpt_interval = config.ckpt_interval

        self.device = config.device
        self.num_workers = config.num_workers

        self.model_name = config.model_name
        self.pretrained_model = config.pretrained_model

        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.client_lr_scaled = config.client_lr_scaled
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay

        if self.client_avg and self.client_lr_scaled:
            self.client_learning_rate = self.learning_rate * self.num_active_clients
        else:
            self.client_learning_rate = self.learning_rate

        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
        self.run_name = config.resume
        if self.is_training:
            if self.resume == None:
                self.run_name = 'fed_savg_{}_cavg_{}_totalclients_{}_run_{}_lr_{}'.format(
                    self.server_avg,
                    self.client_avg,
                    self.num_active_clients,
                    self.run_idx,
                    self.learning_rate
                )
                if self.client_learning_rate != self.learning_rate:
                    self.run_name += '_scaled'

        self.parent_ckpt_save_dir = os.path.join(self.ckpt_save_dir, self.run_name)
        if not os.path.exists(self.parent_ckpt_save_dir):
            os.mkdir(self.parent_ckpt_save_dir)

        self.parent_output_dir = os.path.join(self.output_dir, self.run_name) 
        if not os.path.exists(self.parent_output_dir):
            os.mkdir(self.parent_output_dir)

        self.client_ckpt_save_dirs = []
        self.client_output_dirs = []
        self.client_log_files = []
        for client_idx in range(self.num_active_clients):

            self.client_ckpt_save_dirs.append(os.path.join(self.parent_ckpt_save_dir, 'client_{}'.format(client_idx)))
            if not os.path.exists(self.client_ckpt_save_dirs[client_idx]):
                os.mkdir(self.client_ckpt_save_dirs[client_idx])

            self.client_output_dirs.append(os.path.join(self.parent_output_dir, 'client_{}'.format(client_idx)))
            if not os.path.exists(self.client_output_dirs[client_idx]):
                os.mkdir(self.client_output_dirs[client_idx])

            self.client_log_files.append(os.path.join(self.client_output_dirs[client_idx], 'logs.txt'))

        config_file_copy = os.path.join(self.parent_output_dir, 'config.txt')                                                                                    
        with open(config_file_copy, 'w') as f:
            for arg, value in vars(config).items():
                f.write('--' + arg + '=' + str(value) + '\n')
                print('--' + arg + '=' + str(value))
            f.close()

        # Split the dataset into subsets for clients
        self.split_dataset()
        self.client_train_loaders = []
        for client_idx in range(self.num_active_clients):
            print(len(self.client_train_datasets[client_idx]))
            self.client_train_loaders.append(DataLoader(
                self.client_train_datasets[client_idx],
                batch_size = self.batch_size,
                shuffle = True,
                num_workers = self.num_workers,
                pin_memory = False
            ))
        self.test_loader = DataLoader(
            self.client_test_dataset,
            batch_size = 128,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = False
        )

        # Initialize the clients' model
        self.client_models = []
        self.client_optimizers = []
        self.client_weights = []
        for client_idx in range(self.num_active_clients):
            self.client_models.append(resnet20('client'))
            self.client_models[client_idx].to(self.device)

            self.client_optimizers.append(torch.optim.SGD(
                params = self.client_models[client_idx].parameters(),
                lr = self.client_learning_rate,
                momentum = self.momentum,
                weight_decay = self.weight_decay
            ))
            self.client_models[client_idx].to(self.device)
            self.client_weights.append(self.client_models[client_idx].state_dict())

        # initialize the server's model
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.server_model = resnet20('server')
        self.server_model.to(self.device)
        self.server_optimizer = torch.optim.SGD(
            params = self.server_model.parameters(),
            lr = self.learning_rate,
            momentum = self.momentum,
            weight_decay = self.weight_decay
        )

        self.train()

    def FedAvg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg

    def train(self):

        for epoch in range(self.num_epochs):
            if self.client_avg:
                self.global_client_weight = self.FedAvg(self.client_weights)
                self.client_weights = []
                
            for client_idx in range(self.num_active_clients):
                batch_time = AverageMeter()
                data_time = AverageMeter()
                losses = AverageMeter()
                top1 = AverageMeter()

                if self.client_avg:
                    self.client_models[client_idx].load_state_dict(self.global_client_weight)
                self.client_models[client_idx].train()
            
                with open(self.client_log_files[client_idx], 'a') as log_file:
                    for batch_idx, (imgs, targets, gen_targets) in enumerate(self.client_train_loaders[client_idx]):
                        start_time = time.time()
                        

                        imgs = imgs.to(self.device)
                        targets = targets.to(self.device)
                        if self.generated_label is not None:
                            labels = gen_targets.to(self.device)
                        else:
                            labels = targets.to(self.device)
                        
                        client_outputs = self.client_models[client_idx](imgs)

                        smashed_data = client_outputs.clone().detach().requires_grad_(True)
                        smashed_data = smashed_data.to(self.device)

                        outputs = self.server_model(smashed_data)
                        loss = self.criterion(outputs, labels)

                        self.server_optimizer.zero_grad()
                        loss.backward()
                        
                        client_grads = smashed_data.grad.clone().detach()
                        self.server_optimizer.step()

                        
                        self.client_optimizers[client_idx].zero_grad()
                        client_grads = client_grads.to(self.device)
                        client_outputs.backward(client_grads)
                        self.client_optimizers[client_idx].step()

                        outputs = outputs.float()
                        loss = loss.float()

                        accu = accuracy(outputs.data, targets)[0]
                        losses.update(loss.item(), imgs.size(0))
                        top1.update(accu.item(), imgs.size(0))

                        batch_time.update(time.time() - start_time)

                        log = 'TRAIN | [Epoch] {} | [Batch] {} | [client] {} | [accuracy] {:.2f}/{:.2f} | [train_loss] {:.2f}/{:.2f} | [time] {:.4f}/{:.4f}'.format(
                            epoch, 
                            batch_idx,
                            client_idx,
                            top1.val,
                            top1.avg,
                            losses.val,
                            losses.avg,
                            batch_time.val,
                            batch_time.avg
                        )
                        print(log)
                        log_file.write(log + '\n')
                    log_file.close()
                self.val(client_idx, epoch)
                if self.client_avg:
                    self.client_weights.append(self.client_models[client_idx].state_dict())

    def val(self, client_idx, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        self.client_models[client_idx].eval()
        self.server_model.eval()
        with open(self.client_log_files[client_idx], 'a') as log_file:
            with torch.no_grad():
                for batch_idx, (imgs, targets, gen_targets) in enumerate(self.test_loader):
                    start_time = time.time()
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)

                    if self.generated_label is not None:
                        labels = gen_targets.to(self.device)
                    else:
                        labels = targets.to(self.device)

                    smashed_data = self.client_models[client_idx](imgs)
                    outputs = self.server_model(smashed_data)
                    loss = self.criterion(outputs, labels)

                    outputs = outputs.float()
                    loss = loss.float()

                    accu = accuracy(outputs.data, targets)[0]
                    losses.update(loss.item(), imgs.size(0))
                    top1.update(accu.item(), imgs.size(0))
                    batch_time.update(time.time() - start_time)

                    batch_time.update(time.time() - start_time)

                    log = 'VALID | [Epoch] {} | [Batch] {} | [client] {} | [accuracy] {:.2f}/{:.2f} | [train_loss] {:.2f}/{:.2f} | [time] {:.4f}/{:.4f}'.format(
                        epoch, 
                        batch_idx,
                        client_idx,
                        top1.val,
                        top1.avg,
                        losses.val,
                        losses.avg,
                        batch_time.val,
                        batch_time.avg
                    )
                    print(log)
                    log_file.write(log + '\n')
            log_file.close()
        return top1.avg



    def split_dataset(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = CIFAR10(
            root = self.data_dir,
            train = False,
            client_train = True,
            generated_label = '/home/hihi/FedUSL/data/resnet56_generated.txt',
            train_transform = train_transform,
            test_transform = test_transform
        )
        test_dataset = CIFAR10(
            root = self.data_dir,
            train = False,
            client_train = False,
            generated_label = self.generated_label,
            train_transform = train_transform,
            test_transform = test_transform
        )
        dataset_len = len(test_dataset)
        
        _, self.client_test_dataset = random_split(
            test_dataset, 
            [dataset_len - self.test_size, self.test_size],
            generator = torch.Generator().manual_seed(self.seed[1])
        )
        total_train_dataset, _ = random_split(
            train_dataset, 
            [dataset_len - self.test_size, self.test_size],
            generator = torch.Generator().manual_seed(self.seed[1])
        )
        #total_train_dataset.train = True

        train_dataset_len = len(total_train_dataset)

        client_dataset_len = train_dataset_len // self.num_clients
        client_dataset_len_list = list([client_dataset_len] * self.num_clients)
        remain_len = train_dataset_len - client_dataset_len * self.num_clients
        if remain_len != 0:
            client_dataset_len_list.append(remain_len)

        client_datasets = random_split(
            total_train_dataset, 
            client_dataset_len_list,
            generator = torch.Generator().manual_seed(self.seed[0])
        )
        self.client_train_datasets = [client_datasets[i] for i in range(self.num_active_clients)]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk = (1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    trainer = server_trainer(config)


