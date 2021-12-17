import os
import torch
import time
import datetime

import torch.nn as nn
from torch import random
from dataset import CIFAR10
from model import resnet20
from client_config import config

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

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

class client_trainer():
    def __init__(
        self,
        config
    ) -> None:
        self.num_clients = config.num_clients
        self.num_active_clients = config.num_active_clients
        self.client_idx = config.client_idx
        self.run_idx = config.run_idx
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
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay

        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
        self.run_name = config.resume
        if self.is_training:
            if self.resume == None:
                self.run_name = 'local_client_{}_over_{}_total_{}_run_{}_lr_{}'.format(
                    self.client_idx, 
                    self.num_active_clients,
                    self.num_clients,
                    self.run_idx,
                    self.learning_rate
                )
                if self.generated_label is None:
                    self.run_name = 'gt_' + self.run_name

        self.ckpt_save_dir = os.path.join(self.ckpt_save_dir, self.run_name)
        if not os.path.exists(self.ckpt_save_dir):
            os.mkdir(self.ckpt_save_dir)

        self.output_dir = os.path.join(config.output_dir, self.run_name) 
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        self.log_file = os.path.join(self.output_dir, 'logs.txt')

        config_file_copy = os.path.join(self.output_dir, 'config.txt')                                                                                    
        with open(config_file_copy, 'w') as f:
            for arg, value in vars(config).items():
                f.write('--' + arg + '=' + str(value) + '\n')
                print('--' + arg + '=' + str(value))
            f.close()
        

        self.split_dataset()

        self.client_model = resnet20('full_client')
        self.client_model.to(self.device)

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(
            params = self.client_model.parameters(),
            lr = self.learning_rate,
            momentum = self.momentum,
            weight_decay = self.weight_decay
        )
        self.train_loader = DataLoader(
            self.client_train_datasets[self.client_idx],
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            pin_memory = False
        )
        self.test_loader = DataLoader(
            self.client_test_dataset,
            batch_size = 128,
            shuffle = False,
            num_workers = self.num_workers,
            pin_memory = False
        )

        self.train()

    def train(self):

        for epoch in range(self.num_epochs):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()

            self.client_model.train()
            
            with open(self.log_file, 'a') as log_file:
                for batch_idx, (imgs, targets, gen_targets) in enumerate(self.train_loader):
                    start_time = time.time()
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)
                    if self.generated_label is not None:
                        labels = gen_targets.to(self.device)
                    else:
                        labels = targets.to(self.device)
                    
                    outputs = self.client_model(imgs)
                    loss = self.criterion(outputs, labels)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    outputs = outputs.float()
                    loss = loss.float()

                    accu = accuracy(outputs.data, targets)[0]
                    losses.update(loss.item(), imgs.size(0))
                    top1.update(accu.item(), imgs.size(0))

                    batch_time.update(time.time() - start_time)

                    log = 'TRAIN | [Epoch] {} | [Batch] {} | [client] {} | [accuracy] {:.2f}/{:.2f} | [train_loss] {:.2f}/{:.2f} | [time] {:.4f}/{:.4f}'.format(
                        epoch, 
                        batch_idx,
                        self.client_idx,
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
            self.val(epoch)

    def val(self, epoch):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        self.client_model.eval()
        with open(self.log_file, 'a') as log_file:
            with torch.no_grad():
                for batch_idx, (imgs, targets, gen_targets) in enumerate(self.test_loader):
                    start_time = time.time()
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)

                    if self.generated_label is not None:
                        labels = gen_targets.to(self.device)
                    else:
                        labels = targets.to(self.device)

                    outputs = self.client_model(imgs)
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
                        self.client_idx,
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

if __name__ == '__main__':
    trainer = client_trainer(config)


