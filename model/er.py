# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from .common import MLP, ResNet18, ResNet32
from .common import MaskNet18

import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia.augmentation as K
import kornia
from .buffer import Buffer, Cutout, cutmix_data
from copy import deepcopy
from torchvision import datasets, transforms
from torch import optim

kl = lambda y, t_s, t : F.kl_div(F.log_softmax(y / t, dim=-1), F.softmax(t_s / t, dim=-1), reduce=True) * y.size(0)

class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.reg = args.memory_strength
        self.temp = args.temperature
        self.debugger = 0
        # setup network
        self.is_cifar = any(x in str(args.data_file) for x in ['cifar', 'cub', 'mini'])
        self.is_cifar = False       
        nf = 64 
        self.net = ResNet18(n_outputs, nf=nf)
        #self.net = ResNet32(n_outputs, nf=nf)
        self.lr = args.lr
        
        #self.transforms = nn.Sequential(K.RandomHorizontalFlip(),
        #    transforms.RandomCrop(64))
        mean = (0.485, 0.456, 0.406)
        std =  (0.229, 0.224, 0.225)
        sz = 64
        self.transforms = transforms.Compose([
            transforms.Resize((sz,sz)),
            transforms.RandomCrop(sz, padding=4),
            transforms.RandomHorizontalFlip(),
            #Cutout(16),
            transforms.AutoAugment(transforms.AutoAugmentPolicy('imagenet')),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean,std)]
        )
        self.transforms0 = transforms.Compose([
            transforms.Resize((sz,sz)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean,std)
        ]
        )

        self.beta = args.beta
        #self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        #self.opt = torch.optim.Adam(self.net.parameters(), lr = 3e-4)
        #self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.99995)
        self.update_schedule(True)
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        self.n_outputs = n_outputs
        self.current_task = 0
        self.n_memories = args.n_memories
        self.n_memories = self.n_memories
        self.buffer = Buffer(self.n_memories, mode='ring')
        self.nc_per_task = n_outputs // n_tasks
        self.bsz = args.batch_size 
        self.inner_steps = args.inner_steps
        self.n_outer = args.n_outer
        self.epoch = 0
    def on_epoch_end(self):  
        self.epoch += 1
        pass
    def on_task_end(self):
        self.epoch = 0
        self.update_schedule(True)

    def compute_offsets(self, task):
        end = self.nc_per_task * (task + 1)
        return int(end)

    def update_schedule(self, reset = False):
        if reset:
            self.opt = torch.optim.Adam(self.net.parameters(), lr = 3e-4)
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.99995)
        else:
            self.scheduler.step()

    def forward(self, x, t, return_feat= False):
        if self.training:
            x = self.transforms(x).cuda()
        else:
            #pdb.set_trace()
            x = self.transforms0(x).cuda()
        output = self.net(x)    
        end = self.compute_offsets(t)
        output[:, end:].fill_(-1e10)
        return output
    
    def observe(self, x, t, y):
        
        #t = info[0]
        self.debugger += 1
        if t != self.current_task:
            self.current_task = t
            self.epoch = 0

        self.net.train()
        
        for _ in range(self.inner_steps):
            self.zero_grad()
                        
            offset = self.compute_offsets(t)
            if not self.buffer.is_empty():
                xx , yy, _ = self.buffer.get_data(self.bsz)
                
                x = torch.cat([x,xx], 0)
                y= torch.cat([y,yy], 0)
                
            if np.random.rand(1) < 0.5:    
                logit = self.forward(x,t)
                loss = self.bce(logit, y)
            else:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                logit = self.forward(x, t)
                loss = lam * self.bce(logit, labels_a) + (1-lam)*self.bce(logit, labels_b)
                
            loss.backward()
            self.opt.step()
            self.update_schedule(False)
        #if self.epoch == 0:
        tt = torch.Tensor([t] * y.size(0)).long().cuda()
        self.buffer.add_data(examples = x, labels = y, task_labels = tt)
        
        return 0.
