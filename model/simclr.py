# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from .common import MLP, ResNet18
from .common import MaskNet18

import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia.augmentation as K
import kornia
from .buffer import Buffer
from copy import deepcopy
from torchvision import datasets, transforms

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
        self.net = MaskNet18(n_outputs, nf=nf)
        self.lr = args.lr
        
        self.transforms = nn.Sequential(K.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4))
        self.transforms0 = nn.Sequential(
           K.Normalize(torch.FloatTensor((0.5,0.5,0.5)), torch.FloatTensor((0.5,0.5,0.5))))

        self.beta = args.beta
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,  weight_decay = 1e-4)
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        self.n_outputs = n_outputs
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.n_memories = args.n_memories
        self.mem_cnt = 0       
        
        self.n_memories = n_outputs * self.n_memories
        self.buffer = Buffer(self.n_memories)
        
        self.bsz = args.batch_size
        
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        self.sz = args.replay_batch_size
        self.inner_steps = args.inner_steps
        self.n_outer = args.n_outer
        self.epoch = 0
    def on_epoch_end(self):  
        self.epoch += 1
        pass

    def compute_offsets(self, task):
        return 0, int(self.n_outputs)

    def forward(self, x, t, return_feat= False):
        if not self.training:
            #x = self.transforms0(x).cuda()
            x = x.cuda()
        else:
            x = self.transforms(x).cuda()
        output = self.net(x)    
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
            if not self.buffer.is_empty():
                xx, yy = self.buffer.get_data(self.bsz)
                x1,x2 = self.transforms(xx), self.transforms(xx)
                loss0 = 0.001 *  self.net.SimCLR(x1,x2)
                loss0.backward()
                self.opt.step()
        

        self.zero_grad()
        loss1 = torch.tensor(0.).cuda()
        loss2 = torch.tensor(0.).cuda()
        loss3 = torch.tensor(0.).cuda()
        
        offset1, offset2 = self.compute_offsets(t)
        pred = self.forward(x,t, True)
        loss1 = self.bce(pred, y)
        if not self.buffer.is_empty():
            xx , yy = self.buffer.get_data(self.bsz)
            pred = self.net(xx)
            loss2 += self.bce(pred, yy)
            #loss3 = self.reg * kl(pred , target , self.temp)
        loss = loss1 + loss2 + loss3
        loss.backward()
        self.opt.step()
        
        if self.epoch == 0:
            self.buffer.add_data(examples = x, labels = y)
        return 0.
