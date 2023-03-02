# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from .sparse_common import MLP, ResNet18
from .sparse_common import MaskNet18

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
        self.current_task = 0
        # setup network
        self.is_cifar = any(x in str(args.data_file) for x in ['cifar', 'cub', 'mini'])
        self.is_cifar = False       
        nf = 64
        self.net = MaskNet18(n_outputs, nf=nf)
        self.lr = args.lr
           
        self.transforms = nn.Sequential(K.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4))
        self.mask = args.task_mask
        
        '''
        self.transforms = nn.Sequential(
            K.RandomCrop((32,32)), K.RandomHorizontalFlip(),
            K.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.2, hue=0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur((3,3),(0.1,2.0), p=1.0))
        self.transforms1 = nn.Sequential(
            K.RandomCrop((32,32)), K.RandomHorizontalFlip(),
            K.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.2, hue=0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur((3,3),(0.1,2.0), p=0.1))
        '''


        self.beta = args.beta
        #self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,  weight_decay = 1e-4)
        self.init_optim()
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        self.n_outputs = n_outputs
        
        self.fisher = {}
        self.optpar = {}
        self.n_memories = args.n_memories
        self.mem_cnt = 0       
        
        self.n_memories = n_outputs * self.n_memories
        print(self.n_memories)
        self.buffer = Buffer(self.n_memories)
        
        self.bsz = args.batch_size
        
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        self.sz = args.replay_batch_size
        self.inner_steps = args.inner_steps
        self.n_outer = args.n_outer
        self.epoch = 0
    def init_optim(self):
        if self.current_task > 0:
            del self.opt, self.scheduler
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr,  weight_decay = 1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'min', verbose=True, patience = 25)
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
        mask = output.ge(0.5).fill_(1)
        if isinstance(t, int):
            t = [t] * mask.size(0)
        
        for i in range(mask.size(0)):
            mask[i, self.mask[t[i]]] = False
        output = output.masked_fill(mask, 1e-9)
        return output
    
    def offset(self, t, sz = -1):
        if isinstance(t, int):
            offset = [max(self.mask[t])] * sz
        else:
            offset = [max(self.mask[tt]) for tt in t]
        return torch.tensor(offset).long().cuda()

    def observe(self, x, t, y):

        self.debugger += 1
        if t != self.current_task:
            self.current_task = t
            self.epoch = 0
            #self.init_optim()
        self.net.train()
        

        for _ in range(self.inner_steps):
            self.zero_grad()
            if not self.buffer.is_empty():
                xx, yy, buff_logits, _  = self.buffer.get_data(self.bsz * 2)
                x1,x2 = self.transforms(xx), self.transforms(xx)
                loss0 = 0.0001 *  self.net.BarlowTwins(x1,x2)
                loss0.backward()
                self.opt.step()


        self.zero_grad()
        loss1 = torch.tensor(0.).cuda()
        loss2 = torch.tensor(0.).cuda()
        loss3 = torch.tensor(0.).cuda()
        
        #offset1, offset2 = self.compute_offsets(t)
        #offset0 = self.offset(t, y.size(0))
        #y = y - offset0
        pred = self.forward(x,t, True)
        loss1 = self.bce(pred, y)
        
        if not self.buffer.is_empty():
            xx , yy, buff_logits, t_  = self.buffer.get_data(self.bsz)
            pred_ =  self.forward(xx, t_)
            loss2 +=  self.bce(pred_, yy)
            loss3 += 0.05 * F.mse_loss(pred_, buff_logits)
        loss = loss1 + loss2 + loss3
        loss.backward()
        self.opt.step()

        #if self.epoch == 0:
        tt = torch.Tensor([t] * y.size(0)).long().cuda()
        self.buffer.add_data(examples = x, labels = y, logits = pred.data, task_labels = tt)
        return 0.
