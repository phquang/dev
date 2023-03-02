import os
import sys
import pdb
import torch
import numpy as np
import pickle as pkl
from PIL import Image
from random import shuffle
import pdb
from torchvision import datasets, transforms
from continuum.datasets import CTRL
from continuum.datasets import MNIST, CIFAR10, FashionMNIST, SVHN
import ctrl
import itertools

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):            
        return self.tensors[0].size(0)

class XYDataset(torch.utils.data.Dataset):
    def __init__(self, x,  y,  **kwargs):
        self.x, self.y  = x, y

        # this was to store the inverse permutation in permuted_mnist
        # so that we could 'unscramble' samples and plot them
        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x,  y  = self.x[idx], self.y[idx]
        if type(x) != torch.Tensor:
            x = self.transform(Image.open(x).convert('RGB'))
            y = torch.Tensor(1).fill_(y).long().squeeze()
        else:
            x = x.float() / 255.
            y = y.long()
        

        #return (x - 0.5) * 2,  y
        return x, y
""" Template Dataset for Continual Learning """
class CLDataLoader(object):
    def __init__(self, datasets_per_task, args, train=True):
        bs = args.batch_size if train else 512
       
        self.datasets = datasets_per_task
        self.loaders = [
                torch.utils.data.DataLoader(x, batch_size=bs, shuffle=True, drop_last=False, num_workers=2, pin_memory=False)
                for x in self.datasets ]

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)

class CLwithDomain():
    def __init__(self, train_csv, test_csv, n_tasks, augmentation=False):
        self.train_csv = train_csv
        self.test_csv = test_csv

        self.lb = 0
        self.lb_dict = {}

        self.n_tasks = n_tasks
        self.augmentation = augmentation
        
        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((128,128)), transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((84,84)),
                #transforms.ToTensor()])
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        '''
        self.transform = transforms.Compose([
                transforms.Resize((84,84)),
                transforms.ToTensor()])
        '''
    def get_data(self, setname):
        csv_path = self.train_csv if setname =='train' else self.test_csv
        lines = [x.strip() for x in open(csv_path, 'r').readlines()]
        data = []
        labels = []
        domain_labels = []

        for l in lines:
            path, label = l.split(',')
            if label not in self.lb_dict.keys():
                self.lb_dict[label] = self.lb
                self.lb += 1

            data.append(path)
            labels.append(self.lb_dict[label])
        return np.array(data),  np.array(labels)
    
    def meta_data(self):
        return self.lb, self.total_domains

    def build_benchmark(self, args):
        train_data,  train_label = self.get_data('train')
        test_data,  test_label = self.get_data('test')

        n_classes = int(np.unique(train_label).shape[0])
        
        assert n_classes % self.n_tasks == 0
        n_classes_per_task = n_classes // self.n_tasks

        train_ds, test_ds = [], []
        current_train, current_test = None, None
        cat = lambda x,y: np.concatenate((x,y), axis=0)
        
        for i in range(n_classes):
            train_indices = np.argwhere(train_label == i).reshape(-1)
            test_indices  = np.argwhere(test_label == i).reshape(-1)
            
            x_tr = train_data[train_indices]
            y_tr = train_label[train_indices]

            x_te = test_data[test_indices]
            y_te = test_label[test_indices]

            if current_train is None:
                current_train, current_test = (x_tr, y_tr), (x_te, y_te)
            else:
                current_train = cat(current_train[0], x_tr), cat(current_train[1], y_tr)
                current_test = cat(current_test[0], x_te) , cat(current_test[1], y_te)

            if i % n_classes_per_task == (n_classes_per_task - 1):
                train_ds += [current_train]
                test_ds += [current_test]
                current_train, current_test = None, None

        masks = []
        task_ids = [None for _ in range(self.n_tasks)]
        for task, task_data in enumerate(train_ds):
            labels = np.unique(task_data[1])
            assert labels.shape[0] == n_classes_per_task
            mask = torch.zeros(n_classes).cuda()
            mask[labels] = 1
            masks.append(mask)
            task_ids[task] = labels
        task_ids = torch.from_numpy(np.stack(task_ids)).cuda().long()
        #train_ds, val_ds = make_valid_from_train(train_ds, cut=0.99)
        train_ds = map(lambda x,y: XYDataset(x[0],x[1],**{'source':'data', 'mask':y, 'task_ids':task_ids, 'transform':self.transform}), train_ds, masks)
        #val_ds = map(lambda x, y: XYDataset(x[0],x[1],**{'source':'data', 'mask':y, 'task_ids':task_ids, 'transform':self.transform}), val_ds , masks)
        test_ds = map(lambda x,y: XYDataset(x[0], x[1],**{'source':'data', 'mask':y, 'task_ids':task_ids, 'transform':self.transform}), test_ds , masks)

        #data = (train_ds, val_ds, test_ds)
        #train_loader, val_loader, test_loader  = [CLDataLoader(elem, args, train=t) \
        #        for elem, t in zip(data, [True, False, False])]
        data = (train_ds, test_ds)
        train_loader,  test_loader  = [CLDataLoader(elem, args, train=t) \
                for elem, t in zip(data, [True,  False])]
        return train_loader, 0 , test_loader

def make_valid_from_train(dataset, cut=0.95):
    tr_ds, val_ds = [], []
    for task_ds in dataset:
        x_t,  y_t = task_ds

        # shuffle before splitting
        perm = torch.randperm(len(x_t))
        x_t,  y_t = x_t[perm], y_t[perm]

        split = int(len(x_t) * cut)
        x_tr,  y_tr = x_t[:split], y_t[:split]
        x_val,  y_val = x_t[split:], y_t[split:]

        tr_ds  += [(x_tr, y_tr )]
        val_ds += [(x_val, y_val)]

    return tr_ds, val_ds




class CTRLlong(CTRL):
    def __init__(self, data_path: str = "", split: str = "train", download: bool = True, seed: int = 1):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split must be train, val, or test; not {split}.")
        train = split in ("train", "val")

        rng = np.random.RandomState(seed=seed)

        base_datasets = [
            MNIST(data_path=data_path, train=train, download=download),
            SVHN(data_path=data_path, train=train, download=download),
            FashionMNIST(data_path=data_path, train=train, download=download),
            CIFAR10(data_path=data_path, train=train, download=download)
        ]

        svhn_mnist_share_labels = True
        if svhn_mnist_share_labels:
            task_counter = [0, 0, 10, 10]
        else:
            task_counter = [0, 10, 20, 30]

        proportions_per_class = [1000, 1000, 1000, 500]
        dataset_sample_prob = [0.2, 0.2, 0.3, 0.3]
        nb_classes = 5
        nb_tasks = 30

        datasets, class_counter, class_subsets, proportions = [], [], [], []
        for _ in range(nb_tasks):
            dataset_id = rng.choice([0, 1, 2, 3], p=dataset_sample_prob)
            datasets.append(base_datasets[dataset_id])
            class_counter.append(task_counter[dataset_id])
            class_subsets.append(rng.choice(10, size=nb_classes, replace=False))

            if split == "train":
                proportions.append(proportions_per_class[dataset_id])
            elif split == "val":
                proportions.append(proportions_per_class[dataset_id] // 2)
            else:
                proportions.append(None)

        super().__init__(
            datasets=datasets,
            proportions=proportions,
            class_counter=class_counter,
            class_subsets=class_subsets,
            seed=seed,
            split=split,
            target_size=(32, 32)
        )

def create_ctrl(args):
    task_gen = ctrl.get_stream(args.train_csv, seed = 180)
    datasets = [[],[],[]]
    nb_classes = 0
    nb_tasks = 0
    yy = []
    for task in task_gen:
        nb_tasks += 1
        for split in range(3):
            y = task.get_labels(split, prop=0)
            x = task.get_data(split)
            
            idx = torch.where(y!=-1)
            y = y[idx]
            x = x[idx]
            
            #if single_head:
            if True:
                adjust_y = 0
                for t, old_t in enumerate(task_gen.task_pool):
                    if str(old_t.concepts) == str(task.concepts):
                        break
                    else:
                        adjust_y += old_t.info()['n_classes'][0]
                y+=adjust_y
            datasets[split].append(TensorDataset([x,y], None))
        yy.append(list(set(y.tolist())))
    
    nb_classes = len(set(itertools.chain.from_iterable(yy)))
    tr_data, val_data, te_data = datasets
    train_loader, val_loader,  test_loader  = [CLDataLoader(elem, args, train=t) \
                for elem, t in zip(datasets, [True, False,  False])]
    return train_loader, val_loader, test_loader, nb_classes, nb_tasks, yy


