# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Tuple
from torchvision import transforms
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict, Iterable
from typing import Tuple, Dict, Type, Optional
import pdb

device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, n_tasks=1, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        #self.device = device
        self.device= torch.device('cuda:0')
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                #typ = torch.int64 if attr_str.endswith('els') else torch.float32
                if attr_str.endswith('els'):
                    typ = torch.int64
                elif attr_str.endswith('les'):
                    typ = torch.uint8
                else:
                    typ = torch.float32
                #typ = torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0


class BalancedBuffer(nn.Module):
    def __init__(self, capacity, balance_key = 't', input_shape = [3,32,32]):
        super().__init__()
        self.decreasing_prob_adding = True
        self.rng = np.random.RandomState()
        self.balance_key = balance_key

        bx = torch.zeros([capacity, *input_shape], dtype=torch.float)
        by = torch.zeros([capacity], dtype=torch.long)
        bt = torch.zeros([capacity], dtype=torch.long)
        bl = torch.zeros([capacity, 50], dtype=torch.float)

        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)
        self.register_buffer('bl', bl)

        self.buffers = ['bx', 'by', 'bt', 'bl']
        extra_buffers = {}

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full       = 0
        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    @property
    def unique_tasks_in_buffer(self):
        if self.balance_key is not None:
            return torch.unique(getattr(self, f'b{self.balance_key}'))
        else:
            return torch.tensor([0])

    @property
    def buffer_size(self):
        return self.bx.size(0)

    def calculate_per_task_distribution(self, left_to_place, new = 1):
        def f(x):
            return len(np.array_split(np.arange(min(left_to_place, len(x))), len(self.unique_tasks_in_buffer))[-1])
        return list(map(f, np.array_split(np.arange(self.buffer_size), len(self.unique_tasks_in_buffer)+new)))

    def is_empty(self):
        return self.current_index == 0

    def add_reservoir(self, batch):
        n_elem = batch['x'].size(0)   
        place_left = max(0, self.buffer_size - self.current_index)
        offset = min(place_left, n_elem)
        if place_left:
            for name, data in batch.items():
                buffer = getattr(self, f'b{name}')
                if isinstance(data, Iterable):
                    buffer[self.current_index: self.current_index + offset].data.copy_(data[:offset])
                else:
                    buffer[self.current_index: self.current_index + offset].fill_(data)
            self.current_index += offset
            self.n_seen_so_far += offset
            if offset == batch['x'].size(0):
                return

        x = batch['x']
        left_to_place = n_elem-offset
        unique_tasks = self.unique_tasks_in_buffer
        if batch[self.balance_key] not in unique_tasks:
            space_needed_from_existing_tasks = self.calculate_per_task_distribution(left_to_place=left_to_place, new=1)
            indicies = []

            for i,k in  enumerate(unique_tasks):
                indicies+=list(self.rng.choice(torch.where(getattr(self,f'b{self.balance_key}')==k)[0].cpu(), size=space_needed_from_existing_tasks[i], replace=False))
            idx_buffer = torch.LongTensor(indicies).to(x.device)
            idx_new_data = torch.from_numpy(self.rng.choice(np.arange(left_to_place),idx_buffer.numel(), replace=False)).to(x.device)
            self.n_seen_so_far += idx_buffer.numel()

        else:
            space_per_key = len(np.array_split(np.arange(self.buffer_size), len(unique_tasks))[-1])
            
            idxs_in_buffer = torch.where(getattr(self,f'b{self.balance_key}')==batch[self.balance_key])[0]
            if len(idxs_in_buffer)<space_per_key-1:
                space_needed_from_existing_tasks = self.calculate_per_task_distribution(left_to_place=left_to_place, new=0)
                indicies = []
                for i,k in enumerate(unique_tasks):
                    if k!=batch[self.balance_key]:
                        indicies+=list(self.rng.choice(torch.where(getattr(self,f'b{self.balance_key}')==k)[0].cpu(), size=space_needed_from_existing_tasks[i], replace=False))
                idx_buffer = torch.LongTensor(indicies).to(x.device)
                idx_new_data = torch.from_numpy(self.rng.choice(np.arange(left_to_place),idx_buffer.numel(), replace=False)).to(x.device)
                self.n_seen_so_far += idx_buffer.numel() 
            else:
                indices = torch.FloatTensor(min(left_to_place,len(idxs_in_buffer))).uniform_(0, self.n_seen_so_far).long()
                valid_indices: Tensor = torch.tensor(np.isin(indices, idxs_in_buffer.cpu())).long().to(x.device)
                idx_new_data = valid_indices.nonzero(as_tuple=False).squeeze(-1)
                idx_buffer   = indices[idx_new_data].to(x.device)
                self.n_seen_so_far += left_to_place
                if idx_buffer.numel() == 0:
                    return

    def get_data(self, n_samples, only_task = None, exclude_task = None):
        buffers = OrderedDict()
        n_samples_per_task = map(len, np.array_split(np.arange(min(n_samples, self.buffer_size)), max(1,len(self.unique_tasks_in_buffer) - (int(exclude_task!=None)+int(only_task!=None) ))))
        indicies = []
        for t, n in enumerate(n_samples_per_task):
            if t!=exclude_task:
                if only_task is not None:
                    if t==only_task:
                        ixs_task = (self.bt.cpu()==t).nonzero(as_tuple=False).squeeze()
                        indicies += list(self.rng.choice(ixs_task, min(n,len(ixs_task)), replace=False ))
                else:
                    ixs_task = (self.bt.cpu()==t).nonzero(as_tuple=False).squeeze()
                    indicies += list(self.rng.choice(ixs_task, min(n,len(ixs_task)), replace=False ))
        indicies = torch.LongTensor(indicies).to(device)

        for buffer_name in self.buffers:
            buffers[buffer_name] = getattr(self, buffer_name)[indicies]
        bx = buffers['bx']
        if bx.size(0) < n_samples:
            return OrderedDict({k[1:]: v for (k,v) in buffers.items()})
        else:
            indices_np = self.rng.choice(bx.size(0), n_samples, replace=False)
            indices = torch.from_numpy(indices_np).to(self.bx.device)
        return OrderedDict({k[1:]: v[indices] for (k,v) in buffers.items()})
        
class Cutout:
    def __init__(self, size=16) -> None:
        self.size = size

    def _create_cutout_mask(self, img_height, img_width, num_channels, size):
        """Creates a zero mask used for cutout of shape `img_height` x `img_width`.
        Args:
          img_height: Height of image cutout mask will be applied to.
          img_width: Width of image cutout mask will be applied to.
          num_channels: Number of channels in the image.
          size: Size of the zeros mask.
        Returns:
          A mask of shape `img_height` x `img_width` with all ones except for a
          square of zeros of shape `size` x `size`. This mask is meant to be
          elementwise multiplied with the original image. Additionally returns
          the `upper_coord` and `lower_coord` which specify where the cutout mask
          will be applied.
        """
        # assert img_height == img_width

        # Sample center where cutout mask will be applied
        height_loc = np.random.randint(low=0, high=img_height)
        width_loc = np.random.randint(low=0, high=img_width)

        size = int(size)
        # Determine upper right and lower left corners of patch
        upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
        lower_coord = (
            min(img_height, height_loc + size // 2),
            min(img_width, width_loc + size // 2),
        )
        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = np.ones((img_height, img_width, num_channels))
        zeros = np.zeros((mask_height, mask_width, num_channels))
        mask[
            upper_coord[0] : lower_coord[0], upper_coord[1] : lower_coord[1], :
        ] = zeros
        return mask, upper_coord, lower_coord

    def __call__(self, pil_img):
        pil_img = pil_img.copy()
        img_height, img_width, num_channels = (*pil_img.size, 3)
        _, upper_coord, lower_coord = self._create_cutout_mask(
            img_height, img_width, num_channels, self.size
        )
        pixels = pil_img.load()  # create the pixel map
        for i in range(upper_coord[0], lower_coord[0]):  # for every col:
            for j in range(upper_coord[1], lower_coord[1]):  # For every row
                pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
        return pil_img

def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


