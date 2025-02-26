import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

import numpy as np
import random
from PIL import Image

from globals import CONFIG

"""
def target_random_sample(source_label, target_examples):
    while True:
        target_index = torch.randint(low=0, high=len(target_examples), size=(1,)).item()
        _, target_label = target_examples[target_index]
        if target_label == source_label:
            break
    return target_index
"""

class BaseDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.T = transform
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        x, y = self.examples[index]
        x = Image.open(x).convert('RGB')
        x = self.T(x).to(CONFIG.dtype)
        y = torch.tensor(y).long()
        return x, y
    
class DomainAdaptationDataset(Dataset):
    def __init__(self, source_examples, target_examples, transform):
        self.source_examples = source_examples
        self.target_examples = target_examples
        self.T = transform

    def __len__(self):
        return len(self.source_examples)
    
    def __getitem__(self, index):
        x_source, y_source = self.source_examples[index]
        x_source = Image.open(x_source).convert('RGB')
        x_source = self.T(x_source).to(CONFIG.dtype)
        y_source = torch.tensor(y_source).long()
        # randomly sample from the target domain 
        target_index = torch.randint(low=0, high=len(self.target_examples), size=(1,)).item()
        x_target, _ = self.target_examples[target_index] # target is a tuple (path, 0)
        x_target = Image.open(x_target).convert('RGB')
        x_target = self.T(x_target).to(CONFIG.dtype)
        return x_source, y_source, x_target
    
class DomainGeneralizationDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.T = transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
       (x1, y1), (x2, y2), (x3, y3) = self.examples[index]
       assert(y1 == y2 == y3)
       x1, x2, x3 = Image.open(x1).convert('RGB'), Image.open(x2).convert('RGB'), \
                        Image.open(x3).convert('RGB')
       x1, x2, x3 = self.T(x1).to(CONFIG.dtype), self.T(x2).to(CONFIG.dtype), \
                        self.T(x3).to(CONFIG.dtype)
       y1, y2, y3 = torch.tensor(y1).long(), torch.tensor(y2).long(), torch.tensor(y3).long()
       return (x1, y1), (x2, y2), (x3, y3)
       
    

######################################################
# TODO: modify 'BaseDataset' for the Domain Adaptation setting.
# Hint: randomly sample 'target_examples' to obtain targ_x
#class DomainAdaptationDataset(Dataset):
#    def __init__(self, source_examples, target_examples, transform):
#        self.source_examples = source_examples
#        self.target_examples = target_examples
#        self.T = transform
#    
#    def __len__(self):
#        return len(self.source_examples)
#    
#    def __getitem__(self, index):
#        src_x, src_y = ...
#        targ_x = ...
#
#        src_x = self.T(src_x)
#        targ_x = self.T(targ_x)
#        return src_x, src_y, targ_x

# [OPTIONAL] TODO: modify 'BaseDataset' for the Domain Generalization setting. 
# Hint: combine the examples from the 3 source domains into a single 'examples' list
#class DomainGeneralizationDataset(Dataset):
#    def __init__(self, examples, transform):
#        self.examples = examples
#        self.T = transform
#    
#    def __len__(self):
#        return len(self.examples)
#    
#    def __getitem__(self, index):
#        x1, x2, x3 = self.examples[index]
#        x1, x2, x3 = self.T(x1), self.T(x2), self.T(x3)
#        targ_x = self.T(targ_x)
#        return x1, x2, x3

######################################################

class SeededDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, batch_size=1, shuffle=None, 
                 sampler=None, 
                 batch_sampler=None, 
                 num_workers=0, collate_fn=None, 
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None, 
                 generator=None, *, prefetch_factor=None, persistent_workers=False, 
                 pin_memory_device=""):
        
        if not CONFIG.use_nondeterministic:
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            generator = torch.Generator()
            generator.manual_seed(CONFIG.seed)

            worker_init_fn = seed_worker
        
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, 
                         pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, 
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, 
                         pin_memory_device=pin_memory_device)

