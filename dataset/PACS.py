import torch
import os
import torchvision.transforms as T
from dataset.utils import BaseDataset, DomainAdaptationDataset, DomainGeneralizationDataset
from dataset.utils import SeededDataLoader

from globals import CONFIG

import numpy as np

def get_transform(size, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(T.Resize(256))
        transform.append(T.RandomResizedCrop(size=size, scale=(0.7, 1.0)))
        transform.append(T.RandomHorizontalFlip())
    else:
        transform.append(T.Resize(size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean, std))
    return T.Compose(transform)

def load_domain(domain_path: str):
    domain_examples = []

    with open(os.path.join(CONFIG.dataset_args['root'], f'{domain_path}.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            domain_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))
    
    return domain_examples

def oversample(data: list, N: int):
    while (len(data) < N):
        index = np.random.randint(0, len(data))
        data.append(data[index]) 


def load_data():
    CONFIG.num_classes = 7
    CONFIG.data_input_size = (3, 224, 224)

    # Create transforms
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # ImageNet Pretrain statistics
    train_transform = get_transform(size=224, mean=mean, std=std, preprocess=True)
    test_transform = get_transform(size=224, mean=mean, std=std, preprocess=False)

    # Load examples & create Dataset
    if CONFIG.experiment in ['baseline', 'random', 'domain_adaptation']:
        source_examples, target_examples = [], []

        # Load source
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['source_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            source_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        # Load target
        with open(os.path.join(CONFIG.dataset_args['root'], f"{CONFIG.dataset_args['target_domain']}.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            path, label = line[0].split('/')[1:], int(line[1])
            target_examples.append((os.path.join(CONFIG.dataset_args['root'], *path), label))

        if CONFIG.experiment in ['baseline', 'random']:
            train_dataset = BaseDataset(source_examples, transform=train_transform)
            test_dataset = BaseDataset(target_examples, transform=test_transform)

    ######################################################
    #elif... TODO: Add here how to create the Dataset object for the other experiments


    ######################################################
        elif CONFIG.experiment in ['domain_adaptation']:
            train_dataset = DomainAdaptationDataset(source_examples, target_examples, transform=train_transform)
            test_dataset = BaseDataset(target_examples, transform=test_transform)

    elif CONFIG.experiment in ['domain_generalization']:
        source_domains_examples = []
        source_examples, target_examples = [], []
        print(CONFIG.dataset_args['source_domains'])
        CONFIG.dataset_args['source_domains'] =  CONFIG.dataset_args['source_domains'].split()

        # Load source domains
        for source_domain_path in CONFIG.dataset_args['source_domains']:
            source_domains_examples.append(load_domain(source_domain_path))

        for label in range(2):
            d1, d2, d3 = np.array(source_domains_examples[0]), np.array(source_domains_examples[1]), \
                            np.array(source_domains_examples[2])
            mask1, mask2, mask3 = d1[:, 1] == str(label), d2[:, 1] == str(label), d3[:, 1] == str(label)
            d1, d2, d3 = d1[mask1].tolist(), d2[mask2].tolist(), d3[mask3].tolist()
            num_samples = np.max([len(d1), len(d2), len(d3)])
            for d in [d1, d2, d3]:
                if len(d) < num_samples:
                    oversample(d, num_samples)
            for example in zip(d1, d2, d3):
                example = (example[0], int(example[1])) # since numpy converted the label to string
                source_examples.append(example)
            
        # Load target examples
        target_examples = load_domain(CONFIG.dataset_args['target_domain'])

        # Datasets
        train_dataset = DomainGeneralizationDataset(source_examples, transform=train_transform)
        test_dataset = BaseDataset(target_examples, transform=test_transform)   

    # Dataloaders
    train_loader = SeededDataLoader(
        train_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = SeededDataLoader(
        test_dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        num_workers=CONFIG.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return {'train': train_loader, 'test': test_loader}