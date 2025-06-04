import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from tests.dataset import *

from torch.utils.data import DataLoader

def build_continual_dataloader(args):

    dataloader = list()
    dataset_buf = list()
    if args.Scenario == 'Scenario_1':
        dataset_list = ['Scenario_1_task1_class','Scenario_1_task2_class','Scenario_1_task3_class']

    if args.Scenario == 'Scenario_2':
        dataset_list = ['Scenario_2_task1_class','Scenario_2_task2_class','Scenario_2_task3_class']

    if args.Scenario == 'Scenario_3':
        dataset_list = ['Scenario_2_task1_class','Scenario_2_task2_class','Scenario_2_task3_class','Scenario_1_task1_class']

    if args.Scenario == 'Scenario_4':
        dataset_list = ['Scenario_2_task1_class','Scenario_2_task2_class','Scenario_2_task3_class',
                        'Scenario_1_task1_class','Scenario_1_task2_class']
        
    if args.Scenario == 'Scenario_5':
        dataset_list = ['Scenario_2_task1_class','Scenario_2_task2_class','Scenario_2_task3_class',
                        'Scenario_1_task1_class','Scenario_1_task2_class','Scenario_1_task3_class']

    ngpus_per_node  = torch.cuda.device_count()
    for i in range(args.incremental_steps[args.Scenario]):

        dataset_train, dataset_val = get_dataset(dataset_list[i])
        train_sampler   = None
        val_sampler     = None
        shuffle         = True

        gen = DataLoader(dataset_train, shuffle=shuffle, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                drop_last=True, sampler=train_sampler)
        gen_val = DataLoader(dataset_val, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                                drop_last=False, sampler=val_sampler)
        dataloader.append({'train': gen, 'val': gen_val})
        dataset_buf.append({'train': dataset_train, 'val':dataset_val})

    return dataloader, dataset_buf

def _convert_to_rgb(image):
    return image.convert('RGB')

def get_dataset(dataset):

    transform_train_clip = transforms.Compose(
        [Resize(224, interpolation=Image.BICUBIC), 
        CenterCrop(224),  
        _convert_to_rgb,  
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    transform_val_clip = transforms.Compose(
        [Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    if dataset == 'Scenario_1_task1_class':
        dataset_train = Scenario_1_task1_class([224, 224], random = True, subset='train')
        dataset_val = Scenario_1_task1_class([224, 224], random = False, subset='test') 

    elif dataset == 'Scenario_1_task1_class':
        dataset_train = Scenario_1_task1_class([224, 224], random = True, subset='train')
        dataset_val = Scenario_1_task1_class([224, 224], random = False, subset='test') 
    
    elif dataset == 'Scenario_1_task2_class':
        dataset_train = Scenario_1_task2_class([224, 224], random = True, subset='train')
        dataset_val = Scenario_1_task2_class([224, 224], random = False, subset='test') 

    elif dataset == 'Scenario_1_task3_class':
        dataset_train = Scenario_1_task3_class([224, 224], random = True, subset='train')
        dataset_val = Scenario_1_task3_class([224, 224], random = False, subset='test') 

    elif dataset == 'Scenario_2_task1_class':
        dataset_train = Scenario_2_task1_class([224, 224], random = True, subset='train')
        dataset_val = Scenario_2_task1_class([224, 224], random = False, subset='test') 
    
    elif dataset == 'Scenario_2_task2_class':
        dataset_train = Scenario_2_task2_class([224, 224], random = True, subset='train')
        dataset_val = Scenario_2_task2_class([224, 224], random = False, subset='test') 

    elif dataset == 'Scenario_2_task3_class':
        dataset_train = Scenario_2_task3_class([224, 224], random = True, subset='train')
        dataset_val = Scenario_2_task3_class([224, 224], random = False, subset='test') 

    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val