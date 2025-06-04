import os
import datetime
import json
import math
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists
import torch
import clip
from PIL import Image
from torch import nn
import json
from tests.F30K import ClipDataset30k, dataset_collate, Scenario_2_task1, Scenario_2_task2, Scenario_2_task3
from torch.utils.data import DataLoader
from urtils import fit_one_epoch, LossHistory, fit_one_epoch_prompt, get_lr_scheduler, set_optimizer_lr
from callback import EvalCallback1, EvalCallback_prompt
from clip_multi_prompt.clip_prompt import CLIP
from clip.model import CLIP as CLIP_original
from continue_dataloader import build_continual_dataloader
import logging
_logger = logging.getLogger(__name__)
import torch.nn.functional as F


def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=()):

    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if ntok_new > gs_old ** 2:
        ntok_new -= gs_old ** 2
        posemb_prefix = posemb_prefix.expand(-1, ntok_new, -1)
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb

def resize_text_pos_embed(posemb, posemb_new, num_prefix_tokens=1):
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if ntok_new > 76:
        ntok_new -= 76
        posemb_prefix = posemb_prefix.expand(-1, ntok_new, -1)

    posemb_grid = posemb_grid.unsqueeze(0)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


def main(args):
    device          = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    jit_model = CLIP(
            embed_dim          = 512,
            image_resolution    = 224,
            vision_layers      = 12,
            vision_width       = 768,
            vision_patch_size  = 32,
            context_length     = 77,
            transformer_layers  = 12,
            transformer_width   = 512,
            transformer_heads   = 8,
            vocab_size          = 49408,
            args                = args
            )
    

    model_dict      = jit_model.state_dict()
    pretrained_dict = torch.load('./clip/ViT-B-32.pt', map_location = device)
    pretrained_dict = pretrained_dict.state_dict()
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k == 'visual.positional_embedding' and v.shape[0] != model_dict[k].shape[0]:
            v = resize_pos_embed(
                v.unsqueeze(0),
                model_dict[k].unsqueeze(0),
                1,
                (jit_model.visual.input_resolution // jit_model.visual.patch_size, 
                 jit_model.visual.input_resolution // jit_model.visual.patch_size)
            ).squeeze(0)
        if k == 'positional_embedding' and v.shape[0] != model_dict[k].shape[0]:
            v = resize_text_pos_embed(
                v.unsqueeze(0),
                model_dict[k].unsqueeze(0),
                1
            ).squeeze(0)            
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
   
    model_dict.update(temp_dict)
    jit_model.load_state_dict(model_dict)
    model_train = jit_model.train()
    model_train = model_train.to(device)

    model_original =  CLIP(
            embed_dim          = 512,
            image_resolution    = 224,
            vision_layers      = 12,
            vision_width       = 768,
            vision_patch_size  = 32,
            context_length     = 77,
            transformer_layers  = 12,
            transformer_width   = 512,
            transformer_heads   = 8,
            vocab_size          = 49408,
            args                = args
            )
   
    model_dict      = model_original.state_dict()
    pretrained_dict = torch.load('./clip/ViT-B-32.pt', map_location = device)
    pretrained_dict = pretrained_dict.state_dict()
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k == 'visual.positional_embedding' and v.shape[0] != model_dict[k].shape[0]:
            v = resize_pos_embed(
                v.unsqueeze(0),
                model_dict[k].unsqueeze(0),
                1,
                (jit_model.visual.input_resolution // jit_model.visual.patch_size, 
                 jit_model.visual.input_resolution // jit_model.visual.patch_size)
            ).squeeze(0)
        if k == 'positional_embedding' and v.shape[0] != model_dict[k].shape[0]:
            v = resize_text_pos_embed(
                v.unsqueeze(0),
                model_dict[k].unsqueeze(0),
                1
            ).squeeze(0)            
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model_original.load_state_dict(model_dict)
    model_original = model_original.eval()
    model_original = model_original.to(device)

    for p in model_original.parameters():
        p.requires_grad = False

    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 1e-2
    Init_lr_fit     = 1e-6
    Min_lr_fit      = 1e-8
    optimizer = {
            'adamw' : optim.AdamW(model_train.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'adam'  : optim.Adam(model_train.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
        }[optimizer_type]
    
    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.epochs)
    lf = lambda x: ((1 + math.cos(x * math.pi / 50)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    save_dir            = './output'
    save_period         = 1

    total_incremental_steps = args.incremental_steps[args.Scenario]
    data_loader, dataset= build_continual_dataloader(args)
    image_buf = []
    texts_buf = []

    for task_id in range(total_incremental_steps):

        if task_id > 0:
            weights_dict = torch.load('./outputsw/{}/task{}/best_epoch_weights.pth'.format(args.Scenario, task_id-1)) 
            model_train.load_state_dict(weights_dict, strict=False)   
            model_original.load_state_dict(weights_dict, strict=False) 

        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir+'/'+ args.Scenario +'/'+ 'task' + str(task_id), "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model_train, None)

        if task_id > 0:
            optimizer = {
            'adamw' : optim.AdamW(model_train.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'adam'  : optim.Adam(model_train.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
        }[optimizer_type]

        eval_callback   = EvalCallback_prompt(model_train, data_loader[task_id]['val'], log_dir, True, \
                                        eval_flag=True, period=1, device = device)
        
        for epoch in range(args.epochs):
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch_prompt(model_train = model_train, model_original = model_original, eval_callback = eval_callback, optimizer=optimizer, epoch = epoch,\
                        gen = data_loader[task_id]['train'], gen_val= data_loader[task_id]['val'], Epoch = args.epochs, save_period = save_period, \
                        save_dir = save_dir, args = args, device=device, task_id = task_id, loss_history=loss_history,image_buf=image_buf,texts_buf=texts_buf)
            
        img_buf, text_buf = dataset[task_id]['train'].get_buf()
        image_buf.extend(img_buf)
        texts_buf.extend(text_buf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--Scenario', type=str, default='Scenario_1', 
                        choices=['Scenario_1', 'Scenario_2', 'Scenario_3', 'Scenario_4', 'Scenario_5'])
    parser.add_argument('--incremental_steps', type=dict, default={'Scenario_1':3,'Scenario_2':3,
                                                                   'Scenario_3':4,'Scenario_4':5,'Scenario_5':6})
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr_decay_type', type=str, default='cos')

    parser.add_argument('--dataset', type=str,default='MINIST', help='CIFAR10 or MINIST')
    parser.add_argument('--rate', type=float, default=0.1, help='dataset size')
    parser.add_argument('--model-name', default='', help='create model name')

    parser.add_argument('--num_tasks', type=int, default= 3, help='create model name')

    parser.add_argument('--weights', type=str, default='',help='initial weights path')
    
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--prompt_pool', default=True, type=bool,)
    parser.add_argument('--pool_size', default=6, type=int,)
    parser.add_argument('--prompt_length', default=3,type=int,)
    parser.add_argument('--top_k', default=1, type=int, )
    parser.add_argument('--initializer', default='uniform', type=str,)
    parser.add_argument('--prompt_key', default=True, type=bool,)
    parser.add_argument('--prompt_key_init', default='zero', type=str)
    parser.add_argument('--use_prompt_mask', default=True, type=bool)
    parser.add_argument('--shared_prompt_pool', default=False, type=bool)
    parser.add_argument('--shared_prompt_key', default=False, type=bool)
    parser.add_argument('--batchwise_prompt', default=True, type=bool)
    parser.add_argument('--embedding_key', default='cls', type=str)
    parser.add_argument('--predefined_key', default='', type=str)
    parser.add_argument('--pull_constraint', default=True)
    parser.add_argument('--pull_constraint_coeff', default=0.5, type=float)
    parser.add_argument('--head_type', default='token', type=str)
    parser.add_argument('--prompt_init', default='zero', type=str)
    opt = parser.parse_args()
    main(opt)