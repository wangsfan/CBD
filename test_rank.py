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
from tests.dataset import ClipDataset30k, dataset_collate, Scenario_2_task1, Scenario_2_task2, Scenario_2_task3
from torch.utils.data import DataLoader
from urtils import fit_one_epoch, LossHistory, fit_one_epoch_prompt, get_lr_scheduler, set_optimizer_lr
from callback import EvalCallback1, EvalCallback_prompt
from clip_multi_prompt.clip_prompt import CLIP
from clip.model import CLIP as CLIP_original
from continue_dataloader import build_continual_dataloader
import logging
from urtils import itm_eval, itm_eval1
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
    if not len(gs_new):  
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
    pretrained_dict = torch.load('./output', map_location = device)

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
    model = jit_model.eval()
    model = model.to(device)


    data_loader, dataset = build_continual_dataloader(args)
    ACC = []
    total_incremental_steps = args.incremental_steps[args.Scenario]
    for i in range(total_incremental_steps):
        print(i)
        gen_val = data_loader[i]['val']
        txt_r1 = []
        txt_r5 = []
        img_r1 = []
        img_r5 = []
        epoches = []
        i_features = []
        t_features = []
        for iteration, batch in tqdm(enumerate(gen_val)):
            images, texts = batch
            with torch.no_grad():
                images  = images.to(device)
                    
                cls_features = model.encode_image(images, task_id=None, cls_features=None, train=False, get_feature=True)
                cls_features = cls_features.to(device)

                texts = clip.tokenize(texts).to(device)
       
                text_features = model.encode_text(texts, task_id=None, text_features=None, train=False, get_feature=True)
                text_features = text_features.to(device)

                            
                images_features = model.encode_image(images, task_id=None, cls_features=cls_features, train=False, get_feature=False)
                bs_pair, n, v_n = images_features['logits'].size()
                images_feature = images_features['logits'].view(bs_pair, -1, images_features['logits'].size(-1))
                i_features.append(images_feature)

                texts_features = model.encode_text(texts, task_id=None, text_features=text_features, train=False, get_feature=False)
                bs_pair, n, v_n = texts_features['logits'].size()
                texts_feature = texts_features['logits'].view(bs_pair, -1, texts_features['logits'].size(-1))
                t_features.append(texts_feature)


        i_features = torch.cat(i_features, 0)
        t_features = torch.cat(t_features, 0)
                

        sim_matrix = []
        batch_t_feat = torch.split(t_features, 32)
        batch_v_feat = torch.split(i_features, 32)

        with torch.no_grad():
            for idx1, t_features in enumerate(batch_t_feat):
                each_row = []
                for idx2, i_features in enumerate(batch_v_feat):
                    i_features  = i_features / i_features.norm(dim=-1, keepdim=True)
                    t_features  = t_features / t_features.norm(dim=-1, keepdim=True)
                    retrieve_logits = torch.einsum('atd,bvd->abtv', [t_features, i_features])
                    t2i_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
                    i2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
                    t2i_logits = torch.sum(t2i_logits, dim=2) 
                    i2t_logits = torch.sum(i2t_logits, dim=2) 
                    _retrieve_logits = (t2i_logits + i2t_logits) / 2.0

                    logits = _retrieve_logits.cpu().detach().numpy()
                    each_row.append(logits)

                each_row = np.concatenate(tuple(each_row), axis=-1)
                sim_matrix.append(each_row)

            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        
        logits_per_text  = sim_matrix
        logits_per_image   = sim_matrix.T
        print('logits_per_image:', logits_per_image.shape)

        itm_results = itm_eval1(logits_per_image, logits_per_text)
        txt_r1.append(itm_results['txt_r1'])
        txt_r5.append(itm_results['txt_r5'])
        img_r1.append(itm_results['img_r1'])
        img_r5.append(itm_results['img_r5'])
        ACC.append((itm_results['aptr']+itm_results['apir'])/2)
        print(itm_results)
        print("Get recall done.")

    print(sum(ACC) / len(ACC))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument('--Scenario', type=str, default='Scenario_1', 
                        choices=['Scenario_1', 'Scenario_2', 'Scenario_3', 'Scenario_4', 'Scenario_5'])
    parser.add_argument('--incremental_steps', type=dict, default={'Scenario_1':3,'Scenario_2':3,
                                                                   'Scenario_3':4,'Scenario_4':5,'Scenario_5':6})
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
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
