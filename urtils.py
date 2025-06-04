import numpy as np
from tqdm import tqdm
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
import random
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from timm.utils import accuracy
from timm.optim import create_optimizer
import clip
import math
import os
from copy import deepcopy
from tests.utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import scipy.signal
from functools import partial
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from clip_multi_prompt.until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def compute_ap(ranks):
    ap = 1
    for i,j in enumerate(ranks):
        ap += 1 / (j+1)
    return ap / len(ranks)

def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result

def itm_eval1(scores_i2t, scores_t2i):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        tmp = np.where(inds == index)[0][0]
        rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    aptr = compute_ap(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    apir = compute_ap(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean,
                   'aptr': aptr,
                   'apir': apir,
                   'aver':(aptr+apir)/2}
    
    return eval_result

def fit_one_epoch(model_train, eval_callback, optimizer, epoch, 
                  gen, gen_val, Epoch, save_period, save_dir, args=None,device=None,task_id=None,loss_history=None):
    
    total_loss      = 0
    val_total_loss  = 0
    epoch_step      = len(gen.dataset.image) // args.batch_size
    epoch_step_val  = len(gen_val.dataset.image) // args.batch_size

    print('Start Train')
    pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train = model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, texts = batch

        
        if True:
            images  = images.to(device)
            texts = clip.tokenize(texts).to(device)
            logits_per_image, dfh               = model_train(images, texts)
            logits_per_text                     = logits_per_image.t()

            itm_results = itm_eval1(logits_per_image.cpu().detach().numpy(), logits_per_text.cpu().detach().numpy())
            itms = itm_results['aptr']

            labels                              = torch.arange(len(logits_per_image)).long().to(images.device)

            itms_loss = 1 - itms

            loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss                                = loss_logits_per_image + loss_logits_per_text + itms_loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            
        total_loss += loss.item()

        pbar.set_postfix(**{'total_loss'            : total_loss / (iteration + 1), 
                                'lr'                    : get_lr(optimizer)})
        pbar.update(1)


    pbar.close()
    print('Finish Train')

    
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, texts = batch
        with torch.no_grad():
            images  = images.to(device)
            texts = clip.tokenize(texts).to(device)
            optimizer.zero_grad()
    
            logits_per_image, _                 = model_train(images, texts)
            logits_per_text                     = logits_per_image.t()
            itm_results = itm_eval1(logits_per_image.cpu().numpy(), logits_per_text.cpu().numpy())
            itms = itm_results['aptr']

            labels                              = torch.arange(len(logits_per_image)).long().to(images.device)

            itms_loss = 1 - itms
            loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss                                = loss_logits_per_image + loss_logits_per_text + itms_loss
            
            val_total_loss += loss.item()

        pbar.set_postfix(**{'val_loss'              : val_total_loss / (iteration + 1), 
                                'lr'                    : get_lr(optimizer)})
        pbar.update(1)
            

    pbar.close()
    print('Finish Validation')

    loss_history.append_loss(epoch, total_loss / epoch_step, val_total_loss / epoch_step_val)
    eval_callback.on_epoch_end(epoch + 1, model_train)
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_total_loss / epoch_step_val))
        

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(deepcopy(model_train).half().state_dict(), os.path.join(save_dir +'/'+'prompt'+'/'+ 'task' + str(task_id), 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_total_loss / epoch_step_val)))
            
    if len(loss_history.val_loss) <= 1 or (val_total_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(deepcopy(model_train).half().state_dict(), os.path.join(save_dir +'/'+'prompt'+'/'+ 'task' + str(task_id), "best_epoch_weights.pth"))
            
    torch.save(deepcopy(model_train).half().state_dict(), os.path.join(save_dir +'/'+'prompt'+'/'+ 'task' + str(task_id), "last_epoch_weights.pth"))

def _convert_to_rgb(image):
    return image.convert('RGB')

def build_transform(resolution=224):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    return Compose([
        Resize(resolution, interpolation=Image.BICUBIC),
        CenterCrop(resolution), 
        _convert_to_rgb, 
        ToTensor(),
        normalize,
    ])

def fit_one_epoch_prompt(model_train, model_original, eval_callback, optimizer, epoch, 
                  gen, gen_val, Epoch, save_period, save_dir, args=None,device=None,task_id=None, loss_history=None,image_buf=None,texts_buf=None):
    total_loss      = 0
    val_total_loss  = 0
    epoch_step      = len(gen.dataset.image) // args.batch_size
    epoch_step_val  = len(gen_val.dataset.image) // args.batch_size
    

    print('Start Train')
    pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train = model_train.train()
    model_original = model_original.eval()
    kl = KL()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, texts = batch
        with torch.no_grad():
            images  = images.to(device)
            cls_features = model_train.encode_image(images, task_id=None, cls_features=None, train=True, get_feature=True)
            cls_features = cls_features.to(device)

            text = clip.tokenize(texts).to(device)
            text_features = model_train.encode_text(text, task_id=None, text_features=None, train=True, get_feature=True)
            text_features = text_features.to(device)
        
        if True:
            images  = images.to(device)
            texts = clip.tokenize(texts).to(device)
  
            logits_per_image, _, s_loss, sim, banzhaf     = model_train(images, texts, task_id, cls_features, text_features, train=True, get_feature=False)
            logits_per_text                = logits_per_image.t()

            labels                              = torch.arange(len(logits_per_image)).long().to(images.device)
            loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
            
            if task_id > 0:
                img_buf=[]
                cap_buf=[]
                transformer = build_transform()
                num=[]
                for i in range(30*task_id):  #buffer size
                    num.append(i)
                ids=random.sample(num,8)
                for i in ids:
                    img = Image.open(image_buf[i])
                    img1 = transformer(img)
                    texts = texts_buf[i]
                    text = clip.tokenize(texts).to(device)
                    img_buf.append(img1)
                    cap_buf.append(text)
                image_old= torch.tensor([item.cpu().detach().numpy() for item in img_buf]).cuda(device)  #feature
                cap_old= torch.tensor([item.cpu().detach().numpy() for item in cap_buf]).squeeze(1).cuda(device)
                with torch.no_grad():
                    cls_features = model_train.encode_image(image_old, task_id=None, cls_features=None, train=True, get_feature=True)
                    cls_features = cls_features.to(device)
                    text_features = model_train.encode_text(cap_old, task_id=None, text_features=None, train=True, get_feature=True)
                    text_features = text_features.to(device)

                    logits_per_image_buf_old, _,s_loss_buf_old,sim_buf_old, banzhaf_buf_old     = model_original(image_old, cap_old, task_id, cls_features, text_features, train=True, get_feature=False)
                    logits_per_text                = logits_per_image.t()

                logits_per_image_buf_new, _,s_loss_buf_new,sim_buf_new, banzhaf_buf    = model_train(image_old, cap_old, task_id, cls_features, text_features, train=True, get_feature=False)
                logits_per_text                = logits_per_image.t()

                bid_loss = kl(banzhaf_buf, banzhaf_buf_old)
                cd_loss = kl(logits_per_image_buf_new, logits_per_image_buf_old)
            else:
                cd_loss = 0
                bid_loss = 0

            loss                                = loss_logits_per_image + loss_logits_per_text - 0.1 * sim + s_loss + 0.3*bid_loss + 0.1*cd_loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            
        total_loss += loss.item()

        with torch.no_grad():
            de_parallel(model_train).logit_scale.clamp_(0, math.log(100))


        pbar.set_postfix(**{'total_loss'            : total_loss / (iteration + 1), 
                                'lr'                    : get_lr(optimizer)})
        pbar.update(1)


    pbar.close()
    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    model_original = model_original.eval()
    
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, texts = batch

        with torch.no_grad():
            images  = images.to(device)
            cls_features = model_train.encode_image(images, task_id=None, cls_features=None, train=True, get_feature=True)
            cls_features = cls_features.to(device)

            texts = clip.tokenize(texts).to(device)
            text_features = model_train.encode_text(text, task_id=None, text_features=None, train=True, get_feature=True)
            text_features = text_features.to(device)
                
            optimizer.zero_grad()
    
            logits_per_image, _, s_loss,sim,banzhaf          = model_train(images, texts, task_id, cls_features, text_features, train=True, get_feature=False)
            logits_per_text                     = logits_per_image.t()

            labels                              = torch.arange(len(logits_per_image)).long().to(images.device)
            loss_logits_per_image               = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_logits_per_text                = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss                                = loss_logits_per_image + loss_logits_per_text - 0.1 * sim + s_loss
            
            val_total_loss += loss.item()

        pbar.set_postfix(**{'val_loss'              : val_total_loss / (iteration + 1), 
                                'lr'                    : get_lr(optimizer)})
        pbar.update(1)
            
    pbar.close()
    print('Finish Validation')
    eval_callback.on_epoch_end(epoch + 1, model_train, model_original)
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(deepcopy(model_train).half().state_dict(), os.path.join(save_dir +'/'+args.Scenario +'/'+ 'task' + str(task_id), 'ep%03d-loss%.3f.pth' % (epoch + 1, total_loss / epoch_step)))     
    if len(loss_history.val_loss) <= 1 or (val_total_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(deepcopy(model_train).half().state_dict(), os.path.join(save_dir +'/'+args.Scenario +'/'+ 'task' + str(task_id), "best_epoch_weights.pth"))
    torch.save(deepcopy(model_train).half().state_dict(), os.path.join(save_dir +'/'+args.Scenario +'/'+ 'task' + str(task_id), "last_epoch_weights.pth"))



class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        try:
            self.writer     = SummaryWriter(self.log_dir)
            # dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            # text_input      = ["OK", "OK"]
            # self.writer.add_graph(model, [dummy_input, text_input])
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")





def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func



def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr