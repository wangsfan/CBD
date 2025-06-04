import datetime
import os
import numpy as np
import torch
from torch import nn
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from urtils import itm_eval,itm_eval1
import clip

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        try:
            self.writer     = SummaryWriter(self.log_dir)

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

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model

class EvalCallback():
    def __init__(self, net, gen_val, log_dir, cuda, batch_size=32, eval_flag=True, period=1, device=None):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.gen_val            = gen_val
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.batch_size         = batch_size
        self.eval_flag          = eval_flag
        self.period             = period

        self.txt_r1 = []
        self.txt_r5 = []
        self.img_r1 = []
        self.img_r5 = []
        self.epoches = []
    
    def on_epoch_end(self, epoch, model_eval, device):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = de_parallel(model_eval)
            
            i_features = []
            t_features = []
            for iteration, batch in tqdm(enumerate(self.gen_val)):
                images, texts = batch
                with torch.no_grad():
                    if self.cuda:
                        images  = images.to(device)

                    images_feature = self.net.encode_image(images, train=False)
                    bs_pair, n, v_n = images_feature['logits'].size()
                    images_feature = images_feature['logits'].view(bs_pair, -1, images_feature['logits'].size(-1))
                    i_features.append(images_feature)


                    texts_feature = self.net.encode_text(texts, train=False)
                    bs_pair, n, v_n = texts_feature['logits'].size()
                    texts_feature = texts_feature['logits'].view(bs_pair, -1, texts_feature['logits'].size(-1))
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
            self.txt_r1.append(itm_results['txt_r1'])
            self.txt_r5.append(itm_results['txt_r5'])
            self.img_r1.append(itm_results['img_r1'])
            self.img_r5.append(itm_results['img_r5'])
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_R@1_R@5_R@10.txt"), 'a') as f:
                f.write(str(itm_results))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.txt_r1, 'red', linewidth = 2, label='txt_r1')
            plt.plot(self.epoches, self.txt_r5, 'green', linewidth = 2, label='txt_r5')
            plt.plot(self.epoches, self.img_r1, 'blue', linewidth = 2, label='img_r1')
            plt.plot(self.epoches, self.img_r5, 'pink', linewidth = 2, label='img_r5')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.title('A Recall Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_recall.png"))
            plt.cla()
            plt.close("all")
            print(itm_results)
            print("Get recall done.")

class EvalCallback1():
    def __init__(self, net, gen_val, log_dir, cuda, batch_size=32, eval_flag=True, period=1):
        super(EvalCallback1, self).__init__()
        
        self.net                = net
        self.gen_val            = gen_val
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.batch_size         = batch_size
        self.eval_flag          = eval_flag
        self.period             = period

        self.txt_r1 = []
        self.txt_r5 = []
        self.img_r1 = []
        self.img_r5 = []
        self.epoches = []
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            
            i_features = []
            t_features = []
            for iteration, batch in tqdm(enumerate(self.gen_val)):
                images, texts = batch
                with torch.no_grad():
                    images  = images.cuda(1)
                    images  = images.cuda(1)
                    texts = clip.tokenize(texts).cuda(1)
                    dfh = self.net.logit_scale.exp()

                    
                    images_feature = self.net.encode_image(images)

                    bs_pair, n, v_n = images_feature.size()

                    images_feature = images_feature.view(bs_pair, -1, images_feature.size(-1))


                    i_features.append(images_feature)


                    texts_feature = self.net.encode_text(texts)
                    bs_pair, n, v_n = texts_feature.size()
                    texts_feature = texts_feature.view(bs_pair, -1, images_feature.size(-1))
                    t_features.append(texts_feature)


            i_features = torch.cat(i_features, 0)
            t_features = torch.cat(t_features, 0)
            
            i_features  = i_features / i_features.norm(dim=-1, keepdim=True)
            t_features  = t_features / t_features.norm(dim=-1, keepdim=True)

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

            self.txt_r1.append(itm_results['txt_r1'])
            self.txt_r5.append(itm_results['txt_r5'])
            self.img_r1.append(itm_results['img_r1'])
            self.img_r5.append(itm_results['img_r5'])
            self.epoches.append(epoch)

            
            plt.figure()
            plt.plot(self.epoches, self.txt_r1, 'red', linewidth = 2, label='txt_r1')
            plt.plot(self.epoches, self.txt_r5, 'green', linewidth = 2, label='txt_r5')
            plt.plot(self.epoches, self.img_r1, 'blue', linewidth = 2, label='img_r1')
            plt.plot(self.epoches, self.img_r5, 'pink', linewidth = 2, label='img_r5')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.title('A Recall Curve')
            plt.legend(loc="upper right")

            plt.cla()
            plt.close("all")
            print(itm_results)
            print("Get recall done.")


class EvalCallback_prompt():
    def __init__(self, net, gen_val, log_dir, cuda, batch_size=32, eval_flag=True, period=1,device = None):
        super(EvalCallback_prompt, self).__init__()
        
        self.net                = net
        self.gen_val            = gen_val
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.batch_size         = batch_size
        self.eval_flag          = eval_flag
        self.period             = period
        self.device             = device 

        self.txt_r1 = []
        self.txt_r5 = []
        self.img_r1 = []
        self.img_r5 = []
        self.epoches = []
    
    def on_epoch_end(self, epoch, model_eval, model_original):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            model_original = model_original.to(self.device)
            i_features = []
            t_features = []
            for iteration, batch in tqdm(enumerate(self.gen_val)):
                images, texts = batch
                with torch.no_grad():
                    images  = images.to(self.device)

                    cls_features = self.net.encode_image(images, task_id=None, cls_features=None, train=False, get_feature=True)
                    cls_features = cls_features.to(self.device)

                    texts = clip.tokenize(texts).to(self.device)

                    text_features = self.net.encode_text(texts, task_id=None, text_features=None, train=False, get_feature=True)
                    text_features = text_features.to(self.device)


                    
                    images_features = self.net.encode_image(images, task_id=None, cls_features=cls_features, train=False, get_feature=False)

                    bs_pair, n, v_n = images_features['logits'].size()
 
                    images_feature = images_features['logits'].view(bs_pair, -1, images_features['logits'].size(-1))

                    i_features.append(images_feature)


                    texts_features = self.net.encode_text(texts, task_id=None, text_features=text_features, train=False, get_feature=False)
                    bs_pair, n, v_n = texts_features['logits'].size()
                    texts_feature = texts_features['logits'].view(bs_pair, -1, texts_features['logits'].size(-1))
                    t_features.append(texts_feature)


            i_features = torch.cat(i_features, 0)
            t_features = torch.cat(t_features, 0)
            
            i_features  = i_features / i_features.norm(dim=-1, keepdim=True)
            t_features  = t_features / t_features.norm(dim=-1, keepdim=True)

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

            self.txt_r1.append(itm_results['txt_r1'])
            self.txt_r5.append(itm_results['txt_r5'])
            self.img_r1.append(itm_results['img_r1'])
            self.img_r5.append(itm_results['img_r5'])
            self.epoches.append(epoch)


            
            plt.figure()
            plt.plot(self.epoches, self.txt_r1, 'red', linewidth = 2, label='txt_r1')
            plt.plot(self.epoches, self.txt_r5, 'green', linewidth = 2, label='txt_r5')
            plt.plot(self.epoches, self.img_r1, 'blue', linewidth = 2, label='img_r1')
            plt.plot(self.epoches, self.img_r5, 'pink', linewidth = 2, label='img_r5')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.title('A Recall Curve')
            plt.legend(loc="upper right")

            plt.cla()
            plt.close("all")
            print(itm_results)
            print("Get recall done.")