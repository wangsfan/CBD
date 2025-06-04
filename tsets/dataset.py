import os
from os.path import join, splitext, exists
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import json
from .utils import cvtColor, preprocess_input, pre_caption
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize
import json
import matplotlib.pyplot as plt
import os
import sys
import json
import pickle
import random
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def parse(t):
    res = ""
    flag = False
    fn = t
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "</text>":
                break
            if flag:
                res += " " + line
            if line == "<text>":
                flag = True
    return res

def clean(strings, pattern):

    return [s.replace(pattern, "") for s in strings]



class Scenario_1_task1_class(data.Dataset):
    def __init__(self, input_shape, random, autoaugment_flag=True, subset=None):
        self.input_shape    = input_shape
        self.random         = random

        self.text       = []
        self.image      = []
        self.label      = []
        self.txt2img    = {}
        self.img2txt    = {}
        txt_id          = 0


        data_path_img = './data/Scenario_1_class/Incremental_setting_class/task1/img/'
        data_path_text = './data/Scenario_1_class/Incremental_setting_class/task1/text/'

        train_images, train_label, val_images, val_label, path = read_split_data(data_path_img, 0.3)
        self.train_images = train_images
        train_text = []
        val_text = []

        if subset =='train':
            self.image = train_images
            self.label = train_label
            self.datasets_path  = './data/Scenario_1_class/Incremental_setting_class/task1/img/'

            for img_id, i in enumerate(train_images):
                with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
                    first_line = file.readline()
                self.text.append(pre_caption(first_line, 77))

        if subset =='test':
            self.image = val_images
            self.label = val_label
            self.datasets_path  = './data/Scenario_1_class/Incremental_setting_class/task1/img/'

            for img_id, i in enumerate(val_images):
                with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
                    first_line = file.readline()
                self.text.append(pre_caption(first_line, 77))

        self.autoaugment_flag   = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)
        
    def __len__(self):
        return len(self.image)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def sample_list(self, lst, k):
        return [random.choice(lst) for _ in range(k)]
    
    def get_buf(self):
        text_buf = []
        img_buf = self.sample_list(self.train_images, 35)
        for img_id, i in enumerate(img_buf):
            with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
                first_line = file.readline()
            text_buf.append(pre_caption(first_line, 77))
        return img_buf, text_buf

    def __getitem__(self, index):
        photo_name  = self.image[index]
        label = self.label[index]
        image_path  = photo_name
        caption     = self.text[index]
        
        image       = Image.open(image_path)

        image       = cvtColor(image)
        if self.autoaugment_flag:
            image = self.AutoAugment(image, random=self.random)
        else:
            image = self.get_random_data(image, self.input_shape, random=self.random)
        image       = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))
        return image, caption

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype

        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
        
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        image = self.resize_crop(image)
        
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        image = self.policy(image)
        return image
    
    
class Scenario_1_task2_class(data.Dataset):
    def __init__(self, input_shape, random, autoaugment_flag=True, subset=None):
        self.input_shape    = input_shape
        self.random         = random

        self.text       = []
        self.image      = []
        self.label      = []
        self.txt2img    = {}
        self.img2txt    = {}
        txt_id          = 0


        data_path_img = './data/Scenario_1_class/Incremental_setting_class/task2/img/'
        data_path_text = './data/Scenario_1_class/Incremental_setting_class/task2/text/'

        train_images, train_label, val_images, val_label, path = read_split_data(data_path_img, 0.3)
        self.train_images = train_images
        train_text = []
        val_text = []

        if subset =='train':
            self.image = train_images
            self.label = train_label
            self.datasets_path  = './data/Scenario_1_class/Incremental_setting_class/task2/img/'

            for img_id, i in enumerate(train_images):
                with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
                    first_line = file.readline()
                self.text.append(pre_caption(first_line, 77))

        if subset =='test':
            self.image = val_images
            self.label = val_label
            self.datasets_path  = './data/Scenario_1_class/Incremental_setting_class/task2/img/'

            for img_id, i in enumerate(val_images):
                with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
                    first_line = file.readline()
                self.text.append(pre_caption(first_line, 77))

        self.autoaugment_flag   = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)
        
    def __len__(self):
        return len(self.image)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def sample_list(self, lst, k):
        return [random.choice(lst) for _ in range(k)]
    
    def get_buf(self):
        text_buf = []
        img_buf = self.sample_list(self.train_images, 35)
        for img_id, i in enumerate(img_buf):
            with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
                first_line = file.readline()
            text_buf.append(pre_caption(first_line, 77))
        # print(text_buf)
        # print(img_buf)

        # print('dfh')
        return img_buf, text_buf
    
    def __getitem__(self, index):
        photo_name  = self.image[index]
        label = self.label[index]
        # image_path  = os.path.join(self.datasets_path, photo_name.zfill(16))
        # image_path  = os.path.join(self.datasets_path, photo_name)
        image_path  = photo_name
        caption     = self.text[index]
        
        image       = Image.open(image_path)
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image       = cvtColor(image)
        if self.autoaugment_flag:
            image = self.AutoAugment(image, random=self.random)
        else:
            image = self.get_random_data(image, self.input_shape, random=self.random)
        image       = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))
        return image, caption

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
        
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        #------------------------------------------#
        #   resize并且随即裁剪
        #------------------------------------------#
        image = self.resize_crop(image)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   随机增强
        #------------------------------------------#
        image = self.policy(image)
        return image


class Scenario_1_task3_class(data.Dataset):
    def __init__(self, input_shape, random, autoaugment_flag=True, subset=None):
        self.input_shape    = input_shape
        self.random         = random

        self.text       = []
        self.image      = []
        self.label      = []
        self.txt2img    = {}
        self.img2txt    = {}
        txt_id          = 0


        data_path_img = './data/Scenario_1_class/Incremental_setting_class/task3/img/'
        data_path_text = './data/Scenario_1_class/Incremental_setting_class/task3/text/'

        train_images, train_label, val_images, val_label, path = read_split_data(data_path_img, 0.3)
        self.train_images = train_images
        train_text = []
        val_text = []

        if subset =='train':
            self.image = train_images
            self.label = train_label
            self.datasets_path  = './data/Scenario_1_class/Incremental_setting_class/task3/img/'

            for img_id, i in enumerate(train_images):
                with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
                    first_line = file.readline()
                self.text.append(pre_caption(first_line, 77))

        if subset =='test':
            self.image = val_images
            self.label = val_label
            self.datasets_path  = './data/Scenario_1_class/Incremental_setting_class/task3/img/'

            for img_id, i in enumerate(val_images):
                with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
                    first_line = file.readline()
                self.text.append(pre_caption(first_line, 77))

        self.autoaugment_flag   = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)
        
    def __len__(self):
        return len(self.image)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def sample_list(self, lst, k):
        return [random.choice(lst) for _ in range(k)]
    
    def get_buf(self):
        text_buf = []
        img_buf = self.sample_list(self.train_images, 35)
        for img_id, i in enumerate(img_buf):
            with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
                first_line = file.readline()
            text_buf.append(pre_caption(first_line, 77))
        # print(text_buf)
        # print(img_buf)

        # print('dfh')
        return img_buf, text_buf
    
    def __getitem__(self, index):
        photo_name  = self.image[index]
        label = self.label[index]
        # image_path  = os.path.join(self.datasets_path, photo_name.zfill(16))
        # image_path  = os.path.join(self.datasets_path, photo_name)
        image_path  = photo_name
        caption     = self.text[index]
        
        image       = Image.open(image_path)
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image       = cvtColor(image)
        if self.autoaugment_flag:
            image = self.AutoAugment(image, random=self.random)
        else:
            image = self.get_random_data(image, self.input_shape, random=self.random)
        image       = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))
        return image, caption

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
        
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        #------------------------------------------#
        #   resize并且随即裁剪
        #------------------------------------------#
        image = self.resize_crop(image)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   随机增强
        #------------------------------------------#
        image = self.policy(image)
        return image
    

class Scenario_2_task1_class(data.Dataset):
    def __init__(self, input_shape, random, autoaugment_flag=True, subset=None):
        self.input_shape    = input_shape
        self.random         = random

        self.text       = []
        self.image      = []
        self.txt2img    = {}
        self.img2txt    = {}
        txt_id          = 0

        data_path = './data/Scenario_2_class/Incremental_setting_class/task1/img/'
        data_path_text = './data/Scenario_2_class/Incremental_setting_class/texts/'
        json_path = './data/Scenario_2_class/Incremental_setting_class/wikipedia.json'
        self.json_path = json_path

        train_images, train_label, val_images, val_label, path = read_split_data(data_path, 0.2)
        train_text = []
        val_text = []
        self.train_images = train_images

        if subset =='train':
            self.image = train_images
            self.label = train_label
            self.datasets_path  = './data/Scenario_2_class/Incremental_setting_class/task1/img/'
            data = json.load(open(json_path, 'r', encoding = 'utf-8'))

            for i , j in enumerate(train_images):
                dfh = j.replace('./data/Scenario_2_class/Incremental_setting_class/task1/img/','').partition('/')
                dfh = dfh[2]
                ddd = data[dfh.strip('.jpg')]
                txt_f = join('./data/Scenario_2_class/Incremental_setting_class/texts',"{}.xml".format(ddd))
                sentences = []
                doc = parse(txt_f)  # 手动解析
                words = doc.split()
                for pat in (",", ".", "!", "?", "''", "(", ")", "\"", ":", ";", "{", "}", "[", "]","&","-"):
                    words = clean(words, pat)
                sentences.append(words)
                str = ' '.join(sentences[0])
                self.text.append(pre_caption(str, 77))

        if subset =='test':
            self.image = val_images[:100]
            self.label = val_label
            self.datasets_path  = './data/Scenario_2_class/Incremental_setting_class/task1/img/'
            data = json.load(open(json_path, 'r', encoding = 'utf-8'))

            for i , j in enumerate(val_images[:100]):
                dfh = j.replace('./data/Scenario_2_class/Incremental_setting_class/task1/img/','').partition('/')
                dfh = dfh[2]
                ddd = data[dfh.strip('.jpg')]
                txt_f = join('./data/Scenario_2_class/Incremental_setting_class/texts',"{}.xml".format(ddd))
                sentences = []
                doc = parse(txt_f)  # 手动解析
                words = doc.split()
                for pat in (",", ".", "!", "?", "''", "(", ")", "\"", ":", ";", "{", "}", "[", "]","&","-"):
                    words = clean(words, pat)
                sentences.append(words)
                str = ' '.join(sentences[0])
                self.text.append(pre_caption(str, 77))


        self.autoaugment_flag   = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)
        
    def __len__(self):
        return len(self.image)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def sample_list(self, lst, k):
        return [random.choice(lst) for _ in range(k)]
    
    def get_buf(self):
        text_buf = []
        data = json.load(open(self.json_path, 'r', encoding = 'utf-8'))
        img_buf = self.sample_list(self.train_images, 35)
        for img_id, i in enumerate(img_buf):
            dfh = i.replace('./data/Scenario_2_class/Incremental_setting_class/task1/img/','').partition('/')
            dfh = dfh[2]
            ddd = data[dfh.strip('.jpg')]
            txt_f = join('./data/Scenario_2_class/Incremental_setting_class/texts',"{}.xml".format(ddd))
            sentences = []
            doc = parse(txt_f)  # 手动解析
            words = doc.split()
            for pat in (",", ".", "!", "?", "''", "(", ")", "\"", ":", ";", "{", "}", "[", "]","&","-"):
                words = clean(words, pat)
            sentences.append(words)
            str = ' '.join(sentences[0])
            text_buf.append(pre_caption(str, 77))

            # with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
            #     first_line = file.readline()
            # text_buf.append(pre_caption(first_line, 77))
        # print(text_buf)
        # print(img_buf)

        # print('dfh')
        return img_buf, text_buf

    def __getitem__(self, index):
        photo_name  = self.image[index]
        label = self.label[index]
        # image_path  = os.path.join(self.datasets_path, photo_name.zfill(16))
        # image_path  = os.path.join(self.datasets_path, photo_name)
        image_path  = photo_name
        caption     = self.text[index]
        
        image       = Image.open(image_path)
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image       = cvtColor(image)
        if self.autoaugment_flag:
            image = self.AutoAugment(image, random=self.random)
        else:
            image = self.get_random_data(image, self.input_shape, random=self.random)
        image       = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))
        return image, caption

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
        
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        #------------------------------------------#
        #   resize并且随即裁剪
        #------------------------------------------#
        image = self.resize_crop(image)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   随机增强
        #------------------------------------------#
        image = self.policy(image)
        return image
    

class Scenario_2_task2_class(data.Dataset):
    def __init__(self, input_shape, random, autoaugment_flag=True, subset=None):
        self.input_shape    = input_shape
        self.random         = random

        self.text       = []
        self.image      = []
        self.txt2img    = {}
        self.img2txt    = {}
        txt_id          = 0

        data_path = './data/Scenario_2_class/Incremental_setting_class/task2/img/'
        data_path_text = './data/Scenario_2_class/Incremental_setting_class/texts/'
        json_path = './data/Scenario_2_class/Incremental_setting_class/wikipedia.json'
        self.json_path = json_path

        train_images, train_label, val_images, val_label, path = read_split_data(data_path, 0.2)
        train_text = []
        val_text = []
        self.train_images = train_images

        if subset =='train':
            self.image = train_images
            self.label = train_label
            self.datasets_path  = './data/Scenario_2_class/Incremental_setting_class/task2/img/'
            data = json.load(open(json_path, 'r', encoding = 'utf-8'))

            for i , j in enumerate(train_images):
                dfh = j.replace('./data/Scenario_2_class/Incremental_setting_class/task2/img/','').partition('/')
                dfh = dfh[2]
                ddd = data[dfh.strip('.jpg')]
                txt_f = join('./data/Scenario_2_class/Incremental_setting_class/texts',"{}.xml".format(ddd))
                sentences = []
                doc = parse(txt_f)  # 手动解析
                words = doc.split()
                for pat in (",", ".", "!", "?", "''", "(", ")", "\"", ":", ";", "{", "}", "[", "]","&","-"):
                    words = clean(words, pat)
                sentences.append(words)
                str = ' '.join(sentences[0])
                self.text.append(pre_caption(str, 77))

        if subset =='test':
            self.image = val_images
            self.label = val_label
            self.datasets_path  = './data/Scenario_2_class/Incremental_setting_class/task2/img/'
            data = json.load(open(json_path, 'r', encoding = 'utf-8'))

            for i , j in enumerate(val_images):
                dfh = j.replace('./data/Scenario_2_class/Incremental_setting_class/task2/img/','').partition('/')
                dfh = dfh[2]
                ddd = data[dfh.strip('.jpg')]
                txt_f = join('./data/Scenario_2_class/Incremental_setting_class/texts',"{}.xml".format(ddd))
                sentences = []
                doc = parse(txt_f)  # 手动解析
                words = doc.split()
                for pat in (",", ".", "!", "?", "''", "(", ")", "\"", ":", ";", "{", "}", "[", "]","&","-"):
                    words = clean(words, pat)
                sentences.append(words)
                str = ' '.join(sentences[0])
                self.text.append(pre_caption(str, 77))


        self.autoaugment_flag   = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)
        
    def __len__(self):
        return len(self.image)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def sample_list(self, lst, k):
        return [random.choice(lst) for _ in range(k)]
    
    def get_buf(self):
        text_buf = []
        data = json.load(open(self.json_path, 'r', encoding = 'utf-8'))
        img_buf = self.sample_list(self.train_images, 35)
        for img_id, i in enumerate(img_buf):
            # print(i)
            dfh = i.replace('./data/Scenario_2_class/Incremental_setting_class/task2/img/','')
            dfh = dfh.partition('/')
            # print(dfh)
            dfh = dfh[2]
            # print(dfh)
            ddd = data[dfh.strip('.jpg')]
            txt_f = join('./data/Scenario_2_class/Incremental_setting_class/texts',"{}.xml".format(ddd))
            sentences = []
            doc = parse(txt_f)  # 手动解析
            words = doc.split()
            for pat in (",", ".", "!", "?", "''", "(", ")", "\"", ":", ";", "{", "}", "[", "]","&","-"):
                words = clean(words, pat)
            sentences.append(words)
            str = ' '.join(sentences[0])
            text_buf.append(pre_caption(str, 77))

            # with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
            #     first_line = file.readline()
            # text_buf.append(pre_caption(first_line, 77))
        # print(text_buf)
        # print(img_buf)

        # print('dfh')
        return img_buf, text_buf
    
    def __getitem__(self, index):
        photo_name  = self.image[index]
        label = self.label[index]
        # image_path  = os.path.join(self.datasets_path, photo_name.zfill(16))
        # image_path  = os.path.join(self.datasets_path, photo_name)
        image_path  = photo_name
        caption     = self.text[index]
        
        image       = Image.open(image_path)
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image       = cvtColor(image)
        if self.autoaugment_flag:
            image = self.AutoAugment(image, random=self.random)
        else:
            image = self.get_random_data(image, self.input_shape, random=self.random)
        image       = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))
        return image, caption

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
        
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        #------------------------------------------#
        #   resize并且随即裁剪
        #------------------------------------------#
        image = self.resize_crop(image)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   随机增强
        #------------------------------------------#
        image = self.policy(image)
        return image


class Scenario_2_task3_class(data.Dataset):
    def __init__(self, input_shape, random, autoaugment_flag=True, subset=None):
        self.input_shape    = input_shape
        self.random         = random

        self.text       = []
        self.image      = []
        self.txt2img    = {}
        self.img2txt    = {}
        txt_id          = 0

        data_path = './data/Scenario_2_class/Incremental_setting_class/task3/img/'
        data_path_text = './data/Scenario_2_class/Incremental_setting_class/texts/'
        json_path = './data/Scenario_2_class/Incremental_setting_class/wikipedia.json'
        self.json_path = json_path

        train_images, train_label, val_images, val_label, path = read_split_data(data_path, 0.2)
        train_text = []
        val_text = []
        self.train_images = train_images

        if subset =='train':
            self.image = train_images
            self.label = train_label
            self.datasets_path  = './data/Scenario_2_class/Incremental_setting_class/task3/img/'
            data = json.load(open(json_path, 'r', encoding = 'utf-8'))

            for i , j in enumerate(train_images):
                dfh = j.replace('./data/Scenario_2_class/Incremental_setting_class/task3/img/','').partition('/')
                dfh = dfh[2]
                ddd = data[dfh.strip('.jpg')]
                txt_f = join('./data/Scenario_2_class/Incremental_setting_class/texts',"{}.xml".format(ddd))
                sentences = []
                doc = parse(txt_f)  # 手动解析
                words = doc.split()
                for pat in (",", ".", "!", "?", "''", "(", ")", "\"", ":", ";", "{", "}", "[", "]","&","-"):
                    words = clean(words, pat)
                sentences.append(words)
                str = ' '.join(sentences[0])
                self.text.append(pre_caption(str, 77))

        if subset =='test':
            self.image = val_images[:100]
            self.label = val_label
            self.datasets_path  = './data/Scenario_2_class/Incremental_setting_class/task3/img/'
            data = json.load(open(json_path, 'r', encoding = 'utf-8'))

            for i , j in enumerate(val_images[:100]):
                dfh = j.replace('./data/Scenario_2_class/Incremental_setting_class/task3/img/','').partition('/')
                dfh = dfh[2]
                ddd = data[dfh.strip('.jpg')]
                txt_f = join('./data/Scenario_2_class/Incremental_setting_class/texts',"{}.xml".format(ddd))
                sentences = []
                doc = parse(txt_f)  # 手动解析
                words = doc.split()
                for pat in (",", ".", "!", "?", "''", "(", ")", "\"", ":", ";", "{", "}", "[", "]","&","-"):
                    words = clean(words, pat)
                sentences.append(words)
                str = ' '.join(sentences[0])
                self.text.append(pre_caption(str, 77))


        self.autoaugment_flag   = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)
        
    def __len__(self):
        return len(self.image)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def sample_list(self, lst, k):
        return [random.choice(lst) for _ in range(k)]
    
    def get_buf(self):
        text_buf = []
        data = json.load(open(self.json_path, 'r', encoding = 'utf-8'))
        img_buf = self.sample_list(self.train_images, 35)
        for img_id, i in enumerate(img_buf):
            dfh = i.replace('./data/Scenario_2_class/Incremental_setting_class/task3/img/','').partition('/')
            dfh = dfh[2]
            ddd = data[dfh.strip('.jpg')]
            txt_f = join('./data/Scenario_2_class/Incremental_setting_class/texts',"{}.xml".format(ddd))
            sentences = []
            doc = parse(txt_f)  # 手动解析
            words = doc.split()
            for pat in (",", ".", "!", "?", "''", "(", ")", "\"", ":", ";", "{", "}", "[", "]","&","-"):
                words = clean(words, pat)
            sentences.append(words)
            str = ' '.join(sentences[0])
            text_buf.append(pre_caption(str, 77))

            # with open(i.replace('.jpg', '.txt').replace('img','text'), 'r') as file:
            #     first_line = file.readline()
            # text_buf.append(pre_caption(first_line, 77))
        # print(text_buf)
        # print(img_buf)

        # print('dfh')
        return img_buf, text_buf
    
    def __getitem__(self, index):
        photo_name  = self.image[index]
        label = self.label[index]
        # image_path  = os.path.join(self.datasets_path, photo_name.zfill(16))
        # image_path  = os.path.join(self.datasets_path, photo_name)
        image_path  = photo_name
        caption     = self.text[index]
        
        image       = Image.open(image_path)
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image       = cvtColor(image)
        if self.autoaugment_flag:
            image = self.AutoAugment(image, random=self.random)
        else:
            image = self.get_random_data(image, self.input_shape, random=self.random)
        image       = np.transpose(preprocess_input(np.array(image, dtype='float32')), (2, 0, 1))
        return image, caption

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
        
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        #------------------------------------------#
        #   resize并且随即裁剪
        #------------------------------------------#
        image = self.resize_crop(image)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   随机增强
        #------------------------------------------#
        image = self.policy(image)
        return image


def dataset_collate(batch):
    images      = []
    captions    = []
    for image, caption in batch:
        images.append(image)
        captions.append(caption)

    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    return images, captions