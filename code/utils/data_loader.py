import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np

import time
import logging
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from glob import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90,
    Compose, OneOf, Rotate, Resize
)


class Data_Loader(Dataset):
    def __init__(self, data_path, n_classes=10, 
                 matches=[1,2,3,4,5,6,7,8,9,10], 
                 transform=None, 
                 image_select_thresholds=None,
                 back_ground_index = None,
                 back_ground_threshold=None):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob(os.path.join(data_path, '*.tif'))
        if not n_classes==len(matches):
            raise ValueError('the len(matches) should be same as the n_classes')
        self.n_classes = n_classes
        self.matches = matches
        
        self.transform = transform
        
        # if the class x not selected, the threshold of class x should be 1
        self.select_threshold = image_select_thresholds
        self.back_index = back_ground_index
        self.back_ground_threshold = back_ground_threshold
        if image_select_thresholds is not None:
            assert len(image_select_thresholds)==len(matches)
            if self.back_index is not None:
                assert self.back_index in self.matches
                self.back_index = self.matches.index(self.back_index)
            self.select_img()
            
    def select_img(self):
        imgs_path = glob(os.path.join(self.data_path, '*.tif'))
        selected_imgs_path = []
        print('start select imgs...')
        for i in tqdm(range(len(imgs_path))):
            img_path = imgs_path[i]
            label_path = img_path.replace('tif', 'png')
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            label = self.get_segmentation_array(label)
            class_num, h, w = label.shape
            label = label.reshape((label.shape[0],-1))
            pixel_num = h * w
            rates = label.sum(axis=1)
            """
            if self.back_index is not None:
                if rates[self.back_index] <= self.back_ground_threshold:
                    selected_imgs_path.append(img_path)
                    continue
                    """
            for j in range(len(self.select_threshold)):
                threshold = self.select_threshold[j]
                if rates[j] >= threshold:
                    selected_imgs_path.append(img_path)
                    break
            
        
        self.imgs_path = selected_imgs_path
        print('this datasets has selected '+str(len(self.imgs_path))+' images')
            
                

    def get_segmentation_array(self, label):
        h, w = label.shape
        seg_labels = np.zeros((self.n_classes, h, w))

        for m in self.matches:
            label[label == m] = self.matches.index(m)
        
        for c in range(self.n_classes):
            seg_labels[c, :, :] = (label == c).astype(int)

        return seg_labels
    
    def image_preprocessing(self, image, label):
        #image = image/127.5 - 1
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        
        #image = self.normalize(img=image)

        image = image.transpose(2,0,1)
        # 随机进行数据增强，为2时不做处理
        """
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        """
        label = self.get_segmentation_array(label)
        
        return image, label
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('tif', 'png')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        #label = label.astype(int)
        
        image, label = self.image_preprocessing(image, label)

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)
    

class Class_Data_Loader(Data_Loader):
    def __init__(self, data_path, n_classes=10, 
                 class_index = 1, threshold = 0.01,
                 matches=[1,2,3,4,5,6,7,8,9,10], 
                 transform=None):
        super(Class_Data_Loader, self).__init__(data_path, n_classes, matches, transform)
        if not class_index in matches:
            raise ValueError('class index should be included in matches')
        self.class_imgs_path = []
        self.class_index = class_index
        self.threshold = threshold
        self.initial_loader()
        
    def get_segmentation_array(self, label):
        h, w = label.shape
        seg_label = np.zeros((1, h, w))
        seg_label[0,:,:] = (label==self.class_index).astype(int)

        return seg_label
    
    def initial_loader(self):
        print('The data loader is initializing...')
        for i in tqdm(range(len(self.imgs_path))):
            img_path = self.imgs_path[i]
            lab_path = img_path.replace('tif', 'png')
            label = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
            label = self.get_segmentation_array(label)
            pixel_num = label.sum()
            _, h, w = label.shape
            rate = pixel_num/(h*w)
            if rate>=self.threshold:
                self.class_imgs_path.append(img_path)
                continue
        print('The len of the data loader is '+str(self.__len__()))
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.class_imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('tif', 'png')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        #label = label.astype(int)
        
        image, label = self.image_preprocessing(image, label)

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.class_imgs_path)
        
def get_transform(resize_h=256, resize_w=256, resize_rates=[0,0.25],
                  is_aug=True, is_resize_rate=False,
                  mean=(0.485, 0.456, 0.406, 0.449), 
                  std=(0.229, 0.224, 0.225, 0.226)):
    transform_list=[]
    aug_part = A.OneOf([A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5), A.Transpose(p=0.5)])
    if is_resize_rate:
        transform_list.append(A.RandomScale(scale_limit=resize_rates, p=1.0))
    else:
        transform_list.append(A.Resize(height=resize_h, width=resize_w, p=1.0))
    
    if is_aug:
        transform_list.append(aug_part)
    else:
        transform_list = []
        transform_list.append(A.Resize(height=256, width=256, p=1.0))
    
    transform_list.append(A.Normalize(mean=mean,std=std))
    transform = A.Compose(transform_list)
    
    return transform

def get_binary_datasets(data_path, class_index_list, thresholds, resize_rates=[0, 0.25]):
    assert len(class_index_list)==len(thresholds)
    datasets = []
    for i in range(len(class_index_list)):
        transform = get_transform(resize_rates=resize_rates, is_resize_rate=False, is_aug=True)
        #val_transform = get_transform(is_resize_rate=False, is_aug=False)
        data = Class_Data_Loader(data_path=data_path, class_index=class_index_list[i], threshold=thresholds[i], transform=transform)
        datasets.append(data)
        
    return datasets
    
def get_multiclass_dataset(data_path, n_classes, matches, back_index, back_threshold, select_thresholds,resize_rates=[0, 0.25]):
    transform = get_transform(resize_rates=resize_rates)
    dataset = Data_Loader(data_path=data_path, n_classes=n_classes,
                          matches=matches, transform=transform, 
                          image_select_thresholds=select_thresholds, 
                          back_ground_index=back_index,
                          back_ground_threshold=back_threshold)
    return dataset
    
if __name__ == "__main__":
    dataset = Data_Loader("./data/train")
    print("数据个数：", len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=2, 
                                               shuffle=True)
    i = 0
    for image, label in train_loader:
        print('image.shape:', image.shape)
        print('label.shape:', label.shape)
        i=i+1
        if i>=10:
            break
        