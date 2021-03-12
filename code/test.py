import os

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as D
from torch import optim
from torch.utils.data import SubsetRandomSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.enabled = True

import glob
import numpy as np
import cv2
from tqdm import tqdm

from model.NetWorks import *
from utils.option import *
from utils.data_loader import *

import segmentation_models_pytorch as smp



def test_net(net, opt):
    if opt.is_pretrain:
        print('load pretrain model')
        filename = os.path.join(opt.save_model_path, 'main_model' + 'checkpoint-best.pth')
        #ckpt = torch.load(opt.best_iou_model)
        ckpt = torch.load(filename)
        epoch_start = ckpt['epoch']
        net.load_state_dict(ckpt['state_dict'])
        #optimizer.load_state_dict(ckpt['optimizer'])
        net.to(device=opt.device)
    # 选择设备，有cuda用cuda，没有就用cpu
    device = opt.device
    # 读取所有图片路径
    tests_path = glob(opt.test_path + '/*.tif')
    save_path = opt.test_result_save_path
    # 遍历素有图片
    for test_path in tqdm(tests_path):
        # 保存结果地址
        save_res_path = test_path.replace(opt.test_path, save_path)
        save_res_path = save_res_path.replace('.tif', '.png')
        # 读取图片
        img = cv2.imread(test_path, cv2.IMREAD_UNCHANGED)
        img = normalize(img)
        #img = img/127.5 - 1
        img = img.transpose(2, 0, 1)
        
        # 转为灰度图
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        #print(img.shape)
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=opt.device, dtype=torch.float32)
        
        # 预测
        pred = net(img_tensor)
        pred = pred.data.cpu().numpy()
        if opt.is_tta:     
            pred1 = net(torch.flip(img_tensor, [3]))
            pred2 = net(torch.flip(img_tensor, [2]))
            pred1 = torch.flip(pred1, [3]).data.cpu().numpy()
            pred2 = torch.flip(pred2, [2]).data.cpu().numpy()
            pred = (pred+pred1+pred2)/3.0
        # 提取结果
        #pred = np.array(pred.data.cpu())
        
        pred = np.argmax(pred, axis=1) + 1
        pred = pred.reshape(pred.shape[1], pred.shape[2])
        # 保存图片
        cv2.imwrite(save_res_path, pred)

def normalize(img, max_pixel_value=255.0):
    mean = (0.485, 0.456, 0.406, 0.449)
    std = (0.229, 0.224, 0.225, 0.226)
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


if __name__ == "__main__":
    opt = option()
    device = opt.device
    net = SegModel(n_class=opt.n_classes,in_channel=opt.input_channel, model_name='Unet++', encoder=opt.backbone, activation=opt.last_activation)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    #net.load_state_dict(torch.load(opt.save_model_name, map_location=device))
    # 测试模式
    net.eval()
    #
    test_net(net, opt)
    