import os
import time
import copy

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as D
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.enabled = True

from torch import optim
from torch.utils.data import SubsetRandomSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from utils.option import *
from utils.metric import *
from tqdm import tqdm
from model.NetWorks import *
from model.losses import *
from utils.data_loader import *

import segmentation_models_pytorch as smp
from pytorch_toolbelt import losses as L
from segmentation_models_pytorch.losses import DiceLoss,FocalLoss,SoftCrossEntropyLoss, LovaszLoss

import numpy as np

import traceback

def train_net(net, opt, writer, 
              dataset, optimizer, 
              scheduler, criterion, 
              batch_size=2, lr=0.0001,
              n_classes=10, 
              epoch_start=0,
              epochs=20, 
              model_name='main_model'):
    # create a tensorboard
    train_step = 0
    val_step = 0
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    best_iou = 0
    # 训练epochs次
    for epoch in range(epoch_start, epoch_start + epochs):
        print('training '+str(epoch)+'th step...')
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(opt.val_rate * dataset_size))
        if opt.shuffle_dataset :
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        scaler = GradScaler() 
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                        sampler=valid_sampler)
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        # train
        idx = 0
        
        for image, label in tqdm(train_loader):
            #optimizer.zero_grad()
            train_loader_size = train_loader.__len__()
            if train_step % opt.show_inter == 0:
                train_label_color = label2color(label, n_classes)
            # 将数据拷贝到device中
            image = image.to(device=opt.device, dtype=torch.float32)
            label = label.to(device=opt.device, dtype=torch.float32)
            
            with autocast():
                
                # 使用网络参数，输出预测结果
                pred = net(image)
                if train_step % opt.show_inter == 0:
                    train_pred_color = label2color(pred, n_classes)
                    writer.add_image(model_name+'/train_ground_truth', train_label_color, global_step=train_step)
                    writer.add_image(model_name+'/train_predict_image', train_pred_color, global_step=train_step)
                # 计算loss
                loss = criterion(pred, label)
                writer.add_scalar(model_name+'Loss/train', loss.item(), train_step)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # 更新参数
                #loss.backward()
                #optimizer.step()
            scheduler.step(epoch + idx / train_loader_size)
            idx = idx+1
            train_step= train_step + 1
        del loss, train_loader, train_indices
        
        if epoch%opt.save_inter == 0 and epoch>opt.min_inter:
            state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(opt.save_model_path, model_name + 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
        # validation
        #validation_loss = 0
        net.eval()
        iou=IOUMetric(n_classes)
        for image, label in tqdm(validation_loader):
            optimizer.zero_grad()
            # 将数据拷贝到device中
            if val_step % opt.show_inter == 0:
                label_color = label2color(label, n_classes)
            """
            if n_classes==1:
                n,c,h,w = label.shape
                label = label.reshape(n,h,w)
            """
            image = image.to(device=opt.device, dtype=torch.float32)
            label = label.to(device=opt.device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            
            if val_step % opt.show_inter == 0:
                pred_color = label2color(pred, n_classes)
                writer.add_image(model_name+'/ground_truth', label_color, global_step=val_step)
                writer.add_image(model_name+'/predict_image', pred_color, global_step=val_step)
                del pred_color, label_color
            #writer.add_image('ground_truth', label_color, global_step=val_step)
            #writer.add_image('predict_image', pred_color, global_step=val_step)
            # 计算loss

            val_loss = criterion(pred, label)
            
            writer.add_scalar(model_name+'Loss/val', val_loss.item(), val_step)
            val_step = val_step+1
            
            pred=pred.cpu().data.numpy()
            label = label.cpu().data.numpy()
            if n_classes==1:
                pred[pred>=0.5]=1
                pred[pred<0.5]=0
                label[label>=0.5]=1
                label[label<0.5]=0
                #n,h,w = label.shape
                #label = label.reshape(n,1,h,w)
            else:
                pred= np.argmax(pred,axis=1)
                label = np.argmax(label,axis=1)
            pred = pred.astype(int)
            label = label.astype(int)
            iou.add_batch(pred,label)
            #validation_loss = validation_loss + val_loss.item()
            del val_loss, pred
            
        acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
        writer.add_scalar(model_name+'meanIOU', mean_iu, epoch)
        #validation_loss = validation_loss/split
        #print('Validation Loss: ', validation_loss)
        del val_indices, valid_sampler, validation_loader#, validation_loss
        if mean_iu > best_iou:
            state = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(opt.save_model_path, model_name + 'checkpoint-best.pth')
            torch.save(state, filename)
            best_iou = mean_iu
      
def main(opt):
    # create a tensorboard
    writer = SummaryWriter(opt.train_log)
    
    main_net = SegModel(n_class=opt.n_classes,
                        in_channel=opt.input_channel, 
                        model_name='Unet++', 
                        encoder=opt.backbone, 
                        activation=opt.last_activation)
    multi_class_dataset = get_multiclass_dataset(data_path=opt.train_path, n_classes=opt.n_classes, 
                                                 matches=opt.matches,back_index=opt.back_ground_index,
                                                 back_threshold=opt.back_ground_threshold,
                                                 select_thresholds=opt.select_thresholds)

    if opt.is_pretrain:
        print('load pretrain model')
        filename = os.path.join(opt.save_model_path, 'main_model' + 'checkpoint-best.pth')
        ckpt = torch.load(filename)
        epoch_start = ckpt['epoch']
        main_net.load_state_dict(ckpt['state_dict'])
        #optimizer.load_state_dict(ckpt['optimizer'])
    main_net.to(device=opt.device)
    
    # 定义RMSprop算法
    #optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
    #optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay, amsgrad=False)
    optimizer = optim.AdamW(main_net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    #optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    #optimizer = optim.Adadelta(net.parameters(), lr=opt.lr, rho=0.9, eps=1e-06, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)
    #scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    # 定义Loss算法
    DiceLoss_fn=DiceLoss(mode='multilabel')
    #SoftCrossEntropy_fn = SoftCrossEntropyLoss(smooth_factor=0.1)
    Focal_loss = FocalLoss(mode='multilabel', ignore_index=1)
    #Lovasz_Loss = LovaszLoss(mode='multilabel')
    #criterion = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn, first_weight=0.5, second_weight=0.5).cuda()
    main_loss = L.JointLoss(first=DiceLoss_fn, second=Focal_loss, first_weight=0.5, second_weight=0.5).cuda()
    #criterion = L.JointLoss(first=DiceLoss_fn, second=Lovasz_Loss, first_weight=0.5, second_weight=0.5).cuda()
    
    train_net(net=main_net, opt=opt, 
              writer=writer, 
              dataset=multi_class_dataset, 
              optimizer=optimizer, 
              scheduler=scheduler, 
              criterion=main_loss,
              batch_size=opt.batch_size,
              lr=opt.lr, n_classes=opt.n_classes,
              epoch_start=epoch_start, 
              epochs=opt.epochs)

def label2color(label, n_classes=10):
    if len(label.shape)==4:
        img = get_img(label, n_classes)
    elif len(label.shape)==3:
        img = label
        img = img.cpu()
        img = img.numpy()
        img = img-1
        img = img[0,:,:]
        img = img.reshape(img.shape[0], img.shape[1])
        #print(img.shape)
    else:
        raise ValueError('label shape should be 4 (N, C, H, W) or 3(N, H, W)')
    img = get_color(img,c=n_classes)
    return img

def get_img(class_possibility, n_classes=10):
    if type(class_possibility) is np.ndarray:
        if n_classes==1:
            img[img>=0.5] = 1
            img[img<0.5] = 0
        else:
            img = np.argmax(class_possibility, axis=1)
    elif torch.is_tensor(class_possibility):
        if n_classes==1:
            img = class_possibility.cpu()
            img = img.data.numpy()
            img[img>=0.5] = 1
            img[img<0.5] = 0
            n,c,h,w =img.shape
            img = img.reshape(n,h,w)
        else:
            img = torch.argmax(class_possibility, dim=1)
            img = img.cpu()
            img = img.data.numpy()
    else:
        print(traceback.format_exc())
        raise TypeError('img should be np.ndarray or torch.tensor')
    img = img[0,:,:]
    img = img.reshape(img.shape[0], img.shape[1])
    return img

def get_color(image, c=10):
    assert c<=10
    color_array = np.array([[177, 191, 122],  # farm_land
                            [0, 128, 0],  # forest
                            [128, 168, 93],  # grass
                            [62, 51, 0],  # road
                            [128, 128, 0],  # urban_area
                            [128, 128, 128],  # countryside
                            [192, 128, 0],  # industrial_land
                            [0, 128, 128],  # construction
                            [132, 200, 173],  # water
                            [128, 64, 0]],  # bareland
                           dtype='uint8')
    color_image = np.zeros((3,image.shape[0], image.shape[1]))
    if c==1:
        for i in range(3):
            color_image[i,:,:] = image*128
    else:
        for idx in range(c):
            add_img = np.tile(color_array[idx], (image.shape[0], image.shape[1], 1)).transpose((2, 0, 1))*(image==idx)
            color_image = color_image + add_img
    return color_image

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #visdom.server
    opt = option()
    # 加载网络，图片单通道1，分类为1。
    #net = UNet(4, 10)
    #net = Attention_Unet4(4,10)
    #net = UNet2(4,10)
    """
    net = smp.UnetPlusPlus(encoder_name=backbone, 
                           encoder_weights='imagenet', 
                           decoder_attention_type='scse', 
                           in_channels=4, 
                           classes=10,
                           activation=opt.last_activation)
                           
    net = smp.UnetPlusPlus(encoder_name=opt.backbone, 
                           encoder_weights='imagenet', 
                           decoder_attention_type='scse', 
                           in_channels=4, 
                           classes=10)"""
    # 将网络拷贝到deivce中
    #net.to(device=device)
    #net.to(device=opt.device)
    
    # 指定训练集地址，开始训练
    #data_path = "data/train/"
    #train_net(net, device, data_path)
    
    #train_net(net, opt)
    main(opt)