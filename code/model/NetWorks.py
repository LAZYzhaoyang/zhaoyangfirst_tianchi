import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .Blocks import *
import numpy as np
import segmentation_models_pytorch as smp

#============================Segmentation Model Pytorch============================#

class SegModel(nn.Module):
    def __init__(self, n_class=10, in_channel=4, 
                 model_name='Unet', encoder='resnet34', 
                 activation='softmax'):
        super(SegModel, self).__init__()
        self.available_model_list = ['Unet', 'Unet++', 'MAnet', 
                                     'Linknet', 'FPN', 'PSPnet', 
                                     'PAN', 'DeepLabV3', 'DeepLabV3+']
        self.available_encoder_list = ['resnet18', 'resnet34', 'resnet50', 
                                       'resnet101', 'resnet152', 'resnext50_32x4d',
                                       'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d',
                                       'resnext101_32x32d', 'resnext101_32x48d', 'timm-resnest14d',
                                       'timm-resnest26d', 'timm-resnest50d',  'timm-resnest101e',
                                       'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d',
                                       'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s',
                                       'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s',
                                       'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 
                                       'timm-regnetx_004', 'timm-regnetx_006', 'timm-regnetx_008', 
                                       'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040',
                                       'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120',
                                       'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002',
                                       'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008',
                                       'timm-regnety_016', 'timm-regnety_032', 'timm-regnety_040', 
                                       'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120',
                                       'timm-regnety_160', 'timm-regnety_320','senet154',
                                       'se_resnet50', 'se_resnet101', 'se_resnet152',
                                       'se_resnext50_32x4d', 'se_resnext101_32x4d', 'timm-skresnet18',
                                       'timm-skresnet34', 'timm-skresnet50_32x4d', 'densenet121',
                                       'densenet169', 'densenet201', 'densenet161',
                                       'inceptionresnetv2', 'inceptionv4', 'xception',
                                       'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
                                       'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
                                       'efficientnet-b6', 'efficientnet-b7', 'timm-efficientnet-b0',
                                       'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3',
                                       'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6',
                                       'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 
                                       'timm-efficientnet-lite0', 'timm-efficientnet-lite1', 'timm-efficientnet-lite2',
                                       'timm-efficientnet-lite3', 'timm-efficientnet-lite4', 'mobilenet_v2',
                                       'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn',
                                       'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
        self.available_activation_list = ['sigmoid', 'softmax', 'logsoftmax', 'tanh', 'identity', None]
        
        self.n_class=n_class
        self.in_channel =in_channel
        
        if model_name not in self.available_model_list:
            raise ValueError('model name should be in the available model list')
        if encoder not in self.available_encoder_list:
            raise ValueError('encoder should be in the available encoder list')
        if activation not in self.available_activation_list:
            raise ValueError('activation should be in the available activation list')
        
        self.model_name = model_name
        self.encoder = encoder
        self.activation = activation
        
        self.initial_model()
        
    def initial_model(self):
        if self.model_name == self.available_model_list[0]:
            self.model = smp.Unet(encoder_name=self.encoder, 
                                  in_channels=self.in_channel,
                                  classes=self.n_class,
                                  activation=self.activation,
                                  decoder_attention_type='scse')
        elif self.model_name == self.available_model_list[1]:
            self.model = smp.UnetPlusPlus(encoder_name=self.encoder,
                                          in_channels=self.in_channel,
                                          classes=self.n_class,
                                          activation=self.activation,
                                          decoder_attention_type='scse')
        elif self.model_name == self.available_model_list[2]:
            self.model = smp.MAnet(encoder_name=self.encoder,
                                   in_channels=self.in_channel,
                                   classes=self.n_class,
                                   activation=self.activation)
        elif self.model_name == self.available_model_list[3]:
            self.model = smp.Linknet(encoder_name=self.encoder,
                                     in_channels=self.in_channel,
                                     classes=self.n_class,
                                     activation=self.activation)
        elif self.model_name == self.available_model_list[4]:
            self.model = smp.FPN(encoder_name=self.encoder,
                                 in_channels=self.in_channel,
                                 classes=self.n_class,
                                 activation=self.activation)
        elif self.model_name == self.available_model_list[5]:
            self.model = smp.PSPNet(encoder_name=self.encoder,
                                    in_channels=self.in_channel,
                                    classes=self.n_class,
                                    activation=self.activation)
        elif self.model_name == self.available_model_list[6]:
            self.model = smp.PAN(encoder_name=self.encoder,
                                 in_channels=self.in_channel,
                                 classes=self.n_class,
                                 activation=self.activation)
        elif self.model_name == self.available_model_list[7]:
            self.model = smp.DeepLabV3(encoder_name=self.encoder,
                                       in_channels=self.in_channel,
                                       classes=self.n_class,
                                       activation=self.activation)
        elif self.model_name == self.available_model_list[8]:
            self.model = smp.DeepLabV3Plus(encoder_name=self.encoder,
                                           in_channels=self.in_channel,
                                           classes=self.n_class,
                                           activation=self.activation)
        else:
            raise ValueError('this is a unavailable model name, please check it.')
        
    def forward(self, x):
        out = self.model(x)
        return out


def create_binary_models(model_name_list, backbones):
    assert len(backbones)==len(model_name_list)
    model_list = []
    for i in range(len(model_name_list)):
        model_list.append(SegModel(n_class=1, model_name=model_name_list[i], encoder=backbones[i], activation='sigmoid'))
        
    return model_list

