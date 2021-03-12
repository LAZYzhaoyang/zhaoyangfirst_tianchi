import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch import einsum

#============================Basic Block============================#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding_mode='same', relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        if padding_mode == 'same':
            padding = (kernel_size-1) // 2
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super(ConvPool, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.padding = 0
        self.pool = BasicConv(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, stride=kernel_size, padding_mode='zero', relu=False, bn=False, bias=False)
        
    def forward(self, x):
        x = self.pool(x)
        return x

#============================Net Block============================#
class Spatial_Attention_Module(nn.Module):
    def __init__(self, channel):
        super(Spatial_Attention_Module, self).__init__()
        self.spatial_conv = BasicConv(in_channels=channel, out_channels=channel, kernel_size=1, relu=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        value = x
        attn = self.spatial_conv(x)
        scale = self.sigmoid(attn)
        out = value * scale
        
        return out

class Channel_Attention_Module(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super(Channel_Attention_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c) # squeeze操作
        y1 = self.fc1(y1).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        y2 = self.max_pool(x).view(b, c)
        y2 = self.fc2(y2).view(b, c, 1, 1)
        y = y1 + y2
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return  out 
    
class Non_Local_Module(nn.Module):
    def __init__(self, channel):
        super(Non_Local_Module, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(channel, self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上进行的
        x_phi = self.conv_phi(x).view(b, c, -1)
        # 获取theta特征，维度为[N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # 获取g特征，维度为[N, H * W, C/2]
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # 对phi和theta进行矩阵乘，[N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        # softmax拉到0~1之间
        mul_theta_phi = self.softmax(mul_theta_phi)
        # 与g特征进行矩阵乘运算，[N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # 1X1卷积扩充通道数
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x # 残差连接
        return out
    
class Residual_Block(nn.Module):
    def __init__(self, channel):
        super(Residual_Block, self).__init__()
        self.convblock1 = BasicConv(in_channels=channel, out_channels=channel, bias=True)
        self.convblock2 = BasicConv(in_channels=channel,out_channels=channel, relu=False, bias=True)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        out = self.convblock1(x)
        out = self.convblock2(out)
        out = out + residual
        out = self.relu(out)
        
        return out
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, channels, scale_factor=2, bilinear=False):
        super(Up, self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = BasicConv(in_channels=channels, out_channels=channels//scale_factor, bias=False, bn=False)
        else:
            self.up = nn.ConvTranspose2d(channels, channels // scale_factor, kernel_size=scale_factor, stride=scale_factor)
            
    def forward(self, x):

        out = self.up(x)
        if self.bilinear:
            out = self.conv(out)
        return out

class SegHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(SegHead, self).__init__()
        self.conv = BasicConv(in_channels=in_channels, out_channels=n_classes,relu=False, bn=False, bias=False)
        self.activation = nn.Softmax2d()

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)

        return out


#============================Networks Block============================#
# backbone

class resnet_down(nn.Module):
    def __init__(self, in_channel, out_channel, down_factor=2):
        super(resnet_down, self).__init__()
        self.resnet = Residual_Block(channel=out_channel)
        self.down = ConvPool(in_channels=in_channel, out_channels=out_channel, kernel_size=down_factor)

    def forward(self, x):
        out = self.down(x)
        out = self.resnet(out)

        return out


class resnet_up(nn.Module):
    def __init__(self, in_channel, up_factor=2, bilinear=True):
        super(resnet_up, self).__init__()
        self.up = Up(channels=in_channel, scale_factor=up_factor, bilinear=bilinear)
        self.resnet = Residual_Block(channel= in_channel//up_factor)

    def forward(self, x):
        out = self.up(x)
        out = self.resnet(out)
        return out

class CBAM_Block(nn.Module):
    def __init__(self, in_channel, is_parallel=False, mode='channel_first'):
        super(CBAM_Block, self).__init__()
        self.spatial_att = Spatial_Attention_Module(channel=in_channel)
        self.channel_att = Channel_Attention_Module(in_channel=in_channel)
        self.is_parallel = is_parallel
        self.mode = mode

    def forward(self, x):
        if self.is_parallel:
            spa_att = self.spatial_att(x)
            channel_att = self.channel_att(x)
            out = torch.cat([spa_att, channel_att], dim=1)
        else:
            if self.mode == 'channel_first':
                att = self.channel_att(x)
                out = self.spatial_att(att)
            else:
                att = self.spatial_att(x)
                out = self.channel_att(att)
        return out

#============================Visual Transformer Block============================#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask = None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)




