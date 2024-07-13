"""
   Pixel-wise matrix addition with x, y and z (Pool5_Pool3)
"""

from symbol import xor_expr
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split



# 自定義的注意力模塊 
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.f_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1) # 1X1 conv
        self.g_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.h_conv = nn.Conv2d(in_channels = 256, out_channels = 169 , kernel_size= 1) # revise channel
        self.u_conv = nn.Conv2d(in_channels = 128, out_channels = 169 , kernel_size= 1) # revise channel
        self.channel_conv = nn.Conv2d(in_channels = 169, out_channels = 512 , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 

    def forward(self,x, features_after_fifth_maxpool, features_after_eighth_maxpool):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        # 獲取輸入的尺寸信息
        m_batchsize, C, width, height = x.size()

        # 使用 f_conv 與 g_conv 計算 self attention 的分數
        proj_f = self.f_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # [32, 169, 64] # (B, C, W*H) 轉換為 (B, W*H, C) # 為了使 f 的每個通道成為 self attention 計算中的一個向量
        proj_g = self.g_conv(x).view(m_batchsize, -1, width * height)  # [32, 64, 169] # B X C X W*H

        # 計算能量（分數）
        energy = torch.bmm(proj_f, proj_g)  # 矩陣乘法操作  # [32, 169, 169]
        attention_1 = self.softmax(energy)  # B X (N) X (N) # [32, 169, 169]

        # 使用 h_conv 計算權重調整後的特徵值
        # 使用 Bilinear 插值將特徵圖的大小調整為新的大小，(new_height, new_width)
        y = features_after_eighth_maxpool # [32, 256, 13, 13]
        proj_h = self.h_conv(y).view(m_batchsize, -1, width * height)  # B X C X N # [32, 256, 169]
        attention_2 = attention_1 + proj_h            
        
        new_size = (13, 13)
        z = F.interpolate(features_after_fifth_maxpool, size=new_size, mode='bilinear', align_corners=False) # [32, 128, 27, 27]
        proj_u = self.u_conv(z).view(m_batchsize, -1, width * height)  # B X C X N # [32, 169, 169]
        out = attention_2 + proj_u                             # [32, 169, 169]

        out = out.view(m_batchsize, 169, width, height) # [32, 169, 13, 13]
        out = self.channel_conv(out).view(m_batchsize, -1, width * height) # [32, 512, 169]
        out = out.view(m_batchsize, 512, width, height)  # [32, 512, 13, 13]
        # 對輸出進行 gamma 調整並加上原始輸入特徵圖
        out = self.gamma * out + x

        return out



# 自定義的帶有注意力模塊的 SqueezeNet
class SqueezeNetWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(SqueezeNetWithAttention, self).__init__()
        self.squeezenet = models.squeezenet1_1(pretrained=True)
        self.attention = Self_Attn(in_dim=512, activation='relu') # squeezenet 最後一層 512
        self.fc = nn.Linear(512, num_classes) # 512 -> 6

    def forward(self, input):
        x = self.squeezenet.features(input)
        # 提取第五個 MaxPooling 層之前的特徵圖
        features_before_fifth_maxpool = self.squeezenet.features[:5](input)
        # 提取第五個 MaxPooling 層之後的特徵圖
        features_after_fifth_maxpool = self.squeezenet.features[5](features_before_fifth_maxpool)
        features_before_eighth_maxpool = self.squeezenet.features[:8](input)        
        # 提取第八個 MaxPooling 層之後的特徵圖
        features_after_eighth_maxpool = self.squeezenet.features[8](features_before_eighth_maxpool)
        x = self.attention(x, features_after_fifth_maxpool, features_after_eighth_maxpool)  # [32, 512, 13, 13]
        x = x.mean([2, 3])  # Global average pooling # [32, 512]
        # 將特徵圖展平成一維向量
        x = self.fc(x)
        return x




