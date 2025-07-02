"""
   Pixel-wise matrix addition with x, z and y (Pool3_Pool5) * final version *
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


# Multi-Scale Attention Mechanism
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.f_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1) # 1X1 conv
        self.g_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.h_conv = nn.Conv2d(in_channels = 128, out_channels = 169 , kernel_size= 1) # revise channel # fifth_maxpool [32, 128, 27, 27]
        self.u_conv = nn.Conv2d(in_channels = 256, out_channels = 169 , kernel_size= 1) # revise channel # eighth_maxpool [32, 256, 13, 13]
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
        # Get the input dimensions
        m_batchsize, C, width, height = x.size()

        # Calculate self-attention scores using f_conv and g_conv
        proj_f = self.f_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # Convert (B, C, W*H) to (B, W*H, C) 
        proj_g = self.g_conv(x).view(m_batchsize, -1, width * height)  # B X C X W*H

        # Calculate similarity score
        energy = torch.bmm(proj_f, proj_g)  # Matrix multiplication operation
        attention_1 = self.softmax(energy)  # B X (N) X (N)
        
        # Use h_conv to calculate the weighted adjusted feature values
        # Use Bilinear interpolation to adjust the size of the feature map to the new size (new_height, new_width)
        new_size = (13, 13)
        y = F.interpolate(features_after_fifth_maxpool, size=new_size, mode='bilinear', align_corners=False)
        proj_h = self.h_conv(y).view(m_batchsize, -1, width * height)  # B X C X N 
        attention_2 = attention_1 + proj_h                             

        z = features_after_eighth_maxpool
        proj_u = self.u_conv(z).view(m_batchsize, -1, width * height)  # B X C X N 
        out = attention_2 + proj_u                                    

        out = out.view(m_batchsize, 169, width, height) 
        out = self.channel_conv(out).view(m_batchsize, -1, width * height) 
        out = out.view(m_batchsize, 512, width, height) 
        # Apply gamma adjustment to the output and add the original input feature map
        out = self.gamma * out + x
        return out


# Multi-Scale Attention Mechanism Network
class SqueezeNetWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(SqueezeNetWithAttention, self).__init__()
        self.squeezenet = models.squeezenet1_1(pretrained=True)
        self.attention = Self_Attn(in_dim=512, activation='relu') # The last layer of SqueezeNet has 512 channels
        self.fc = nn.Linear(512, num_classes) # 512 -> 6

    def forward(self, input):
        x = self.squeezenet.features(input)
        # Extract the feature map before the fifth MaxPooling layer
        features_before_fifth_maxpool = self.squeezenet.features[:5](input)
        # Extract the feature map after the fifth MaxPooling layer
        features_after_fifth_maxpool = self.squeezenet.features[5](features_before_fifth_maxpool)
        # Extract the feature map before the eighth MaxPooling layer
        features_before_eighth_maxpool = self.squeezenet.features[:8](input)        
        # Extract the feature map after the eighth MaxPooling layer
        features_after_eighth_maxpool = self.squeezenet.features[8](features_before_eighth_maxpool)
        x = self.attention(x, features_after_fifth_maxpool, features_after_eighth_maxpool) 
        x = x.mean([2, 3])  # Global average pooling # [32, 512]
        x = self.fc(x)      # Flatten the feature map into a 1D vector
        return x


