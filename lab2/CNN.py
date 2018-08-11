# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 19:08:46 2018

@author: Gavin
"""

import torch
import torch.nn as nn


shortcut = []
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample = 0, skip_kernal = 0):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_down = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(inplanes)
        self.bn_skip = nn.BatchNorm2d(skip_kernal)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.skip = nn.Conv2d(inplanes, skip_kernal, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_skip = nn.Conv2d(inplanes+skip_kernal, inplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample1 = nn.Upsample(scale_factor=1, mode='bilinear')
        self.downsample = downsample
        self.skip_kernal = skip_kernal
    def encode(self, x):
        x = self.conv(x)
        x = self.conv_down(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        shortcut.append(x)
        return x
    def decode(self, x):
        if self.skip_kernal != 0 :
            res = self.relu(self.bn_skip(self.skip(shortcut[self.downsample-1])))
            x = torch.cat((x, res), 1)
            x = self.upsample1(x)
            x = self.conv_skip(x)
        else:
            x = self.upsample1(x)
            x = self.conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.upsample(x)
        return x    
    def forward(self, x):
        global shortcut
        if self.downsample == 0:
            x = self.encode(x)
        else :
            x = self.decode(x)

        return x

class CNN(nn.Module):
    def __init__(self, block, in_channel, layers, skips):
        super(CNN, self).__init__()
        self.d_layer0 = self._make_layer(block, in_channel, layers[0], 0, 0)
        self.d_layer1 = self._make_layer(block, layers[0], layers[1], 0, 0)
        self.d_layer2 = self._make_layer(block, layers[1], layers[2], 0, 0)
        self.d_layer3 = self._make_layer(block, layers[2], layers[3], 0, 0)
        self.d_layer4 = self._make_layer(block, layers[3], layers[4], 0, 0)
        self.q_layer0 = self._make_layer(block, layers[0], 3, 1, skips[0])
        self.q_layer1 = self._make_layer(block, layers[1], layers[0], 2, skips[1])
        self.q_layer2 = self._make_layer(block, layers[2], layers[1], 3, skips[2])
        self.q_layer3 = self._make_layer(block, layers[3], layers[2], 4, skips[3])
        self.q_layer4 = self._make_layer(block, layers[4], layers[3], 5, skips[4])
    def _make_layer(self, block, in_planes, out_planes, down, skip):
        layers = block(in_planes, out_planes, stride=1, downsample = down, skip_kernal = skip)
        return nn.Sequential(layers)

    def forward(self, x):
        global shortcut
        x = self.d_layer4(self.d_layer3(self.d_layer2(self.d_layer1(self.d_layer0(x)))))
        x = self.q_layer0(self.q_layer1(self.q_layer2(self.q_layer3(self.q_layer4(x)))))
        shortcut = []
        return x

def task1():
    return CNN(BasicBlock, 3, [8, 16, 32, 64, 128], [0, 0, 0, 4, 4])

def task2():
    return CNN(BasicBlock, 32, [128, 128, 128, 128, 128], [4, 4, 4, 4, 4])

def task3():
    return CNN(BasicBlock, 32, [128, 128, 128, 128, 128], [4, 4, 4, 4, 4])