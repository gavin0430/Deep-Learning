# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 12:43:58 2018

@author: Gavin
"""



import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from logger import Logger
import numpy as np
import cv2 
import CNN

use_cuda = torch.cuda.is_available()
#hyperarameter
LR = .1
targets = cv2.imread('0.jpg')                   #image
weight, height, channels = targets.shape
inputs = torch.FloatTensor(3, weight, height).normal_(0, .1)
#targets= torch.FloatTensor(3, weight, height).normal_(0, 1)   #U(0,1) noise


#np.random.shuffle(targets)            #image shuffled

targets = np.transpose(targets, (2, 0, 1))
inputs = np.expand_dims(inputs, axis=0)
targets = np.expand_dims(targets, axis=0)
inputs = torch.Tensor(inputs)
targets = torch.Tensor(targets)
inputs  = Variable(inputs)
targets = Variable(targets)
#setting
cnn = CNN.task1()
if use_cuda:
    cnn.cuda()
    cnn = torch.nn.DataParallel(cnn, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
loss_function = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=LR)
logger = Logger('./logs')


def train(inputs,targets):


    cnn.train()
    sigma = 1/30
    for i in range(2400):
        inputs = inputs.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        tmp = inputs.data + sigma * torch.randn(inputs.shape).cuda()
        outputs = cnn(Variable(tmp))
        #targets.data = targets.data + sigma * torch.randn(targets.shape).cuda()        #image+noise
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        print(i)
        print('Train : Loss: %.3f' % (loss.data[0]))
        info = {'Train_loss' : loss.data[0]}
        for tag, value in info.items():
            logger.scalar_summary(tag, value, i)



##main
train(inputs,targets)
tmp = cnn(inputs)
    
img = tmp[0].data.cpu().numpy()
img = np.transpose(img, (1, 2, 0))
cv2.imwrite('task1.jpg', np.array(img))
    
