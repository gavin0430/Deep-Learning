# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 17:54:49 2018

@author: Gavin
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2 
import CNN

use_cuda = torch.cuda.is_available()
#hyperarameter
LR = .1
targets = cv2.imread('noise_image.png')
weight, height, ch = targets.shape
inputs = torch.FloatTensor(32, weight, height).normal_(0, .1)

targets = np.transpose(targets, (2, 0, 1))
inputs = np.expand_dims(inputs, axis=0)
targets = np.expand_dims(targets, axis=0)
inputs = torch.Tensor(inputs)
targets = torch.Tensor(targets)
inputs  = Variable(inputs)
targets = Variable(targets)
#setting
cnn = CNN.task2()
if use_cuda:
    cnn.cuda()
    cnn = torch.nn.DataParallel(cnn, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
loss_function = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr=LR)



def train(inputs,targets):


    cnn.train()
    j = 0
    sigma = 1/30
    for i in range(1800):
        inputs = inputs.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        tmp = inputs.data + sigma * torch.randn(inputs.shape).cuda()
        outputs = cnn(Variable(tmp))
        if i % 360 == 0:
            
            f_name = 'task2_' + str(j) + '.jpg'
            j += 1
            img = outputs[0].data.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            cv2.imwrite(f_name, np.array(img))
            
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
    
        print(i)
        print('Train : Loss: %.3f' % (loss.data[0]))
    



##main
train(inputs,targets)
tmp = cnn(inputs)
    
img = tmp[0].data.cpu().numpy()
img = np.transpose(img, (1, 2, 0))
cv2.imwrite('task2_final.jpg', np.array(img))
    
