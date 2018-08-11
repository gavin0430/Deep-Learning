# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:20:17 2018

@author: Gavin
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import downsampler as D
import numpy as np
import cv2 
import CNN
use_cuda = torch.cuda.is_available()
#hyperarameter
LR = .1
targets = cv2.imread('LowResolution.png')
weight, height, ch = targets.shape
inputs = torch.FloatTensor(32, 4*weight, 4*height).normal_(0, .1)

targets = np.transpose(targets, (2, 0, 1))
inputs = np.expand_dims(inputs, axis=0)
targets = np.expand_dims(targets, axis=0)
inputs = torch.Tensor(inputs)
targets = torch.Tensor(targets)
inputs  = Variable(inputs)
targets = Variable(targets)
#setting
cnn = CNN.task3()
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
    for i in range(2000):
        inputs = inputs.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        tmp = inputs.data + sigma * torch.randn(inputs.shape).cuda()
        outputs = cnn(Variable(tmp))
        if i % 400 == 0:
	    
            f_name = 'task3_' + str(j) + '.jpg'
            j += 1
            img = outputs[0].data.cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            cv2.imwrite(f_name, np.array(img))
        outputs = Down(outputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        print(i)
        print('Train : Loss: %.3f' % (loss.data[0]))



Down = D.Downsampler(n_planes = 3, factor = 4, kernel_type = 'lanczos', kernel_width=4, support = True, phase=0, preserve_size=True).type(torch.cuda.FloatTensor)


##main
train(inputs,targets)
tmp = cnn(inputs)
    
img = tmp[0].data.cpu().numpy()
img = np.transpose(img, (1, 2, 0))
cv2.imwrite('task3_final.jpg', np.array(img))
    
    

