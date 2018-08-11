
# coding: utf-8

# In[1]:


import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.utils.data as torch_data
import pickle
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter


# In[2]:


height, width =320, 160
batch_size = 8
latent_dim = 10
channels = 3
n_residual_blocks = 6
lr = 0.0002


# In[3]:


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
    
        self.fc = nn.Linear(latent_dim, channels*(height)*(width))
    
        self.l1 = nn.Sequential(nn.Conv2d(channels*2, 64, 3, 1, 1), nn.ReLU(inplace=True))
    
        resblocks = []
        for _ in range(n_residual_blocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)
        self.tanh = nn.Tanh()
        self.l2 = nn.Sequential(nn.Conv2d(64, channels, 3, 1, 1))#, nn.Tanh())


    def forward(self, img, z):
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)
    
        return self.tanh(img_)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    
        self.model = nn.Sequential(
        nn.Conv2d(channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        validity = self.model(img)
    
        return validity

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 36, 5, stride=2, padding=1),
            nn.ELU(0.2, inplace=True),
            nn.Conv2d(36, 48, 3, stride=2, padding=1),
            nn.ELU(0.2, inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.ELU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ELU(0.2, inplace=True),
            nn.Dropout(p=0.5)			
        )
        input_size = (height) * (width) // 2**4
        self.output_layer = nn.Sequential(
            nn.Linear(12800,100),
            nn.ELU(0.2, inplace=True),
            nn.Linear(100,50),
            nn.ELU(0.2, inplace=True),
            nn.Linear(50,10),
            nn.ELU(0.2, inplace=True),
            nn.Linear(10,1),
        )
    def forward(self, img):
            feature_repr = self.model(img)
            feature_repr = feature_repr.view(feature_repr.size(0), -1)
            label = self.output_layer(feature_repr)
            return label


# In[4]:


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
classifier = Classifier()
# Optimizers
optimizer_G = torch.optim.Adam( itertools.chain(generator.parameters(), classifier.parameters()),	#task
                                lr=lr, betas=(0.5, 0.99))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.99))


# In[5]:


class car_data(torch_data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        with open(root, 'rb') as fp:
            contents = pickle.load(fp)
            if train==True:
                self.data = contents['train']['imgs']
                self.label = contents['train']['steers']
            else:
                self.data = contents['test']['imgs']
                self.label = contents['test']['steers']
        
        self.shape = self.data.shape
        
    def __getitem__(self, index):
        img = self.data[index]
        label = self.label[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, label
    
    def __len__(self):
        return self.data.shape[0]

with open('../data/cardata/mountain.pkl', 'rb') as fp:
     mountain= pickle.load(fp)
with open('../data/cardata/sand.pkl', 'rb') as fp:
     sand= pickle.load(fp)
T = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
mountain_dset = car_data('../data/cardata/mountain.pkl', transform=T)
T = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
sand_dset = car_data('../data/cardata/sand.pkl', transform=T)


# In[6]:


source_loader = torch_data.DataLoader(mountain_dset, batch_size=batch_size, shuffle=True)
target_loader = torch_data.DataLoader(sand_dset, batch_size=batch_size, shuffle=True)


# In[7]:


def main():
    # Initialize weights

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)		
    classifier.apply(weights_init_normal)

    #generator = torch.load('./original/final_G')
    #discriminator = torch.load('./original/final_D')
    #classifier = torch.load('./original/final_T')
    # Loss weights
    lambda_adv =  1
    lambda_task = 0.1	#change

    generator.cuda()
    discriminator.cuda()
    classifier.cuda()
    writer = SummaryWriter()
    #adversarial_loss.cuda()
    #task_loss.cuda()
    FloatTensor = torch.cuda.FloatTensor
    valid = Variable(FloatTensor(batch_size, 1, 10, 20).fill_(1.0), requires_grad=False)
    fake = Variable(FloatTensor(batch_size, 1, 10, 20).fill_(0.0), requires_grad=False)
    '''
    training
    '''
    save_point = 0
    i = 0
    _g = 0.
    _d = 0.
    _t = 0.
    while True:
        source_data, source_label = source_loader.__iter__().next()
        target_data, target_label = target_loader.__iter__().next()
        source_data = Variable(source_data).cuda()
        source_label = Variable(source_label).cuda()
        target_data = Variable(target_data).cuda()
        source_label = (source_label).type(torch.cuda.FloatTensor)
    
        optimizer_G.zero_grad()
        z = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, latent_dim))))
        fake_img = generator(source_data, z)
        label_pred_G = classifier(fake_img)
        label_pred_G = label_pred_G.view(batch_size)
        label_pred_S = classifier(source_data)
        label_pred_S = label_pred_S.view(batch_size)
        task_loss_ =    (F.mse_loss(label_pred_G, source_label) + F.mse_loss(label_pred_S, source_label)) / 2

        #task_loss_ = 0.3 * (F.mse_loss(fake_B, imgs_B) + 0.7 * F.mse_loss(fake_B, imgs_A) )
        g_loss =    lambda_adv * F.binary_cross_entropy(discriminator(fake_img), valid) + lambda_task * task_loss_
        g_loss.backward()
        optimizer_G.step()
        optimizer_D.zero_grad()
        real_loss = F.binary_cross_entropy(discriminator(target_data), valid)
        fake_loss = F.binary_cross_entropy(discriminator(fake_img.detach()), fake)
        d_loss = (real_loss + fake_loss)/2
        d_loss.backward()
        optimizer_D.step()

        _g = _g + g_loss.data[0]
        _d = _d + d_loss.data[0]
        _t = _t + task_loss_.data[0]
        
        if task_loss_.data >10 :
            os_exit(0)
        if i%100 == 0:
            writer.add_scalars('G_loss', {'loss': _g/100.}, i)
            writer.add_scalars('D_loss', {'loss': _d/100.}, i)
            writer.add_scalars('T_loss', {'loss': _t/100.}, i)
            _g = 0.
            _d = 0.
            _t = 0.
            print("[Step : %d]"% (i))
            print("G : ", g_loss.data)
            print("D : ", d_loss.data)
            print("T : ", task_loss_.data)
            sample = fake_img[0].detach()
            save_image(sample, 'images%d.png' % ((i/100)%10), normalize=True)
            torch.save(generator, './G')
            torch.save(classifier, './T')
            torch.save(discriminator, './D')
        i += 1

    '''
    training
    '''


# In[ ]:


if __name__ == '__main__':
    main()

