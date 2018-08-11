from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
from logger import Logger
logger = Logger('./logs_cvae')
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

device = torch.cuda.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(784, 400), nn.ReLU())
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Sequential(nn.Linear(30, 392), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(11, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
	    nn.ReLU(inplace=True),
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),
	    nn.BatchNorm2d(1),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(2, 11, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(11),
	    nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(11, 3, kernel_size=3, stride=1, padding=1),
	    nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())

    def encode(self, x, c):
        x = x.view(self.batch_size, 1, 28, 28)
        onehot = torch.zeros([self.batch_size, 10, 784])
        for i in range(self.batch_size):
			onehot[i][c[i].data[0]] = 1
        onehot = onehot.view(self.batch_size, 10, 28, 28).cuda()
        onehot = Variable(onehot)
	x = torch.cat((x, onehot), 1)
        #x = Variable(x)
        x = self.conv1(x)
        #print(x.shape)
        h1 = self.fc1(x.view(-1, 784))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
	#print(type(std))
	#print(type(mu))
        eps = torch.randn(std.size())
	eps = Variable(eps).cpu()
	std = std.cpu()
	mu = mu.cpu()
	#print(type(eps))
        return eps.mul(std).add_(mu).cuda()

    def decode(self, z, c):
        self.batch_size = c.shape[0]
        onehot = torch.zeros([self.batch_size, 10])
        for i in range(self.batch_size):
			onehot[i][c[i].data[0]] = 1
        onehot = Variable(onehot)
        z = torch.cat((z, onehot.cuda()), 1)
        h3 = self.fc3(z)
        h3 = h3.view(self.batch_size, 2, 14, 14).cuda()
        h3 = self.conv2(h3)
        return h3.view(-1, 784)

    def forward(self, x, c):
        self.batch_size = c.shape[0]
        mu, logvar = self.encode(x.view(-1, 784), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


#model = VAE().to(device)
model = VAE().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = 0.5 * torch.sum(mu.pow(2)+ logvar.pow(2) - torch.log(logvar.pow(2)) - 1)
    #KLD = -0.5 * torch.sum(1 + torch.log(logvar.pow(2)) - mu.pow(2) - logvar.pow(2))

    return BCE + KLD

best_acc = 0.
def train(epoch):
    global best_acc
    model.train()
    train_loss = 1000
    for batch_idx, (data, label) in enumerate(train_loader):
        #data = data.to(device)
        label, data = label.cuda(), data.cuda()
        label, data = Variable(label), Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, label)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        #train_loss += loss.item()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                #loss.item() / len(data)))
                loss.data[0] / len(data)))
            info = {'Train_loss' : loss.data[0] / len(data)}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch)
            if best_acc > 100. * loss.data[0] / len(data):
                best_acc = 100. * loss.data[0] / len(data)
                torch.save(model, './model_cvae')

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0

    for i, (data, label) in enumerate(test_loader):
        #data = data.to(device)
        label, data = label.cuda(), data.cuda()
        label, data = Variable(label, requires_grad=False), Variable(data, requires_grad=False)
        recon_batch, mu, logvar = model(data, label)
        #test_loss += loss_function(recon_batch, data, mu, logvar).item()
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 10)
            comparison = torch.cat([data[:n],recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
	    comparison = comparison.cpu().data
            save_image(comparison,'results_cvae/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    sample = torch.randn(100, 20).cuda()
    sample = Variable(sample)
    vec = [0,1,2,3,4,5,6,7,8,9] * 10
    onehot = torch.LongTensor(vec).cuda()
    onehot = Variable(onehot)
    #decode = model.decode(sample, torch.tensor(finish, ).cuda())
    sample = model.decode(sample, onehot).cpu()

    #sample = decode().cpu()
    sample = sample.data
    save_image(sample.view(100,1,28,28),'results_cvae/sample_' + str(epoch) + '.png', nrow=10)
