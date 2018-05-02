__author__ = 'github.com/leaxp'

import os
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
import visdom
import time, datetime
import math
torch.backends.cudnn.deterministic = True

from data_load import ReadDataset, Rescale, PlotLabels, ToTensor
from os.path import expanduser
home = expanduser("~")

# set timing stamp temp folder name
timestamp = time.strftime("%d%m%y_%H%M%S", time.localtime())

# build autoencoder network
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=1)
        self.conv4 = nn.Conv2d(8, 1, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(1)
        self.convt1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.convt2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.convt3 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)
                nn.init.normal(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(self.bn1(x)))  # out [16, 104, 104, 1]
        x = self.conv2(x)
        x = self.pool(F.relu(self.bn2(x))) # out [8, 52, 52, 1]
        x = self.conv3(x)
        x = self.pool(F.relu(self.bn2(x))) # out [8, 26, 26, 1]

        x = self.convt1(x)
        x = F.relu(self.bn2(x)) # out [8, 52, 52, 1]
        x = self.convt2(x)
        x = F.relu(self.bn1(x)) # out [16, 104, 104, 1]
        x = self.convt3(x)
        x = F.relu(self.bn2(x)) # out [8, 208, 208, 1]
        x = self.conv4(x)
        x = F.relu(self.bn4(x)) # out [1, 208, 208, 1]

        return x

model = autoencoder().cuda()

def train(epochs, lr=1e-3, batch_size=32, seed=23, kernel_width=5, kernel_fwhm=3, verbose=True, save=True, load_model=False, model_path=None):

    if load_model:
        model.load_state_dict(torch.load(model_path))

    torch.cuda.manual_seed_all(seed)

    trsfm = transforms.Compose([Rescale(8),
                                PlotLabels(100),
                                ToTensor()
                                ])

    train = ReadDataset(csv_file = home + "/data/dataset_cae/train_label.csv",
                        tif_file =  home + "/data/dataset_cae/train_data.tif",
                        transform=trsfm)
    val = ReadDataset(csv_file = home + "/data/dataset_cae/val_label.csv",
                        tif_file =  home + "/data/dataset_cae/val_data.tif",
                        transform=trsfm)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)
    val_iter = iter(val_loader)

    # function to save and close
    def saveit():
        if save:
            os.makedirs(home + "/data/temp/{}".format(timestamp))
            file_path = home + '/data/temp/{}/cae_model_{}.pt'.format(timestamp, timestamp)
            torch.save(model.state_dict(), file_path)
            print('  -------------------------------')
            print("model saved:", file_path)
        else:
            print('  -------------------------------')

    # make a gaussian kernel
    def Gauss(size, fwhm = 3, center=None):
        """ Make a square gaussian kernel.

        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]

        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2) # Gaussian kernel

    kernel = Variable(torch.FloatTensor(Gauss(kernel_width, kernel_fwhm).reshape(1, 1, kernel_width, kernel_width)).cuda())

    # define the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print('  epochs: ', epochs, ' batch_size: ', batch_size, ' lr: ', lr, 'seed: ', torch.cuda.initial_seed())
    print('Training... ')
    print('  |  epoch|  train_loss| n_batch|')

    #  Training the ConvNet auto-encoder
    try:
        for epoch in range(epochs):  # loop over the dataset multiple times
            t = tqdm(train_loader, ncols=80, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
            running_loss = []
            for i, data in enumerate(t):
                # =======inputs/labels=======
                inputs, labels = data['image'], data['positions']
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                # ==========forward==========
                outputs = model(inputs)
                loss = criterion(F.conv2d(outputs, kernel, padding=2), F.conv2d(labels, kernel, padding=2))
                # ==========backward==========
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # ============================

                running_loss.append(loss.data[0])

                # tqdm update
                t.set_description('  |{:5.0f}  |{:12.6f}|{:8.0f}|'.format(epoch+1, np.mean(running_loss), i+1))
                t.refresh()

            # write to visdom
            if verbose:
                if epoch==0:
                    vis=visdom.Visdom()
                    label_win = vis.images(utils.make_grid(labels.cpu().data[:4], padding=5, pad_value=1, normalize=True, scale_each=True), opts=dict(title='label images'))
                    pred_win = vis.images(utils.make_grid(outputs.cpu().data[:4], padding=5, pad_value=1, normalize=True, scale_each=True), opts=dict(title='prediction images'))
                    loss_win = vis.line(X=np.array([epoch+1]), Y= np.array([np.mean(running_loss)]), opts=dict(width=850, xlabel='<b>epochs</b>', ylabel='training loss', markersize=5, markers=True, title="<b> Conv_Autoencoder </b> training loss"))
                else:
                    vis.images(utils.make_grid(labels.cpu().data[:4], padding=5, pad_value=1, normalize=True, scale_each=True), win=label_win, opts=dict(title='label images'))
                    vis.images(utils.make_grid(outputs.cpu().data[:4], padding=5, pad_value=1, normalize=True, scale_each=True), win=pred_win, opts=dict(title='prediction images'))
                    vis.line(X=np.array([epoch+1]), Y= np.array([np.mean(running_loss)]), win=loss_win, update='append', opts=dict(width=850, xlabel='<b>epochs</b>', ylabel='training loss', markersize=5, markers=True, title="<b> Conv_Autoencoder </b> training loss"))
        saveit()
    except KeyboardInterrupt:
        saveit()

if __name__ == "__main__":

    model_path = home + "/data/storage/050418_162056/cae_model_050418_162056.pt"

    train(epochs=100, lr=1e-4, kernel_width=5, kernel_fwhm=3, seed=99, save=False, load_model=True, model_path=model_path)
