__autor__ = 'github.com/leaxp'

import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
import matplotlib.pyplot as plt
# import pandas as pd
from skimage import exposure
from skimage.io import imread
from skimage.transform import resize
import time, datetime

from data_load import ReadDataset, Rescale, PlotLabels, ToTensor
from os.path import expanduser
home = expanduser("~")

model_path = "model/autoencoder_model.pt"

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
model.load_state_dict(torch.load(model_path))

# %% scale bar to matplotlib
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
fontprops = fm.FontProperties(size=12)

# %% Image Rescale
def img_rescale(img, p1=0, p2=99.9):
    p1_value, p2_value = np.percentile(img, (p1, p2))
    img_rescaled = exposure.rescale_intensity(img, in_range=(p1_value, p2_value))
    return img_rescaled

# %% load tubes image

file_path = "sequence/"

# Parameters
# red square
red_x = 30
red_y = 2
red_sqr = 26
# green square
green_x = red_x + (20/12.5)
green_y = red_y
green_sqr = 15

# %% decoded all frames

def decode_img(img_zoom):
    img_zoom_scale = resize(img_zoom, (red_sqr*8, red_sqr*8), preserve_range=True, mode='constant')
    img_zoom_scale = (img_zoom_scale - img_zoom_scale.mean())/img_zoom_scale.std() # Normalization
    img_zoom_scale = Variable(torch.FloatTensor(img_zoom_scale).cuda().view(1, 1, 208, 208))
    decoded_image = model(img_zoom_scale)
    return decoded_image.data.view(208,208)

start_time = time.clock()

image = np.zeros((208,208))

for frame in range(360):
    img = imread(file_path + "{0:05d}.tif".format(frame+1))[red_y:(red_y+red_sqr),red_x:(red_x+red_sqr)]
    image += decode_img(img)

# elapsed time
print(time.clock() - start_time, "seconds")

# %% plot construction

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.set_title("Localisation image reconstruction ")
ax1.imshow(img_rescale(image, 0, 99), cmap="gray", zorder=1)
ax1_scalebar = AnchoredSizeBar(ax1.transData, 40, '500nm', 'lower right', color='white', pad=0.5, frameon=False, size_vertical=1.2, fontproperties=fontprops)
ax1.add_artist(ax1_scalebar)
ax1.set_xlim(0, 207)
ax1.set_ylim(207, 8)

ax2.set_title("Region Zoom")
ax2.imshow(img_rescale(image, 0, 99), cmap="gray", zorder=1)
ax2_scalebar = AnchoredSizeBar(ax2.transData, 40, '500nm', 'lower right', color='white', pad=0.5, frameon=False, size_vertical=1.2, fontproperties=fontprops)
ax2.add_artist(ax2_scalebar)
ax2.set_xlim(50, 150)
ax2.set_ylim(120, 20)
plt.show()

if __name__ == "__main__":

    plt.savefig("localization.png",  bbox_inches='tight')
