import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class RCNNBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channel),
                                   nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channel))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channel))
        self.conv4 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channel))
        self.conv5 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channel))


    def forward(self, x):
        x1 = self.conv1(x)
        x_n = self.conv2(x1)
        x_n = x1 + x_n
        x_n = self.conv3(x_n)
        x_n = x1 + x_n
        x_n = self.conv4(x_n)
        x_n = x1 + x_n
        x_n = self.conv5(x_n)
        x_n = x1 + x_n
        return x_n


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False, dropout=False, up_W=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if up_W:
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            else:
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=[1, 4], stride=[1, 2], padding=[0, 1])
        self.conv = RCNNBlock(out_channels*2, out_channels)
        self.dropout = nn.Dropout2d(p=0.2) if dropout else None

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self, bilinear=False):
        super(Decoder, self).__init__()

        self.bilinear = bilinear

        dim=32
        self.up1 = Up(dim*16, dim*8, bilinear, dropout=True, up_W=True)
        self.up2 = Up(dim*8, dim*4, bilinear, dropout=True, up_W=True)
        self.up3 = Up(dim*4, dim*2, bilinear, dropout=True, up_W=True)
        self.up4 = Up(dim*2, dim, bilinear, up_W=True)

        # last channels
        self.last_channels = dim

        print("Using RCNN decoder")

    def forward(self, x5, skips):
        [x4, x3, x2, x1] = skips
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth
