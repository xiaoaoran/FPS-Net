# encoding=utf-8
""" Parts of the U-Net model """

import torch
import torch.nn as nn
# import torch.nn.functional as F
from common.context_block import ContextBlock

import pdb

class RDBlock(nn.Module):
  def __init__(self, inplanes, outplanes, block_index, layer_num=2, bn_d=0.1):
    super(RDBlock, self).__init__()
    self.layer_num = layer_num
    self.block_index = block_index

    # self.global_conv = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False)
    self.ms_conv = MultiScaleBlock(inplanes, outplanes)

    dim_list = [outplanes]
    for i in range(self.layer_num):
      conv_name = 'downblock'+str(self.block_index)+'_conv%s'%i
      relu_name = 'downblock'+str(self.block_index)+'_relu%s'%i

      in_dim = sum(dim_list)
      conv = nn.Sequential( nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1, bias=False))

      # conv = nn.Sequential(nn.Conv2d(inplanes*(i+1), inplanes*(i+1), kernel_size=3, stride=1, padding=1, bias=False))
      setattr(self, conv_name, conv)
      setattr(self, relu_name, nn.ReLU())

      dim_list.append(in_dim)
      in_dim = sum(dim_list)
      self.local_conv = nn.Sequential(nn.Conv2d(in_dim, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.ReLU())

    # self.local_conv = nn.Sequential(nn.Conv2d(inplanes*4, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
    #                                 nn.ReLU())

    # # GCnet Attention
    # self.gc_att = ContextBlock(inplanes=outplanes, ratio=0.25)

  def forward(self, x):
    # pdb.set_trace()
    # global branch
    # global_out = self.global_conv(x)
    x = self.ms_conv(x)

    global_out = x

    # local branch
    li = [x]
    out = x
    for i in range(self.layer_num):
      conv_name = 'downblock' + str(self.block_index) + '_conv%s' % i
      relu_name = 'downblock' + str(self.block_index) + '_relu%s' % i
      conv = getattr(self, conv_name)
      relu = getattr(self, relu_name)
      out = conv(out)
      out = relu(out)

      li.append(out)
      out = torch.cat(li, 1)

    local_out = self.local_conv(out)

    # residual learning
    out = global_out + local_out
    return out

class MultiScaleBlock(nn.Module):
    """multi-scale,插入不同位置"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ms1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.ms2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.ms3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.ms4 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)

        self.activate = nn.Sequential(nn.BatchNorm2d(out_channels*4),
                                      nn.LeakyReLU())

        self.conv = nn.Sequential(nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, padding=0),
                                  nn.BatchNorm2d(out_channels),
                                  nn.LeakyReLU())

    def forward(self, x):
        x1 = self.ms1(x)
        x2 = self.ms2(x)
        x3 = self.ms3(x)
        x4 = self.ms4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        # x = x1 + x2 + x3 + x4

        x = self.activate(x)

        x = self.conv(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, block_index, bn_d=0.01, dropout=False, down_W=True):
        super().__init__()
        if down_W:
            stride= 2
        else:
            stride = [1, 2]
        if not dropout:
            self.rdb = RDBlock(inplanes=in_channels, outplanes=in_channels, block_index=block_index)
            self.down = nn.Sequential(
                # nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=stride, dilation=1, padding=1, bias=False),  # decrease shape, increase dim
                nn.BatchNorm2d(out_channels, momentum=bn_d),
                nn.ReLU())
        else:
            self.rdb = RDBlock(inplanes=in_channels, outplanes=in_channels, block_index=block_index)
            self.down = nn.Sequential(
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),

                # nn.Conv2d(in_channels, out_channels,
                #           kernel_size=3, stride=stride, dilation=1, padding=1, bias=False),  # decrease shape, increase dim
                nn.BatchNorm2d(out_channels, momentum=bn_d),
                nn.ReLU(),
                nn.Dropout2d(p=0.2)
            )

    def forward(self, x):
        skip = self.rdb(x)
        # pdb.set_trace()
        down_x = self.down(skip)
        return down_x, skip


class Backbone(nn.Module):
    def __init__(self, params):
        super(Backbone, self).__init__()
        self.use_range = params["input_depth"]["range"]
        self.use_xyz = params["input_depth"]["xyz"]
        self.use_remission = params["input_depth"]["remission"]
        self.drop_prob = params["dropout"]
        self.bn_d = params["bn_d"]
        self.OS = params["OS"]
        self.layers = params["extra"]["layers"]
        print("Using distvary_ms_rdb_unet" + str(self.layers) + " Backbone")

        self.input_depth = 0
        self.input_idxs = []
        if self.use_range:
            self.input_depth += 1
            self.input_idxs.append(0)
        if self.use_xyz:
            self.input_depth += 3
            self.input_idxs.extend([1, 2, 3])
        if self.use_remission:
            self.input_depth += 1
            self.input_idxs.append(4)

        dim = 32
        # self.inc = DoubleConv(self.input_depth, dim)
        # self.inc = MultiScaleBlock(self.input_depth, dim)
        self.inc_range = RDBlock(1, dim, block_index='range')
        self.inc_zxy = RDBlock(3, dim, block_index='zxy')
        self.inc_remission = RDBlock(1, dim, block_index='remission')
        self.merge = nn.Sequential(nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0),
                                   nn.BatchNorm2d(dim),
                                   nn.LeakyReLU())

        self.down1 = Down(dim, dim*2, 1, self.bn_d, down_W=True)
        self.down2 = Down(dim*2, dim*4, 2, self.bn_d, dropout=True, down_W=True)
        self.down3 = Down(dim*4, dim*8, 3, self.bn_d, dropout=True, down_W=True)
        self.down4 = Down(dim*8, dim*16, 4, self.bn_d, dropout=True, down_W=True)
        self.mid = RDBlock(dim*16, dim*16, 5)

        # last channels
        self.last_channels = dim * 16

        # GCnet Attention
        # self.gc_att = ContextBlock(inplanes=self.last_channels, ratio=0.25)

    def forward(self, x):
        range = x[:,0,:,:].unsqueeze(1)
        zxy = x[:,1:4,:,:]
        remission = x[:,-1,:,:].unsqueeze(1)

        range = self.inc_range(range)
        zxy = self.inc_zxy(zxy)
        remission = self.inc_remission(remission)
        x = torch.cat((range, zxy, remission), dim=1)
        x = self.merge(x)

        x, x1 = self.down1(x)
        x, x2 = self.down2(x)
        x, x3 = self.down3(x)
        x, x4 = self.down4(x)
        x = self.mid(x)
        # x = self.gc_att(x)
        self.range = range
        self.zxy=zxy
        self.remission=remission
        return x, [x4, x3, x2, x1]

    def get_last_depth(self):
        return self.last_channels

    def get_input_depth(self):
        return self.input_depth