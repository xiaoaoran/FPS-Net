#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
from tasks.semantic.postproc.CRF import CRF
import __init__ as booger
import pdb


class Segmentator(nn.Module):
  def __init__(self, ARCH, nclasses, path=None, path_append="", strict=False, log_dir=None):
    super().__init__()
    self.ARCH = ARCH
    self.nclasses = nclasses
    self.path = path
    self.path_append = path_append
    self.strict = False

    # get the model, depends on training or testing
    if log_dir is None:  # training
      bboneModule = imp.load_source("bboneModule",
                                    booger.TRAIN_PATH + '/backbones/' +
                                    self.ARCH["backbone"]["name"] + '.py')
      self.backbone = bboneModule.Backbone(params=self.ARCH["backbone"])
    else:  # testing
      bboneModule = imp.load_source("bboneModule", log_dir[:-5] + 'encoder_' + self.ARCH["backbone"]["name"] + '.py')
      self.backbone = bboneModule.Backbone(params=self.ARCH["backbone"])
      print("Test Phase: enoder loaded from ", log_dir + '/encoder_' + self.ARCH["backbone"]["name"] + '.py')

    # do a pass of the backbone to initialize the skip connections
    stub = torch.zeros((1, 5, self.ARCH["dataset"]["sensor"]["img_prop"]["height"],  # self.backbone.get_input_depth() -> 5
                        self.ARCH["dataset"]["sensor"]["img_prop"]["width"]))

    if torch.cuda.is_available():
      stub = stub.cuda()
      self.backbone.cuda()
    # pdb.set_trace()
    stub = torch.cat((stub, stub), 0)
    _, stub_skips = self.backbone(stub)

    if log_dir is None:
      decoderModule = imp.load_source("decoderModule",
                                      booger.TRAIN_PATH + '/tasks/semantic/decoders/' +
                                      self.ARCH["decoder"]["name"] + '.py')
      self.decoder = decoderModule.Decoder(params=self.ARCH["decoder"],
                                           stub_skips=stub_skips,
                                           OS=self.ARCH["backbone"]["OS"],
                                           feature_depth=self.backbone.get_last_depth())
    else:  # testing
      decoderModule = imp.load_source("decoderModule", log_dir[:-5] + 'decoder_' + self.ARCH["decoder"]["name"] + '.py')
      self.decoder = decoderModule.Decoder(params=self.ARCH["decoder"],
                                           stub_skips=stub_skips,
                                           OS=self.ARCH["backbone"]["OS"],
                                           feature_depth=self.backbone.get_last_depth())
      print("Test Phase: deoder loaded from ", log_dir + '/decoder_' + self.ARCH["decoder"]["name"] + '.py')

    if  'deeplab' in self.ARCH["backbone"]["name"]:
      self.head = nn.Sequential()
    else:
      self.head = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                                  nn.Conv2d(self.decoder.get_last_depth(),
                                            self.nclasses, kernel_size=3,
                                            stride=1, padding=1))

    if self.ARCH["post"]["CRF"]["use"]:
      self.CRF = CRF(self.ARCH["post"]["CRF"]["params"], self.nclasses)
    else:
      self.CRF = None

    # train backbone?
    if not self.ARCH["backbone"]["train"]:
      for w in self.backbone.parameters():
        w.requires_grad = False

    # train decoder?
    if not self.ARCH["decoder"]["train"]:
      for w in self.decoder.parameters():
        w.requires_grad = False

    # train head?
    if not self.ARCH["head"]["train"]:
      for w in self.head.parameters():
        w.requires_grad = False

    # train CRF?
    if self.CRF and not self.ARCH["post"]["CRF"]["train"]:
      for w in self.CRF.parameters():
        w.requires_grad = False

    # print number of parameters and the ones requiring gradients
    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel() for p in self.parameters())
    weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # breakdown by layer
    weights_enc = sum(p.numel() for p in self.backbone.parameters())
    weights_dec = sum(p.numel() for p in self.decoder.parameters())
    weights_head = sum(p.numel() for p in self.head.parameters())
    print("Param encoder ", weights_enc)
    print("Param decoder ", weights_dec)
    print("Param head ", weights_head)
    if self.CRF:
      weights_crf = sum(p.numel() for p in self.CRF.parameters())
      print("Param CRF ", weights_crf)

    # get weights
    if path is not None:
      # try backbone
      try:
        w_dict = torch.load(path + "/backbone" + path_append,
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model backbone weights")
      except Exception as e:
        print()
        print("Couldn't load backbone, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try decoder
      try:
        w_dict = torch.load(path + "/segmentation_decoder" + path_append,
                            map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model decoder weights")
      except Exception as e:
        print("Couldn't load decoder, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try head
      try:
        w_dict = torch.load(path + "/segmentation_head" + path_append,
                            map_location=lambda storage, loc: storage)
        self.head.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try CRF
      if self.CRF:
        try:
          w_dict = torch.load(path + "/segmentation_CRF" + path_append,
                              map_location=lambda storage, loc: storage)
          self.CRF.load_state_dict(w_dict, strict=True)
          print("Successfully loaded model CRF weights")
        except Exception as e:
          print("Couldn't load CRF, using random weights. Error: ", e)
          if strict:
            print("I'm in strict mode and failure to load weights blows me up :)")
            raise e
    else:
      print("No path to pretrained, using random init.")

  def forward(self, x, mask=None):
    # pdb.set_trace()
    y, skips = self.backbone(x)  # x: [16, 5, 64, 1024]  y:[16, 1024, 64, 32] || skip存着每次变小前的feature map
    y = self.decoder(y, skips)  # y:[16, 1024, 64, 32]
    y = self.head(y)

    # if self.ARCH["backbone"]["name"] != 'deeplabv3':
    y = F.softmax(y, dim=1)


    if self.CRF:
      assert(mask is not None)
      y = self.CRF(x, y, mask)

    return y
    # range, zxy, remission = self.backbone.range, self.backbone.zxy, self.backbone.remission
    # return y, range, zxy, remission

  def save_checkpoint(self, logdir, suffix=""):
    # Save the weights
    torch.save(self.backbone.state_dict(), logdir +
               "/backbone" + suffix)
    torch.save(self.decoder.state_dict(), logdir +
               "/segmentation_decoder" + suffix)
    torch.save(self.head.state_dict(), logdir +
               "/segmentation_head" + suffix)
    if self.CRF:
      torch.save(self.CRF.state_dict(), logdir +
                 "/segmentation_CRF" + suffix)
