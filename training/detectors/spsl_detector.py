

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import random

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='spsl')
class SpslDetector(AbstractDetector):
    def __init__(self):
        super().__init__()
   
        self.backbone = self.build_backbone()
        self.loss_func = self.build_loss()


    def build_backbone(self):
        # prepare the backbone
        backbone_class = BACKBONE['xception']
        
        model_config ={'mode': 'original',
                                   'num_classes': 1, 'inc': 4, 'dropout': False}
        backbone = backbone_class(model_config)

        # To get a good performance, use the ImageNet-pretrained Xception model
        state_dict = torch.load('./pretrained/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}

        # remove conv1 from state_dict
        conv1_data = state_dict.pop('conv1.weight')

        backbone.load_state_dict(state_dict, False)
        print('Load pretrained model successfully!')

        # copy on conv1
        # let new conv1 use old param to balance the network
        backbone.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
        avg_conv1_data = conv1_data.mean(dim=1, keepdim=True)  # average across the RGB channels
        backbone.conv1.weight.data = avg_conv1_data.repeat(1, 4, 1, 1)  # repeat the averaged weights across the 4 new channels
        print('Copy conv1 from pretrained model')
        return backbone

    def build_loss(self):
        # prepare the loss function
        loss_class = LOSSFUNC['bce']
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict, phase_fea) -> torch.tensor:
        features = torch.cat((data_dict['image'], phase_fea), dim=1)
        return self.backbone.features(features)

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        pred = pred.squeeze(1)
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def get_test_metrics(self):
        pass

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the phase features
        phase_fea = self.phase_without_amplitude(data_dict['image'])
        # bp
        features = self.features(data_dict, phase_fea)
        # get the prediction by classifier
        pred = self.classifier(features)

        # build the prediction dict for each output
        pred_dict = {'cls': pred}

        return pred_dict

    def phase_without_amplitude(self, img):
        # Convert to grayscale
        gray_img = torch.mean(img, dim=1, keepdim=True) # shape: (batch_size, 1, 256, 256)
        # Compute the DFT of the input signal
        X = torch.fft.fftn(gray_img)
        # Extract the phase information from the DFT
        phase_spectrum = torch.angle(X)
        # Create a new complex spectrum with the phase information and zero magnitude
        reconstructed_X = torch.exp(1j * phase_spectrum)
        # Use the IDFT to obtain the reconstructed signal
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X))
        return reconstructed_x
