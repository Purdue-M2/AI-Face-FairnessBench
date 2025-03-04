

import os
import datetime
import logging
import random
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

from metrics.base_metrics_class import calculate_metrics_for_train,calculate_metrics_for_train_softmax

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from efficientnet_pytorch import EfficientNet

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='core')
class CoreDetector(AbstractDetector):
    def __init__(self):
        super().__init__()

        self.backbone = self.build_backbone()
        self.loss_func = self.build_loss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        
    def build_backbone(self):
        # prepare the backbone
        backbone_class = BACKBONE['xception']
        model_config = {'mode': 'original',
                                   'num_classes': 2, 'inc': 3, 'dropout': False}
        backbone = backbone_class(model_config)
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load('./pretrained/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        # logger.info('Load pretrained model successfully!')
        print('Load pretrained model successfully!')
        return backbone
    
    def build_loss(self):
        # prepare the loss function
        loss_class = LOSSFUNC['consistency_loss']
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image'])

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        core_feat = pred_dict['core_feat']
        loss = self.loss_func(core_feat, pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train_softmax(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def get_test_metrics(self):
        pass
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the core_feat for loss
        core_feat = nn.ReLU(inplace=False)(features)
        core_feat= F.adaptive_avg_pool2d(core_feat, (1, 1))
        core_feat = core_feat.view(core_feat.size(0), -1)
        # get the prediction by classifier
        pred = self.classifier(features)

        pred_dict = {'cls': pred, 'core_feat': core_feat}


        return pred_dict
