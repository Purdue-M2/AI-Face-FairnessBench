
import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from scipy import optimize
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

logger = logging.getLogger(__name__)

@DETECTOR.register_module(module_name='daw_fdd')
class DawFddDetector(AbstractDetector):
    def __init__(self):
        super().__init__()
        self.backbone = self.build_backbone()
        self.loss_func = self.build_loss()
        
    def build_backbone(self):
        # prepare the backbone
        backbone_class = BACKBONE['xception']
        backbone = backbone_class({'mode': 'original',
                                   'num_classes': 1, 'inc': 3, 'dropout': False})
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load('./pretrained/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        print('Load pretrained model successfully!')
        return backbone
    
    def build_loss(self):
        # prepare the loss function
        loss_class = LOSSFUNC['daw_bce']
        loss_func = loss_class()
        return loss_func

    def threshplus_tensor(self, x):
        y = x.clone()
        pros = torch.nn.ReLU()
        z = pros(y)
        return z
    
    def search_func(self, losses, alpha):
        return lambda x: x + (1.0/alpha)*(self.threshplus_tensor(losses-x).mean().item())

    def searched_lamda_loss(self, losses, searched_lamda, alpha):
        return searched_lamda + ((1.0/alpha)*torch.mean(self.threshplus_tensor(losses-searched_lamda))) 
    
    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict['image']) #32,3,256,256

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # defualt 0.9
        inner_alpha = 0.9
        outer_alpha = 0.5
        label = data_dict['label']
        intersec_label = data_dict['intersec_label']
        pred = pred_dict['cls']
        outer_loss = []
        inter_index = list(torch.unique(intersec_label))
        loss_entropy = self.loss_func(pred, label)
        for index in inter_index:
            ori_inter_loss = loss_entropy[intersec_label == index]
            lamda_i_search_func = self.search_func(ori_inter_loss,inner_alpha)
            searched_lamda_i = optimize.fminbound(lamda_i_search_func, np.min(ori_inter_loss.cpu().detach().numpy()) - 1000.0, np.max(ori_inter_loss.cpu().detach().numpy()))
            inner_loss = self.searched_lamda_loss(ori_inter_loss, searched_lamda_i, inner_alpha)
            outer_loss.append(inner_loss)
        outer_loss = torch.stack(outer_loss)
        lamda_search_func = self.search_func(outer_loss, outer_alpha)
        searched_lamda = optimize.fminbound(lamda_search_func, np.min(outer_loss.cpu().detach().numpy()) - 1000.0, np.max(outer_loss.cpu().detach().numpy()))
        loss = self.searched_lamda_loss(outer_loss, searched_lamda, outer_alpha)
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
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)

        pred_dict = {'cls': pred}

        return pred_dict