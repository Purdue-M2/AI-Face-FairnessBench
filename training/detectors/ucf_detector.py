

import os
import datetime
import logging
import random
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict
from scipy import optimize

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


@DETECTOR.register_module(module_name='ucf')
class UCFDetector(AbstractDetector):
    def __init__(self):
        super().__init__()

        self.num_classes = 1
        self.encoder_feat_dim = 512
        self.half_fingerprint_dim = self.encoder_feat_dim//2

        self.encoder_f = self.build_backbone()
        self.encoder_c = self.build_backbone()

        self.loss_func = self.build_loss()
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        # basic function
        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # conditional gan
        self.con_gan = Conditional_UNet()


        specific_task_number = 4
        self.head_spe = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim,
            out_f=specific_task_number
        )
        self.head_sha = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim,
            out_f=self.num_classes
        )
        self.block_spe = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim,
            out_f=self.half_fingerprint_dim
        )
        self.block_sha = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim,
            out_f=self.half_fingerprint_dim
        )

    def build_backbone(self):
        # prepare the backbone
        backbone_class = BACKBONE['xception']

        backbone = backbone_class({'mode': 'adjust_channel',
                                   'num_classes': 1, 'inc': 3, 'dropout': False})
        state_dict = torch.load('./pretrained/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        print('Load pretrained model successfully!')
        return backbone

    def build_loss(self):

        cls_loss_class = LOSSFUNC['bce']
        spe_loss_class = LOSSFUNC['cross_entropy']
        con_loss_class = LOSSFUNC['contrastive_regularization']
        rec_loss_class = LOSSFUNC['l1loss']
        cls_loss_func = cls_loss_class()
        spe_loss_func = spe_loss_class()
        con_loss_func = con_loss_class(margin=3.0)
        rec_loss_func = rec_loss_class()
        loss_func = {
            'cls': cls_loss_func,
            'spe': spe_loss_func,
            'con': con_loss_func,
            'rec': rec_loss_func,
        }
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        cat_data = data_dict['image']
        # encoder
        f_all = self.encoder_f.features(cat_data)
        c_all = self.encoder_c.features(cat_data)
        feat_dict = {'forgery': f_all, 'content': c_all}
        return feat_dict

    def classifier(self, features: torch.tensor) -> torch.tensor:
        # classification, multi-task
        # split the features into the specific and common forgery
        f_spe = self.block_spe(features)
        f_share = self.block_sha(features)
        return f_spe, f_share

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        if 'label_spe' in data_dict and 'recontruction_imgs' in pred_dict:
            return self.get_train_losses(data_dict, pred_dict)
        # else:  # test mode
        #     return self.get_test_losses(data_dict, pred_dict)

    def get_train_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        # get combined, real, fake imgs
        cat_data = data_dict['image']
        real_img, fake_img = cat_data.chunk(2, dim=0)
        # get the reconstruction imgs
        reconstruction_image_1, \
            reconstruction_image_2, \
            self_reconstruction_image_1, \
            self_reconstruction_image_2 \
            = pred_dict['recontruction_imgs']
        # get label
        label = data_dict['label']
        label_spe = data_dict['label_spe']
        # get pred
        pred = pred_dict['cls']

        pred_spe = pred_dict['cls_spe']

        loss_sha = self.loss_func['cls'](pred, label)

        loss_spe = self.loss_func['spe'](pred_spe, label_spe)

        self_loss_reconstruction_1 = self.loss_func['rec'](
            fake_img, self_reconstruction_image_1)
        self_loss_reconstruction_2 = self.loss_func['rec'](
            real_img, self_reconstruction_image_2)
        cross_loss_reconstruction_1 = self.loss_func['rec'](
            fake_img, reconstruction_image_2)
        cross_loss_reconstruction_2 = self.loss_func['rec'](
            real_img, reconstruction_image_1)
        loss_reconstruction = \
            self_loss_reconstruction_1 + self_loss_reconstruction_2 + \
            cross_loss_reconstruction_1 + cross_loss_reconstruction_2

        common_features = pred_dict['feat']
        specific_features = pred_dict['feat_spe']
        loss_con = self.loss_func['con'](
            common_features, specific_features, label_spe)

        loss = loss_sha + 0.1*loss_spe + 0.3*loss_reconstruction + 0.05*loss_con
        loss_dict = {
            'overall': loss,
            'common': loss_sha,
            'specific': loss_spe,
            'reconstruction': loss_reconstruction,
            'contrastive': loss_con,
        }
        return loss_dict



    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        def get_accracy(label, output):
            _, prediction = torch.max(output, 1)    # argmax
            correct = (prediction == label).sum().item()
            accuracy = correct / prediction.size(0)
            return accuracy

        # get pred and label
        label = data_dict['label']
        pred = pred_dict['cls']
        pred = pred.squeeze(1)
        label_spe = data_dict['label_spe']
        pred_spe = pred_dict['cls_spe']

        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(), pred.detach())
        acc_spe = get_accracy(label_spe.detach(), pred_spe.detach())
        metric_batch_dict = {'acc': acc, 'acc_spe': acc_spe,
                             'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self):
        pass


    def forward(self, data_dict: dict, inference=False) -> dict:
        # split the features into the content and forgery
        features = self.features(data_dict)
        forgery_features, content_features = features['forgery'], features['content']
        # get the prediction by classifier (split the common and specific forgery)
        f_spe, f_share = self.classifier(forgery_features)


        if inference:
            # inference only consider share loss
            out_sha, sha_feat = self.head_sha(f_share)
            out_spe, spe_feat = self.head_spe(f_spe)


            pred_dict = {'cls': out_sha, 'feat': sha_feat}
            return pred_dict


        f_all = torch.cat((f_spe, f_share), dim=1)

        f2, f1 = f_all.chunk(2, dim=0)
        c2, c1 = content_features.chunk(2, dim=0)

        self_reconstruction_image_1 = self.con_gan(f1, c1)

        self_reconstruction_image_2 = self.con_gan(f2, c2)

        reconstruction_image_1 = self.con_gan(f1, c2)
        reconstruction_image_2 = self.con_gan(f2, c1)

        out_spe, spe_feat = self.head_spe(f_spe)
        out_sha, sha_feat = self.head_sha(f_share)

        pred_dict = {
            'cls': out_sha,
            'feat': sha_feat,
            'cls_spe': out_spe,
            'feat_spe': spe_feat,
            'feat_content': content_features,
            'recontruction_imgs': (
                reconstruction_image_1,
                reconstruction_image_2,
                self_reconstruction_image_1,
                self_reconstruction_image_2
            )
        }
        return pred_dict


def sn_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, in_channels, 3, padding=1)),
        nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=2)),
        nn.LeakyReLU(0.2, inplace=True)
    )


def r_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class AdaIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        # self.l1 = nn.Linear(num_classes, in_channel*4, bias=True) #bias is good :)

    def c_norm(self, x, bs, ch, eps=1e-7):
        # assert isinstance(x, torch.cuda.FloatTensor)
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y):
        assert x.size(0) == y.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        y_ = y.reshape(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
        y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
        out = ((x - x_mean.expand(size)) / x_std.expand(size)) \
            * y_std.expand(size) + y_mean.expand(size)
        return out


class Conditional_UNet(nn.Module):

    def init_weight(self, std=0.2):
        for m in self.modules():
            cn = m.__class__.__name__
            if cn.find('Conv') != -1:
                m.weight.data.normal_(0., std)
            elif cn.find('Linear') != -1:
                m.weight.data.normal_(1., std)
                m.bias.data.fill_(0)

    def __init__(self):
        super(Conditional_UNet, self).__init__()

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)
        # self.dropout_half = HalfDropout(p=0.3)

        self.adain3 = AdaIN()
        self.adain2 = AdaIN()
        self.adain1 = AdaIN()

        self.dconv_up3 = r_double_conv(512, 256)
        self.dconv_up2 = r_double_conv(256, 128)
        self.dconv_up1 = r_double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, 3, 1)
        self.up_last = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.activation = nn.Tanh()
        # self.init_weight()

    def forward(self, c, x):  # c is the style and x is the content
        x = self.adain3(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up3(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up3(c)

        x = self.adain2(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up2(x)
        c = self.upsample(c)
        c = self.dropout(c)
        c = self.dconv_up2(c)

        x = self.adain1(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = self.dconv_up1(x)

        x = self.conv_last(x)
        out = self.up_last(x)

        return self.activation(out)


class MLP(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(MLP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        x = self.pool(x)
        x = self.mlp(x)
        return x


class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(hidden_dim, out_f, 1, 1),)

    def forward(self, x):
        x = self.conv2d(x)
        return x


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Head, self).__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        bs = x.size()[0]
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat
