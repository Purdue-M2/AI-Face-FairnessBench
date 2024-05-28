'''
Reference:
@inproceedings{lin2024preserving,
  title={Preserving Fairness Generalization in Deepfake Detection},
  author={Lin, Li and He, Xinan and Ju, Yan and Wang, Xin and Ding, Feng and Hu, Shu},
  booktitle={CVPR},
  year={2024}
}
'''


import logging
import numpy as np
from sklearn import metrics
from scipy import optimize

import torch
import torch.nn as nn
from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='fair_df_detector')
class FairDetector(AbstractDetector):
    def __init__(self):
        super().__init__()

        self.num_classes = 2

        self.num_classes_target = 1

        self.encoder_feat_dim = 512
        self.half_fingerprint_dim = self.encoder_feat_dim//2


        self.encoder_f = self.build_backbone()
        self.encoder_c = self.build_backbone()
        self.encoder_fair = self.build_backbone()

        self.loss_func = self.build_loss()


        # basic function
        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # conditional gan
        self.con_gan = Conditional_UNet()
        self.adain = AdaIN()

        # head

        specific_task_number = 4
        fair_task_number = 8

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
        self.head_fair = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim,
            out_f=fair_task_number
        )
        self.head_fused = Head(
            in_f=self.half_fingerprint_dim,
            hidden_dim=self.encoder_feat_dim,
            out_f=self.num_classes_target
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
        self.block_fair = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim,
            out_f=self.half_fingerprint_dim
        )
        self.block_fused = Conv2d1x1(
            in_f=self.encoder_feat_dim,
            hidden_dim=self.half_fingerprint_dim,
            out_f=self.half_fingerprint_dim
        )

    def build_backbone(self):
        # prepare the backbone

        backbone_class = BACKBONE['xception']
        backbone = backbone_class({'mode': 'adjust_channel',
                                   'num_classes': 2, 'inc': 3, 'dropout': False})
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load(
            './pretrained/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model successfully!')
        return backbone

    def build_loss(self):

        cls_loss_class = LOSSFUNC['cross_entropy']
        spe_loss_class = LOSSFUNC['cross_entropy']
        con_loss_class = LOSSFUNC['contrastive_regularization']
        rec_loss_class = LOSSFUNC['l1loss']
        fair_loss_class = LOSSFUNC['balance']
        fuse_loss_class = LOSSFUNC['daw_bce']
        cls_loss_func = cls_loss_class()
        spe_loss_func = spe_loss_class()
        con_loss_func = con_loss_class(margin=3.0)
        rec_loss_func = rec_loss_class()
        fair_loss_func = fair_loss_class(
            cls_num_list=[2475, 25443, 1468, 4163, 8013, 31281, 1111, 2185])
        fuse_loss_func = fuse_loss_class()
        loss_func = {
            'cls_ag': cls_loss_func,
            'spe': spe_loss_func,
            'con': con_loss_func,
            'rec': rec_loss_func,
            'fair': fair_loss_func,
            'fuse': fuse_loss_func
        }
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        cat_data = data_dict['image']
        # encoder
        f_all = self.encoder_f.features(cat_data)
        c_all = self.encoder_c.features(cat_data)
        fair_all = self.encoder_fair.features(cat_data)
        feat_dict = {'forgery': f_all, 'content': c_all, 'fairness': fair_all}
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

    #  setting all negative values to zero
    def threshplus_tensor(self, x):
        y = x.clone()
        pros = torch.nn.ReLU()
        z = pros(y)
        return z

    def search_func(self, losses, alpha):
        return lambda x: x + (1.0/alpha)*(self.threshplus_tensor(losses-x).mean().item())

    def search_func_smooth(self, losses, alpha, tau1, tau2):
        return lambda x: x + (tau2 / (2 * alpha)) * (x ** 2) + (1.0 / alpha) * self.threshplus_tensor(tau1 * torch.log(1 + torch.exp((losses - x) / tau1))).mean().item() if not torch.isinf(torch.exp((losses - x) / tau1)).any() else np.finfo(float).max

    def searched_lamda_loss(self, losses, searched_lamda, alpha):
        return searched_lamda + ((1.0/alpha)*torch.mean(self.threshplus_tensor(losses-searched_lamda)))

    def searched_lamda_loss_smooth(self, losses, searched_lamda, alpha, tau1, tau2):
        return searched_lamda + (tau2/(2*alpha))*(searched_lamda ** 2) + (1.0/alpha)*(torch.mean(self.threshplus_tensor(tau1*torch.log(1+torch.exp((losses-searched_lamda)/tau1)))))

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
        label_fair = data_dict['intersec_label']
        intersec_label = data_dict['intersec_label']

        # get pred
        pred = pred_dict['cls_ag']
        
        pred_spe = pred_dict['cls_spe']
        pred_fair = pred_dict['cls_fair']

        
        pred_fuse = pred_dict['cls']

        
        loss_sha = self.loss_func['cls_ag'](pred, label)

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

        
        loss_fair = self.loss_func['fair'](pred_fair, label_fair)

        outer_loss = []
        inter_index = list(torch.unique(intersec_label))
        loss_fuse_entropy = self.loss_func['fuse'](pred_fuse, label)
        for index in inter_index:
            ori_inter_loss = loss_fuse_entropy[intersec_label == index]
            lamda_i_search_func = self.search_func(
                ori_inter_loss, 0.9)
            searched_lamda_i = optimize.fminbound(lamda_i_search_func, np.min(ori_inter_loss.cpu(
            ).detach().numpy()) - 1000.0, np.max(ori_inter_loss.cpu().detach().numpy()))
            inner_loss = self.searched_lamda_loss(
                ori_inter_loss, searched_lamda_i, 0.9)
            outer_loss.append(inner_loss)
        outer_loss = torch.stack(outer_loss)
        lamda_search_func = self.search_func_smooth(
            outer_loss, 0.5, 0.001, 0.0001)

        searched_lamda = optimize.fminbound(lamda_search_func, np.min(outer_loss.cpu(
        ).detach().numpy()) - 1000.0, np.max(outer_loss.cpu().detach().numpy()))
        loss_fuse = self.searched_lamda_loss_smooth(
            outer_loss, searched_lamda, 0.5, 0.001, 0.0001)

        loss = loss_sha + 0.1*loss_spe + 0.3 * \
            loss_reconstruction + 0.05*loss_con + 0.1*loss_fair + loss_fuse
        loss_dict = {
            'overall': loss,
            'common': loss_sha,
            'specific': loss_spe,
            'reconstruction': loss_reconstruction,
            'contrastive': loss_con,
            'fairness': loss_fair,
            'fusion': loss_fuse
        }
        return loss_dict


    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        def get_accracy(label, output):
            _, prediction = torch.max(output, 1)    
            correct = (prediction == label).sum().item()
            accuracy = correct / prediction.size(0)
            return accuracy

        # get pred and label
        label = data_dict['label']

        pred = pred_dict['cls']
        pred = pred.squeeze(1)
        label_spe = data_dict['label_spe']
        pred_spe = pred_dict['cls_spe']
        label_fair = data_dict['intersec_label']
        pred_fair = pred_dict['cls_fair']

        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(
            label.detach(), pred.detach())
        acc_spe = get_accracy(label_spe.detach(), pred_spe.detach())
        acc_fair = get_accracy(label_fair.detach(), pred_fair.detach())
        metric_batch_dict = {'acc_fused': acc, 'acc_spe': acc_spe, 'acc_fair': acc_fair,
                             'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_test_metrics(self):

        pass

    def forward(self, data_dict: dict, inference=False) -> dict:
        # split the features into the content and forgery,fairness
        features = self.features(data_dict)
        forgery_features, content_features, fair_features = features[
            'forgery'], features['content'], features['fairness']

        # get the prediction by classifier (split the common and specific forgery)
        f_spe, f_share = self.classifier(forgery_features)
        
        f_fair = self.block_fair(fair_features)
        fused_features = self.adain(f_fair, f_share)  # [16, 256, 8, 8]

        if inference:
            # inference only consider share loss
            out_sha, sha_feat = self.head_sha(f_share)
            out_spe, spe_feat = self.head_spe(f_spe)
            out_fused, fused_feat = self.head_fused(fused_features)

            pred_dict = {'cls_ag': out_sha, 'feat': sha_feat,
                'cls': out_fused, 'feat_fused': fused_feat}
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
        out_fair, fair_feat = self.head_fair(f_fair)
        out_fused, fused_feat = self.head_fused(fused_features)



        pred_dict = {
            'cls_ag': out_sha,
            'feat': sha_feat,
            'cls_spe': out_spe,
            'feat_spe': spe_feat,
            'cls_fair': out_fair,
            'feat_fair': fair_feat,
            'cls': out_fused,
            'feat_fused': fused_feat,
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

    def forward(self, c, x):  
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
