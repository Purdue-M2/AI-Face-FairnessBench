import sys
from detectors import DETECTOR
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os.path as osp
from log_utils import Logger
import torch.backends.cudnn as cudnn
from dataset.pair_dataset import pairDataset
from dataset.datasets_train import ImageDataset_Train, ImageDataset_Test
import csv
import argparse
from tqdm import tqdm
import sklearn.metrics as metrics
import time
import os
from transform import get_albumentations_transforms_vit_clip
from fairness_metrics import acc_fairness


parser = argparse.ArgumentParser("Example")

parser.add_argument('--lr', type=float, default=0.0005,
                    help="learning rate for training")
parser.add_argument('--train_batchsize', type=int, default=128, help="batch size")
parser.add_argument('--test_batchsize', type=int, default=32, help="test batch size")
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--fake_datapath', type=str,
                    default='dataset/')
parser.add_argument('--real_datapath', type=str,
                    default='dataset/')
parser.add_argument('--datapath', type=str,
                    default='../dataset/')
parser.add_argument("--continue_train", default=False, action='store_true')
parser.add_argument("--checkpoints", type=str, default='',
                    help="continue train model path")
parser.add_argument("--model", type=str, default='UnivFD')

parser.add_argument("--dataset_type", type=str, default='no_pair',
                    help="detector name[pair,no_pair]")

#################################test##############################
# in intersectional attribute skintone1 represent Light (tone1-3)
parser.add_argument("--inter_attribute", type=str,
                    default='nomale,skintone1-nomale,skintone2-nomale,skintone3-male,skintone1-male,skintone2-male,skintone3-child-young-adult-middle-senior')
parser.add_argument("--test_datapath", type=str,
                        default='../dataset/test.csv', help="test data path")
parser.add_argument("--savepath", type=str,
                        default='../results')

args = parser.parse_args()

from models.clip import clip 
from PIL import Image
import torch.nn as nn




class CLIPModel(nn.Module):
    def __init__(self, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load('ViT-L/14', device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( 768, num_classes )
 

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)

###### import data transform #######
from transform import clip_default_data_transforms as data_transforms
test_transforms = get_albumentations_transforms_vit_clip([''], model_type='clip')
###### load data ######
if args.dataset_type == 'pair':
    train_dataset = pairDataset(args.datapath + 'train_fake_spe.csv', args.datapath + 'train_real.csv', data_transforms['train'])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=16, pin_memory=True, collate_fn=train_dataset.collate_fn)
    train_dataset_size = len(train_dataset)
    
else:
    train_dataset = ImageDataset_Train(args.datapath + 'train.csv', data_transforms['train'])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=16, pin_memory=True)
    train_dataset_size = len(train_dataset)


device = torch.device('cuda:0')


def cleanup_npy_files(directory):
    """
    Deletes all .npy files in the given directory.
    :param directory: The directory to clean up .npy files in.
    """
    for item in os.listdir(directory):
        if item.endswith(".npy"):
            os.remove(os.path.join(directory, item))
    print("Cleaned up .npy files in directory:", directory)


# train and evaluation
def train(model, criterion, optimizer, scheduler, num_epochs, start_epoch):

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        phase = 'train'
        model.train()

        total_loss = 0.0
        all_labels = []
        all_probabilities = []

        for idx, data_dict in enumerate(tqdm(train_dataloader)):

            imgs = data_dict['image'].to(device)
            labels = data_dict['label'].to(device).float()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(imgs)
                outputs = outputs.squeeze(1)
                losses = criterion(outputs, labels)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += losses.item() * imgs.size(0)

    
            with torch.no_grad():
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())


            if idx % 100 == 0:
                
                labels_array = np.array(all_labels)
                probabilities_array = np.array(all_probabilities)

               
                batch_accuracy = metrics.accuracy_score(labels_array, (probabilities_array > 0.5))
                batch_auc = metrics.roc_auc_score(labels_array, probabilities_array)
                batch_precision = metrics.average_precision_score(labels_array, probabilities_array)
                
                fpr, tpr, thresholds = metrics.roc_curve(labels_array, probabilities_array)
                batch_eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]  
               
                print(f'#{idx} - Acc: {batch_accuracy:.4f}, AUC: {batch_auc:.4f}, AP: {batch_precision:.4f}, EER: {batch_eer:.4f}')

                
                all_labels = []
                all_probabilities = []


        epoch_loss = total_loss / train_dataset_size
        print('Epoch: {} Loss: {:.4f}'.format(epoch, epoch_loss))
        scheduler.step()


        # evaluation

        if (epoch+1) % 1 == 0:

            savepath = './checkpoints/'+args.model


            temp_model = savepath+"/"+args.model+str(epoch)+'.pth'
            torch.save(model.state_dict(), temp_model)

            print()
            print('-' * 10)

            phase = 'test'
            model.eval()

            interattributes = args.inter_attribute.split('-')


            for eachatt in interattributes:
                test_dataset = ImageDataset_Test(args.test_datapath, eachatt,test_transforms)

                test_dataloader = DataLoader(
                    test_dataset, batch_size=args.test_batchsize, shuffle=False,num_workers=16, pin_memory=True)

                print('Testing: ', eachatt)
                print('-' * 10)
            
                pred_list = []
                label_list = []

                for idx, data_dict in enumerate(tqdm(test_dataloader)):

                    imgs = data_dict['image'].to(device)
                    labels = data_dict['label'].to(device)
                        
                    with torch.no_grad():
                        output = model(imgs)
                        pred = output
                        pred = pred.cpu().data.numpy().tolist()
            

                        pred_list += pred
                        label_list += labels.cpu().data.numpy().tolist()


                label_list = np.array(label_list)
                pred_list = np.array(pred_list)
                savepath = args.savepath + '/' + eachatt
                np.save(savepath+'labels.npy', label_list)
                np.save(savepath+'predictions.npy', pred_list)

                print()
                # print('-' * 10)
            acc_fairness(args.savepath + '/', [['nomale', 'male'], ['skintone1', 'skintone2', 'skintone3'],['child', 'young', 'adult','middle','senior']])
            cleanup_npy_files(args.savepath)

    return model, epoch


def main():

    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()
    fix_backbone = True


    sys.stdout = Logger(osp.join('./checkpoints/'+args.model+'/log_training.txt'))


    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # # model = models.vit_b_16(pretrained=True)
    model = CLIPModel()


    torch.nn.init.normal_(model.fc.weight.data, 0.0, 0.02)


    if fix_backbone:
        params = []
        for name, p in model.named_parameters():
            if  name=="fc.weight" or name=="fc.bias": 
                params.append(p) 
            else:
                p.requires_grad = False
    else:
        print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
        import time 
        time.sleep(3)
        params = model.parameters()



    model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)


    start_epoch = 0

    # optimize
    # params_to_update = model.parameters()
    optimizer4nn = optim.SGD(params, lr=args.lr,
                             momentum=0.9, weight_decay=5e-03)

    optimizer = optimizer4nn

    # print(params, optimizer)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=60, gamma=0.9)

    model, epoch = train(model, criterion, optimizer,
                         exp_lr_scheduler, num_epochs=10, start_epoch=start_epoch)

    if epoch == 9:
        print("training finished!")
        exit()


if __name__ == '__main__':
    main()
