import sys
from detectors import DETECTOR
import torch
import torch.nn as nn
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
import time
import os


from transform import get_albumentations_transforms

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
parser.add_argument("--model", type=str, default='xception',
                    help="detector name[xception, fair_df_detector,daw_fdd, efficientnet,...]")

parser.add_argument("--dataset_type", type=str, default='no_pair',
                    help="detector name[pair,no_pair]")
parser.add_argument("--post_processing", default=False)

#################################test##############################

parser.add_argument("--inter_attribute", type=str,
                    default='male,asian-male,white-male,black-male,others-nonmale,asian-nonmale,white-nonmale,black-nonmale,others-young-middle-senior-ageothers')
parser.add_argument("--single_attribute", type=str,
                    default='young-middle-senior-ageothers')
parser.add_argument("--test_datapath", type=str,
                        default='../dataset/test.csv', help="test data path")
parser.add_argument("--savepath", type=str,
                        default='../results')

args = parser.parse_args()

if args.model=='srm' or args.model=='core':
    from fairness_metrics_srm import acc_fairness_softmax
else:
    from fairness_metrics import acc_fairness

###### import data transform #######
from transform import default_data_transforms as data_transforms
test_transforms = get_albumentations_transforms([''])
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

# prepare the model (detector)
model_class = DETECTOR[args.model]


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
def train(model,  optimizer, scheduler, num_epochs, start_epoch):

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        phase = 'train'
        model.train()

        total_loss = 0.0

        for idx, data_dict in enumerate(tqdm(train_dataloader)):

            imgs, labels, intersec_labels = data_dict['image'], data_dict[
                'label'], data_dict['intersec_label']
            if 'label_spe' in data_dict:
                label_spe = data_dict['label_spe']
                data_dict['label_spe'] = label_spe.to(device)
            data_dict['image'], data_dict['label'], data_dict['intersec_label'] = imgs.to(
                device), labels.to(device), intersec_labels.to(device),

            with torch.set_grad_enabled(phase == 'train'):

                preds = model(data_dict)
                if isinstance(model, torch.nn.DataParallel):
                    losses = model.module.get_losses(data_dict, preds)
                else:
                    losses = model.get_losses(data_dict, preds)
                
                losses = losses['overall']
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            if idx % 200 == 0:
                # compute training metric for each batch data
                if isinstance(model, torch.nn.DataParallel):
                    batch_metrics = model.module.get_train_metrics(data_dict, preds)
                else:
                    batch_metrics = model.get_train_metrics(data_dict, preds)


                print('#{} batch_metric{}'.format(idx, batch_metrics))

            total_loss += losses.item() * imgs.size(0)

        epoch_loss = total_loss / train_dataset_size
        print('Epoch: {} Loss: {:.4f}'.format(epoch, epoch_loss))

        # update learning rate
        if phase == 'train':
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
                if args.post_processing:
                  test_dataset = ImageDataset_Test(args.test_datapath, eachatt, test_transforms, args.post_processing)
                else:
                  test_dataset = ImageDataset_Test(args.test_datapath, eachatt, data_transforms['test'], args.post_processing)

                test_dataloader = DataLoader(
                    test_dataset, batch_size=args.test_batchsize, shuffle=False,num_workers=32, pin_memory=True)

                print('Testing: ', eachatt)
                print('-' * 10)
                # print('%d batches int total' % len(test_dataloader))

                pred_list = []
                label_list = []

                for idx, data_dict in enumerate(tqdm(test_dataloader)):
                    imgs, labels = data_dict['image'], data_dict['label']
                    if 'label_spe' in data_dict:
                        data_dict.pop('label_spe')  # remove the specific label

                    data_dict['image'], data_dict['label'] = imgs.to(
                        device), labels.to(device)
                        
                    with torch.no_grad():
                        output = model(data_dict, inference=True)
                        pred = output['cls']
                        pred = pred.cpu().data.numpy().tolist()
            

                        pred_list += pred
                        label_list += labels.cpu().data.numpy().tolist()


               

                label_list = np.array(label_list)
                pred_list = np.array(pred_list)
                savepath = args.savepath + '/' + eachatt
                np.save(savepath+'labels.npy', label_list)
                np.save(savepath+'predictions.npy', pred_list)

                print()
               
            acc_fairness(args.savepath + '/', [['male', 'nonmale'], ['asian', 'white', 'black', 'others'],['young', 'middle','senior','ageothers']])
            cleanup_npy_files(args.savepath)

    return model, epoch


def main():

    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()


    sys.stdout = Logger(osp.join('./checkpoints/'+args.model+'/log_training.txt'))


    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    model = model_class()
    model.to(device)

    start_epoch = 0
    if args.continue_train and args.checkpoints != '':
        state_dict = torch.load(args.checkpoints)
        model.load_state_dict(state_dict)
        start_epoch = 0
        

    # optimize
    params_to_update = model.parameters()
    optimizer4nn = optim.SGD(params_to_update, lr=args.lr,
                             momentum=0.9, weight_decay=5e-03)

    optimizer = optimizer4nn

    print(params_to_update, optimizer)

    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=60, gamma=0.9)

    model, epoch = train(model, optimizer,
                         exp_lr_scheduler, num_epochs=5, start_epoch=start_epoch)

    if epoch == 4:
        print("training finished!")
        exit()


if __name__ == '__main__':
    main()
