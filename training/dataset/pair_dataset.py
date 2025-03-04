'''
The code is designed for scenarios such as disentanglement-based methods where it is necessary to ensure an equal number of positive and negative samples.
'''

import torch
import random
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from PIL import Image
import random



class pairDataset(Dataset):
    def __init__(self, csv_fake_file, csv_real_file, owntransforms):

        # Get real and fake image lists
       
        self.fake_image_list = pd.read_csv(csv_fake_file)
        self.real_image_list = pd.read_csv(csv_real_file)
        self.transform = owntransforms


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fake_img_path = self.fake_image_list.loc[idx, 'Image Path']
        real_idx = random.randint(0, len(self.real_image_list) - 1)
        real_img_path = self.real_image_list.loc[real_idx, 'Image Path']

        if fake_img_path != 'Image Path':
            fake_img = Image.open(fake_img_path)
            fake_trans = self.transform(fake_img)
            fake_label = np.array(self.fake_image_list.loc[idx, 'Target'])

          
            fake_spe_label = np.array(self.fake_image_list.loc[idx, 'Specific'])
            fake_intersec_label = np.array(self.fake_image_list.loc[idx, 'Intersection'])
          
        if real_img_path != 'Image Path':
            real_img = Image.open(real_img_path)
            real_trans = self.transform(real_img)
            real_label = np.array(self.real_image_list.loc[real_idx, 'Target'])
            real_spe_label = np.array(self.real_image_list.loc[real_idx, 'Target'])
            real_intersec_label = np.array(
                self.real_image_list.loc[real_idx, 'Intersection'])
           

        return {"fake": (fake_trans, fake_label, fake_spe_label, fake_intersec_label),
                "real": (real_trans, real_label, real_spe_label, real_intersec_label)}

    def __len__(self):
        return len(self.fake_image_list)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor
                    

        Returns:
            A tuple containing the image tensor, the label tensor
        """
        # Separate the image, label,  tensors for fake and real data
        fake_images, fake_labels, fake_spe_labels, fake_intersec_labels = zip(
            *[data["fake"] for data in batch])
  
        fake_labels = tuple(x.item() for x in fake_labels)
        fake_spe_labels = tuple(x.item() for x in fake_spe_labels)
        fake_intersec_labels = tuple(x.item() for x in fake_intersec_labels)
   
        real_images, real_labels, real_spe_labels, real_intersec_labels = zip(
            *[data["real"] for data in batch])
        real_labels = tuple(x.item() for x in real_labels)
        real_spe_labels = tuple(x.item() for x in real_spe_labels)
        real_intersec_labels = tuple(x.item() for x in real_intersec_labels)


        # Stack the image, label, tensors for fake and real data
        fake_images = torch.stack(fake_images, dim=0)
        fake_labels = torch.LongTensor(fake_labels)
        fake_spe_labels = torch.LongTensor(fake_spe_labels)
        fake_intersec_labels = torch.LongTensor(fake_intersec_labels)


        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)
        real_spe_labels = torch.LongTensor(real_spe_labels)
        real_intersec_labels = torch.LongTensor(real_intersec_labels)


        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        spe_labels = torch.cat([real_spe_labels, fake_spe_labels], dim=0)
        intersec_labels = torch.cat(
            [real_intersec_labels, fake_intersec_labels], dim=0)
    

        data_dict = {
            'image': images,
            'label': labels,
            'label_spe': spe_labels,
            'intersec_label': intersec_labels,
        }
        return data_dict
