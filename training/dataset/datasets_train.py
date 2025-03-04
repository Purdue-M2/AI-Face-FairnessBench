import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle
import os
import pandas as pd
from PIL import Image
import random


class ImageDataset_Train(Dataset):
 

    def __init__(self, csv_file, owntransforms):
        super(ImageDataset_Train, self).__init__()
        self.img_path_label = pd.read_csv(csv_file)
        self.transform = owntransforms

    def __len__(self):
        return len(self.img_path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        img_path = self.img_path_label.iloc[idx, 0]

        if not isinstance(img_path, str) or not os.path.exists(img_path):
            raise ValueError(f"Expected img_path to be a valid file path, got {type(img_path)}: {img_path}")



        if img_path != 'Image Path':
            img = Image.open(img_path)
            img = self.transform(img)
           
            label = np.array(self.img_path_label.loc[idx, 'Target'])
            intersec_label = np.array(self.img_path_label.loc[idx, 'Intersection'])

        return {'image': img, 'label': label, 'intersec_label': intersec_label}

class ImageDataset_Test_Ind(Dataset):


    def __init__(self, csv_file, owntransforms):
        super(ImageDataset_Test_Ind, self).__init__()
        self.img_path_label = pd.read_csv(csv_file)
        self.transform = owntransforms

    def __len__(self):
        return len(self.img_path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_path_label.iloc[idx, 0]

        if img_path != 'Image Path':
            img = Image.open(img_path)
            img = self.transform(img)
           
            label = np.array(self.img_path_label.loc[idx, 'Target'])

        return {'image': img, 'label': label}



class ImageDataset_Test(Dataset):
    def __init__(self, csv_file, attribute, owntransforms):
        self.transform = owntransforms
        self.img = []
        self.label = []
        
        # Mapping from attribute strings to (intersec_label, age_label) tuples
        # Note: if an attribute doesn't correspond to an age label, we use None
        attribute_to_labels = {
            'nomale,skintone1': (0, None), 'nomale,skintone2': (1, None), 'nomale,skintone3': (2, None),
            'male,skintone1': (3, None), 'male,skintone2': (4, None),
            'male,skintone3': (5, None), 'child': (None, 0),
            'young': (None, 1), 'adult': (None, 2), 'middle': (None, 3),'senior': (None, 4)
        }

        # Check if the attribute is valid
        if attribute not in attribute_to_labels:
            raise ValueError(f"Attribute {attribute} is not recognized.")
        
        intersec_label, age_label = attribute_to_labels[attribute]

        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Iterate through the DataFrame rows
        for _, row in df.iterrows():
            img_path = row['Image Path'] 
            mylabel = int(row['Target'])  

            # Depending on the attribute, check the corresponding label
            if intersec_label is not None and int(row['Intersection']) == intersec_label:
                self.img.append(img_path)
                self.label.append(mylabel)
            elif age_label is not None and int(row['Predicted Age']) == age_label:  
                self.img.append(img_path)
                self.label.append(mylabel)

    def __getitem__(self, index):
        path = self.img[index]
        img = np.array(Image.open(path))
        label = self.label[index]
        augmented = self.transform(image=img)
        img = augmented['image']  


        data_dict = {
            'image': img,
            'label': label
        }

        return data_dict


    def __len__(self):
        return len(self.img)
