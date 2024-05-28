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
    '''
    Data format in .csv file each line:
    Image Path,Predicted Gender,Predicted Age,Predicted Race,Reliability Score Gender,Reliability Score Age,Reliability Score Race,Ground Truth Gender,Ground Truth Age,Ground Truth Race,Intersection,Target,Specific
    '''

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
                # Debug print to verify the content of img_path
        # print(f"Image path at index {idx}: {img_path}")
        # Check if the img_path is indeed a string and exists
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            raise ValueError(f"Expected img_path to be a valid file path, got {type(img_path)}: {img_path}")



        if img_path != 'Image Path':
            img = Image.open(img_path)
            img = self.transform(img)
            # label = np.array(self.img_path_label.iloc[idx, 1])
            label = np.array(self.img_path_label.loc[idx, 'Target'])

            # intersec_label = np.array(self.img_path_label.iloc[idx, 6])
            intersec_label = np.array(self.img_path_label.loc[idx, 'Intersection'])

        return {'image': img, 'label': label, 'intersec_label': intersec_label}

class ImageDataset_Test_Ind(Dataset):
    '''
    Data format in .csv file each line:
    Image Path,Predicted Gender,Predicted Age,Predicted Race,Reliability Score Gender,Reliability Score Age,Reliability Score Race,Ground Truth Gender,Ground Truth Age,Ground Truth Race,Intersection,Target,Specific
    '''

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
            # label = np.array(self.img_path_label.iloc[idx, 1])
            label = np.array(self.img_path_label.loc[idx, 'Target'])

            # intersec_label = np.array(self.img_path_label.iloc[idx, 6])
            # intersec_label = np.array(self.img_path_label.loc[idx, 'Intersection'])

        return {'image': img, 'label': label}

class ImageDataset_Test_Cross_Domain(Dataset):
    def __init__(self, csv_file, attribute, owntransforms):
        self.transform = owntransforms
        self.img = []
        self.label = []

        attribute_to_labels = {
            'nonmale': (0), 'male': (1)
        }

        # Check if the attribute is valid
        # print(attribute)
        if attribute not in attribute_to_labels:
            
            raise ValueError(f"Attribute {attribute} is not recognized.")
        
        gender_label = attribute_to_labels[attribute]

        with open(csv_file, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            next(rows)  # Skip the header row
            for row in rows:
                img_path = row[0]
                mylabel = int(row[1])
                
                # Depending on the attribute, check the corresponding label
                if gender_label is not None and int(row[2]) == gender_label:
                    self.img.append(img_path)
                    self.label.append(mylabel)
        # print(attribute, len(self.img), len(self.label))

    def __getitem__(self, index):
        path = self.img[index]
        # img = Image.open(path)  
        img = np.array(Image.open(path))
        label = self.label[index]
        # img = self.transform(img)
        # img = self.transform(image=img)['image']
        augmented = self.transform(image=img)
        img = augmented['image']  # This is now a PyTorch tensor


        data_dict = {
            'image': img,
            'label': label
        }

        return data_dict


    def __len__(self):
        return len(self.img)


class ImageDataset_Test_old(Dataset):
    def __init__(self, csv_file, attribute, owntransforms):
        self.transform = owntransforms
        self.img = []
        self.label = []

        att_list = attribute.split(',')

        with open(csv_file, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            line_count = 0
            for row in rows:
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    img_path = row[0]
                    if img_path != 'Image Path':
                        mylabel = int(row[11])

                        intersec_label = int(row[10])
                        age_label = int(row[8])
                        if len(att_list) == 2:
                            if attribute == 'male,asian':
                                if intersec_label == 0:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)
                            if attribute == 'male,white':
                                if intersec_label == 1:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)
                            if attribute == 'male,black':
                                if intersec_label == 2:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)
                            if attribute == 'male,others':
                                if intersec_label == 3:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)
                            if attribute == 'nonmale,asian':
                                if intersec_label == 4:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)
                            if attribute == 'nonmale,white':
                                if intersec_label == 5:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)
                            if attribute == 'nonmale,black':
                                if intersec_label == 6:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)
                            if attribute == 'nonmale,others':
                                if intersec_label == 7:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)
                        else:
                            if attribute == 'young':
                                if age_label == 0:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)
                            if attribute == 'middle':
                                if age_label == 1:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)
                            if attribute == 'senior':
                                if age_label == 2:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)
                            if attribute == 'ageothers':
                                if age_label == 3:
                                    self.img.append(img_path)
                                    self.label.append(mylabel)                    
                                        
        print(attribute, len(self.img), len(self.label))

    def __getitem__(self, index):

        path = self.img[index % len(self.img)]

        img = Image.open(path)
        label = self.label[index % len(self.label)]
        img = self.transform(img)
        data_dict = {}
        data_dict['image'] = img
        data_dict['label'] = label

        return data_dict

    def __len__(self):
        return len(self.img)


class ImageDataset_Test(Dataset):
    def __init__(self, csv_file, attribute, owntransforms):
        self.transform = owntransforms
        self.img = []
        self.label = []
        
        # Mapping from attribute strings to (intersec_label, age_label) tuples
        # Note: if an attribute doesn't correspond to an age label, we use None
        attribute_to_labels = {
            'male,asian': (0, None), 'male,white': (1, None), 'male,black': (2, None),
            'male,others': (3, None), 'nonmale,asian': (4, None), 'nonmale,white': (5, None),
            'nonmale,black': (6, None), 'nonmale,others': (7, None), 'young': (None, 0),
            'middle': (None, 1), 'senior': (None, 2), 'ageothers': (None, 3)
        }

        # Check if the attribute is valid
        if attribute not in attribute_to_labels:
            raise ValueError(f"Attribute {attribute} is not recognized.")
        
        intersec_label, age_label = attribute_to_labels[attribute]

        with open(csv_file, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            next(rows)  # Skip the header row
            for row in rows:
                img_path = row[0]
                mylabel = int(row[11])
                
                # Depending on the attribute, check the corresponding label
                if intersec_label is not None and int(row[10]) == intersec_label:
                    self.img.append(img_path)
                    self.label.append(mylabel)
                elif age_label is not None and int(row[8]) == age_label:
                    self.img.append(img_path)
                    self.label.append(mylabel)
        # print(attribute, len(self.img), len(self.label))

    def __getitem__(self, index):
        path = self.img[index]
        # img = Image.open(path)  
        img = np.array(Image.open(path))
        label = self.label[index]
        # img = self.transform(img)
        # img = self.transform(image=img)['image']
        augmented = self.transform(image=img)
        img = augmented['image']  # This is now a PyTorch tensor


        data_dict = {
            'image': img,
            'label': label
        }

        return data_dict


    def __len__(self):
        return len(self.img)