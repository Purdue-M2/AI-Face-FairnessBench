
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2
# print(dir(A))

# Define the transformation pipeline using Albumentations
def get_albumentations_transforms(use_methods):
    transforms_list = [A.Resize(256, 256)]
    if 'gaussian_blur' in use_methods:
        transforms_list.append(A.GaussianBlur(blur_limit=(5, 5), p=1.0))
        print('gaussian_blur kernel size 5')
    if 'jpeg_compression' in use_methods:
        transforms_list.append(A.JpegCompression(quality_lower=60, quality_upper=60, p=1.0))
        print('using jpeg compression 100')
    if 'random_crop' in use_methods:
        transforms_list.append(A.RandomCrop(height=224, width=224, p=1.0))
        print('random_crop_224')
    if 'center_crop' in use_methods:
        transforms_list.append(A.CenterCrop(height=224, width=224, p=1.0))
    if 'hue_saturation_value' in use_methods:
        transforms_list.append(A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50, p=1.0))
        print('HueSaturationValue applied with a hue shift limit of 50, saturation shift limit of 50, and value shift limit of 50.')
    if 'random_brightness_contrast' in use_methods:
        transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0.8, contrast_limit=0.8, p=1.0))
        print('BrightnessContrast applied with a brightness limit of 0.8 and a contrast limit of 0.8.')
    if 'rotation' in use_methods:
        transforms_list.append(A.Rotate(limit=45, p=1.0))
        print('rotation applied with limit 15')

    # Converting the image to PyTorch tensor and normalize
    transforms_list.extend([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)

def get_albumentations_transforms_vit_clip(use_methods, model_type):
    if model_type == 'vit':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif model_type == 'clip':
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        raise ValueError("Unsupported model type. Choose either 'vit' or 'clip'.")

    transforms_list = [A.Resize(256, 256) if 'random_crop' in use_methods else A.Resize(224, 224)]

    if 'gaussian_blur' in use_methods:
        transforms_list.append(A.GaussianBlur(blur_limit=(3, 3), p=1.0))
        print('gaussian_blur kernel size 3')
    if 'jpeg_compression' in use_methods:
        transforms_list.append(A.JpegCompression(quality_lower=80, quality_upper=80, p=1.0))
        print('using jpeg compression 80')
    if 'random_crop' in use_methods:
        transforms_list.append(A.RandomCrop(height=224, width=224, p=1.0))
        print('random_crop_224')
    if 'center_crop' in use_methods:
        transforms_list.append(A.CenterCrop(height=224, width=224, p=1.0))
    if 'hue_saturation_value' in use_methods:
        transforms_list.append(A.HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=50, p=1.0))
        print('HueSaturationValue applied with a hue shift limit of 50, saturation shift limit of 50, and value shift limit of 50.')
    if 'random_brightness_contrast' in use_methods:
        transforms_list.append(A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0))
        print('BrightnessContrast applied with a brightness limit of 0.4 and a contrast limit of 0.4.')
    if 'rotation' in use_methods:
        transforms_list.append(A.Rotate(limit=30, p=1.0))
        print('rotation applied with limit 30')
    
    # Converting the image to PyTorch tensor and normalize
    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


vit_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

clip_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ]),
}

