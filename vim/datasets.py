# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

import numpy as np

from torch.utils.data import Subset

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    print(args.data_set)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'FLAME':
        train_dataset = datasets.ImageFolder(args.data_path)

        train_transforms = train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
            transforms.RandomVerticalFlip(p=0.5),  # Vertical flip
            transforms.RandomRotation(degrees=10),  # Rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),  # Gaussian noise
            transforms.ToTensor(),  # Convert to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize
            transforms.ToTensor(),  # Convert to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        nb_classes = 2

        # Oversampling to balance classes
        indices = list(range(len(train_dataset)))
        targets = [sample[1] for sample in train_dataset.samples]  # Labels for each sample

        ros = RandomOverSampler(sampling_strategy=0.8, random_state=42)
        resampled_indices, resampled_targets = ros.fit_resample(np.array(indices).reshape(-1, 1), targets)
        resampled_indices = resampled_indices.flatten()

        # Create oversampled subset
        train_oversampled = Subset(train_dataset, resampled_indices)

        train_indices, val_indices = train_test_split(
            range(len(train_oversampled)), 
            test_size=0.2,        
            random_state=42,           
            stratify=resampled_targets,        
        )

        # Create subsets for train and validation
        # dtype: torch.utils.data.dataset.Subset
        train_dataset = Subset(train_oversampled, train_indices)
        val_dataset = Subset(train_oversampled, val_indices)

        print(f"Training Set Size: {len(train_dataset)} images")
        print(f"Validation Set Size: {len(val_dataset)} images")
        
        # dtype: torchvision.datasets.folder.ImageFolder
        train_dataset.dataset.dataset.transform = train_transforms
        val_dataset.dataset.dataset.transform = test_transforms

        
        return train_dataset, val_dataset, nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
