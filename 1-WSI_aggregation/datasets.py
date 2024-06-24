# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mostly copy-paste from https://github.com/facebookresearch/deit/blob/main/datasets.py
"""

import os
import json
import glob
import random
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data import Dataset, DataLoader
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torchvision.datasets.folder import *
from typing import *
from sklearn.model_selection import KFold

class FilterableImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            valid_classes: List = None
    ):
        self.valid_classes = valid_classes
        super(FilterableImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes = [valid_class for valid_class in classes if valid_class in self.valid_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_to_idx

class UrineDataset(Dataset):
    """
    Code for reading the ImageNet
    """
    def __init__(self, dataset_path='',transform=None):
        train_data_path = dataset_path
        self.transform = transform
        # self.imagenet = datasets.ImageFolder(root=train_data_path)
        self.imagenet = FilterableImageFolder(root=train_data_path, valid_classes=['benign','cancer','atypical', 'suspicious'])

    def __len__(self):
        return len(self.imagenet)

    def __getitem__(self, index):
        image_tensor = self.transform(self.imagenet[index][0])
        label_tensor = int(self.imagenet[index][1])
        # print(label_tensor)
        return image_tensor, label_tensor

class EmbedDataset(Dataset):
    def __init__(self, dataset_path='',is_train=True, aug_mix=False,
                 class_names=['cancer', 'benign', 'atypical', 'suspicious'], sample_ratio=[0.01, 0.2]):
        real_root_list = dataset_path.split('/')[:-1]
        self.real_root = '/'.join(real_root_list)
        self.filenames = glob.glob(dataset_path + '/*' + '/BD*')
        self.filenames = sorted(self.filenames)
        
        # if is_train==False:
        #     excluded_directories = ['atypical', 'suspicious']
        #     self.filenames = [file_name for file_name in self.filenames if all(excluded_dir not in file_name for excluded_dir in excluded_directories)]

        self.EmbedList = []
        self.aug_mix = aug_mix
        for name in self.filenames:
            if (all(chr.isdigit() for chr in name.split('-')[-1].split('.')[0])):
                self.EmbedList.append(name)
        self.class_names = class_names
        self.Embedwlabel = {k: [] for k in class_names}
        ### aggregate embedding file acorrding to label
        self.agg_emb()
        self.sample_ratio = sample_ratio
        self.aug_dict = {
            'cancer': ['cancer','benign'],
            'benign': ['benign'],
        }

    def __len__(self):
        return len(self.EmbedList)

    def mix_augmentation(self, label_name, data_or):
        for key in self.aug_dict.keys():
            if label_name in key:
                cls_name = random.choice(self.aug_dict[key])
                mix_query_list = self.Embedwlabel[cls_name]
                break
        # print(mix_query_list)
        # assert 2==3
        filename = random.choice(mix_query_list)
        data = np.load(filename)

        if filename.split('/')[-4] == 'scale256to128':
            k = 400
        else:
            k  =100
        while data.shape[0] != k:
            randcol = random.randint(0, data.shape[0] - 1)
            data = np.concatenate([data, np.expand_dims(data[randcol, :], 0)], axis=0)
        sm_ratio = random.uniform(self.sample_ratio[0], self.sample_ratio[1])
        index_list = list(range(k))
        insert_num = int(k * sm_ratio)
        rm_num = k - insert_num
        index_s = random.choices(index_list, k=insert_num)
        index_r = random.choices(index_list, k=rm_num)
        data = torch.tensor(data)
        data_aug = torch.cat((torch.index_select(data, 0, torch.tensor(index_s)), torch.index_select(data_or, 0, torch.tensor(index_r))), dim=0)
        # print(data_aug.shape)
        return data_aug

    def agg_emb(self):
        for name in self.filenames:
            if (all(chr.isdigit() for chr in name.split('-')[-1].split('.')[0])):
                filename = name
                if filename.split('/')[-2] == 'cancer':
                    self.Embedwlabel['cancer'].append(name)
                elif filename.split('/')[-2] == 'benign':
                    self.Embedwlabel['benign'].append(name)
                elif filename.split('/')[-2] == 'atypical':
                    self.Embedwlabel['atypical'].append(name)
                elif filename.split('/')[-2] == 'suspicious':
                    self.Embedwlabel['suspicious'].append(name)
                else:
                    None
    def __getitem__(self, index):
        filename = self.EmbedList[index]

        data = np.load(filename)
        if filename.split('/')[-4] == 'scale256to128':
            k = 400
        else:
            k=100
        while data.shape[0] != k:
            randcol = random.randint(0, data.shape[0]-1)
            data = np.concatenate([data,np.expand_dims(data[randcol,:],0)],axis=0)

        data = torch.tensor(data)
        if filename.split('/')[-2] == 'cancer':
            label = [int(1)]
        if filename.split('/')[-2] == 'benign':
            label = [int(0)]
        if filename.split('/')[-2] == 'atypical':
            label = [int(0)]
        if filename.split('/')[-2] == 'suspicious':
            label = [int(1)]

        label = torch.Tensor(label)

        if self.aug_mix:
            data_aug = self.mix_augmentation(filename.split('/')[-2], data)
            return data_aug, label
        else:
            return data, label

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

# Assuming CustomDataset is the original dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single data sample and its corresponding label
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


def build_dataset(is_train, args,nth_fold):
    print(is_train)
    transform = build_transform(is_train, args)

    if args.data_set == 'urine':
        if nth_fold > 0:
            ### VPU
            if args.features == 'VPU':
                root1 = os.path.join(args.data_path, 'VPU/scale128_'+str(nth_fold) , 'train' if is_train else 'test')
                root2 = os.path.join(args.data_path, 'VPU/scale256_'+str(nth_fold)  , 'train' if is_train else 'test')
            ### PLIP
            if args.features == 'PLIP':
                root1 = os.path.join(args.data_path, 'dinov2/scale128_'+str(nth_fold) , 'train' if is_train else 'test')
                root2 = os.path.join(args.data_path, 'dinov2/scale256_'+str(nth_fold)  , 'train' if is_train else 'test')
        else:
            ### VPU
            if args.features == 'VPU':
                root1 = os.path.join(args.data_path,'VPU/scale128_ep10'  , 'train'  if is_train else 'test')
                root2 = os.path.join(args.data_path,'VPU/scale256_ep10' , 'train'  if is_train else 'test')
            ### PLIP
            if args.features == 'PLIP':
                root1 = os.path.join(args.data_path,'dinov2/scale128'  , 'train'  if is_train else 'test')
                root2 = os.path.join(args.data_path,'dinov2/scale256' , 'train'  if is_train else 'test')

        dataset1 = EmbedDataset(dataset_path=root1,is_train=is_train)
        dataset2 = EmbedDataset(dataset_path=root2,is_train=is_train)

        # root = os.path.join(args.data_path, 'train'  if is_train else 'test')
        # dataset = UrineDataset(dataset_path=root, transform = transform)
        nb_classes = 2
    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform,download=True)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
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

    return dataset1,dataset2, nb_classes


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
        size = int(args.crop_ratio * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
