"""
Partly imported from

https://github.com/YU1ut/MixMatch-pytorch/blob/master/dataset/cifar10.py
"""
import glob
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.folder import *
from typing import *
from sklearn.model_selection import GroupKFold
mean = [0.8615, 0.8761, 0.9014]
std = [0.1821, 0.1439, 0.0756]
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
        #增加了这下面这句
        classes = [valid_class for valid_class in classes if valid_class in self.valid_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

class UrineSlideDataset(Dataset):
    def __init__(self, dataset_path=''):
        train_data_path = dataset_path
        self.imagenet = datasets.ImageFolder(root=train_data_path)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=0),
             # transforms.Pad(4),
             # transforms.RandomCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             # -- cropped cell --
             # transforms.Normalize(mean=[0.4454, 0.8467, 1.4581], std=[1.5059, 1.4546, 1.3584])
             # -- tile --
             transforms.Normalize(mean=mean, std=std)
             ])
    def __len__(self):
        return len(self.imagenet)
    def __getitem__(self, index):
        image_tensor = self.transform(self.imagenet[index][0])
        label_tensor = int(self.imagenet[index][1])
        return image_tensor, label_tensor


class EmbedDataset(Dataset):
    def __init__(self, dataset_path=''):
        real_root_list = dataset_path.split('/')[:-1]
        self.real_root = '/'.join(real_root_list)
        self.filenames = glob.glob(dataset_path + '/*' + '/BD*')
        self.filenames = sorted(self.filenames )
        self.EmbedList = []
        for name in self.filenames:
            if (all(chr.isdigit() for chr in name.split('-')[-1].split('.')[0])):
                self.EmbedList.append(name)

    def __len__(self):
        return len(self.EmbedList)

    def __getitem__(self, index):
        filename = self.EmbedList[index]

        data = np.load(filename)
        if filename.split('/')[-4] == 'scale256to128':
            k = 400
        if filename.split('/')[-4] == 'scale256' or filename.split('/')[-4] == 'scale128':
            k = 100
        else:
            k=100
        while data.shape[0] < k:
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
        return data, label

class urine_labeled_dataset(Dataset):
    """
       Code for reading the ImageNet
       """

    def __init__(self, dataset_path='', target_transform=None, data_class = None):
        self.target_transform = target_transform
        self.data_class = data_class
        # train_data_path = os.path.join(dataset_path, 'train')
        train_data_path = dataset_path
        if data_class == 'p':
            self.imagenet = FilterableImageFolder(root=train_data_path, valid_classes=['benign'])
        elif data_class == 'u':
            self.imagenet = FilterableImageFolder(root=train_data_path, valid_classes=['cancer'])
        else:
            # self.imagenet = datasets.ImageFolder(root=train_data_path)
            self.imagenet = FilterableImageFolder(root=train_data_path,valid_classes=['benign','cancer'])
        self.transform = transforms.Compose(
             [transforms.Resize((256, 256), interpolation=0),
             #   transforms.Pad(4),
             #   transforms.RandomCrop(224),
               transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
              # transforms.Normalize(mean=[0.4454, 0.8467, 1.4581], std=[1.5059, 1.4546, 1.3584])
              # -- tile --
              transforms.Normalize(mean=mean, std=std)
             ])

    def __len__(self):
        return len(self.imagenet)

    def __getitem__(self, index):
        image_tensor = self.transform(self.imagenet[index][0])
        label_tensor = int(self.imagenet[index][1])
        if self.target_transform is not None:
            label_tensor = self.target_transform(label_tensor)
        if self.data_class == 'p':
            label_tensor = label_tensor
            # print('p',label_tensor)
        elif self.data_class == 'u':
            label_tensor = -1
            # print('u', label_tensor)
        return image_tensor, label_tensor


class UrineSlideDataset(Dataset):
    """
       Code for reading the ImageNet
       """

    def __init__(self, dataset_path='',target_transform=None):
        self.target_transform = target_transform
        # train_data_path = os.path.join(dataset_path, 'train')
        self.dataset_path = dataset_path
        self.imagenet = datasets.ImageFolder(root=dataset_path)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=0),
            # transforms.Pad(4),
            # transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    def __len__(self):
        return len(self.imagenet)

    def __getitem__(self, index):
        image_tensor = self.transform(self.imagenet[index][0])
        # label_tensor = int(self.imagenet[index][1])
        if self.dataset_path.split('/')[-2] == 'cancer':
            label_tensor = [int(0)]
        if self.dataset_path.split('/')[-2] == 'benign':
            label_tensor = [int(1)]
        if self.dataset_path.split('/')[-2] == 'atypical':
            label_tensor = [int(1)]
        if self.dataset_path.split('/')[-2] == 'suspicious':
            label_tensor = [int(0)]
        label_tensor = torch.Tensor(label_tensor)
        return image_tensor, label_tensor


def get_val_labeled(labels, val_idxs, n_val_labeled, positive_label_list):
    val_labeled_idxs = []
    np.random.shuffle(val_idxs)
    labels = np.array(labels)
    n_labeled_per_class = int(n_val_labeled / len(positive_label_list))
    for i in positive_label_list:
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        val_labeled_idxs.extend(idxs[0:n_labeled_per_class])
    return val_labeled_idxs


def train_val_split(labels, n_labeled, positive_label_list):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []
    n_labeled_per_class = int(n_labeled / len(positive_label_list))

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        if i in positive_label_list:
            train_labeled_idxs.extend(idxs[:n_labeled_per_class])
            train_unlabeled_idxs.extend(idxs[0:-500])
        else:
            train_unlabeled_idxs.extend(idxs[0:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


def transpose(x, source='NHWC', target='NCHW'):
    '''
    N: batch size
    H: height
    W: weight
    C: channel
    '''
    return x.transpose([source.index(d) for d in target])




class urine_unlabeled_dataset(urine_labeled_dataset):

    def __init__(self, root, data_class=None,target_transform=None):
        super(urine_unlabeled_dataset, self).__init__(root, data_class=data_class,target_transform=target_transform)
        if data_class == 'u':
            self.label_tensor = np.array([-1 for i in range(len(self.label_tensor))])
        else:
            pass

def get_urine_data(positive_label_list,nth_fold):
    target_transform = lambda x: 1 if x in positive_label_list else 0
    if nth_fold > 0:
        train_labeled_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128_rand100_new_kfold/'+str(nth_fold)+'/train', data_class='p', target_transform=target_transform)
        train_unlabeled_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128_rand100_new_kfold/'+str(nth_fold)+'/train', data_class ='u', target_transform=target_transform)
        val_unlabeled_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128_rand100_new_kfold/'+str(nth_fold)+'/test', data_class ='u', target_transform=target_transform)
        val_labeled_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128_rand100_new_kfold/'+str(nth_fold)+'/test', data_class ='p',target_transform=target_transform)
        test_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128_rand100_new_kfold/'+str(nth_fold)+'/test',target_transform=target_transform)
    else:
        train_labeled_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128to256_rand100_new/train',data_class='p', target_transform=target_transform)
        train_unlabeled_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128to256_rand100_new/train', data_class='u', target_transform=target_transform)
        val_unlabeled_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128to256_rand100_new/test',data_class='u', target_transform=target_transform)
        val_labeled_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128to256_rand100_new/test',data_class='p', target_transform=target_transform)
        test_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128to256_rand100_new/test',target_transform=target_transform)

    return train_labeled_dataset, train_unlabeled_dataset, val_labeled_dataset, val_unlabeled_dataset, test_dataset

def get_urine_loaders(positive_label_list, batch_size=500,nth_fold=''):
    train_labeled_dataset, train_unlabeled_dataset, val_labeled_dataset, val_unlabeled_dataset, test_dataset = get_urine_data(positive_label_list=positive_label_list,nth_fold=nth_fold)
    p_loader = DataLoader(dataset=train_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    x_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_p_loader = DataLoader(dataset=val_labeled_dataset, batch_size=batch_size, shuffle=False)
    val_x_loader = DataLoader(dataset=val_unlabeled_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return x_loader, p_loader, val_x_loader, val_p_loader, test_loader

def get_urine_data_inference(batch_size=500,nth_fold=''):
    train_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128to256_rand100_new/train',target_transform=target_transform)
    test_dataset = urine_labeled_dataset('/bigdata/projects/beidi/data/tile128to256_rand100_new/test',target_transform=target_transform)
    return train_dataset, test_dataset

def get_urine_loaders_inference(positive_label_list, batch_size=500,nth_fold=''):
    train_labeled_dataset, train_unlabeled_dataset, val_labeled_dataset, val_unlabeled_dataset, test_dataset = get_urine_data_inference(nth_fold=nth_fold)
    x_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return x_loader, test_loader
