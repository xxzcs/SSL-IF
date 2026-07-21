# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import gc
import copy
import json
import random
import torch
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset


mean, std = {}, {}
mean['bus'] = [0.371, 0.371, 0.372]
std['bus'] = [0.167, 0.167, 0.167]


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def get_bus(args, alg, name, num_labels, num_classes, data_dir='../uda_data', lpath='', ulpath='', include_lb_to_ulb=False):
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        # transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        # transforms.RandomCrop((img_size, img_size)),
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['bus'], std['bus'])
    ])

    transform_medium = transforms.Compose([
        # transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        # RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['bus'], std['bus'])
    ])

    transform_strong = transforms.Compose([
        # transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        # RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(2, 10),   #RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['bus'], std['bus'])
    ])

    transform_val = transforms.Compose([
        # transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['bus'], std['bus'])
    ])

    # data_dir = os.path.join(data_dir, name.lower())

    # dataset = BUSDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, ulb=False, alg=alg)
    # percentage = num_labels / len(dataset)

    lb_dset = BUSDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, ulb=False, alg=alg, db_path=lpath)

    ulb_dset = BUSDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, alg=alg, ulb=True, medium_transform=transform_medium, strong_transform=transform_strong, 
                          db_path=ulpath, include_lb_to_ulb=include_lb_to_ulb) #, lb_index=lb_dset.lb_idx)

    eval_dset = ImageFolder(os.path.join(data_dir, "val"), transform=transform_val)

    test_dset = ImageFolder(os.path.join(data_dir, "test"), transform=transform_val)

    return lb_dset, ulb_dset, eval_dset, test_dset

def load_db(db_path, class_to_idx):  
    db = torch.load(db_path)

    images = []

    for key in sorted(db.keys()):

        for image_path in db[key]:
            ori_image_path = '../' + image_path
            images.append((ori_image_path, class_to_idx[key]))
    return images

class BUSDataset(BasicDataset, ImageFolder):
    def __init__(self, root, transform, ulb, alg, medium_transform=None, strong_transform=None, db_path='', include_lb_to_ulb=False): #, lb_index=None):
        self.alg = alg
        self.is_ulb = ulb
        # self.percentage = percentage
        self.transform = transform
        self.root = root
        self.include_lb_to_ulb = include_lb_to_ulb
        # self.lb_index = lb_index
        self.db_path = db_path

        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self.find_classes(self.root)
        samples = load_db(self.db_path, class_to_idx)
        random.shuffle(samples)
        
        # samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data = [s[0] for s in samples]
        self.targets = [s[1] for s in samples]

        self.medium_transform = medium_transform
        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ['sequencematch'], f"alg {self.alg} requires strong augmentation"
        self.strong_transform = strong_transform
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch', 'refixmatch'], f"alg {self.alg} requires strong augmentation"

    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        target = self.targets[index]
        return sample, target

    def make_dataset(
            self,
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
    ):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)
        
        lb_idx = {}
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                random.shuffle(fnames)
                if self.percentage != -1:
                    fnames = fnames[:int(len(fnames) * self.percentage)]
                if self.percentage != -1:
                    lb_idx[target_class] = fnames
                for fname in fnames:
                    if not self.include_lb_to_ulb:
                        if fname in self.lb_index[target_class]:
                            continue
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        gc.collect()
        self.lb_idx = lb_idx
        return instances
    
    