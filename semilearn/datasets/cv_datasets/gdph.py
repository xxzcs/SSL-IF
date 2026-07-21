# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import os
import random

from PIL import Image
from torchvision import transforms

from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset


mean, std = {}, {}
mean['gdph'] = [0.485, 0.456, 0.406]
std['gdph'] = [0.229, 0.224, 0.225]


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


def _get_row_value(row, candidates):
    lowered_map = {str(key).replace('\ufeff', '').strip().lower(): value for key, value in row.items()}
    for candidate in candidates:
        if candidate.lower() in lowered_map:
            return lowered_map[candidate.lower()]
    raise KeyError(f"Cannot find any of columns {candidates} in csv row with keys {list(row.keys())}")


def _resolve_image_path(img_dir, image_id):
    image_id = str(image_id).strip()
    direct_path = os.path.join(img_dir, image_id)
    if os.path.exists(direct_path):
        return direct_path

    _, ext = os.path.splitext(image_id)
    if not ext:
        for suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp']:
            candidate = os.path.join(img_dir, f"{image_id}{suffix}")
            if os.path.exists(candidate):
                return candidate

    raise FileNotFoundError(f"Cannot locate image for id '{image_id}' under {img_dir}")


def _read_csv(csv_path, img_dir):
    samples = []
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = _get_row_value(row, ['ID'])
            fold = int(_get_row_value(row, ['fold']))
            label = int(_get_row_value(row, ['label']))
            img_path = _resolve_image_path(img_dir, img_id)
            samples.append((img_path, label, fold))
    return samples


def _split_by_fold(samples, fold_id):
    train = []
    val = []
    for path, label, fold in samples:
        if fold == fold_id:
            val.append((path, label))
        else:
            train.append((path, label))
    return train, val


def _stratified_sample(samples, num_labels, num_classes, seed=0):
    random.seed(seed)
    class_to_items = {class_idx: [] for class_idx in range(num_classes)}
    for index, (_, label) in enumerate(samples):
        class_to_items[label].append(index)

    per_class = max(1, num_labels // num_classes)
    selected = set()
    for class_idx in range(num_classes):
        random.shuffle(class_to_items[class_idx])
        selected.update(class_to_items[class_idx][:per_class])

    if len(selected) < num_labels:
        remain = [index for index in range(len(samples)) if index not in selected]
        random.shuffle(remain)
        selected.update(remain[: num_labels - len(selected)])

    return sorted(list(selected))


def get_gdph(args, alg, name, num_labels, num_classes, data_dir='../uda_data/GDPH', include_lb_to_ulb=False):
    img_size = args.img_size
    fold_id = getattr(args, 'fold', 1)

    transform_weak = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['gdph'], std['gdph'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(2, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['gdph'], std['gdph'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['gdph'], std['gdph'])
    ])

    csv_path = os.path.join(data_dir, 'label.csv')
    img_dir = os.path.join(data_dir, 'img')
    all_samples = _read_csv(csv_path, img_dir)

    train_samples, val_samples = _split_by_fold(all_samples, fold_id)


    label_ratio = getattr(args, 'label_ratio', None)
    if label_ratio is not None and label_ratio == 1.0:
        lb_samples = train_samples
    else:
        if label_ratio is not None:
            actual_num_labels = int(len(train_samples) * label_ratio)
        elif num_labels < 1.0:
            actual_num_labels = int(len(train_samples) * num_labels)
        else:
            actual_num_labels = int(num_labels)

        actual_num_labels = max(actual_num_labels, num_classes)

        split_seed = getattr(args, 'split_seed', None)
        if split_seed is None:
            split_seed = args.seed

        lb_idx = _stratified_sample(train_samples, actual_num_labels, num_classes, seed=split_seed)
        lb_samples = [train_samples[index] for index in lb_idx]

    if include_lb_to_ulb:
        ulb_samples = train_samples
    else:
        if label_ratio is not None and label_ratio == 1.0:
            ulb_samples = []
        else:
            lb_idx_set = set(lb_idx)
            ulb_samples = [sample for index, sample in enumerate(train_samples) if index not in lb_idx_set]

    lb_dset = GDPHDataset(lb_samples, transform_weak, ulb=False, alg=alg)
    ulb_dset = GDPHDataset(ulb_samples, transform_weak, ulb=True, alg=alg, strong_transform=transform_strong)
    eval_dset = GDPHDataset(val_samples, transform_val, ulb=False, alg=alg)
    test_dset = None

    return lb_dset, ulb_dset, eval_dset, test_dset


class GDPHDataset(BasicDataset):
    def __init__(self, samples, transform, ulb, alg, strong_transform=None):
        self.alg = alg
        self.is_ulb = ulb
        self.transform = transform
        self.strong_transform = strong_transform

        self.data = [sample[0] for sample in samples]
        self.targets = [sample[1] for sample in samples]

        self.loader = default_loader

    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        target = self.targets[index]
        return sample, target