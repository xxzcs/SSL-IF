import os
import random
import xml.etree.ElementTree as ET

from PIL import Image
from torchvision import transforms

from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset


mean, std = {}, {}
mean['tn5000'] = [0.230, 0.230, 0.231]
std['tn5000'] = [0.196, 0.196, 0.196]


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


def get_label(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    names = [obj.find('name').text for obj in root.findall('object') if obj.find('name') is not None]
    if not names:
        return -1

    name = str(names[0]).strip().lower()
    if name in {'0', 'benign'}:
        return 0
    if name in {'1', 'malignant'}:
        return 1
    return -1


def _read_tn5000_split(txt_path, img_dir, ann_dir):
    samples = []
    if not os.path.exists(txt_path):
        return samples

    with open(txt_path, 'r') as f:
        img_ids = [line.strip() for line in f.readlines() if line.strip()]

    for img_id in img_ids:
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        xml_path = os.path.join(ann_dir, f'{img_id}.xml')
        if not os.path.exists(img_path) or not os.path.exists(xml_path):
            continue

        label = get_label(xml_path)
        if label != -1:
            samples.append((img_path, label))

    return samples


def _stratified_sample(samples, num_labels, num_classes, seed=0):
    rng = random.Random(seed)
    class_to_items = {class_idx: [] for class_idx in range(num_classes)}
    for index, (_, label) in enumerate(samples):
        class_to_items[label].append(index)

    per_class = max(1, num_labels // num_classes)
    selected = set()
    for class_idx in range(num_classes):
        rng.shuffle(class_to_items[class_idx])
        selected.update(class_to_items[class_idx][:per_class])

    if len(selected) < num_labels:
        remain = [index for index in range(len(samples)) if index not in selected]
        rng.shuffle(remain)
        selected.update(remain[: num_labels - len(selected)])

    return sorted(list(selected))


def get_tn5000(args, alg, name, num_labels, num_classes, data_dir='../uda_data/TN5000', include_lb_to_ulb=False):
    img_size = args.img_size

    transform_weak = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['tn5000'], std['tn5000'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(2, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['tn5000'], std['tn5000'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['tn5000'], std['tn5000'])
    ])

    img_dir = os.path.join(data_dir, 'JPEGImages')
    ann_dir = os.path.join(data_dir, 'Annotations')
    train_txt = os.path.join(data_dir, 'ImageSets', 'Main', 'train.txt')
    val_txt = os.path.join(data_dir, 'ImageSets', 'Main', 'val.txt')
    test_txt = os.path.join(data_dir, 'ImageSets', 'Main', 'test.txt')

    train_samples = _read_tn5000_split(train_txt, img_dir, ann_dir)
    val_samples = _read_tn5000_split(val_txt, img_dir, ann_dir)
    test_samples = _read_tn5000_split(test_txt, img_dir, ann_dir)

    label_ratio = getattr(args, 'label_ratio', None)
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
        lb_idx_set = set(lb_idx)
        ulb_samples = [sample for index, sample in enumerate(train_samples) if index not in lb_idx_set]

    lb_dset = TN5000Dataset(lb_samples, transform_weak, ulb=False, alg=alg)
    ulb_dset = TN5000Dataset(ulb_samples, transform_weak, ulb=True, alg=alg, strong_transform=transform_strong)
    eval_dset = TN5000Dataset(val_samples, transform_val, ulb=False, alg=alg)
    test_dset = TN5000Dataset(test_samples, transform_val, ulb=False, alg=alg)

    return lb_dset, ulb_dset, eval_dset, test_dset


class TN5000Dataset(BasicDataset):
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
