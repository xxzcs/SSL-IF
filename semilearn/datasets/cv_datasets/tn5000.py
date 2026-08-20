import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict

from PIL import Image, ImageEnhance, ImageOps
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

    selected = set()
    total_count = len(samples)
    if total_count == 0:
        return []

    # Keep the original class prior in the labeled subset instead of forcing
    # a class-balanced labeled split. This is the protocol locked for TN5000.
    raw_targets = {}
    floor_targets = {}
    remainders = []
    for class_idx in range(num_classes):
        class_count = len(class_to_items[class_idx])
        raw_target = (class_count / total_count) * num_labels
        floor_target = min(class_count, int(raw_target))
        raw_targets[class_idx] = raw_target
        floor_targets[class_idx] = floor_target
        remainders.append((raw_target - floor_target, class_idx))

    allocated = sum(floor_targets.values())
    remaining_budget = min(num_labels, total_count) - allocated

    for _, class_idx in sorted(remainders, reverse=True):
        if remaining_budget <= 0:
            break
        if floor_targets[class_idx] >= len(class_to_items[class_idx]):
            continue
        floor_targets[class_idx] += 1
        remaining_budget -= 1

    if allocated == 0 and num_labels >= num_classes:
        # Safety fallback for extremely small label budgets.
        for class_idx in range(num_classes):
            if class_to_items[class_idx]:
                floor_targets[class_idx] = min(
                    len(class_to_items[class_idx]),
                    max(1, floor_targets[class_idx]),
                )

    for class_idx in range(num_classes):
        rng.shuffle(class_to_items[class_idx])
        selected.update(class_to_items[class_idx][:floor_targets[class_idx]])

    if len(selected) < num_labels:
        remain = [index for index in range(len(samples)) if index not in selected]
        rng.shuffle(remain)
        selected.update(remain[: num_labels - len(selected)])

    return sorted(list(selected))


def _sample_exact_count(indices, target_count, seed):
    rng = random.Random(seed)
    if target_count <= 0:
        return []
    indices = list(indices)
    rng.shuffle(indices)
    return indices[: min(target_count, len(indices))]


def _oversample_to_class_max(samples, num_classes, seed=0):
    rng = random.Random(seed)
    class_to_samples = {class_idx: [] for class_idx in range(num_classes)}
    for sample in samples:
        class_to_samples[sample[1]].append(sample)

    class_sizes = [len(class_to_samples[class_idx]) for class_idx in range(num_classes)]
    target_size = max(class_sizes) if class_sizes else 0
    balanced = []
    for class_idx in range(num_classes):
        items = list(class_to_samples[class_idx])
        if not items:
            continue
        if len(items) < target_size:
            items.extend(rng.choices(items, k=target_size - len(items)))
        balanced.extend(items)
    rng.shuffle(balanced)
    return balanced


def _make_offline_augmented_sample(sample, aug_name):
    path, label = sample[:2]
    return (path, label, aug_name)


def _minority_augmented_variants(sample):
    return [
        _make_offline_augmented_sample(sample, 'hflip'),
        _make_offline_augmented_sample(sample, 'rot_p7'),
        _make_offline_augmented_sample(sample, 'rot_m7_bright'),
    ]


def _build_balanced_dup_counts(samples, num_classes, seed=0):
    rng = random.Random(seed)
    class_to_indices = {class_idx: [] for class_idx in range(num_classes)}
    for index, (_, label) in enumerate(samples):
        class_to_indices[label].append(index)

    dup_counts = defaultdict(int)
    class_counts = {class_idx: len(class_to_indices[class_idx]) for class_idx in range(num_classes)}
    target_size = max(class_counts.values()) if class_counts else 0

    for class_idx in range(num_classes):
        indices = list(class_to_indices[class_idx])
        if not indices:
            continue
        rng.shuffle(indices)
        base = target_size // len(indices)
        rem = target_size % len(indices)
        for pos, index in enumerate(indices):
            dup_counts[index] = base + (1 if pos < rem else 0)
    return dup_counts, target_size


def _split_balanced_train_then_partition(samples, num_classes, label_ratio, seed=0):
    rng = random.Random(seed)
    dup_counts, per_class_target = _build_balanced_dup_counts(samples, num_classes, seed)
    label_budget_total = int((per_class_target * num_classes) * label_ratio)
    label_budget_total = max(label_budget_total, num_classes)

    per_class_budget = {class_idx: label_budget_total // num_classes for class_idx in range(num_classes)}
    for class_idx in range(label_budget_total % num_classes):
        per_class_budget[class_idx] += 1

    class_to_indices = {class_idx: [] for class_idx in range(num_classes)}
    for index, (_, label) in enumerate(samples):
        class_to_indices[label].append(index)

    lb_base_indices = set()
    for class_idx in range(num_classes):
        indices = list(class_to_indices[class_idx])
        rng.shuffle(indices)
        target = per_class_budget[class_idx]
        selected = []
        cumulative = 0
        for index in indices:
            if cumulative >= target:
                break
            candidate = dup_counts[index]
            if not selected:
                selected.append(index)
                cumulative += candidate
                continue
            if abs((cumulative + candidate) - target) <= abs(cumulative - target):
                selected.append(index)
                cumulative += candidate
        if not selected and indices:
            selected = [indices[0]]
        lb_base_indices.update(selected)

    lb_samples = []
    ulb_samples = []
    for index, sample in enumerate(samples):
        copies = max(1, dup_counts[index])
        if index in lb_base_indices:
            lb_samples.extend([sample] * copies)
        else:
            ulb_samples.extend([sample] * copies)

    rng.shuffle(lb_samples)
    rng.shuffle(ulb_samples)
    return lb_samples, ulb_samples


def _build_augmented_balanced_groups(samples, num_classes, seed=0):
    rng = random.Random(seed)
    class_to_indices = {class_idx: [] for class_idx in range(num_classes)}
    for index, (_, label) in enumerate(samples):
        class_to_indices[label].append(index)

    class_counts = {class_idx: len(class_to_indices[class_idx]) for class_idx in range(num_classes)}
    target_size = max(class_counts.values()) if class_counts else 0
    groups = {}

    for index, sample in enumerate(samples):
        groups[index] = [sample]

    for class_idx in range(num_classes):
        indices = list(class_to_indices[class_idx])
        class_count = len(indices)
        if class_count == 0 or class_count >= target_size:
            continue

        variants_by_index = {
            index: _minority_augmented_variants(samples[index])
            for index in indices
        }
        gap = target_size - class_count
        expanded = []
        while len(expanded) < gap:
            shuffled = list(indices)
            rng.shuffle(shuffled)
            for index in shuffled:
                for variant in variants_by_index[index]:
                    expanded.append((index, variant))
                    if len(expanded) >= gap:
                        break
                if len(expanded) >= gap:
                    break

        for index, variant in expanded:
            groups[index].append(variant)

    return groups, target_size


def _split_augmented_balanced_train_then_partition(samples, num_classes, label_ratio, seed=0):
    rng = random.Random(seed)
    groups, per_class_target = _build_augmented_balanced_groups(samples, num_classes, seed)
    label_budget_total = int((per_class_target * num_classes) * label_ratio)
    label_budget_total = max(label_budget_total, num_classes)

    per_class_budget = {class_idx: label_budget_total // num_classes for class_idx in range(num_classes)}
    for class_idx in range(label_budget_total % num_classes):
        per_class_budget[class_idx] += 1

    class_to_indices = {class_idx: [] for class_idx in range(num_classes)}
    for index, (_, label) in enumerate(samples):
        class_to_indices[label].append(index)

    lb_base_indices = set()
    for class_idx in range(num_classes):
        indices = list(class_to_indices[class_idx])
        rng.shuffle(indices)
        target = per_class_budget[class_idx]
        selected = []
        cumulative = 0
        for index in indices:
            if cumulative >= target:
                break
            candidate = len(groups[index])
            if not selected:
                selected.append(index)
                cumulative += candidate
                continue
            if abs((cumulative + candidate) - target) <= abs(cumulative - target):
                selected.append(index)
                cumulative += candidate
        if not selected and indices:
            selected = [indices[0]]
        lb_base_indices.update(selected)

    lb_samples = []
    ulb_samples = []
    for index, group_samples in groups.items():
        if index in lb_base_indices:
            lb_samples.extend(group_samples)
        else:
            ulb_samples.extend(group_samples)

    rng.shuffle(lb_samples)
    rng.shuffle(ulb_samples)
    return lb_samples, ulb_samples


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

    balance_mode = getattr(args, 'tn5000_balance_mode', 'none')
    if balance_mode == 'full_train_balanced_split':
        effective_label_ratio = getattr(args, 'label_ratio', None)
        if effective_label_ratio is None:
            effective_label_ratio = actual_num_labels / max(len(train_samples), 1)
        lb_samples, ulb_samples = _split_balanced_train_then_partition(
            train_samples, num_classes, effective_label_ratio, seed=split_seed
        )
    elif balance_mode == 'full_train_aug_balanced_split':
        effective_label_ratio = getattr(args, 'label_ratio', None)
        if effective_label_ratio is None:
            effective_label_ratio = actual_num_labels / max(len(train_samples), 1)
        lb_samples, ulb_samples = _split_augmented_balanced_train_then_partition(
            train_samples, num_classes, effective_label_ratio, seed=split_seed
        )
    else:
        lb_idx = _stratified_sample(train_samples, actual_num_labels, num_classes, seed=split_seed)
        lb_samples = [train_samples[index] for index in lb_idx]

        if balance_mode == 'label_balanced':
            lb_samples = _oversample_to_class_max(lb_samples, num_classes, seed=split_seed)

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
        self.sample_meta = [sample[2] if len(sample) > 2 else None for sample in samples]

        self.loader = default_loader

    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        aug_name = None
        if hasattr(self, 'sample_meta') and index < len(self.sample_meta):
            aug_name = self.sample_meta[index]
        if aug_name is not None:
            sample = self.apply_offline_augmentation(sample, aug_name)
        target = self.targets[index]
        return sample, target

    def apply_offline_augmentation(self, sample, aug_name):
        if aug_name == 'hflip':
            return ImageOps.mirror(sample)
        if aug_name == 'rot_p7':
            return sample.rotate(7, resample=Image.BILINEAR)
        if aug_name == 'rot_m7_bright':
            rotated = sample.rotate(-7, resample=Image.BILINEAR)
            return ImageEnhance.Brightness(rotated).enhance(1.08)
        return sample
