import argparse
import glob
import importlib.util
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


BUS_MEAN = [0.371, 0.371, 0.372]
BUS_STD = [0.167, 0.167, 0.167]


def load_resnet18_builder():
    path = os.path.join("semilearn", "nets", "resnet", "resnet.py")
    spec = importlib.util.spec_from_file_location("ssl_resnet18_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resnet18


def strip_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        new_state_dict[key] = value
    return new_state_dict


def build_loader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(BUS_MEAN, BUS_STD),
    ])
    dataset = ImageFolder(os.path.join(data_dir, "test"), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataset, loader


def evaluate(model, loader, device):
    probs_all = []
    labels_all = []
    preds_all = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            probs_all.append(probs.cpu().numpy())
            labels_all.append(labels.numpy())
            preds_all.append(preds.cpu().numpy())

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    preds = np.concatenate(preds_all)

    auc = roc_auc_score(labels, probs[:, 1])
    acc = accuracy_score(labels, preds)
    sen = recall_score(labels, preds, pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(labels, preds, zero_division=0)
    return {"AUC": auc, "ACC": acc, "Sen": sen, "Spe": spe, "F1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", required=True)
    parser.add_argument("--data_dir", default="../uda_data")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    ckpts = sorted(glob.glob(args.glob))
    if not ckpts:
        raise SystemExit("no checkpoints matched")

    resnet18 = load_resnet18_builder()
    _, loader = build_loader(args.data_dir, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for ckpt_path in ckpts:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = strip_module_prefix(checkpoint["ema_model"])
        model = resnet18(num_classes=2, pretrained=False, pretrained_path="")
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        metrics = evaluate(model, loader, device)
        rows.append((os.path.dirname(ckpt_path), metrics))

    for name, metrics in rows:
        metric_str = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        print(f"{name} {metric_str}")

    print("SUMMARY")
    for key in ["AUC", "ACC", "Sen", "Spe", "F1"]:
        values = np.array([row[1][key] for row in rows], dtype=float)
        print(f"{key} mean={values.mean():.6f} std={values.std(ddof=0):.6f}")


if __name__ == "__main__":
    main()
