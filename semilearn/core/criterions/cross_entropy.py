# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch
import torch.nn as nn

from torch.nn import functional as F


def ce_loss(logits, targets, reduction='none', weight=None):
    """
    cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction, weight=weight)


class CELoss(nn.Module):
    """
    Wrapper for ce loss
    """
    def __init__(self, class_weights=None, focal_gamma=0.0):
        super().__init__()
        if class_weights:
            weights = [float(x.strip()) for x in class_weights.split(",") if x.strip()]
            self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))
        else:
            self.class_weights = None
        self.focal_gamma = float(focal_gamma)

    def forward(self, logits, targets, reduction='none'):
        weight = None
        if self.class_weights is not None and logits.shape != targets.shape:
            weight = self.class_weights.to(logits.device)
        if self.focal_gamma <= 0:
            return ce_loss(logits, targets, reduction, weight=weight)

        if logits.shape == targets.shape:
            raise NotImplementedError("Focal loss only supports hard labels in this repo.")

        log_pred = F.log_softmax(logits, dim=-1)
        log_pt = log_pred.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_weight = (1.0 - pt).pow(self.focal_gamma)
        if weight is not None:
            focal_weight = focal_weight * weight.gather(dim=0, index=targets)
        loss = -focal_weight * log_pt

        if reduction == 'none':
            return loss
        if reduction == 'sum':
            return loss.sum()
        return loss.mean()
