# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.utils import SSL_Argument, str2bool


class MeanTeacherAP_Net(nn.Module):
    def __init__(self, base, proj_size=128, pred_size=128):
        super().__init__()
        self.backbone = base
        self.num_features = base.num_features
        self.projector = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.BatchNorm1d(self.num_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_features, proj_size),
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_size, pred_size),
            nn.BatchNorm1d(pred_size),
            nn.ReLU(inplace=True),
            nn.Linear(pred_size, proj_size),
        )

    def forward(self, x, use_predictor=True, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        proj = F.normalize(self.projector(feat), dim=1)
        pred = F.normalize(self.predictor(proj), dim=1) if use_predictor else proj
        return {"logits": logits, "feat": feat, "proj": proj, "pred": pred}

    def group_matcher(self, coarse=False):
        return self.backbone.group_matcher(coarse, prefix="backbone.")


@ALGORITHMS.register("meanteacher_ap")
class MeanTeacherAP(AlgorithmBase):
    """
    MeanTeacher + additional positive comparator for the MICCAI contrastive baseline.
    The implementation keeps the teacher-student EMA structure and adds:
    1. BYOL-style prediction loss between student strong view and teacher weak view.
    2. Additional-positive loss, where each unlabeled sample selects one labeled
       reference via closed-form last-layer TracIn and aligns to the teacher projection
       of that reference.
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.init(
            unsup_warm_up=args.unsup_warm_up,
            byol_loss_ratio=args.byol_loss_ratio,
            ap_loss_ratio=args.ap_loss_ratio,
            ap_temperature=args.ap_temperature,
            ap_warmup_epochs=args.ap_warmup_epochs,
            ap_topk=args.ap_topk,
            ap_target=args.ap_target,
            ap_mean_reduce=args.ap_mean_reduce,
        )

    def init(
        self,
        unsup_warm_up=0.4,
        byol_loss_ratio=1.0,
        ap_loss_ratio=0.5,
        ap_temperature=1.0,
        ap_warmup_epochs=0,
        ap_topk=1,
        ap_target="soft",
        ap_mean_reduce=True,
    ):
        self.unsup_warm_up = unsup_warm_up
        self.byol_loss_ratio = byol_loss_ratio
        self.ap_loss_ratio = ap_loss_ratio
        self.ap_temperature = ap_temperature
        self.ap_warmup_epochs = ap_warmup_epochs
        self.ap_topk = ap_topk
        self.ap_target = ap_target
        self.ap_mean_reduce = ap_mean_reduce

    def set_model(self):
        model = super().set_model()
        return MeanTeacherAP_Net(
            model,
            proj_size=self.args.proj_size,
            pred_size=self.args.pred_size,
        )

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = MeanTeacherAP_Net(
            ema_model,
            proj_size=self.args.proj_size,
            pred_size=self.args.pred_size,
        )
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model

    @staticmethod
    def _neg_cosine(p, z):
        z = z.detach()
        return 2 - 2 * (F.normalize(p, dim=1) * F.normalize(z, dim=1)).sum(dim=1).mean()

    def _ap_warmup_coef(self):
        if self.ap_warmup_epochs <= 0:
            return 1.0
        return 0.0 if self.epoch < self.ap_warmup_epochs else 1.0

    def _select_additional_positive(self, logits_lb, y_lb, logits_u, phi_lb, phi_u):
        num_cls = self.num_classes
        L = logits_lb.shape[0]
        U = logits_u.shape[0]
        if L == 0 or U == 0:
            return None

        p_l = F.softmax(logits_lb.detach(), dim=1)
        g_l = p_l - F.one_hot(y_lb, num_cls).float()

        p_u = F.softmax(logits_u.detach(), dim=1)
        if self.ap_target == "soft":
            g_u = p_u * logits_u.detach().sum(1, keepdim=True) - logits_u.detach()
        else:
            y_u = F.one_hot(p_u.argmax(1), num_cls).float()
            g_u = p_u - y_u

        if self.ap_mean_reduce:
            g_l = g_l / max(L, 1)
            g_u = g_u / max(U, 1)

        infl = (g_u @ g_l.t()) * (phi_u.detach() @ phi_lb.detach().t())
        if self.ap_temperature and self.ap_temperature != 1.0:
            infl = infl / self.ap_temperature

        topk = min(max(self.ap_topk, 1), L)
        top_idx = infl.topk(topk, dim=1).indices
        return top_idx[:, 0]

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        with self.amp_cm():
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb["logits"]
            feats_x_lb = outs_x_lb["feat"]
            proj_x_lb = outs_x_lb["proj"]

            self.ema.apply_shadow()
            with torch.no_grad():
                self.bn_controller.freeze_bn(self.model)
                outs_x_ulb_w_t = self.model(x_ulb_w, use_predictor=False)
                outs_x_lb_t = self.model(x_lb, use_predictor=False)
                self.bn_controller.unfreeze_bn(self.model)
            self.ema.restore()

            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb_s = self.model(x_ulb_s)
            self.bn_controller.unfreeze_bn(self.model)

            logits_x_ulb_w = outs_x_ulb_w_t["logits"]
            feats_x_ulb_w = outs_x_ulb_w_t["feat"]
            proj_x_ulb_w = outs_x_ulb_w_t["proj"]

            logits_x_ulb_s = outs_x_ulb_s["logits"]
            feats_x_ulb_s = outs_x_ulb_s["feat"]
            pred_x_ulb_s = outs_x_ulb_s["pred"]

            proj_x_lb_t = outs_x_lb_t["proj"]

            feat_dict = {
                "x_lb": feats_x_lb,
                "x_ulb_w": feats_x_ulb_w,
                "x_ulb_s": feats_x_ulb_s,
            }

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction="mean")
            unsup_loss = self.consistency_loss(
                logits_x_ulb_s,
                self.compute_prob(logits_x_ulb_w.detach()),
                "mse",
            )

            byol_loss = self._neg_cosine(pred_x_ulb_s, proj_x_ulb_w)

            ap_idx = self._select_additional_positive(
                logits_x_lb, y_lb, logits_x_ulb_w, feats_x_lb, feats_x_ulb_w
            )
            if ap_idx is None:
                ap_loss = pred_x_ulb_s.new_zeros(())
            else:
                ap_targets = proj_x_lb_t[ap_idx]
                ap_loss = self._neg_cosine(pred_x_ulb_s, ap_targets)

            unsup_warmup = np.clip(
                self.it / (self.unsup_warm_up * self.num_train_iter),
                a_min=0.0,
                a_max=1.0,
            )
            total_loss = (
                sup_loss
                + self.lambda_u * unsup_loss * unsup_warmup
                + self.byol_loss_ratio * byol_loss
                + self.ap_loss_ratio * self._ap_warmup_coef() * ap_loss
            )

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(
            sup_loss=sup_loss.item(),
            unsup_loss=unsup_loss.item(),
            byol_loss=byol_loss.item(),
            ap_loss=ap_loss.item(),
            total_loss=total_loss.item(),
        )
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument("--unsup_warm_up", float, 0.4, "warm up ratio for unsupervised loss"),
            SSL_Argument("--proj_size", int, 128),
            SSL_Argument("--pred_size", int, 128),
            SSL_Argument("--byol_loss_ratio", float, 1.0),
            SSL_Argument("--ap_loss_ratio", float, 0.5),
            SSL_Argument("--ap_temperature", float, 1.0),
            SSL_Argument("--ap_warmup_epochs", int, 0),
            SSL_Argument("--ap_topk", int, 1),
            SSL_Argument("--ap_target", str, "soft"),
            SSL_Argument("--ap_mean_reduce", str2bool, True),
        ]
