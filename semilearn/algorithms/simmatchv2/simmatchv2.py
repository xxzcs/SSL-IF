# SimMatchV2 (ICCV2023) ported into the USB framework, single-GPU (non-DDP).
# Graph-consistency math follows the official SimMatchV2-main/models/simmatchv2.py.
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignQueueHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument


class SimMatchV2_Net(nn.Module):
    def __init__(self, base, proj_size=128):
        super(SimMatchV2_Net, self).__init__()
        self.backbone = base
        self.num_features = base.num_features
        self.mlp_proj = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(inplace=False),
            nn.Linear(self.num_features, proj_size),
        )

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        feat_proj = F.normalize(self.mlp_proj(feat))
        return {'logits': logits, 'feat': feat_proj}

    def group_matcher(self, coarse=False):
        return self.backbone.group_matcher(coarse, prefix='backbone.')


@ALGORITHMS.register('simmatchv2')
class SimMatchV2(AlgorithmBase):
    """
    SimMatchV2 (https://arxiv.org/abs/2308.06692) — graph consistency.
    单卡(非DDP)实现, 图一致性数学与官方 SimMatchV2-main 一致.
    新增超参: t, alpha, topn, K(无标注队列长度), lambda_ee, lambda_ne,
    proj_size, p_cutoff, da_len. (lambda_nn 复用 ulb_loss_ratio)
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.t = args.t
        self.alpha = args.alpha
        self.topn = args.topn
        self.p_cutoff = args.p_cutoff
        self.proj_size = args.proj_size
        self.lambda_nn = args.ulb_loss_ratio
        self.lambda_ee = args.lambda_ee
        self.lambda_ne = args.lambda_ne
        self.da_len = args.da_len
        # 标注集大小 (USB 框架在构建数据后写入 lb_dest_len)
        self.n_label = args.lb_dest_len
        # 无标注队列长度 K: 需可被 (batch_size*uratio) 整除
        bu = args.batch_size * args.uratio
        self.K = max(bu, (args.K // bu) * bu) if args.K >= bu else bu

        # 标注 bank: 特征 + onehot 标签 (按标注样本 index 索引)
        self.l_bank = F.normalize(torch.randn(self.n_label, self.proj_size).cuda(self.gpu), dim=1)
        self.l_labels = torch.zeros(self.n_label, self.num_classes).cuda(self.gpu)
        # 无标注队列: 特征 + 预测概率
        self.u_bank = F.normalize(torch.randn(self.K, self.proj_size).cuda(self.gpu), dim=1)
        self.u_labels = torch.ones(self.K, self.num_classes).cuda(self.gpu) / self.num_classes
        self.u_ptr = 0

    def set_hooks(self):
        # DA target distribution: 'uniform' (default, original behaviour) forces ulb
        # pseudo-labels toward 1/num_classes; 'gt' aligns to a known true class prior
        # passed via --da_p_target (e.g. "0.368,0.632" for GDPH); 'model' tracks the
        # labeled-batch prior online.
        da_p_target_type = getattr(self.args, 'da_p_target_type', 'uniform')
        da_p_target = getattr(self.args, 'da_p_target', None)
        if da_p_target_type == 'gt':
            assert da_p_target is not None, "--da_p_target must be set when da_p_target_type='gt'"
            if isinstance(da_p_target, str):
                da_p_target = np.array([float(v) for v in da_p_target.split(',')], dtype=np.float32)
        else:
            da_p_target = None
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len,
                               p_target_type=da_p_target_type, p_target=da_p_target),
            "DistAlignHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = SimMatchV2_Net(model, proj_size=self.args.proj_size)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = SimMatchV2_Net(ema_model, proj_size=self.args.proj_size)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    @torch.no_grad()
    def _update_u_bank(self, feat, prob):
        bs = feat.shape[0]
        assert self.K % bs == 0, f"K({self.K}) 不能被无标注 batch({bs}) 整除"
        self.u_bank[self.u_ptr:self.u_ptr + bs] = feat.detach()
        self.u_labels[self.u_ptr:self.u_ptr + bs] = prob.detach()
        self.u_ptr = (self.u_ptr + bs) % self.K

    @torch.no_grad()
    def _update_l_bank(self, feat, y, index):
        self.l_bank[index] = feat.detach()
        self.l_labels[index] = F.one_hot(y, num_classes=self.num_classes).float()

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        idx_lb = idx_lb.cuda(self.gpu)

        with self.amp_cm():
            l_bank = self.l_bank.clone().detach()
            l_labels = self.l_labels.clone().detach()
            u_bank = self.u_bank.clone().detach()
            u_labels = self.u_labels.clone().detach()

            # ---- teacher (EMA) 弱视图: 特征/概率 + 图传播伪标签 ----
            self.ema.apply_shadow()
            with torch.no_grad():
                out_k = self.model(torch.cat([x_lb, x_ulb_w]))
                logits_k, feat_k = out_k['logits'], out_k['feat']
                feat_kx = feat_k[:num_lb]
                feat_ku = feat_k[num_lb:]
                prob_ku = F.softmax(logits_k[num_lb:], dim=1)
                if getattr(self.args, 'use_da', True):
                    prob_ku = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=prob_ku.detach())

                # 边-边关系 (对无标注队列)
                relation_ku = F.softmax((feat_ku @ u_bank.T) / self.t, dim=-1)
                # 图传播得到伪标签 (用标注 bank 的 topn 近邻)
                topn = min(self.topn, l_bank.shape[0])
                l_sim_index = torch.topk(feat_ku @ l_bank.T, k=topn, largest=True, sorted=False, dim=-1)[1].flatten()
                l_nn_feat = l_bank[l_sim_index].reshape(num_ulb, topn, -1)
                l_nn_label = l_labels[l_sim_index].reshape(num_ulb, topn, -1)
                l_cat_feat = torch.cat([feat_ku.unsqueeze(1), l_nn_feat], dim=1)
                l_cat_label = torch.cat([prob_ku.unsqueeze(1), l_nn_label], dim=1)
                masks = torch.eye(topn + 1, device=feat_ku.device)
                A = l_cat_feat @ l_cat_feat.transpose(-2, -1) / self.t - masks * 1e9
                A = F.softmax(A, dim=-1)
                A = (1 - self.alpha) * torch.inverse(masks - self.alpha * A)
                pseudo_label = (A @ l_cat_label)[:, 0]
            self.ema.restore()

            # ---- student 强视图 ----
            out_q = self.model(torch.cat([x_lb, x_ulb_s]))
            logits_q, feat_q = out_q['logits'], out_q['feat']
            logits_qx = logits_q[:num_lb]
            logits_qu = logits_q[num_lb:]
            feat_qu = feat_q[num_lb:]

            sup_loss = self.ce_loss(logits_qx, y_lb, reduction='mean')

            # 图一致性损失
            relation_qu = F.softmax((feat_qu @ u_bank.T) / self.t, dim=1)
            loss_ee = torch.sum(-relation_qu.log() * relation_ku.detach(), dim=1).mean()
            nn_qu = relation_qu @ u_labels
            loss_ne = torch.sum(-nn_qu.log() * prob_ku.detach(), dim=1).mean()
            if torch.isnan(loss_ee) or torch.isinf(loss_ee):
                loss_ee = torch.zeros(1, device=feat_qu.device).mean()
            if torch.isnan(loss_ne) or torch.isinf(loss_ne):
                loss_ne = torch.zeros(1, device=feat_qu.device).mean()

            # 伪标签一致性 (强视图分类), 阈值掩码
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=pseudo_label, softmax_x_ulb=False)
            nn_loss = self.consistency_loss(logits_qu, pseudo_label, 'ce', mask=mask)

            total_loss = sup_loss + self.lambda_nn * nn_loss + self.lambda_ee * loss_ee + self.lambda_ne * loss_ne

            # 更新两个 bank
            self._update_l_bank(feat_kx, y_lb, idx_lb)
            self._update_u_bank(feat_ku, prob_ku)

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         nn_loss=nn_loss.item(),
                                         loss_ee=loss_ee.item(),
                                         loss_ne=loss_ne.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['l_bank'] = self.l_bank.cpu()
        save_dict['l_labels'] = self.l_labels.cpu()
        save_dict['u_bank'] = self.u_bank.cpu()
        save_dict['u_labels'] = self.u_labels.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.l_bank = checkpoint['l_bank'].cuda(self.gpu)
        self.l_labels = checkpoint['l_labels'].cuda(self.gpu)
        self.u_bank = checkpoint['u_bank'].cuda(self.gpu)
        self.u_labels = checkpoint['u_labels'].cuda(self.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--t', float, 0.1),
            SSL_Argument('--alpha', float, 0.1),
            SSL_Argument('--topn', int, 128),
            SSL_Argument('--K', int, 2560),
            SSL_Argument('--proj_size', int, 128),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--lambda_ee', float, 5.0),
            SSL_Argument('--lambda_ne', float, 5.0),
            SSL_Argument('--da_len', int, 256),
            SSL_Argument('--da_p_target_type', str, 'uniform'),
            SSL_Argument('--da_p_target', str, None),
        ]
