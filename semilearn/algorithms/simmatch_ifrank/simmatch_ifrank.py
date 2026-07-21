# SimMatch + Influence-Rank Consistency (新增插件, 不修改原 simmatch.py)
# 思路: 在 SimMatch v1 的损失之外, 增加一项"影响排序一致性"损失 Lifrank。
#   - 用 TracIn 的 logit 级闭式梯度算无标注样本对 in-batch 有标注样本的 influence:
#       g = softmax(logit) - onehot(target)  (CE 对 logit 的梯度)
#       influence(u, l) = g_u · g_l
#     —— 无需逐样本 autograd, 一个 matmul 向量化完成。
#   - 对每个无标注样本, 按弱视图 influence 选 top-k 个有标注参考, 弱/强视图各得一个
#     对这 k 个参考的 affinity 分布, 转成 Plackett-Luce 排列分布, 做弱->强的 KL 一致性。
#   - 默认 ifrank_lambda 由命令行/配置给; 该算法是独立的 'simmatch_ifrank', 原 SimMatch 不受影响。
import itertools

import torch
import torch.nn.functional as F

from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.simmatch import SimMatch
from semilearn.algorithms.utils import SSL_Argument


@ALGORITHMS.register('simmatch_ifrank')
class SimMatchIFRank(SimMatch):
    """
    继承自 SimMatch, 仅在 train_step 末尾追加 influence-rank 一致性损失。
    新增超参:
        --ifrank_lambda  影响排序一致性损失权重 (0 则退化为原版 SimMatch)
        --ifrank_T       affinity softmax 温度 (对应原方法的 corrT)
        --ifrank_k       参考样本数 num_references (排列复杂度 k!, 默认 4)
        --ifrank_combine ''=纯 influence; 'multiply'=influence × 余弦相似度
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.ifrank_lambda = float(getattr(args, 'ifrank_lambda', 1.0))
        self.ifrank_T = float(getattr(args, 'ifrank_T', 0.5))
        self.ifrank_k = int(getattr(args, 'ifrank_k', 4))
        self.ifrank_combine = getattr(args, 'ifrank_combine', '') or ''
        # 预生成 k! 个排列 [k!, k]
        self._ifrank_perms = torch.tensor(
            list(itertools.permutations(range(self.ifrank_k))), dtype=torch.long)

    # ---- Plackett-Luce: affinity 分布 [U,k] -> 排列分布 [U, k!] (与原 prob2rank 一致) ----
    def _plackett_luce(self, aff):
        perms = self._ifrank_perms.to(aff.device)          # [k!, k]
        C = aff[:, perms]                                  # [U, k!, k]
        eps = 1e-10
        rank = torch.ones(aff.shape[0], perms.shape[0], device=aff.device, dtype=aff.dtype)
        for i in range(perms.shape[1]):
            rank = rank * (C[:, :, i] / (C[:, :, i:].sum(dim=-1) + eps))
        return rank                                        # [U, k!]

    # ---- 向量化 influence-rank 一致性损失 (弱视图为 teacher, 梯度只走强视图) ----
    def _ifrank_consistency_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s,
                                 feats_x_lb=None, feats_x_ulb_w=None, feats_x_ulb_s=None):
        num_cls = self.num_classes
        L = logits_x_lb.shape[0]
        k = min(self.ifrank_k, L)
        if k < 2:
            return logits_x_ulb_s.new_zeros(())

        # ---- teacher 分支 (弱视图), 全程 no_grad ----
        with torch.no_grad():
            p_l = F.softmax(logits_x_lb.detach(), dim=1)              # [L,C]
            g_l = p_l - F.one_hot(y_lb, num_cls).float()             # [L,C] CE对logit梯度
            p_uw = F.softmax(logits_x_ulb_w.detach(), dim=1)         # [U,C]
            y_u = F.one_hot(p_uw.argmax(dim=1), num_cls).float()    # [U,C] 伪标签
            g_uw = p_uw - y_u                                        # [U,C]
            infl_uw = g_uw @ g_l.t()                                 # [U,L] 弱视图影响
            if self.ifrank_combine == 'multiply' and feats_x_ulb_w is not None:
                infl_uw = infl_uw * (feats_x_ulb_w.detach() @ feats_x_lb.detach().t())
            ref_idx = infl_uw.topk(k, dim=1).indices                # [U,k] 选参考
            aff_w = F.softmax(torch.gather(infl_uw, 1, ref_idx) / self.ifrank_T, dim=1)
            rank_w = self._plackett_luce(aff_w)                     # [U, k!] (target)

        # ---- student 分支 (强视图), 梯度回传 ----
        g_us = F.softmax(logits_x_ulb_s, dim=1) - y_u               # y_u 已 detach
        infl_us = g_us @ g_l.t()                                    # g_l 已 detach
        if self.ifrank_combine == 'multiply' and feats_x_ulb_s is not None:
            infl_us = infl_us * (feats_x_ulb_s @ feats_x_lb.detach().t())
        aff_s = F.softmax(torch.gather(infl_us, 1, ref_idx) / self.ifrank_T, dim=1)
        rank_s = self._plackett_luce(aff_s)                        # [U, k!]

        return F.kl_div((rank_s + 1e-10).log(), rank_w, reduction='batchmean')

    def train_step(self, idx_lb, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        num_ulb = len(x_ulb_w['input_ids']) if isinstance(x_ulb_w, dict) else x_ulb_w.shape[0]
        idx_lb = idx_lb.cuda(self.gpu)

        with self.amp_cm():
            bank = self.mem_bank.clone().detach()

            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits, feats = outputs['logits'], outputs['feat']
                logits_x_lb, ema_feats_x_lb = logits[:num_lb], feats[:num_lb]
                ema_logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                ema_feats_x_ulb_w, feats_x_ulb_s = feats[num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb, ema_feats_x_lb = outs_x_lb['logits'], outs_x_lb['feat']
                outs_x_ulb_w = self.model(x_ulb_w)
                ema_logits_x_ulb_w, ema_feats_x_ulb_w = outs_x_ulb_w['logits'], outs_x_ulb_w['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s, feats_x_ulb_s = outs_x_ulb_s['logits'], outs_x_ulb_s['feat']

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            self.ema.apply_shadow()
            with torch.no_grad():
                if self.use_ema_teacher:
                    ema_feats_x_lb = self.model(x_lb)['feat']
                ema_probs_x_ulb_w = F.softmax(ema_logits_x_ulb_w, dim=-1)
                if getattr(self.args, 'use_da', True):
                    ema_probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=ema_probs_x_ulb_w.detach())
            self.ema.restore()
            feat_dict = {'x_lb': ema_feats_x_lb, 'x_ulb_w': ema_feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            with torch.no_grad():
                teacher_logits = ema_feats_x_ulb_w @ bank
                teacher_prob_orig = F.softmax(teacher_logits / self.T, dim=1)
                factor = ema_probs_x_ulb_w.gather(1, self.labels_bank.expand([num_ulb, -1]))
                teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                if self.smoothing_alpha < 1:
                    bs = teacher_prob_orig.size(0)
                    aggregated_prob = torch.zeros([bs, self.num_classes], device=teacher_prob_orig.device)
                    aggregated_prob = aggregated_prob.scatter_add(1, self.labels_bank.expand([bs, -1]), teacher_prob_orig)
                    probs_x_ulb_w = ema_probs_x_ulb_w * self.smoothing_alpha + aggregated_prob * (1 - self.smoothing_alpha)
                else:
                    probs_x_ulb_w = ema_probs_x_ulb_w

            student_logits = feats_x_ulb_s @ bank
            student_prob = F.softmax(student_logits / self.T, dim=1)
            in_loss = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1).mean()
            if self.epoch == 0:
                in_loss *= 0.0
                probs_x_ulb_w = ema_probs_x_ulb_w

            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s, probs_x_ulb_w, 'ce', mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_in * in_loss

            # ===== 新增: influence-rank 一致性损失 =====
            ifrank_loss = self._ifrank_consistency_loss(
                logits_x_lb, y_lb, ema_logits_x_ulb_w, logits_x_ulb_s,
                feats_x_lb=ema_feats_x_lb, feats_x_ulb_w=ema_feats_x_ulb_w, feats_x_ulb_s=feats_x_ulb_s)
            if self.epoch == 0:
                ifrank_loss = ifrank_loss * 0.0
            total_loss = total_loss + self.ifrank_lambda * ifrank_loss

            self.update_bank(ema_feats_x_lb, y_lb, idx_lb)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         in_loss=in_loss.item(),
                                         ifrank_loss=ifrank_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        argument = SimMatch.get_argument()
        argument.extend([
            SSL_Argument('--ifrank_lambda', float, 1.0),
            SSL_Argument('--ifrank_T', float, 0.5),
            SSL_Argument('--ifrank_k', int, 4),
            SSL_Argument('--ifrank_combine', str, ''),
        ])
        return argument
