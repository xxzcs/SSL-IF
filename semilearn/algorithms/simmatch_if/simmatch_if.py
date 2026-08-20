# SimMatch + 闭式 last-layer TracIn 影响排序一致性 (新增插件, 不修改原 simmatch.py)
#
# IF(影响)用闭式 last-layer TracIn, 无需重建 resnet/逐样本 autograd:
#     logits = W·φ,  ∂CE/∂W = (p-y)⊗φ   (φ = fc输入, backbone 512维特征)
#     IF(u,l) = ⟨(p_u-y_u)⊗φ_u, (p_l-y_l)⊗φ_l⟩ = [(p_u-y_u)·(p_l-y_l)] × [φ_u·φ_l]
#   == 你 fixmatch_if 的 Captum TracInCPFast(final_fc_layer) (已数值证等价)。
#
# 关键(对齐你的实现): IF 是 **detached 引导**(决定看哪些参考/怎么加权), 真正可训练的梯度
#   走 **强视图的余弦一致性**。弱视图=teacher(全 detached), KL(强‖弱)。
#
# 融合 (--ifrank_combine, 复刻你 compute_corrfea 的三种):
#   multiply          : softmax((if_λ·zscore(IF) + csim_λ·cos_raw)/T)   你最佳: nostrong T0.5 / strong T0.9
#   multiply_balanced : softmax((if_λ·zscore(IF) + csim_λ·zscore(cos))/T)
#   multiplyo         : softmax((if_λ·IF_raw + csim_λ·cos_raw)/T)       注: 闭式IF是原始φ大尺度, if_λ/T 需重调
#
# 三种损失模式 (--ifrank_mode):
#   add     : 在 SimMatch in_loss 之外, 外挂一项 rank-consistency loss (KL)。
#   replace : 用 rank_loss 替换 in_loss (会丢掉 in_loss 的稠密实例一致性, 实测掉点)。
#   in_fuse : 不外挂 loss, 而是把 IF 作为乘性因子融进 in_loss 的 teacher 分布:
#             teacher_prob ∝ softmax(feat@bank/T) × 语义factor × 影响factor。
#             影响factor 由 in-batch 有标注样本的闭式 IF 聚合到"每类", 再经 labels_bank 广播到整个 bank
#             (与语义factor 同粒度)。--if_fuse_strength=0 时退化为原版 SimMatch in_loss。
import itertools

import torch
import torch.nn.functional as F

from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.simmatch import SimMatch
from semilearn.algorithms.utils import SSL_Argument, str2bool


@ALGORITHMS.register('simmatch_if')
class SimMatchIF(SimMatch):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.ifrank_loss_weight = float(getattr(args, 'ifrank_loss_weight', 1.0))
        self.corrT = float(getattr(args, 'corrT', 0.5))
        self.num_references = int(getattr(args, 'num_references', 4))
        self.ifrank_combine = getattr(args, 'ifrank_combine', 'multiply') or 'multiply'
        self.if_lambda = float(getattr(args, 'if_lambda', 1.0))
        self.csim_lambda = float(getattr(args, 'csim_lambda', 1.0))
        self.use_strong_if = bool(getattr(args, 'use_strong_if', False))
        self.ifrank_mode = getattr(args, 'ifrank_mode', 'add') or 'add'
        assert self.ifrank_mode in ('add', 'replace', 'in_fuse', 'teacher_fuse'), f"未知 ifrank_mode: {self.ifrank_mode}"
        self.ifrank_warmup_epochs = int(getattr(args, 'ifrank_warmup_epochs', 1))
        self.ifrank_warmup_mode = getattr(args, 'ifrank_warmup_mode', 'zero') or 'zero'
        assert self.ifrank_warmup_mode in ('zero', 'rampup'), f"未知 ifrank_warmup_mode: {self.ifrank_warmup_mode}"
        # in_fuse: 把 IF 作为乘性因子融进 SimMatch in_loss 的 teacher 分布。
        # strength=0 -> 各类影响权重均匀 -> 对 in_loss 无影响(=原版 SimMatch); 越大 IF 调制越强。
        self.if_fuse_strength = float(getattr(args, 'if_fuse_strength', 1.0))
        # TracIn 的 η(学习率)因子: 完整 TracIn = Σ η·⟨∇,∇⟩。Captum TracInCPFast 乘的是
        # checkpoints_load_func 的返回值(源码 tracincp_fast_rand_proj.py:313/340), 不是优化器 lr;
        # 原版 fixmatch_if 的 checkpoints_load_func_inbatch 恒 return 1. => η=1(单位学习率简化版)。
        # 故忠实复刻默认取 1.0, 而非 lr。仅 multiplyo/signedadd(原始尺度)才生效;
        # multiply/multiply_balanced/in_fuse 对 IF 做 z-score/除std, 此常数被约掉、无影响。
        self.if_tracin_scale = float(getattr(args, 'if_tracin_scale', 0.0)) or 1.0
        # 代表性样本选择器 (复刻 compute_corrfea.py):
        #   topk        : 纯弱 IF top-k (原插件行为, 作对照)
        #   by_instance : 原版默认 = 弱/强共同 top-(k//2) 支持 + 共同 top-(k-k//2) 反对
        #   orthogonal  : rankmatch-style 正交锚点 (全局共享)
        #   random      : 每个 batch 随机参考
        #   support_only: 仅支持样本, 不再混入反对样本
        # ref_cand_k = 候选深度 (原版 args.k=8), 先取 top-cand 支持/反对再求弱强共同元素。
        self.ref_select = getattr(args, 'ref_select', 'topk') or 'topk'
        assert self.ref_select in ('topk', 'by_instance', 'orthogonal', 'random', 'support_only'), f"未知 ref_select: {self.ref_select}"
        self.ref_cand_k = int(getattr(args, 'ref_cand_k', 8))
        self.ifrank_score_mode = getattr(args, 'ifrank_score_mode', 'fused') or 'fused'
        assert self.ifrank_score_mode in ('fused', 'cosine', 'if'), f"未知 ifrank_score_mode: {self.ifrank_score_mode}"
        self._perms = torch.tensor(
            list(itertools.permutations(range(self.num_references))), dtype=torch.long)

    def ifrank_warmup_coef(self):
        warmup = self.ifrank_warmup_epochs
        if warmup <= 0:
            return 1.0
        if self.ifrank_warmup_mode == 'rampup':
            return min(1.0, max(0.0, float(self.epoch + 1) / float(warmup)))
        return 0.0 if self.epoch < warmup else 1.0

    def _plackett_luce(self, aff):
        perms = self._perms.to(aff.device)
        C = aff[:, perms]
        eps = 1e-10
        rank = torch.ones(aff.shape[0], perms.shape[0], device=aff.device, dtype=aff.dtype)
        for i in range(perms.shape[1]):
            rank = rank * (C[:, :, i] / (C[:, :, i:].sum(dim=-1) + eps))
        return rank

    @staticmethod
    def _row_zscore(x):
        return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-10)

    @staticmethod
    def _early_common(l1, l2, n):
        """复刻 compute_corrfea.find_early_common_elements: 取 l1、l2 的共同元素(保 l1 顺序)前 n 个;
        不足 n 时退回 l1[:n]。"""
        s2 = set(l2)
        common = [x for x in l1 if x in s2]
        if len(common) < n:
            return l1[:n]
        return common[:n]

    def _select_global_orthogonal_ref_idx(self, phi_lb, k, batch_size):
        L = phi_lb.shape[0]
        if L <= k:
            base_idx = torch.arange(L, device=phi_lb.device, dtype=torch.long)
        else:
            phi_lb_n = F.normalize(phi_lb.detach(), dim=1)
            selected_mask = torch.zeros(L, dtype=torch.bool, device=phi_lb.device)
            selected = []

            first_idx = torch.randint(0, L, (1,), device=phi_lb.device).item()
            selected.append(first_idx)
            selected_mask[first_idx] = True

            for _ in range(1, k):
                selected_stack = phi_lb_n[selected]
                cos_sim = torch.matmul(phi_lb_n, selected_stack.t()).abs()
                max_cos = cos_sim.max(dim=1).values
                max_cos[selected_mask] = 10.0
                next_idx = max_cos.argmin().item()
                selected.append(next_idx)
                selected_mask[next_idx] = True

            base_idx = torch.tensor(selected, device=phi_lb.device, dtype=torch.long)

        return base_idx.unsqueeze(0).expand(batch_size, -1)

    def _select_random_ref_idx(self, L, k, batch_size, device):
        if L <= k:
            base = torch.arange(L, device=device, dtype=torch.long)
            return base.unsqueeze(0).expand(batch_size, -1)
        rows = [torch.randperm(L, device=device)[:k] for _ in range(batch_size)]
        return torch.stack(rows, dim=0)

    def _select_ref_idx(self, IF_uw, IF_us, phi_lb, k):
        """返回每个无标注样本的 k 个参考(有标注)索引 [U,k]。
        topk        : 弱 IF top-k (原行为)。
        by_instance : 弱/强共同 top-(k//2) 支持 + 共同 top-(k-k//2) 反对 (原版默认)。
        orthogonal  : rankmatch-style 正交锚点。
        random      : 随机参考。
        support_only: 仅保留支持样本。"""
        if self.ref_select == 'orthogonal':
            return self._select_global_orthogonal_ref_idx(phi_lb, k, IF_uw.shape[0])
        if self.ref_select == 'random':
            return self._select_random_ref_idx(IF_uw.shape[1], k, IF_uw.shape[0], IF_uw.device)
        if self.ref_select == 'topk':
            return IF_uw.topk(k, dim=1).indices
        # by_instance
        L = IF_uw.shape[1]
        cand = min(self.ref_cand_k, L)
        if self.ref_select == 'support_only':
            prop_uw = IF_uw.topk(cand, dim=1).indices.tolist()
            prop_us = IF_us.topk(cand, dim=1).indices.tolist()
            rows = []
            for i in range(len(prop_uw)):
                sel = self._early_common(prop_uw[i], prop_us[i], k)
                if len(sel) < k:
                    for c in prop_uw[i]:
                        if c not in sel:
                            sel.append(c)
                            if len(sel) == k:
                                break
                rows.append(sel[:k])
            return torch.tensor(rows, device=IF_uw.device, dtype=torch.long)
        num_prop = max(1, k // 2)
        num_oppo = k - num_prop
        prop_uw = IF_uw.topk(cand, dim=1).indices.tolist()
        prop_us = IF_us.topk(cand, dim=1).indices.tolist()
        oppo_uw = IF_uw.topk(cand, dim=1, largest=False).indices.tolist()
        oppo_us = IF_us.topk(cand, dim=1, largest=False).indices.tolist()
        rows = []
        for i in range(len(prop_uw)):
            sel = self._early_common(prop_uw[i], prop_us[i], num_prop)
            sel = sel + self._early_common(oppo_uw[i], oppo_us[i], num_oppo)
            # 保证恰好 k 个: 不足则从弱支持候选补齐(L 太小时的兜底)
            if len(sel) < k:
                for c in prop_uw[i]:
                    if c not in sel:
                        sel.append(c)
                        if len(sel) == k:
                            break
            rows.append(sel[:k])
        return torch.tensor(rows, device=IF_uw.device, dtype=torch.long)

    def _influence_class_factor(self, logits_x_lb, y_lb, phi_lb,
                                logits_x_ulb_w, phi_uw, labels_bank, num_ulb):
        """in_fuse 模式: 闭式 last-layer TracIn 影响 -> 每类影响权重 -> 广播到 memory bank,
        作为 in_loss teacher 分布的乘性因子。全程 detached。返回 [U, bank] 的正因子。

        IF(u,l) = scale·(g_u·g_l)×(φ_u·φ_l), g=softmax(logit)-target(伪/真)。
        按 in-batch 有标注样本的类别求平均 -> 每类影响 [U,C], z-score 去尺度,
        softmax(strength··) 得每类权重; strength=0 时均匀 -> 对 in_loss 无影响。
        """
        num_cls = self.num_classes
        with torch.no_grad():
            p_l = F.softmax(logits_x_lb, dim=1)
            g_l = p_l - F.one_hot(y_lb, num_cls).float()                       # [L,C]
            p_uw = F.softmax(logits_x_ulb_w, dim=1)
            y_u = F.one_hot(p_uw.argmax(1), num_cls).float()
            g_uw = p_uw - y_u                                                  # [U,C]
            IF_uw = self.if_tracin_scale * (g_uw @ g_l.t()) * (phi_uw @ phi_lb.t())  # [U,L] 闭式IF×η
            y_lb_oh = F.one_hot(y_lb, num_cls).float()                         # [L,C]
            class_cnt = y_lb_oh.sum(0).clamp(min=1.0)                          # [C] 各类样本数
            class_infl = (IF_uw @ y_lb_oh) / class_cnt                         # [U,C] 每类平均影响
            # 行内去均值=类间相对影响; 再除全 batch 标量 std=去掉闭式IF任意量纲, 但保留样本间强弱
            # (注意 C=2 时按行 z-score 会塌成 ±常数, 抹掉幅度, 故不用 _row_zscore)。
            class_infl = class_infl - class_infl.mean(dim=1, keepdim=True)
            class_infl = class_infl / (class_infl.std() + 1e-10)
            infl_w = F.softmax(self.if_fuse_strength * class_infl, dim=1)      # [U,C]; strength=0 -> 均匀
            infl_factor = infl_w.gather(1, labels_bank.expand([num_ulb, -1]))  # [U, bank] 广播到 bank
        return infl_factor

    def _influence_anchor_factor(self, logits_x_lb, y_lb, phi_lb,
                                 logits_x_ulb_w, phi_uw,
                                 logits_x_ulb_s, phi_us, idx_lb, num_ulb):
        """teacher_fuse 模式: 保留 batch-8 的 2+2 by_instance 选择(选锚点用弱+强 IF, 忠实原设计),
        把每个无标注样本对其 k 个锚点的「弱视图 IF+余弦融合 affinity」折进 in_loss 的 K-维 teacher
        (乘到锚点对应 bank 位置, 非锚点=1)。全程 detached。返回 [U, K] 正因子。
        与 in_fuse(每类坍缩)不同: 这里保留逐锚点粒度, 不和每类 factor 冗余; 但只调制 k/K 个位置,
        强度由 if_fuse_strength 控制(=0 时全 1 -> 退化为原版 SimMatch in_loss)。"""
        num_cls = self.num_classes
        L = logits_x_lb.shape[0]
        k = min(self.num_references, L)
        infl = torch.ones(num_ulb, self.K, device=phi_uw.device, dtype=phi_uw.dtype)
        if k < 2:
            return infl
        with torch.no_grad():
            phi_lb_n = F.normalize(phi_lb, dim=1)
            p_l = F.softmax(logits_x_lb, dim=1)
            g_l = p_l - F.one_hot(y_lb, num_cls).float()                       # [L,C]
            p_uw = F.softmax(logits_x_ulb_w, dim=1)
            y_u = F.one_hot(p_uw.argmax(1), num_cls).float()                   # [U,C]
            g_uw = p_uw - y_u
            g_us = F.softmax(logits_x_ulb_s, dim=1) - y_u
            IF_uw = self.if_tracin_scale * (g_uw @ g_l.t()) * (phi_uw @ phi_lb.t())  # [U,L]
            IF_us = self.if_tracin_scale * (g_us @ g_l.t()) * (phi_us @ phi_lb.t())
            ref_idx = self._select_ref_idx(IF_uw, IF_us, k)                    # [U,k] 你的 2+2
            # teacher 目标权重: 弱视图 IF+余弦融合(teacher=弱视图, detached)
            if_w = torch.gather(IF_uw, 1, ref_idx)
            cos_w = torch.gather(F.normalize(phi_uw, dim=1) @ phi_lb_n.t(), 1, ref_idx)
            aff_w = self._fuse(if_w, cos_w)                                    # [U,k] softmax over k
            # 转成中心在 1 的乘性因子: 均匀(1/k)->1; if_fuse_strength 调强弱
            factor_k = (aff_w * k).clamp(min=1e-6) ** self.if_fuse_strength    # [U,k]
            bank_pos = idx_lb[ref_idx]                                        # [U,k] 锚点 -> bank 索引
            infl.scatter_(1, bank_pos, factor_k)
        return infl

    def _fuse(self, ifs, cos):
        if self.ifrank_score_mode == 'cosine':
            return F.softmax(cos / self.corrT, dim=1)
        if self.ifrank_score_mode == 'if':
            if self.ifrank_combine == 'multiplyo':
                if_score = ifs
            else:
                if_score = self._row_zscore(ifs)
            return F.softmax(if_score / self.corrT, dim=1)
        # ifs: detached 影响分 [U,k]; cos: 余弦 [U,k] (强支 live)
        if self.ifrank_combine == 'gate':
            # 相乘门控: cos 相似度分布(基座) × softplus(β·zscore(IF)) 单调正门控, 再归一化。
            # IF 高的参考放大、低的抑制, 永不翻转; zscore 尺度无关; β=if_lambda, 温度=corrT。
            base = F.softmax(self.csim_lambda * cos / self.corrT, dim=1)
            gate = F.softplus(self.if_lambda * self._row_zscore(ifs))
            aff = base * gate
            return aff / (aff.sum(dim=1, keepdim=True) + 1e-10)
        if self.ifrank_combine == 'multiplyo':           # 原始 IF + 原始余弦
            score = self.if_lambda * ifs + self.csim_lambda * cos
        elif self.ifrank_combine == 'multiply_balanced':  # IF、余弦都 z-score
            score = self.if_lambda * self._row_zscore(ifs) + self.csim_lambda * self._row_zscore(cos)
        else:                                            # 'multiply': zscore(IF) + 原始余弦
            score = self.if_lambda * self._row_zscore(ifs) + self.csim_lambda * cos
        return F.softmax(score / self.corrT, dim=1)

    def _ifrank_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s, phi_lb, phi_uw, phi_us):
        num_cls = self.num_classes
        L = logits_x_lb.shape[0]
        k = min(self.num_references, L)
        if k < 2:
            return logits_x_ulb_s.new_zeros(())

        phi_lb_d = phi_lb.detach()
        phi_lb_n = F.normalize(phi_lb_d, dim=1)            # 参考特征(余弦用)

        # ---- IF(detached 引导) + 弱视图 teacher(全 detached) ----
        U = logits_x_ulb_w.shape[0]
        if_target = getattr(self.args, 'if_target', 'soft') or 'soft'
        _mr = getattr(self.args, 'if_mean_reduce', True)
        mean_reduce = _mr if isinstance(_mr, bool) else str(_mr).lower() not in ('false', '0', 'none', '')
        with torch.no_grad():
            lg_l = logits_x_lb.detach(); lg_uw = logits_x_ulb_w.detach(); lg_us = logits_x_ulb_s.detach()
            p_l = F.softmax(lg_l, dim=1)
            g_l = p_l - F.one_hot(y_lb, num_cls).float()                  # [L,C] 有标注=真标签硬CE(同 my_celoss)
            p_uw = F.softmax(lg_uw, dim=1); p_us = F.softmax(lg_us, dim=1)
            if if_target == 'soft':
                # 忠实复刻 my_celoss(以"原始 logits"为软目标): g = p·Σlogits − logits
                # 实测: 该软 g 的 IF 与你 Captum corr≈0.905; 硬 argmax 仅 0.52。
                g_uw = p_uw * lg_uw.sum(1, keepdim=True) - lg_uw
                g_us = p_us * lg_us.sum(1, keepdim=True) - lg_us
            else:
                y_u = F.one_hot(p_uw.argmax(1), num_cls).float()          # 旧的硬 argmax(不忠实, 仅对照)
                g_uw = p_uw - y_u
                g_us = p_us - y_u
            if mean_reduce:
                # 复刻 my_celoss 的 .mean() 批规约: g_l÷L, g_u÷U。balanced 经 zscore 不受此常数影响;
                # multiplyo 靠它把 IF 缩到你 Captum 量级、避免 T0.05 饱和。
                g_l = g_l / max(L, 1); g_uw = g_uw / max(U, 1); g_us = g_us / max(U, 1)
            IF_uw = self.if_tracin_scale * (g_uw @ g_l.t()) * (phi_uw.detach() @ phi_lb_d.t())  # [U,L] 闭式IF×η
            IF_us = self.if_tracin_scale * (g_us @ g_l.t()) * (phi_us.detach() @ phi_lb_d.t())
            ref_idx = self._select_ref_idx(IF_uw, IF_us, phi_lb_d, k)   # [U,k] 参考选择策略
            if_w = torch.gather(IF_uw, 1, ref_idx)
            if_s = torch.gather(IF_us if self.use_strong_if else IF_uw, 1, ref_idx)
            cos_w = torch.gather(F.normalize(phi_uw.detach(), dim=1) @ phi_lb_n.t(), 1, ref_idx)
            aff_w = self._fuse(if_w, cos_w)
            rank_w = self._plackett_luce(aff_w)                          # target

        # ---- 强视图余弦(LIVE, 梯度由此回传) ----
        cos_s = torch.gather(F.normalize(phi_us, dim=1) @ phi_lb_n.t(), 1, ref_idx)
        aff_s = self._fuse(if_s, cos_s)
        rank_s = self._plackett_luce(aff_s)

        if getattr(self.args, 'if_mask', False):
            # 置信度门控: 只对弱视图 max-prob >= p_cutoff 的无标注样本算 IF 排序损失(降低不确定样本噪声)
            with torch.no_grad():
                conf = F.softmax(logits_x_ulb_w.detach(), dim=1).max(1).values
                m = (conf >= float(getattr(self.args, 'p_cutoff', 0.95))).float()
            kl_per = F.kl_div((rank_s + 1e-10).log(), rank_w, reduction='none').sum(1)
            loss = (kl_per * m).sum() / m.sum().clamp(min=1.0)
        else:
            loss = F.kl_div((rank_s + 1e-10).log(), rank_w, reduction='batchmean')
        if __import__('os').environ.get('IFPROBE2'):
            print(f"[IFPROBE2 combine={self.ifrank_combine} T={self.corrT} ifl={self.if_lambda}] "
                  f"IF|abs|mean={if_w.abs().mean().item():.4g} cos|abs|mean={cos_w.abs().mean().item():.3g} "
                  f"aff_w.maxprob={aff_w.max(1).values.mean().item():.3f} aff_s.maxprob={aff_s.max(1).values.mean().item():.3f} "
                  f"argmax_agree={(aff_w.argmax(1)==aff_s.argmax(1)).float().mean().item():.2f} "
                  f"ifrank_loss={loss.item():.5f}", flush=True)
        return loss

    def train_step(self, idx_lb, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        num_ulb = len(x_ulb_w['input_ids']) if isinstance(x_ulb_w, dict) else x_ulb_w.shape[0]
        idx_lb = idx_lb.cuda(self.gpu)

        # 显式前向, 额外取 fc 输入特征 φ(512维) 给 last-layer TracIn (一次 backbone 前向)
        net = self.model.module if hasattr(self.model, 'module') else self.model
        assert not getattr(self.args, 'use_epass', False), "simmatch_if 暂只支持 use_epass=False"

        def _fwd(x):
            phi = net.backbone(x, only_feat=True)
            logits = net.backbone(phi, only_fc=True)
            proj = net.l2norm(net.mlp_proj(phi))
            return phi, logits, proj

        with self.amp_cm():
            bank = self.mem_bank.clone().detach()

            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                phi, logits, feats = _fwd(inputs)
                logits_x_lb, ema_feats_x_lb = logits[:num_lb], feats[:num_lb]
                ema_logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                ema_feats_x_ulb_w, feats_x_ulb_s = feats[num_lb:].chunk(2)
                phi_x_lb = phi[:num_lb]
                phi_x_ulb_w, phi_x_ulb_s = phi[num_lb:].chunk(2)
            else:
                phi_x_lb, logits_x_lb, ema_feats_x_lb = _fwd(x_lb)
                phi_x_ulb_w, ema_logits_x_ulb_w, ema_feats_x_ulb_w = _fwd(x_ulb_w)
                phi_x_ulb_s, logits_x_ulb_s, feats_x_ulb_s = _fwd(x_ulb_s)

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
                if self.ifrank_mode == 'in_fuse':
                    # IF 融进 teacher: softmax(feat@bank/T) × 语义factor × 影响factor(每类坍缩版)
                    infl_factor = self._influence_class_factor(
                        logits_x_lb.detach(), y_lb, phi_x_lb.detach(),
                        ema_logits_x_ulb_w.detach(), phi_x_ulb_w.detach(),
                        self.labels_bank, num_ulb)
                    teacher_prob = teacher_prob_orig * factor * infl_factor
                elif self.ifrank_mode == 'teacher_fuse':
                    # 保留 batch-8 的 2+2 选择, 4 锚点 IF+余弦 affinity 折进 teacher(逐锚点粒度)
                    infl_factor = self._influence_anchor_factor(
                        logits_x_lb.detach(), y_lb, phi_x_lb.detach(),
                        ema_logits_x_ulb_w.detach(), phi_x_ulb_w.detach(),
                        logits_x_ulb_s.detach(), phi_x_ulb_s.detach(), idx_lb, num_ulb)
                    teacher_prob = teacher_prob_orig * factor * infl_factor
                else:
                    teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                if __import__('os').environ.get('IFPROBE') and self.ifrank_mode in ('in_fuse', 'teacher_fuse'):
                    # 探针: 量 IF 因子对 in_loss teacher 的实际影响 (重合度/稀释诊断)
                    base = teacher_prob_orig * factor
                    base = base / base.sum(dim=1, keepdim=True)
                    kl_move = (teacher_prob * (teacher_prob.clamp_min(1e-10).log()
                                               - base.clamp_min(1e-10).log())).sum(dim=1).mean()
                    fa = infl_factor - infl_factor.mean(dim=1, keepdim=True)
                    fb = factor - factor.mean(dim=1, keepdim=True)
                    corr = (fa * fb).sum(dim=1) / (fa.norm(dim=1) * fb.norm(dim=1) + 1e-10)
                    onbank = (teacher_prob * (infl_factor != 1.0).float()).sum(dim=1).mean()  # teacher_fuse: 落在被调制位置的质量
                    print(f"[IFPROBE it={self.it} mode={self.ifrank_mode}] "
                          f"kl_move={kl_move.item():.5f}  corr(infl,semantic)={corr.mean().item():.4f}  "
                          f"modulated_mass={onbank.item():.4f}", flush=True)

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

            # ===== 闭式 last-layer 影响排序一致性 (特征因子用 fc 输入 φ, 512维) =====
            if self.ifrank_mode in ('in_fuse', 'teacher_fuse'):
                # IF 已融进上面的 in_loss teacher 分布, 不再外挂 rank loss。
                ifrank_loss = logits_x_ulb_s.new_zeros(())
            else:
                ifrank_loss = self._ifrank_loss(
                    logits_x_lb, y_lb, ema_logits_x_ulb_w, logits_x_ulb_s,
                    phi_x_lb, phi_x_ulb_w, phi_x_ulb_s)
            ifrank_loss = ifrank_loss * self.ifrank_warmup_coef()

            total_loss = sup_loss + self.lambda_u * unsup_loss
            if self.ifrank_mode in ('add', 'in_fuse', 'teacher_fuse'):
                total_loss = total_loss + self.lambda_in * in_loss        # 保留 SimMatch 实例相似度(in_fuse/teacher_fuse 下其 teacher 已含 IF)
            total_loss = total_loss + self.ifrank_loss_weight * ifrank_loss

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
            SSL_Argument('--ifrank_loss_weight', float, 1.0),
            SSL_Argument('--corrT', float, 0.5),
            SSL_Argument('--num_references', int, 4),
            SSL_Argument('--ifrank_combine', str, 'multiply'),
            SSL_Argument('--if_lambda', float, 1.0),
            SSL_Argument('--csim_lambda', float, 1.0),
            SSL_Argument('--use_strong_if', str2bool, False),
            SSL_Argument('--ifrank_mode', str, 'add'),
            SSL_Argument('--if_fuse_strength', float, 1.0),
            SSL_Argument('--ref_select', str, 'topk'),
            SSL_Argument('--ref_cand_k', int, 8),
            SSL_Argument('--ifrank_score_mode', str, 'fused'),  # fused | cosine | if
            SSL_Argument('--if_target', str, 'soft'),        # soft=忠实my_celoss(p·Σlogits−logits); hard=旧argmax
            SSL_Argument('--if_mean_reduce', str2bool, True), # 复刻my_celoss批规约(multiplyo需要;balanced无所谓)
            SSL_Argument('--if_mask', str2bool, False),       # 置信度门控: 只对高置信无标注算IF排序损失
            SSL_Argument('--ifrank_warmup_epochs', int, 1),
            SSL_Argument('--ifrank_warmup_mode', str, 'zero'),
        ])
        return argument
