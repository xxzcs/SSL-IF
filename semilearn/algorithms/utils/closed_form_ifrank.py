# 闭式 last-layer TracIn 影响排序一致性 —— 可复用 Mixin (从 simmatch_if 抽出, 逐字对齐)。
# 供无关系项框架 (FreeMatch/FlexMatch/FixMatch...) 以纯 add 方式接入同一套 IF-rank 损失,
# 保证跨框架通用性验证时 IF 实现完全一致。
#
# IF(u,l) = if_tracin_scale·[(p_u-y_u)·(p_l-y_l)]×[φ_u·φ_l]  (φ = fc 输入 512 维特征 = 'feat')
# IF 是 detached 引导(决定看哪些参考/怎么加权); 可训练梯度走强视图余弦一致性; 弱视图=teacher(detached); KL(强‖弱)。
import itertools
import torch
import torch.nn.functional as F
import warnings

from semilearn.algorithms.utils import SSL_Argument, str2bool


class ClosedFormIFRankMixin:
    def init_ifrank(self, args):
        self.ifrank_loss_weight = float(getattr(args, 'ifrank_loss_weight', 1.0))
        self.corrT = float(getattr(args, 'corrT', 0.9))
        self.num_references = int(getattr(args, 'num_references', 4))
        self.ifrank_combine = getattr(args, 'ifrank_combine', 'multiply_balanced') or 'multiply_balanced'
        self.if_lambda = float(getattr(args, 'if_lambda', 1.0))
        self.csim_lambda = float(getattr(args, 'csim_lambda', 1.0))
        self.use_strong_if = bool(getattr(args, 'use_strong_if', True))
        raw_if_scale = getattr(args, 'if_tracin_scale', None)
        self.if_tracin_scale = 1.0 if raw_if_scale is None else float(raw_if_scale)
        self.ref_select = getattr(args, 'ref_select', 'by_instance') or 'by_instance'
        assert self.ref_select in ('topk', 'by_instance', 'orthogonal', 'random', 'support_only'), f"未知 ref_select: {self.ref_select}"
        self.ref_cand_k = int(getattr(args, 'ref_cand_k', 8))
        self.if_target = getattr(args, 'if_target', 'soft') or 'soft'
        self.if_mean_reduce = bool(getattr(args, 'if_mean_reduce', True))
        self.ifrank_score_mode = getattr(args, 'ifrank_score_mode', 'fused') or 'fused'
        assert self.ifrank_score_mode in ('fused', 'cosine', 'if'), f"未知 ifrank_score_mode: {self.ifrank_score_mode}"
        if self.ifrank_score_mode == 'if':
            raise ValueError(
                "ClosedFormIFRankMixin does not support ifrank_score_mode='if'. "
                "In this implementation, IF scores are computed in a detached branch, "
                "so IF-only ranking produces a numeric loss but no gradient to the model. "
                "Use 'fused' or 'cosine', or implement a differentiable IF-only variant first."
            )
        if raw_if_scale == 0.0:
            warnings.warn(
                "if_tracin_scale is explicitly set to 0.0; IF scores will be zeroed out.",
                RuntimeWarning,
                stacklevel=2,
            )
        # warmup: 前 warmup_epochs 个 epoch 抑制 ifrank_loss。
        #   mode='zero'   -> epoch < warmup_epochs 时系数=0，之后=1 (默认 warmup_epochs=1 == 旧的"只关 epoch0")
        #   mode='rampup' -> 系数 = min(1, epoch / warmup_epochs) 线性 0->1
        self.ifrank_warmup_epochs = int(getattr(args, 'ifrank_warmup_epochs', 1))
        self.ifrank_if_ramp_epochs = int(getattr(args, 'ifrank_if_ramp_epochs', self.ifrank_warmup_epochs))
        self.ifrank_warmup_mode = getattr(args, 'ifrank_warmup_mode', 'zero') or 'zero'
        assert self.ifrank_warmup_mode in (
            'zero',
            'rampup',
            'cosine_then_fused',
            'cosine_then_if_ramp',
            'if_ramp_from_start',
        ), f"未知 ifrank_warmup_mode: {self.ifrank_warmup_mode}"
        self._perms = torch.tensor(
            list(itertools.permutations(range(self.num_references))), dtype=torch.long)

    def ifrank_warmup_coef(self):
        w = self.ifrank_warmup_epochs
        if w <= 0:
            return 1.0
        if self.ifrank_warmup_mode == 'rampup':
            return min(1.0, float(self.epoch) / float(w))
        if self.ifrank_warmup_mode in ('cosine_then_fused', 'cosine_then_if_ramp', 'if_ramp_from_start'):
            return 1.0
        return 0.0 if self.epoch < w else 1.0

    def ifrank_if_coef(self):
        w = self.ifrank_warmup_epochs
        r = self.ifrank_if_ramp_epochs
        cur = float(self.epoch)
        if self.ifrank_warmup_mode == 'cosine_then_fused':
            return 0.0 if cur < w else 1.0
        if self.ifrank_warmup_mode == 'cosine_then_if_ramp':
            if cur < w:
                return 0.0
            if r <= 0:
                return 1.0
            return min(1.0, max(0.0, (cur - float(w)) / float(r)))
        if self.ifrank_warmup_mode == 'if_ramp_from_start':
            if w <= 0:
                return 1.0
            return min(1.0, cur / float(w))
        return 1.0

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
            base_idx = torch.arange(L, device=device, dtype=torch.long)
            return base_idx.unsqueeze(0).expand(batch_size, -1)
        rows = [torch.randperm(L, device=device)[:k] for _ in range(batch_size)]
        return torch.stack(rows, dim=0)

    def _select_ref_idx(self, IF_uw, IF_us, phi_lb, k):
        if self.ref_select == 'orthogonal':
            return self._select_global_orthogonal_ref_idx(phi_lb, k, IF_uw.shape[0])
        if self.ref_select == 'random':
            return self._select_random_ref_idx(IF_uw.shape[1], k, IF_uw.shape[0], IF_uw.device)
        if self.ref_select == 'topk':
            return IF_uw.topk(k, dim=1).indices
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
            if len(sel) < k:
                for c in prop_uw[i]:
                    if c not in sel:
                        sel.append(c)
                        if len(sel) == k:
                            break
            rows.append(sel[:k])
        return torch.tensor(rows, device=IF_uw.device, dtype=torch.long)

    def _fuse(self, ifs, cos):
        if_coef = self.ifrank_if_coef()
        if self.ifrank_score_mode == 'cosine':
            return F.softmax(cos / self.corrT, dim=1)
        if self.ifrank_score_mode == 'if':
            if self.ifrank_combine == 'multiplyo':
                if_score = ifs
            else:
                if_score = self._row_zscore(ifs)
            return F.softmax(if_score / self.corrT, dim=1)
        z_if = self._row_zscore(ifs)
        z_cos = self._row_zscore(cos)
        if self.ifrank_combine == 'multiplyo':
            score = self.if_lambda * if_coef * ifs + self.csim_lambda * cos
        elif self.ifrank_combine == 'multiply_balanced':
            score = self.if_lambda * if_coef * z_if + self.csim_lambda * z_cos
        elif self.ifrank_combine == 'product_balanced':
            score = (self.if_lambda * if_coef * z_if) * (self.csim_lambda * z_cos)
        elif self.ifrank_combine == 'residual_product_balanced':
            gate = torch.clamp(1.0 + self.if_lambda * if_coef * z_if, min=0.0)
            score = (self.csim_lambda * z_cos) * gate
        elif self.ifrank_combine == 'gated_cosine_balanced':
            gate = torch.sigmoid(self.if_lambda * if_coef * z_if)
            score = (self.csim_lambda * z_cos) * gate
        else:  # 'multiply'
            score = self.if_lambda * if_coef * z_if + self.csim_lambda * cos
        return F.softmax(score / self.corrT, dim=1)

    def _hard_label_grad(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes).float()
        grad = probs - one_hot

        focal_gamma = float(getattr(self.ce_loss, 'focal_gamma', 0.0))
        if focal_gamma <= 0:
            return grad

        class_weights = getattr(self.ce_loss, 'class_weights', None)
        alpha = logits.new_ones(targets.shape[0])
        if class_weights is not None:
            alpha = class_weights.to(logits.device).gather(0, targets)

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
        one_minus_pt = (1.0 - pt).clamp_min(1e-12)
        focal_scale = one_minus_pt.pow(focal_gamma)
        focal_corr = -focal_gamma * pt * one_minus_pt.pow(focal_gamma - 1.0) * torch.log(pt)
        return (alpha * (focal_scale + focal_corr)).unsqueeze(1) * grad

    def ifrank_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s, phi_lb, phi_uw, phi_us):
        num_cls = self.num_classes
        L = logits_x_lb.shape[0]
        k = min(self.num_references, L)
        if k < 2:
            return logits_x_ulb_s.new_zeros(())
        phi_lb_d = phi_lb.detach()
        phi_lb_n = F.normalize(phi_lb_d, dim=1)
        U = logits_x_ulb_w.shape[0]
        if_target = getattr(self, 'if_target', 'soft')
        mean_reduce = getattr(self, 'if_mean_reduce', True)
        with torch.no_grad():
            lg_l = logits_x_lb.detach(); lg_uw = logits_x_ulb_w.detach(); lg_us = logits_x_ulb_s.detach()
            g_l = self._hard_label_grad(lg_l, y_lb)
            p_uw = F.softmax(lg_uw, dim=1); p_us = F.softmax(lg_us, dim=1)
            if if_target == 'soft':
                # 忠实 my_celoss(原始 logits 软目标): g = p·Σlogits − logits (实测与 Captum corr≈0.905)
                g_uw = p_uw * lg_uw.sum(1, keepdim=True) - lg_uw
                g_us = p_us * lg_us.sum(1, keepdim=True) - lg_us
            else:
                y_u = F.one_hot(p_uw.argmax(1), num_cls).float()   # 旧硬 argmax(仅对照)
                y_u_idx = p_uw.argmax(1)
                g_uw = self._hard_label_grad(lg_uw, y_u_idx)
                g_us = self._hard_label_grad(lg_us, y_u_idx)
            if mean_reduce:
                g_l = g_l / max(L, 1); g_uw = g_uw / max(U, 1); g_us = g_us / max(U, 1)
            IF_uw = self.if_tracin_scale * (g_uw @ g_l.t()) * (phi_uw.detach() @ phi_lb_d.t())
            IF_us = self.if_tracin_scale * (g_us @ g_l.t()) * (phi_us.detach() @ phi_lb_d.t())
            ref_idx = self._select_ref_idx(IF_uw, IF_us, phi_lb_d, k)
            if_w = torch.gather(IF_uw, 1, ref_idx)
            if_s = torch.gather(IF_us if self.use_strong_if else IF_uw, 1, ref_idx)
            import os as _os_pr
            if _os_pr.environ.get('IFPROBE'):
                _a=if_w.flatten().float(); _b=if_s.flatten().float()
                _c=(((_a-_a.mean())*(_b-_b.mean())).sum()/((_a-_a.mean()).norm()*(_b-_b.mean()).norm()+1e-10)).item()
                _ifuw=IF_uw.std().item(); _ifus=IF_us.std().item()
                _ws=if_w.std().item(); _ss=if_s.std().item(); _diff=(if_w-if_s).abs().mean().item()
                print("[CFPROBE] IF_uw_std=%.4g IF_us_std=%.4g | wsel_std=%.4g ssel_std=%.4g | ws_absdiff=%.4g corr_ws=%.4f" % (_ifuw,_ifus,_ws,_ss,_diff,_c), flush=True)
            cos_w = torch.gather(F.normalize(phi_uw.detach(), dim=1) @ phi_lb_n.t(), 1, ref_idx)
            aff_w = self._fuse(if_w, cos_w)
            rank_w = self._plackett_luce(aff_w)
        cos_s = torch.gather(F.normalize(phi_us, dim=1) @ phi_lb_n.t(), 1, ref_idx)
        aff_s = self._fuse(if_s, cos_s)
        rank_s = self._plackett_luce(aff_s)
        return F.kl_div((rank_s + 1e-10).log(), rank_w, reduction='batchmean')

    @staticmethod
    def ifrank_arguments():
        return [
            SSL_Argument('--ifrank_loss_weight', float, 1.0),
            SSL_Argument('--corrT', float, 0.9),
            SSL_Argument('--num_references', int, 4),
            SSL_Argument('--ifrank_combine', str, 'multiply_balanced'),
            SSL_Argument('--if_lambda', float, 1.0),
            SSL_Argument('--csim_lambda', float, 1.0),
            SSL_Argument('--use_strong_if', str2bool, True),
            SSL_Argument('--if_tracin_scale', float, 0.0),
            SSL_Argument('--ref_select', str, 'by_instance'),
            SSL_Argument('--ref_cand_k', int, 8),
            SSL_Argument('--ifrank_score_mode', str, 'fused'),  # fused | cosine | if
            SSL_Argument('--if_target', str, 'soft'),         # soft=忠实(p·Σlogits−logits); hard=旧argmax
            SSL_Argument('--if_mean_reduce', str2bool, True),
            SSL_Argument('--ifrank_warmup_epochs', int, 1),   # 前 N epoch 抑制 (默认1=只关epoch0)
            SSL_Argument('--ifrank_if_ramp_epochs', int, 1),  # IF 分支线性增长跨度
            SSL_Argument('--ifrank_warmup_mode', str, 'zero'),  # zero | rampup | cosine_then_fused | cosine_then_if_ramp | if_ramp_from_start
        ]
