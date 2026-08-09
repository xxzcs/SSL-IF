import itertools

import torch
import torch.nn.functional as F

from semilearn.algorithms.utils import SSL_Argument, str2bool


class DifferentiableIFRankMixin:
    """
    Fully differentiable IF-cosine fusion for in-batch ranking consistency.

    Design:
    - weak branch acts as detached teacher;
    - strong branch remains differentiable;
    - labeled anchors stay detached for stability;
    - reference selection is discrete and based on detached weak/strong IF scores.
    """

    def init_dif(self, args):
        self.ifrank_loss_weight = float(getattr(args, 'ifrank_loss_weight', 1.0))
        self.corrT = float(getattr(args, 'corrT', 0.9))
        self.num_references = int(getattr(args, 'num_references', 4))
        self.ifrank_combine = getattr(args, 'ifrank_combine', 'multiply_balanced') or 'multiply_balanced'
        self.if_lambda = float(getattr(args, 'if_lambda', 1.0))
        self.csim_lambda = float(getattr(args, 'csim_lambda', 1.0))
        self.use_strong_if = bool(getattr(args, 'use_strong_if', True))
        self.if_tracin_scale = float(getattr(args, 'if_tracin_scale', 1.0))
        self.ref_select = getattr(args, 'ref_select', 'by_instance') or 'by_instance'
        assert self.ref_select in ('topk', 'by_instance', 'orthogonal', 'random', 'support_only'), f"未知 ref_select: {self.ref_select}"
        self.ref_cand_k = int(getattr(args, 'ref_cand_k', 8))
        self.if_target = getattr(args, 'if_target', 'soft') or 'soft'
        self.if_mean_reduce = bool(getattr(args, 'if_mean_reduce', True))
        self.ifrank_score_mode = getattr(args, 'ifrank_score_mode', 'fused') or 'fused'
        assert self.ifrank_score_mode in ('fused', 'cosine', 'if'), f"未知 ifrank_score_mode: {self.ifrank_score_mode}"
        self.ifrank_warmup_epochs = int(getattr(args, 'ifrank_warmup_epochs', 1))
        self.ifrank_warmup_mode = getattr(args, 'ifrank_warmup_mode', 'zero') or 'zero'
        assert self.ifrank_warmup_mode in ('zero', 'rampup'), f"未知 ifrank_warmup_mode: {self.ifrank_warmup_mode}"
        self._perms = torch.tensor(
            list(itertools.permutations(range(self.num_references))), dtype=torch.long
        )

    def ifrank_warmup_coef(self):
        w = self.ifrank_warmup_epochs
        if w <= 0:
            return 1.0
        if self.ifrank_warmup_mode == 'rampup':
            return min(1.0, float(self.epoch) / float(w))
        return 0.0 if self.epoch < w else 1.0

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
        if self.ifrank_score_mode == 'cosine':
            return F.softmax(cos / self.corrT, dim=1)
        if self.ifrank_score_mode == 'if':
            if self.ifrank_combine == 'multiplyo':
                if_score = ifs
            else:
                if_score = self._row_zscore(ifs)
            return F.softmax(if_score / self.corrT, dim=1)
        if self.ifrank_combine == 'multiplyo':
            score = self.if_lambda * ifs + self.csim_lambda * cos
        elif self.ifrank_combine == 'multiply_balanced':
            score = self.if_lambda * self._row_zscore(ifs) + self.csim_lambda * self._row_zscore(cos)
        else:
            score = self.if_lambda * self._row_zscore(ifs) + self.csim_lambda * cos
        return F.softmax(score / self.corrT, dim=1)

    def _grad_surrogate(self, logits, target):
        p = F.softmax(logits, dim=1)
        if self.if_target == 'soft':
            return p * logits.sum(dim=1, keepdim=True) - logits
        return p - target

    def dif_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s, phi_lb, phi_uw, phi_us):
        num_cls = self.num_classes
        L = logits_x_lb.shape[0]
        k = min(self.num_references, L)
        if k < 2:
            return logits_x_ulb_s.new_zeros(())

        # labeled anchors are detached; rank teacher is built from weak view only.
        lg_l = logits_x_lb.detach()
        phi_lb_d = phi_lb.detach()
        phi_lb_n = F.normalize(phi_lb_d, dim=1)
        U = logits_x_ulb_w.shape[0]

        p_l = F.softmax(lg_l, dim=1)
        g_l = p_l - F.one_hot(y_lb, num_cls).float()
        y_u = F.one_hot(F.softmax(logits_x_ulb_w.detach(), dim=1).argmax(1), num_cls).float()

        g_uw = self._grad_surrogate(logits_x_ulb_w.detach(), y_u)
        g_us = self._grad_surrogate(logits_x_ulb_s, y_u)
        if self.if_mean_reduce:
            g_l = g_l / max(L, 1)
            g_uw = g_uw / max(U, 1)
            g_us = g_us / max(U, 1)

        IF_uw = self.if_tracin_scale * (g_uw @ g_l.t()) * (phi_uw.detach() @ phi_lb_d.t())
        IF_us = self.if_tracin_scale * (g_us @ g_l.t()) * (phi_us @ phi_lb_d.t())

        with torch.no_grad():
            ref_idx = self._select_ref_idx(IF_uw.detach(), IF_us.detach(), phi_lb_d, k)
            if_w = torch.gather(IF_uw.detach(), 1, ref_idx)
            cos_w = torch.gather(F.normalize(phi_uw.detach(), dim=1) @ phi_lb_n.t(), 1, ref_idx)
            rank_w = self._plackett_luce(self._fuse(if_w, cos_w)).detach()

        if_s = torch.gather(IF_us if self.use_strong_if else IF_uw.detach(), 1, ref_idx)
        cos_s = torch.gather(F.normalize(phi_us, dim=1) @ phi_lb_n.t(), 1, ref_idx)
        rank_s = self._plackett_luce(self._fuse(if_s, cos_s))
        return F.kl_div((rank_s + 1e-10).log(), rank_w, reduction='batchmean')

    @staticmethod
    def dif_arguments():
        return [
            SSL_Argument('--ifrank_loss_weight', float, 1.0),
            SSL_Argument('--corrT', float, 0.9),
            SSL_Argument('--num_references', int, 4),
            SSL_Argument('--ifrank_combine', str, 'multiply_balanced'),
            SSL_Argument('--if_lambda', float, 1.0),
            SSL_Argument('--csim_lambda', float, 1.0),
            SSL_Argument('--use_strong_if', str2bool, True),
            SSL_Argument('--if_tracin_scale', float, 1.0),
            SSL_Argument('--ref_select', str, 'by_instance'),
            SSL_Argument('--ref_cand_k', int, 8),
            SSL_Argument('--ifrank_score_mode', str, 'fused'),
            SSL_Argument('--if_target', str, 'hard'),
            SSL_Argument('--if_mean_reduce', str2bool, True),
            SSL_Argument('--ifrank_warmup_epochs', int, 1),
            SSL_Argument('--ifrank_warmup_mode', str, 'zero'),
        ]
