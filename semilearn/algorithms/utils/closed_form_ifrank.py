# 闭式 last-layer TracIn 影响排序一致性 —— 可复用 Mixin (从 simmatch_if 抽出, 逐字对齐)。
# 供无关系项框架 (FreeMatch/FlexMatch/FixMatch...) 以纯 add 方式接入同一套 IF-rank 损失,
# 保证跨框架通用性验证时 IF 实现完全一致。
#
# IF(u,l) = if_tracin_scale·[(p_u-y_u)·(p_l-y_l)]×[φ_u·φ_l]  (φ = fc 输入 512 维特征 = 'feat')
# IF 是 detached 引导(决定看哪些参考/怎么加权); 可训练梯度走强视图余弦一致性; 弱视图=teacher(detached); KL(强‖弱)。
import itertools
import torch
import torch.nn.functional as F

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
        self.if_tracin_scale = float(getattr(args, 'if_tracin_scale', 0.0)) or 1.0
        self.ref_select = getattr(args, 'ref_select', 'by_instance') or 'by_instance'
        assert self.ref_select in ('topk', 'by_instance'), f"未知 ref_select: {self.ref_select}"
        self.ref_cand_k = int(getattr(args, 'ref_cand_k', 8))
        self.if_target = getattr(args, 'if_target', 'soft') or 'soft'
        self.if_mean_reduce = bool(getattr(args, 'if_mean_reduce', True))
        # warmup: 前 warmup_epochs 个 epoch 抑制 ifrank_loss。
        #   mode='zero'   -> epoch < warmup_epochs 时系数=0，之后=1 (默认 warmup_epochs=1 == 旧的"只关 epoch0")
        #   mode='rampup' -> 系数 = min(1, epoch / warmup_epochs) 线性 0->1
        self.ifrank_warmup_epochs = int(getattr(args, 'ifrank_warmup_epochs', 1))
        self.ifrank_warmup_mode = getattr(args, 'ifrank_warmup_mode', 'zero') or 'zero'
        assert self.ifrank_warmup_mode in ('zero', 'rampup'), f"未知 ifrank_warmup_mode: {self.ifrank_warmup_mode}"
        self._perms = torch.tensor(
            list(itertools.permutations(range(self.num_references))), dtype=torch.long)

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

    def _select_ref_idx(self, IF_uw, IF_us, k):
        if self.ref_select == 'topk':
            return IF_uw.topk(k, dim=1).indices
        L = IF_uw.shape[1]
        cand = min(self.ref_cand_k, L)
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
        if self.ifrank_combine == 'multiplyo':
            score = self.if_lambda * ifs + self.csim_lambda * cos
        elif self.ifrank_combine == 'multiply_balanced':
            score = self.if_lambda * self._row_zscore(ifs) + self.csim_lambda * self._row_zscore(cos)
        else:  # 'multiply'
            score = self.if_lambda * self._row_zscore(ifs) + self.csim_lambda * cos
        return F.softmax(score / self.corrT, dim=1)

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
            p_l = F.softmax(lg_l, dim=1)
            g_l = p_l - F.one_hot(y_lb, num_cls).float()
            p_uw = F.softmax(lg_uw, dim=1); p_us = F.softmax(lg_us, dim=1)
            if if_target == 'soft':
                # 忠实 my_celoss(原始 logits 软目标): g = p·Σlogits − logits (实测与 Captum corr≈0.905)
                g_uw = p_uw * lg_uw.sum(1, keepdim=True) - lg_uw
                g_us = p_us * lg_us.sum(1, keepdim=True) - lg_us
            else:
                y_u = F.one_hot(p_uw.argmax(1), num_cls).float()   # 旧硬 argmax(仅对照)
                g_uw = p_uw - y_u; g_us = p_us - y_u
            if mean_reduce:
                g_l = g_l / max(L, 1); g_uw = g_uw / max(U, 1); g_us = g_us / max(U, 1)
            IF_uw = self.if_tracin_scale * (g_uw @ g_l.t()) * (phi_uw.detach() @ phi_lb_d.t())
            IF_us = self.if_tracin_scale * (g_us @ g_l.t()) * (phi_us.detach() @ phi_lb_d.t())
            ref_idx = self._select_ref_idx(IF_uw, IF_us, k)
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
            SSL_Argument('--if_target', str, 'soft'),         # soft=忠实(p·Σlogits−logits); hard=旧argmax
            SSL_Argument('--if_mean_reduce', str2bool, True),
            SSL_Argument('--ifrank_warmup_epochs', int, 1),   # 前 N epoch 抑制 (默认1=只关epoch0)
            SSL_Argument('--ifrank_warmup_mode', str, 'zero'),  # zero | rampup
        ]
