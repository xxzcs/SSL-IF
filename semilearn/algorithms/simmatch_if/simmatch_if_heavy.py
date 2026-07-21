# SimMatch + 【重型 TracIn】影响排序一致性: 直接调用用户 bus_test_if 的原版 Captum computeIF
# + compute_corrfea, 用于验证"用户原版方法在 SimMatch 上是否真有效"(排除闭式 0.905 近似误差)。
# 最小侵入: 继承 SimMatchIF, 只重写 _ifrank_loss(其余 SimMatch/闭式管线不变)。
# if_model = SimMatch 自己的 backbone(patch 成输出 logits), 保证 IF 用 SimMatch 真实特征。
import os, sys, time, types
import torch
import torch.nn.functional as F

from semilearn.core.utils import ALGORITHMS, get_net_builder
from semilearn.algorithms.simmatch_if.simmatch_if import SimMatchIF
from semilearn.algorithms.utils import SSL_Argument

# 引入用户原版 bus_test_if 代码
_BUS = '/home/xiexiaozheng/bus_test_if'
if _BUS not in sys.path:
    sys.path.insert(0, _BUS)


def _interleave(x, size):  # 复刻 train 脚本的 interleave(与 de_interleave 配对)
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def _logit_forward(self, x, *a, **k):
    o = self._orig_forward(x)
    return o['logits'] if isinstance(o, dict) else o


@ALGORITHMS.register('simmatch_if_heavy')
class SimMatchIFHeavy(SimMatchIF):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        from types import SimpleNamespace
        # 供 computeIF/compute_corrfea 用的 args(其余 getattr 有默认)
        self._ifargs = SimpleNamespace(
            batch_size=args.batch_size, mu=args.uratio, k=int(getattr(args, 'ref_cand_k', 8)),
            debias=False, device=torch.device('cuda'), combine=self.ifrank_combine,
            epochs=args.epoch, num_classes=args.num_classes,
            if_lambda=self.if_lambda, csim_lambda=self.csim_lambda, use_strong_if=self.use_strong_if)
        # 独立 if_model(与 backbone 同架构), 输出 logits 供 Captum
        m = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
        m._orig_forward = m.forward
        m.forward = types.MethodType(_logit_forward, m)
        self._ifmodel = m.cuda(self.gpu)
        self._heavy_imgs = None
        self._heavy_t = 0.0; self._heavy_n = 0

    def train_step(self, idx_lb, x_lb, y_lb, x_ulb_w, x_ulb_s):
        # 暂存原始图像给重型 _ifrank_loss(Captum 需要), 再走原 SimMatch 逻辑
        self._heavy_imgs = (x_lb, x_ulb_w, x_ulb_s)
        return super().train_step(idx_lb, x_lb, y_lb, x_ulb_w, x_ulb_s)

    def _ifrank_loss(self, logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s, phi_lb, phi_uw, phi_us):
        from compute_IF import computeIF_inbatch  # noqa
        from compute_corrfea import (select_reference_features_by_instance,
                                     compute_correlation_by_ifscore_csim, prob2rank_classification)
        x_lb, x_ulb_w, x_ulb_s = self._heavy_imgs
        num_lb = x_lb.shape[0]; num_ulb = x_ulb_w.shape[0]
        mu = num_ulb // max(num_lb, 1)
        self._ifargs.mu = mu; self._ifargs.batch_size = num_lb; self._ifargs.epochs = self.epochs
        num_refs = min(self.num_references, num_lb)

        net = self.model.module if hasattr(self.model, 'module') else self.model
        backbone = net.backbone
        state = {k: v.detach().clone() for k, v in backbone.state_dict().items()}

        _t0 = time.time()
        with torch.no_grad():
            inputs = _interleave(torch.cat((x_lb, x_ulb_w, x_ulb_s)), 2 * mu + 1).to(x_lb.device)
            prop_uw, oppo_uw, prop_us, oppo_us, ps_uw, ps_us, pi_uw, pi_us = computeIF_inbatch(
                self._ifargs, self._ifmodel, state, inputs, y_lb,
                logits_x_ulb_w.detach(), logits_x_ulb_s.detach())
        self._heavy_t += time.time() - _t0; self._heavy_n += 1
        if self._heavy_n % 50 == 0:
            print(f"[HEAVY_TIME] computeIF 平均 {self._heavy_t/self._heavy_n:.3f}s/step (n={self._heavy_n})", flush=True)

        # 选参考 + 融合(用户原版; 弱支特征 detached=teacher, 强支 live=梯度回传)
        ref_feats, sel_idx = select_reference_features_by_instance(
            phi_lb.detach(), prop_uw, oppo_uw, prop_us, oppo_us, num_refs)
        weak_corr, strong_corr = compute_correlation_by_ifscore_csim(
            self._ifargs, self.epoch, ps_uw, ps_us, pi_uw, pi_us,
            phi_uw.detach(), phi_us, ref_feats, sel_idx, self.corrT)
        weak_rank, strong_rank = prob2rank_classification(weak_corr, strong_corr, num_refs)
        return F.kl_div((strong_rank + 1e-10).log(), weak_rank, reduction='batchmean')

    @staticmethod
    def get_argument():
        return SimMatchIF.get_argument()
