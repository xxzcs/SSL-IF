# SoftMatch + 闭式 last-layer TracIn 影响排序一致性 (纯 add; SoftMatch 无关系项)。
# 不修改原 softmatch.py。train_step 忠实复刻 SoftMatch + 追加 ifrank_loss(软目标 mixin)。
import torch

from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.softmatch.softmatch import SoftMatch
from semilearn.algorithms.utils.closed_form_ifrank import ClosedFormIFRankMixin


@ALGORITHMS.register('softmatch_ifcf')
class SoftMatchIFcf(ClosedFormIFRankMixin, SoftMatch):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init_ifrank(args)

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']; feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']; feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']; feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=logits_x_ulb_w, use_hard_label=self.use_hard_label, T=self.T)
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

            # ---- 闭式 IF 排序一致性 (纯新增关系项) ----
            ifrank_loss = self.ifrank_loss(logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s,
                                           feats_x_lb, feats_x_ulb_w, feats_x_ulb_s)
            ifrank_loss = ifrank_loss * self.ifrank_warmup_coef()

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.ifrank_loss_weight * ifrank_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(),
                                         ifrank_loss=ifrank_loss.item(), total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return SoftMatch.get_argument() + ClosedFormIFRankMixin.ifrank_arguments()
