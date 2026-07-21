# DeFixMatch + 闭式 last-layer TracIn 影响排序一致性 (纯 add)。
# 不修改原 defixmatch.py。train_step 逐字复刻 DeFixMatch(含去偏 anti_unsup_loss)+ 追加 ifrank_loss。
# IF 参考端用弱增强 labeled (x_lb) 的 feats/logits，与其他框架一致。
import torch

from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.defixmatch.defixmatch import DeFixMatch
from semilearn.algorithms.utils.closed_form_ifrank import ClosedFormIFRankMixin


@ALGORITHMS.register('defixmatch_ifcf')
class DeFixMatchIFcf(ClosedFormIFRankMixin, DeFixMatch):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init_ifrank(args)

    def train_step(self, x_lb, x_lb_s, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_lb_s, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb, logits_x_lb_s = outputs['logits'][:2*num_lb].chunk(2)
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][2*num_lb:].chunk(2)
                feats_x_lb, feats_x_lb_s = outputs['feat'][:2*num_lb].chunk(2)
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][2*num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']; feats_x_lb = outs_x_lb['feat']
                outs_x_lb_s = self.model(x_lb_s)
                logits_x_lb_s = outs_x_lb_s['logits']; feats_x_lb_s = outs_x_lb_s['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']; feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']; feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb': feats_x_lb, 'x_lb_s': feats_x_lb_s, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            sup_loss = (1/2) * (self.ce_loss(logits_x_lb, y_lb, reduction='mean')
                                + self.ce_loss(logits_x_lb_s, y_lb, reduction='mean'))

            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            probs_x_lb = self.compute_prob(logits_x_lb.detach())

            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())
                probs_x_lb = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_lb.detach())

            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            mask_lb = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_lb, softmax_x_ulb=False)

            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w, use_hard_label=self.use_hard_label,
                                          T=self.T, softmax=False)
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

            anti_pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                               logits=probs_x_lb, use_hard_label=self.use_hard_label,
                                               T=self.T, softmax=False)
            del probs_x_lb
            anti_unsup_loss = self.consistency_loss(logits_x_lb_s, anti_pseudo_label, 'ce', mask=mask_lb)

            # ---- 闭式 IF 排序一致性 (纯新增关系项) ----
            ifrank_loss = self.ifrank_loss(logits_x_lb, y_lb, logits_x_ulb_w, logits_x_ulb_s,
                                           feats_x_lb, feats_x_ulb_w, feats_x_ulb_s)
            ifrank_loss = ifrank_loss * self.ifrank_warmup_coef()

            total_loss = sup_loss + self.lambda_u * (unsup_loss - anti_unsup_loss) \
                + self.ifrank_loss_weight * ifrank_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), unsup_loss=unsup_loss.item(),
                                         anti_unsup_loss=anti_unsup_loss.item(),
                                         ifrank_loss=ifrank_loss.item(), total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item(),
                                         util_ratio_lb=mask_lb.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return DeFixMatch.get_argument() + ClosedFormIFRankMixin.ifrank_arguments()
