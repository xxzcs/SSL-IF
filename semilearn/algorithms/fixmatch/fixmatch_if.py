import torch

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import FixedThresholdingHook, PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.utils.compute_IF import computeIF_inbatch
from semilearn.algorithms.utils.compute_corrfea import (
    compute_correlation_by_ifscore_csim_with_mapping,
    compute_correlation_by_ifscore_with_mapping,
    prob2rank_classification,
    select_reference_features_by_instance,
)


@ALGORITHMS.register('fixmatch_if')
class FixMatch_IF(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
        self.ifrank_loss_weight = args.ifrank_loss_weight
        self.if_target = getattr(args, 'if_target', 'soft')  # 'soft'=logits软目标(默认) / 'hard'=argmax硬标签喂给TracInCPFast

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), 'PseudoLabelingHook')
        self.register_hook(FixedThresholdingHook(), 'MaskingHook')
        super().set_hooks()

    def rank_loss_fn(self, feat_dict, prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, pindices_uw, oindices_uw, pindices_us, oindices_us, pscores_uw, oscores_uw, pscores_us, oscores_us):
        feas_x = feat_dict['x_lb']
        feas_u_w = feat_dict['x_ulb_w']
        feas_u_s = feat_dict['x_ulb_s']
        num_references = getattr(self.args, 'num_references', 4)
        corr_temperature = getattr(self.args, 'corrT', 0.5)

        reference_features, selected_indices = select_reference_features_by_instance(
            feas_x, prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, num_references
        )

        if not getattr(self.args, 'combine', None):
            weak_corr, strong_corr = compute_correlation_by_ifscore_with_mapping(
                pindices_uw,
                oindices_uw,
                pindices_us,
                oindices_us,
                pscores_uw,
                oscores_uw,
                pscores_us,
                oscores_us,
                selected_indices,
                corr_temperature,
            )
        else:
            weak_corr, strong_corr = compute_correlation_by_ifscore_csim_with_mapping(
                self.args,
                pindices_uw,
                oindices_uw,
                pindices_us,
                oindices_us,
                pscores_uw,
                oscores_uw,
                pscores_us,
                oscores_us,
                feas_u_w,
                feas_u_s,
                reference_features,
                selected_indices,
                corr_temperature,
            )

        weak_rank, strong_rank = prob2rank_classification(weak_corr, strong_corr, num_references)
        weak_rank = weak_rank.detach()

        weak_rank = weak_rank + 1e-12
        weak_rank = weak_rank / weak_rank.sum(dim=-1, keepdim=True)
        strong_rank_log = (strong_rank + 1e-12).log()

        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        return kl_loss(strong_rank_log, weak_rank)

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
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            if self.registered_hook('DistAlignHook'):
                probs_x_ulb_w = self.call_hook('dist_align', 'DistAlignHook', probs_x_ulb=probs_x_ulb_w.detach())

            mask = self.call_hook('masking', 'MaskingHook', logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            pseudo_label = self.call_hook(
                'gen_ulb_targets',
                'PseudoLabelingHook',
                logits=probs_x_ulb_w,
                use_hard_label=self.use_hard_label,
                T=self.T,
                softmax=False,
            )

            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

            current_model = self.model.module if hasattr(self.model, 'module') else self.model
            state = current_model.state_dict()
            if not hasattr(self.args, 'mu'):
                self.args.mu = self.args.uratio

            try:
                if getattr(self, 'if_target', 'soft') == 'hard':
                    _ifw_tgt = logits_x_ulb_w.argmax(1).detach(); _ifs_tgt = logits_x_ulb_s.argmax(1).detach()
                else:
                    _ifw_tgt = logits_x_ulb_w.detach(); _ifs_tgt = logits_x_ulb_s.detach()
                prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, pscores_uw, oscores_uw, pscores_us, oscores_us = computeIF_inbatch(
                    self.args, state, x_lb, x_ulb_w, x_ulb_s, y_lb, _ifw_tgt, _ifs_tgt
                )
                rank_loss = self.rank_loss_fn(
                    feat_dict,
                    prop_list_uw,
                    oppo_list_uw,
                    prop_list_us,
                    oppo_list_us,
                    prop_list_uw,
                    oppo_list_uw,
                    prop_list_us,
                    oppo_list_us,
                    pscores_uw,
                    oscores_uw,
                    pscores_us,
                    oscores_us,
                )
                if torch.isnan(rank_loss):
                    self.print_fn(f'WARNING: rank_loss is NaN at iter {self.it}. Setting to 0.0.')
                    rank_loss = torch.tensor(0.0, device=sup_loss.device)
            except Exception as error:
                self.print_fn(f'ERROR in IF/rank loss calculation: {error}. Skipping rank loss for this step.')
                rank_loss = torch.tensor(0.0, device=sup_loss.device)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.ifrank_loss_weight * rank_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(
            sup_loss=sup_loss.item(),
            unsup_loss=unsup_loss.item(),
            total_loss=total_loss.item(),
            util_ratio=mask.float().mean().item(),
            rank_loss=rank_loss.item(),
        )
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--ifrank_loss_weight', float, 1.0),
            SSL_Argument('--k', int, 8),
            SSL_Argument('--num_references', int, 4),
            SSL_Argument('--corrT', float, 0.5),
            SSL_Argument('--combine', str, 'multiplyo'),
            SSL_Argument('--if_lambda', float, 10.0),
            SSL_Argument('--csim_lambda', float, 1.0),
            SSL_Argument('--use_strong_if', str2bool, False),
            SSL_Argument('--if_target', str, 'soft'),
        ]