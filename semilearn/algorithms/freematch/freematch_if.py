import torch
import torch.nn.functional as F

from .utils import FreeMatchThresholingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.utils.compute_IF import computeIF_inbatch
from semilearn.algorithms.utils.compute_corrfea import select_reference_features_by_instance, compute_correlation_by_ifscore_csim_with_mapping, prob2rank_classification, compute_correlation_by_ifscore_with_mapping


#incorporate influence function based rank loss into Freematch
# TODO: move these to .utils or algorithms.utils.loss
def replace_inf_to_zero(val):
    val[~torch.isfinite(val)] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()

@ALGORITHMS.register('freematch_if')
class FreeMatch_IF(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile, clip_thresh=args.clip_thresh)
        self.lambda_e = args.ent_loss_ratio
        self.ifrank_loss_weight = args.ifrank_loss_weight

    def init(self, T, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False):
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh


    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FreeMatchThresholingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")
        super().set_hooks()

    def rank_loss_fn(self, feat_dict, prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, pindices_uw, oindices_uw, pindices_us, oindices_us, pscores_uw, oscores_uw, pscores_us, oscores_us):
        feas_x = feat_dict['x_lb']
        feas_u_w = feat_dict['x_ulb_w']
        feas_u_s = feat_dict['x_ulb_s']
        num_references = getattr(self.args, 'num_references', 4)
        corr_temperature = getattr(self.args, 'corrT', 0.5)
        epoch = self.it // (self.num_train_iter // self.epochs) if self.epochs > 0 else 0

        # STEP 1. 选择参考特征向量
        reference_features, selected_indices, selected_sources = select_reference_features_by_instance(
            feas_x, prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, num_references)

        # STEP 2. 计算相关性
        if not getattr(self.args, 'combine', None):
             # 3' correlation is computed by using influence score instead of the cosine similarity
             weak_corr, strong_corr = compute_correlation_by_ifscore_with_mapping(
                 pindices_uw, oindices_uw, pindices_us, oindices_us,
                 pscores_uw, oscores_uw, pscores_us, oscores_us, selected_indices, selected_sources, corr_temperature)
        else:
             # 4' correlation is computed by combining influence score and cosine similarity
             weak_corr, strong_corr = compute_correlation_by_ifscore_csim_with_mapping(
                 self.args, pindices_uw, oindices_uw, pindices_us, oindices_us,
                 pscores_uw, oscores_uw, pscores_us, oscores_us,
                 feas_u_w, feas_u_s, reference_features, selected_indices, selected_sources, corr_temperature)
                
        # STEP 3. 转换为排名分布
        weak_rank, strong_rank = prob2rank_classification(weak_corr, strong_corr, num_references) 

        # STEP 4. 计算一致性损失
        # 核心修改：将弱增强分支（Teacher）设为固定目标，梯度仅通过强增强分支（Student）回传
        # 这样可以防止 IF 带来的额外梯度干扰 FreeMatch 对弱增强特征的基准统计（即 p_model/threshold）
        weak_rank = weak_rank.detach()

        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        # 增加 eps 并确保 weak_rank 也要满足归一化要求，防止 KL 散度出现负值
        weak_rank = weak_rank + 1e-12
        weak_rank = weak_rank / weak_rank.sum(dim=-1, keepdim=True)
        
        strong_rank_log = (strong_rank + 1e-12).log()
        rank_loss = kl_loss(strong_rank_log, weak_rank)

        return rank_loss

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
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
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}


            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # calculate mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w) #"masking" is the method need to be called in MaskingHook function


            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            # 
            # calculate unlabeled loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)
            
            # calculate entropy loss
            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist) 
            else:
               ent_loss = 0.0
            
            # calculate rank loss
            current_model = self.model.module if hasattr(self.model, "module") else self.model 
            state = current_model.state_dict()
            
            if not hasattr(self.args, 'mu'):
                self.args.mu = self.args.uratio
            
            # 使用 try-except 捕获可能导致 NaN 的 IF 计算过程，并添加详细日志
            try:
                # 检查输入数据是否有 NaN
                if torch.isnan(x_lb).any() or torch.isnan(x_ulb_w).any() or torch.isnan(x_ulb_s).any():
                    self.print_fn("CRITICAL: Input data contains NaN before IF calculation")
                
                # --- 移除冗余 Logits 范围检查 ---

                prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, pscores_uw, oscores_uw, pscores_us, oscores_us = computeIF_inbatch(
                    self.args, state, x_lb, x_ulb_w, x_ulb_s, y_lb, logits_x_ulb_w.detach(), logits_x_ulb_s.detach())
                
                # --- 每 100 个 iter 输出一次详细的 IF 数值分布，兼顾监控与性能 ---
                if self.it % 100 == 0:
                    with torch.no_grad():
                        def get_stats(t, name):
                            t_flat = t.detach().cpu()
                            mean = t_flat.mean().item()
                            max_val = t_flat.max().item()
                            min_val = t_flat.min().item()
                            std = t_flat.std().item()
                            return f"{name}: mean={mean:.4f}, max={max_val:.4f}, min={min_val:.4f}, std={std:.4f}"

                        self.print_fn(f"\n[IF Statistics at Iter {self.it}]")
                        self.print_fn(get_stats(pscores_uw, "Weak_IF"))
                        self.print_fn(get_stats(pscores_us, "Strong_IF"))
                # -------------------------------------------------------------
                
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
                
                # 如果 rank_loss 本身是 NaN，打印警告并置零，防止污染整个模型
                if torch.isnan(rank_loss):
                    self.print_fn(f"WARNING: rank_loss is NaN at iter {self.it}. Setting to 0.0 to prevent gradient pollution.")
                    rank_loss = torch.tensor(0.0, device=rank_loss.device)

            except Exception as e:
                self.print_fn(f"ERROR in IF/Rank loss calculation: {e}. Skipping rank_loss for this step.")
                rank_loss = torch.tensor(0.0, device=sup_loss.device)
            
            # Print losses and rank_loss to debug NaN
            if self.it % self.args.num_log_iter == 0:
                self.print_fn(f"Iter {self.it}: sup {sup_loss.item():.4f}, unsup {unsup_loss.item():.4f}, ent {ent_loss.item() if isinstance(ent_loss, torch.Tensor) else ent_loss:.4f}, rank {rank_loss.item():.4f}")

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss + self.ifrank_loss_weight * rank_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item(),
                                         ent_loss=ent_loss.item() if isinstance(ent_loss, torch.Tensor) else ent_loss,
                                         rank_loss=rank_loss.item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['MaskingHook'].p_model.cpu()
        save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        save_dict['label_hist'] = self.hooks_dict['MaskingHook'].label_hist.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--ent_loss_ratio', float, 0.01),
            SSL_Argument('--ifrank_loss_weight', float, 1.0),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
            SSL_Argument('--k', int, 8),
            SSL_Argument('--num_references', int, 4),
            SSL_Argument('--corrT', float, 0.5),
            SSL_Argument('--combine', str, 'multiplyo'),
            SSL_Argument('--if_lambda', float, 10.0),
            SSL_Argument('--csim_lambda', float, 1.0),
            SSL_Argument('--use_strong_if', str2bool, False),
        ]