import os
import random
import time
import numpy as np
import torch
from captum.influence import TracInCPFast
from torch.utils.data import Dataset
import torch.nn.functional as F
# from models.resnet import resnet18 
from itertools import permutations
from collections import Counter, OrderedDict

def select_reference_features(labeled_features, prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, num_references):
    """
    根据特征距离选择参考向量
    根据IF得分选择有标注数据特征空间中几个有代表性的样本作为参考向量
    
    Args:
        labeled_features: 有标注样本的特征 [B_labeled, C] [8,512]
        num_references: 参考向量数量
    
    Returns:
        reference_features: 参考特征 [num_references, C]
        selected_indices: 选中的索引 [num_references]
    """
    B, C = labeled_features.shape
    if B <= num_references:
        return labeled_features, torch.arange(B)  # 如果样本数不足，返回所有样本
    
    '''
    # 根据特征选择参考向量，计算特征间的欧氏距离矩阵
    distances = torch.cdist(labeled_features, labeled_features, p=2)  # [B, B]
    
    # 贪心选择距离最远的参考向量
    selected_indices = []
    
    # 随机选择第一个参考点
    first_idx = torch.randint(0, B, (1,)).item()
    selected_indices.append(first_idx)
    
    # 依次选择与已选点距离最远的点
    for i in range(1, num_references):
        # 计算每个未选点到已选点的最小距离
        min_distances = []
        for j in range(B):
            if j in selected_indices:
                min_distances.append(-1)  # 已选点标记为-1
            else:
                # 计算到已选点的最小距离
                dist_to_selected = [distances[j, idx].item() for idx in selected_indices]
                min_distances.append(min(dist_to_selected))
        
        # 选择最小距离最大的点
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)
    '''
    
    # select four consistent sample features for all unlabeled images in batch from prop_list and oppo_list
    candidate_list = [prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us]
    selected_indices = []
    used_indices = set()

    all_frequency_stats = []

    for listname in candidate_list:
        first_four = [row[:4] for row in listname]
        flattend = [item for sublist in first_four for item in sublist]

        freq_counter = Counter(flattend)
        all_frequency_stats.append(freq_counter)

    # 为每个list按顺序选择最佳索引
    for i, freq_counter in enumerate(all_frequency_stats):
        # 获取当前list的频率排序
        sorted_candidates = freq_counter.most_common()
        
        # 选择第一个未被使用的索引
        chosen_idx = None
        for idx, freq in sorted_candidates:
            if idx not in used_indices:
                chosen_idx = idx
                break
        
        # 如果所有索引都被用了，选择频率最高的未在selected_indices中的索引
        if chosen_idx is None:
            for idx, freq in sorted_candidates:
                if idx not in selected_indices:
                    chosen_idx = idx
                    break
        
        selected_indices.append(chosen_idx)
        used_indices.add(chosen_idx)

    #     return selected_indices, {
    #     'frequency_stats': all_frequency_stats,
    #     'selection_order': selected_indices,
    #     'used_indices': list(used_indices)
    # }


    # 返回选中的参考特征和索引
    reference_features = labeled_features[selected_indices]  # [num_references, C]
    selected_indices_tensor = torch.tensor(selected_indices, device=labeled_features.device)
    
    
    return reference_features, selected_indices_tensor

# select four orthogonal instances from labeled features for all unlabeled features
def find_orthogonal_neighbors(labeled_features, num_references):
    """
    From labeled features, greedily select K orthogonal reference samples.
    Then extract corresponding features from unlabeled weak/strong features using the same indices.

    This set is shared across ALL unlabeled samples in the batch.
    """    

    L, D = labeled_features.shape
    # Step 1: Normalize
    labeled_norm = F.normalize(labeled_features, p=2, dim=-1)

    # Step 2: Greedy orthogonal selection
    selected_mask = torch.zeros(L, dtype=torch.bool, device=labeled_features.device)
    selected_features = []
    indices = []
    for k in range(num_references):
        if k == 0:
            # Start with the most energetic sample (best heuristic)
            energy = (labeled_features ** 2).sum(dim=-1)
            idx = energy.argmax().item()
        else:
            # Compute max cosine similarity to all previously selected
            selected_stack = torch.stack(selected_features, dim=0)  # (k, D)
            cos_sim_to_selected = torch.einsum('l c, k c -> l k', labeled_norm, selected_stack).abs()  # (L, K)
            max_cos_sim = cos_sim_to_selected.max(dim=1).values  # (L,)
            max_cos_sim[selected_mask] = 10.0  # mask out already selected
            idx = max_cos_sim.argmin().item()

        selected_features.append(labeled_norm[idx])
        selected_mask[idx] = True
        indices.append(idx)

    # Final reference set: (K, D)
    ref_features = torch.stack(selected_features, dim=0)

    return ref_features, indices  # (K, D)


################
def find_early_common_elements(list1, list2):
    common_elements = list(set(list1) & set(list2))
    if len(common_elements) < 2:
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # print(list1)
        return list1[:2]
    common_elements.sort(key=lambda x: list1.index(x))
    # print(common_elements[:2])
    # if len(common_elements) < 2:            
    #     common_elements = list1
    
    return common_elements[:2]

def select_reference_features_by_instance(labeled_features, prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, num_references):
    # select four different sample features for different unlabeled paired images in batch from prop_list and oppo_list
    """
    为每个锚点特征选择个性化参考样本
    
    Args:
        list: 锚点特征 [UB, num_references] - 弱增强或强增强样本对应的有标注图像的IF索引组成的list
        labeled_features: 有标注样本特征 [B_labeled, C]
        num_references: 选择的参考数量
    
    Returns:
        references: 为每个锚点选择的参考特征 [B, num_references, C]
    """
    indices = []
    # 保证输入为tensor时转为list
    prop_list_uw_list = prop_list_uw.tolist() if isinstance(prop_list_uw, torch.Tensor) else prop_list_uw
    prop_list_us_list = prop_list_us.tolist() if isinstance(prop_list_us, torch.Tensor) else prop_list_us
    oppo_list_uw_list = oppo_list_uw.tolist() if isinstance(oppo_list_uw, torch.Tensor) else oppo_list_uw
    oppo_list_us_list = oppo_list_us.tolist() if isinstance(oppo_list_us, torch.Tensor) else oppo_list_us

    for (list1, list2), (list3, list4) in zip(
            zip(prop_list_uw_list, prop_list_us_list),
            zip(oppo_list_uw_list, oppo_list_us_list)):
        current_indice_prop = find_early_common_elements(list1, list2)
        current_indice_oppo = find_early_common_elements(list3, list4)
        current_indice = current_indice_prop + current_indice_oppo
        indices.append(current_indice)
    indices = torch.tensor(indices, device=labeled_features.device)
    B, N = indices.shape
    features_dim = labeled_features.shape[1]
    flat_indices = indices.view(-1)
    select_reference_features = labeled_features[flat_indices]
    referenced_features = select_reference_features.view(B, N, features_dim)
    return referenced_features, indices

def find_early_common_elements_diverse(list1, list2, sim_matrix, already_selected, num_to_pick):
    """
    在两个列表中寻找共同的前几名，且要求选择的样本与已选样本相似度不能太高。
    """
    from collections import OrderedDict
    # 合并两个列表并保持顺序（优先级）
    candidates = list(OrderedDict.fromkeys(list1 + list2))
    selected = []
    
    for c in candidates:
        if c in already_selected: continue
        
        # 计算与已选样本的最大相似度
        current_full_set = already_selected + selected
        if len(current_full_set) == 0:
            max_sim = 0
        else:
            max_sim = max([sim_matrix[c, s].item() for s in current_full_set])
        
        # 如果相似度小于 0.9，或者这是第一个备选，或者没得选了，就选中它
        if max_sim < 0.9 or len(selected) == 0:
            selected.append(c)
        
        if len(selected) == num_to_pick:
            break
            
    # 如果没选够（比如大家都太像了），就放宽要求补齐
    if len(selected) < num_to_pick:
        for c in candidates:
            if c not in selected and c not in already_selected:
                selected.append(c)
                if len(selected) == num_to_pick: break
                
    return selected
# 2' first select top-2 prop and top-2 oppo candidates, then select 4 most orthogonal samples from them
def select_reference_features_by_instance_diverse(labeled_features, prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, num_references):
    """
    为每个锚点特征选择个性化参考样本，增加多样性约束（独立方法）
    """
    indices = []
    prop_list_uw_list = prop_list_uw.tolist() if isinstance(prop_list_uw, torch.Tensor) else prop_list_uw
    prop_list_us_list = prop_list_us.tolist() if isinstance(prop_list_us, torch.Tensor) else prop_list_us
    oppo_list_uw_list = oppo_list_uw.tolist() if isinstance(oppo_list_uw, torch.Tensor) else oppo_list_uw
    oppo_list_us_list = oppo_list_us.tolist() if isinstance(oppo_list_us, torch.Tensor) else oppo_list_us

    labeled_norm = F.normalize(labeled_features, p=2, dim=-1)
    sim_matrix = torch.matmul(labeled_norm, labeled_norm.T)

    for i in range(len(prop_list_uw_list)):
        current_selected = find_early_common_elements_diverse(
            prop_list_uw_list[i], prop_list_us_list[i], sim_matrix, [], 2
        )
        current_selected += find_early_common_elements_diverse(
            oppo_list_uw_list[i], oppo_list_us_list[i], sim_matrix, current_selected, 2
        )
        indices.append(current_selected)
        
    indices = torch.tensor(indices, device=labeled_features.device)
    B, N = indices.shape
    features_dim = labeled_features.shape[1]
    referenced_features = labeled_features[indices.view(-1)].view(B, N, features_dim)
    return referenced_features, indices
# 3' first select top-3 prop and top-3 oppo candidates, then select 4 most orthogonal samples from them
def select_reference_features_by_if_diversity(labeled_features, prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, num_references):
    """
    策略 A：在 IF 高分样本中寻找多样性。
    逻辑：先找到正向和负向影响最高的候选者，然后用贪心法选出其中最正交的样本。
    """
    indices = []
    prop_list = prop_list_uw.tolist() if isinstance(prop_list_uw, torch.Tensor) else prop_list_uw
    oppo_list = oppo_list_uw.tolist() if isinstance(oppo_list_uw, torch.Tensor) else oppo_list_uw
    
    labeled_norm = F.normalize(labeled_features, p=2, dim=-1)
    sim_matrix = torch.matmul(labeled_norm, labeled_norm.T) 

    for i in range(len(prop_list)):
        # 候选池：取正向 Top-3 和 负向 Top-3
        candidates = list(OrderedDict.fromkeys(prop_list[i][:3] + oppo_list[i][:3]))
        
        selected = []
        # 第一个必选 IF 分数最高的
        selected.append(candidates[0])
        
        # 剩下的从候选池里选跟已经选中的最不像的
        for _ in range(num_references - 1):
            best_val = 1.0
            best_idx = -1
            for c in candidates:
                if c in selected: continue
                # 计算到已选集合的最大相似度
                max_sim = max([sim_matrix[c, s].item() for s in selected])
                if max_sim < best_val:
                    best_val = max_sim
                    best_idx = c
            if best_idx != -1:
                selected.append(best_idx)
            else:
                # 兜底：如果候选不够，顺着填入
                for c in candidates:
                    if c not in selected: 
                        selected.append(c)
                        if len(selected) == num_references: break
        
        indices.append(selected[:num_references])

    indices = torch.tensor(indices, device=labeled_features.device)
    B, N = indices.shape
    features_dim = labeled_features.shape[1]
    referenced_features = labeled_features[indices.view(-1)].view(B, N, features_dim)
    return referenced_features, indices

# 1' first select two most orthogonal samples globally, then select two highest IF samples for each unlabeled sample
def select_reference_features_hybrid(labeled_features, prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, num_references):
    """
    策略 B：优化版 2+2。
    目标：修正昨天实验中 RefSim 0.98 的问题。
    逻辑：强制选择 2 个共同正向 (Prop) 和 2 个共同负向 (Oppo) 样本。
    区别：相比 by_instance，增加了在负样本中寻找与正样本差异最大的过程，强制打破塌陷。
    """
    B_ul = len(prop_list_uw)
    labeled_norm = F.normalize(labeled_features, p=2, dim=-1)
    sim_matrix = torch.matmul(labeled_norm, labeled_norm.T)

    indices = []
    p_uw = prop_list_uw.tolist() if isinstance(prop_list_uw, torch.Tensor) else prop_list_uw
    p_us = prop_list_us.tolist() if isinstance(prop_list_us, torch.Tensor) else prop_list_us
    o_uw = oppo_list_uw.tolist() if isinstance(oppo_list_uw, torch.Tensor) else oppo_list_uw
    o_us = oppo_list_us.tolist() if isinstance(oppo_list_us, torch.Tensor) else oppo_list_us

    for i in range(B_ul):
        # 1. 选出指定数量的一半作为共同正向点 (Proponents)
        # 比如 num_references=4，则选 2 个
        num_prop = num_references // 2
        num_oppo = num_references - num_prop

        current_prop = find_early_common_elements(p_uw[i], p_us[i])
        selected_prop = current_prop[:num_prop]
        
        # 2. 从共同负向点 (Opponents) 中选出与正样本差异最大的
        common_oppos = list(OrderedDict.fromkeys(o_uw[i] + o_us[i]))
        
        # 多样性打分：计算该负样本到已选正样本的最大相似度
        scored_oppos = []
        for oc in common_oppos:
            if oc in selected_prop: continue
            # 防止 selected_prop 为空
            if not selected_prop:
                max_sim = 0
            else:
                max_sim = max([sim_matrix[oc, p].item() for p in selected_prop])
            scored_oppos.append((oc, max_sim))
        
        # 排序：取相似度最低的 num_oppo 个负样本（最正交）
        scored_oppos.sort(key=lambda x: x[1])
        selected_oppo = [x[0] for x in scored_oppos[:num_oppo]]
        
        # 汇总
        current_selected = selected_prop + selected_oppo
        
        # 兜底补齐逻辑
        if len(current_selected) < num_references:
            for cand in (common_oppos + p_uw[i]):
                if cand not in current_selected:
                    current_selected.append(cand)
                    if len(current_selected) == num_references: break

        indices.append(current_selected[:num_references])

    indices = torch.tensor(indices, device=labeled_features.device)
    features_dim = labeled_features.shape[1]
    referenced_features = labeled_features[indices.view(-1)].view(B_ul, num_references, features_dim)
    return referenced_features, indices

# 5' Global Orthogonal Anchors + Local Prop/Oppo (2+1+1)
def select_reference_features_global_hybrid(labeled_features, prop_list_uw, oppo_list_uw, prop_list_us, oppo_list_us, num_references):
    """
    方案 1'（进阶版）：2个全局正交锚点 + 1个局部最高支持者 + 1个局部最高反对者。
    - 结构：2 (Global Orthogonal) + 1 (Local Prop) + 1 (Local Oppo)。
    - 优点：比纯支持者方案更防坍缩，因为它在排名中引入了“负向拉力”。
    """
    B_ul = len(prop_list_uw)
    B_l = labeled_features.shape[0]
    labeled_norm = F.normalize(labeled_features, p=2, dim=-1)
    sim_matrix = torch.matmul(labeled_norm, labeled_norm.T)

    # 1. 全局阶段：找 Batch 内最正交的两个点作为骨架 (Global Anchors)
    dist_matrix = sim_matrix.clone()
    mask = torch.eye(B_l, device=labeled_features.device).bool()
    dist_matrix[mask] = 2.0  # 排除对角线
    
    min_val, min_idx = torch.min(dist_matrix.view(-1), dim=0)
    idx1 = min_idx // B_l
    idx2 = min_idx % B_l
    global_anchors = [idx1.item(), idx2.item()]

    # 2. 局部阶段：挑选最具代表性的 正/负 样本
    indices = []
    p_uw = prop_list_uw.tolist() if isinstance(prop_list_uw, torch.Tensor) else prop_list_uw
    p_us = prop_list_us.tolist() if isinstance(prop_list_us, torch.Tensor) else prop_list_us
    o_uw = oppo_list_uw.tolist() if isinstance(oppo_list_uw, torch.Tensor) else oppo_list_uw
    o_us = oppo_list_us.tolist() if isinstance(oppo_list_us, torch.Tensor) else oppo_list_us

    for i in range(B_ul):
        # 寻找共同支持者和反对者
        common_prop = find_early_common_elements(p_uw[i], p_us[i])
        common_oppo = find_early_common_elements(o_uw[i], o_us[i])
        
        # 挑选 1 个局部最高支持者（排除已选锚点）
        best_prop = []
        for p in common_prop:
            if p not in global_anchors:
                best_prop = [p]
                break
        
        # 挑选 1 个局部最高反对者（排除已选锚点）
        best_oppo = []
        for o in common_oppo:
            if o not in global_anchors and o not in best_prop:
                best_oppo = [o]
                break
        
        # 汇总
        current_selected = global_anchors + best_prop + best_oppo
        
        # 兜底补齐 (如果数量不够)
        if len(current_selected) < num_references:
            for cand in (common_prop + common_oppo):
                if cand not in current_selected:
                    current_selected.append(cand)
                    if len(current_selected) == num_references: break
        
        indices.append(current_selected[:num_references])

    indices = torch.tensor(indices, device=labeled_features.device)
    features_dim = labeled_features.shape[1]
    referenced_features = labeled_features[indices.view(-1)].view(B_ul, num_references, features_dim)
    return referenced_features, indices

def compute_feature_correlation(features, reference_features, temperature):
    """
    计算分类特征与参考向量的相关性（余弦相似度）
    
    Args:
        features: 输入特征 [B, C]
        reference_features: 参考特征 [num_refs, C]
    
    Returns:
        correlation: 相关性矩阵 [B, num_refs]
    """
    B, C = features.shape
    num_refs = reference_features.shape[0]
    
    # 计算余弦相似度
    # features: [B, C]
    # reference_features: [num_refs, C]
    correlation = F.cosine_similarity(
        features.unsqueeze(1),  # [B, 1, C]
        reference_features.unsqueeze(0),  # [1, num_refs, C]
        dim=2
    )  # [B, num_refs]
    
    # 归一化为概率分布
    # temperature = 0.05
    correlation_prob = F.softmax(correlation / temperature, dim=-1)
    
    return correlation_prob

def compute_correlation_by_ifscore(pscores_uw, pscores_us, indices, temperature):
    """
    计算分类特征与参考向量的相关性（余弦相似度）
    
    Args:
        features: 输入特征 [B, C]
        reference_features: 参考特征 [num_refs, C]
    
    Returns:
        correlation: 相关性矩阵 [B, num_refs]
    """
    # B, C = features.shape
    # num_refs = reference_features.shape[0]

    
    # 归一化为概率分布
    # temperature = 0.05
    # correlation_prob = F.softmax(correlation / temperature, dim=-1)
    # method 1: select 4 indices randomly for different uw/us pairs
    # index_matrix = np.array([random.sample(range(8), 4) for _ in range(240)])
    # print("!!!!!!!!!!!!!!!!!!!!!!")
    # print(indices)
    # print(indices.shape)

    # method 2: select 4 common indices by using the frequency in four prop/oppo lists for different uw/us pairs
    # indices = indices.unsqueeze(0).repeat(240, 1)
    # print(indices)
    # print(indices.shape)

    # indices = torch.from_numpy(index_matrix)
    # method 3: IF score is gathered from the selected indices

    if not indices.is_cuda and torch.cuda.is_available():
        indices = indices.to('cuda')
    selected_uw_scores = torch.gather(
        pscores_uw,
        1,
        indices
    )
    selected_us_scores = torch.gather(
        pscores_us,
        1,
        indices
    )
    # temperature = 1
    weak_corr_prob = F.softmax(torch.clamp(selected_uw_scores / temperature, min=-50, max=50), dim=-1)
    strong_corr_prob = F.softmax(torch.clamp(selected_us_scores / temperature, min=-50, max=50), dim=-1)
    # print(indices[:5])
    
    return weak_corr_prob, strong_corr_prob


def gather_if_scores_by_indices(selected_indices, prop_indices, prop_scores):
    if not selected_indices.is_cuda and prop_indices.is_cuda:
        selected_indices = selected_indices.to(prop_indices.device)
    batch_size, num_refs = selected_indices.shape
    gathered_scores = torch.zeros(batch_size, num_refs, device=prop_scores.device, dtype=prop_scores.dtype)
    for batch_idx in range(batch_size):
        for ref_idx in range(num_refs):
            labeled_index = selected_indices[batch_idx, ref_idx]
            match = (prop_indices[batch_idx] == labeled_index).nonzero(as_tuple=False)
            if match.numel() > 0:
                gathered_scores[batch_idx, ref_idx] = prop_scores[batch_idx, match[0, 0]]
    return gathered_scores


def _zscore_per_sample(scores):
    score_mean = scores.mean(dim=1, keepdim=True)
    score_std = scores.std(dim=1, keepdim=True) + 1e-10
    return (scores - score_mean) / score_std


def build_signed_if_scores(selected_scores, normalize=True):
    if normalize:
        return _zscore_per_sample(selected_scores.abs())
    else:
        return selected_scores.abs()


def compute_correlation_by_ifscore_with_mapping(
    prop_indices_uw,
    oppo_indices_uw,
    prop_indices_us,
    oppo_indices_us,
    pscores_uw,
    oscores_uw,
    pscores_us,
    oscores_us,
    indices,
    temperature,
):
    selected_uw_scores = gather_if_scores_by_indices(indices, prop_indices_uw, pscores_uw)
    selected_us_scores = gather_if_scores_by_indices(indices, prop_indices_us, pscores_us)

    weak_corr_prob = F.softmax(torch.clamp(selected_uw_scores / temperature, min=-50, max=50), dim=-1)
    strong_corr_prob = F.softmax(torch.clamp(selected_us_scores / temperature, min=-50, max=50), dim=-1)
    return weak_corr_prob, strong_corr_prob

def compute_correlation_by_ifscore_csim(args, pscores_uw, pscores_us, feas_u_w, feas_u_s, ref_features, indices, temperature):
    """
    计算分类特征与参考向量的相关性（余弦相似度）
    
    Args:
        features: 输入特征 [B, C]
        reference_features: 参考特征 [num_refs, C]
    
    Returns:
        correlation: 相关性矩阵 [B, num_refs]
    """
   
    # method 4: calculate the correlation score by combing IF score and cosine similarity score

    if not indices.is_cuda and torch.cuda.is_available():
        indices = indices.to('cuda')

    selected_uw_scores = torch.gather(
        pscores_uw,
        1,
        indices
    )
    selected_us_scores = torch.gather(
        pscores_us,
        1,
        indices
    )
    cosine_correlation_uw = F.cosine_similarity(
        feas_u_w.unsqueeze(1),  # [B, 1, C]
        ref_features,             # [B, num_refs, C]
        dim=2
    )  # [B, num_refs]
    
    cosine_correlation_us = F.cosine_similarity(
        feas_u_s.unsqueeze(1),  # [B, 1, C]
        ref_features,             # [B, num_refs, C]
        dim=2
    )  # [B, num_refs]

    # 直接进入合并逻辑，删掉此处容易导致溢出的冗余 softmax
    
    if args.combine == 'add':
        # ... (此处逻辑保持不变)
        weak_corr_prob = F.softmax(selected_uw_scores / temperature, dim=-1)
        strong_corr_prob = F.softmax(selected_us_scores / temperature, dim=-1)
        cosine_correlation_wprob = F.softmax(cosine_correlation_uw / temperature, dim=-1)
        cosine_correlation_sprob = F.softmax(cosine_correlation_us / temperature, dim=-1)
        weak_corr_prob += cosine_correlation_wprob
        strong_corr_prob += cosine_correlation_sprob
        need_norm = True
    elif args.combine == 'multiply':
        # 模式 1：仅对影响函数 (IF) 得分做标准化，保持余弦相似度 (CosSim) 原始尺度
        if_weight = getattr(args, 'if_lambda', 1.0)
        csim_weight = getattr(args, 'csim_lambda', 1.0)
        use_strong = getattr(args, 'use_strong_if', False)
        
        # 1. 标准化 IF 得分
        if_mean_uw = selected_uw_scores.mean(dim=1, keepdim=True)
        if_std_uw = selected_uw_scores.std(dim=1, keepdim=True) + 1e-10
        norm_if_uw = (selected_uw_scores - if_mean_uw) / if_std_uw

        # 确定强分支的指导分数
        if use_strong:
            if_mean_us = selected_us_scores.mean(dim=1, keepdim=True)
            if_std_us = selected_us_scores.std(dim=1, keepdim=True) + 1e-10
            norm_if_guidance = (selected_us_scores - if_mean_us) / if_std_us
        else:
            norm_if_guidance = norm_if_uw

        # 融合：标准化的 IF + 原始尺度 CosSim
        # 使用 clamp 限制数值范围，防止 softmax 溢出产生 NaN
        weak_score = (if_weight * norm_if_uw + csim_weight * cosine_correlation_uw) / temperature
        strong_score = (if_weight * norm_if_guidance + csim_weight * cosine_correlation_us) / temperature
        
        weak_corr_prob = F.softmax(torch.clamp(weak_score, min=-50, max=50), dim=-1)
        strong_corr_prob = F.softmax(torch.clamp(strong_score, min=-50, max=50), dim=-1)
        need_norm = False
    
    elif args.combine == 'multiply_balanced':
        # 模式 2：平衡模式，对 IF 和 CosSim 都进行标准化，确保量级对等
        if_weight = getattr(args, 'if_lambda', 1.0)
        csim_weight = getattr(args, 'csim_lambda', 1.0)
        use_strong = getattr(args, 'use_strong_if', False)
        
        # 1. 标准化 IF 得分
        if_mean_uw = selected_uw_scores.mean(dim=1, keepdim=True)
        if_std_uw = selected_uw_scores.std(dim=1, keepdim=True) + 1e-10
        norm_if_uw = (selected_uw_scores - if_mean_uw) / if_std_uw

        if use_strong:
            if_mean_us = selected_us_scores.mean(dim=1, keepdim=True)
            if_std_us = selected_us_scores.std(dim=1, keepdim=True) + 1e-10
            norm_if_guidance = (selected_us_scores - if_mean_us) / if_std_us
        else:
            norm_if_guidance = norm_if_uw
        
        # 2. 标准化 CosSim
        cs_mean_uw = cosine_correlation_uw.mean(dim=1, keepdim=True)
        cs_std_uw = cosine_correlation_uw.std(dim=1, keepdim=True) + 1e-10
        norm_cosine_uw = (cosine_correlation_uw - cs_mean_uw) / cs_std_uw

        cs_mean_us = cosine_correlation_us.mean(dim=1, keepdim=True)
        cs_std_us = cosine_correlation_us.std(dim=1, keepdim=True) + 1e-10
        norm_cosine_us = (cosine_correlation_us - cs_mean_us) / cs_std_us

        # 3. 融合
        weak_corr_prob = F.softmax((if_weight * norm_if_uw * csim_weight * norm_cosine_uw) / temperature, dim=-1)
        strong_corr_prob = F.softmax((if_weight * norm_if_guidance * csim_weight * norm_cosine_us) / temperature, dim=-1)
        need_norm = False

    elif args.combine == 'multiplyo':
        # 模式 3：最初的样子，IF 和 CosSim 都不做标准化
        if_weight = getattr(args, 'if_lambda', 1.0)
        csim_weight = getattr(args, 'csim_lambda', 1.0)
        use_strong = getattr(args, 'use_strong_if', False)

        if_guidance = selected_us_scores if use_strong else selected_uw_scores
        
        # 限制数值范围防止 softmax 溢出产生 NaN
        weak_score = (if_weight * selected_uw_scores + csim_weight * cosine_correlation_uw) / temperature
        strong_score = (if_weight * if_guidance + csim_weight * cosine_correlation_us) / temperature
        
        weak_corr_prob = F.softmax(torch.clamp(weak_score, min=-50, max=50), dim=-1)
        strong_corr_prob = F.softmax(torch.clamp(strong_score, min=-50, max=50), dim=-1)
        need_norm = False
    # elif args.combine == 'addn':
    #     if epoch < args.epochs * 0.3:
    #         weak_corr_prob = cosine_correlation_wprob
    #         strong_corr_prob = cosine_correlation_sprob
    #         need_norm = False
    #     else:
    #         weak_corr_prob += cosine_correlation_wprob
    #         strong_corr_prob += cosine_correlation_sprob   
    #         need_norm = True
    # else:
    #     if epoch < args.epochs * 0.3:
    #         weak_corr_prob = cosine_correlation_wprob
    #         strong_corr_prob = cosine_correlation_sprob
    #         need_norm = False
    #     else:
    #         weak_corr_prob *= cosine_correlation_wprob
    #         strong_corr_prob *= cosine_correlation_sprob   
    #         need_norm = True
    else:
        raise ValueError(f"Unknown combine method: {args.combine}")

    if need_norm:
        weak_corr_prob = F.softmax(weak_corr_prob / temperature, dim=-1)
        strong_corr_prob = F.softmax(strong_corr_prob / temperature, dim=-1)

    return weak_corr_prob, strong_corr_prob


def compute_correlation_by_ifscore_csim_with_mapping(
    args,
    prop_indices_uw,
    oppo_indices_uw,
    prop_indices_us,
    oppo_indices_us,
    pscores_uw,
    oscores_uw,
    pscores_us,
    oscores_us,
    feas_u_w,
    feas_u_s,
    ref_features,
    indices,
    temperature,
):
    selected_uw_scores = gather_if_scores_by_indices(indices, prop_indices_uw, pscores_uw)
    selected_us_scores = gather_if_scores_by_indices(indices, prop_indices_us, pscores_us)

    cosine_correlation_uw = F.cosine_similarity(
        feas_u_w.unsqueeze(1),
        ref_features,
        dim=2,
    )
    cosine_correlation_us = F.cosine_similarity(
        feas_u_s.unsqueeze(1),
        ref_features,
        dim=2,
    )
    # 打印统计信息
    # detach and move to cpu for numpy compatibility
    uw = selected_uw_scores.detach().cpu().numpy()
    us = selected_us_scores.detach().cpu().numpy()
    cos_uw = cosine_correlation_uw.detach().cpu().numpy()
    cos_us = cosine_correlation_us.detach().cpu().numpy()

    print("=== IF 统计 ===")
    print(f"weak范围: [{uw.min():.6f}, {uw.max():.6f}]")
    print(f"宽度: {uw.max() - uw.min():.6f}")
    print(f"均值: {uw.mean():.6f}")
    print(f"标差: {uw.std():.6f}")
    print(f"四分位距: {np.percentile(uw, 75) - np.percentile(uw, 25):.6f}")
    print(f"strong范围: [{us.min():.6f}, {us.max():.6f}]")
    print(f"宽度: {us.max() - us.min():.6f}")
    print(f"均值: {us.mean():.6f}")
    print(f"标差: {us.std():.6f}")
    print(f"四分位距: {np.percentile(us, 75) - np.percentile(us, 25):.6f}")
    print("\n=== cos 统计 ===")
    print(f"weak范围: [{cos_uw.min():.6f}, {cos_uw.max():.6f}]")
    print(f"宽度: {cos_uw.max() - cos_uw.min():.6f}")
    print(f"均值: {cos_uw.mean():.6f}")
    print(f"标差: {cos_uw.std():.6f}")
    print(f"四分位距: {np.percentile(cos_uw, 75) - np.percentile(cos_uw, 25):.6f}")
    print(f"strong范围: [{cos_us.min():.6f}, {cos_us.max():.6f}]")
    print(f"宽度: {cos_us.max() - cos_us.min():.6f}")
    print(f"均值: {cos_us.mean():.6f}")
    print(f"标差: {cos_us.std():.6f}")
    print(f"四分位距: {np.percentile(cos_us, 75) - np.percentile(cos_us, 25):.6f}")
    print("\n=== 关键对比 ===")

    if_range = uw.max() - uw.min()
    cos_range = cos_uw.max() - cos_uw.min()
    ratio = cos_range / if_range if if_range > 0 else float('inf')
    print(f"范围比例 (cos_range / IF_range): {ratio:.2f}")


    if args.combine == 'add':
        weak_corr_prob = F.softmax(selected_uw_scores / temperature, dim=-1)
        strong_corr_prob = F.softmax(selected_us_scores / temperature, dim=-1)
        cosine_correlation_wprob = F.softmax(cosine_correlation_uw / temperature, dim=-1)
        cosine_correlation_sprob = F.softmax(cosine_correlation_us / temperature, dim=-1)
        weak_corr_prob += cosine_correlation_wprob
        strong_corr_prob += cosine_correlation_sprob
        need_norm = True
    elif args.combine in ('multiply', 'multiply_balanced'):
        # 'multiply' 与 'multiply_balanced' 在本函数里等价: IF 和 cos 都按行 z-score 后加权相加。
        # (与 mixin ClosedFormIFRankMixin._fuse 的 multiply_balanced 一致; 修复原先缺 multiply_balanced 分支掉进 else 只用IF的 bug)
        if_weight = getattr(args, 'if_lambda', 1.0)
        csim_weight = getattr(args, 'csim_lambda', 1.0)
        use_strong = getattr(args, 'use_strong_if', False)

        if_mean_uw = selected_uw_scores.mean(dim=1, keepdim=True)
        if_std_uw = selected_uw_scores.std(dim=1, keepdim=True) + 1e-10
        norm_if_uw = (selected_uw_scores - if_mean_uw) / if_std_uw

        if use_strong:
            if_mean_us = selected_us_scores.mean(dim=1, keepdim=True)
            if_std_us = selected_us_scores.std(dim=1, keepdim=True) + 1e-10
            norm_if_us = (selected_us_scores - if_mean_us) / if_std_us
        else:
            norm_if_us = norm_if_uw
        
        if_mean_cuw = cosine_correlation_uw.mean(dim=1, keepdim=True)
        if_std_cuw = cosine_correlation_uw.std(dim=1, keepdim=True) + 1e-10
        norm_if_cuw = (cosine_correlation_uw - if_mean_cuw) / if_std_cuw

        if_mean_cus = cosine_correlation_us.mean(dim=1, keepdim=True)
        if_std_cus = cosine_correlation_us.std(dim=1, keepdim=True) + 1e-10
        norm_if_cus = (cosine_correlation_us - if_mean_cus) / if_std_cus

        # combined_uw = if_weight * norm_if_uw + csim_weight * cosine_correlation_uw
        # combined_us = if_weight * norm_if_us + csim_weight * cosine_correlation_us
        combined_uw = if_weight * norm_if_uw + csim_weight * norm_if_cuw
        combined_us = if_weight * norm_if_us + csim_weight * norm_if_cus

        weak_corr_prob = F.softmax(torch.clamp(combined_uw / temperature, min=-50, max=50), dim=-1)
        strong_corr_prob = F.softmax(torch.clamp(combined_us / temperature, min=-50, max=50), dim=-1)
        need_norm = False
    elif args.combine == 'multiplyo':
        # No normalization, direct weighted sum of IF and CosSim (original multiplyo logic)
        if_weight = getattr(args, 'if_lambda', 1.0)
        csim_weight = getattr(args, 'csim_lambda', 1.0)
        use_strong = getattr(args, 'use_strong_if', False)

        if_guidance = selected_us_scores if use_strong else selected_uw_scores

        weak_score = (if_weight * selected_uw_scores + csim_weight * cosine_correlation_uw) / temperature
        strong_score = (if_weight * if_guidance + csim_weight * cosine_correlation_us) / temperature

        weak_corr_prob = F.softmax(torch.clamp(weak_score, min=-50, max=50), dim=-1)
        strong_corr_prob = F.softmax(torch.clamp(strong_score, min=-50, max=50), dim=-1)
        need_norm = False
    elif args.combine == 'signedadd':
        if_weight = getattr(args, 'if_lambda', 1.0)
        csim_weight = getattr(args, 'csim_lambda', 1.0)
        use_strong = getattr(args, 'use_strong_if', False)

        signed_if_uw = build_signed_if_scores(selected_uw_scores, normalize=True)
        if use_strong:
            signed_if_us = build_signed_if_scores(selected_us_scores, normalize=True)
        else:
            signed_if_us = signed_if_uw

        combined_uw = if_weight * signed_if_uw + csim_weight * cosine_correlation_uw
        combined_us = if_weight * signed_if_us + csim_weight * cosine_correlation_us
        weak_corr_prob = F.softmax(torch.clamp(combined_uw / temperature, min=-50, max=50), dim=-1)
        strong_corr_prob = F.softmax(torch.clamp(combined_us / temperature, min=-50, max=50), dim=-1)
        need_norm = False
    else:
        weak_corr_prob = F.softmax(torch.clamp(selected_uw_scores / temperature, min=-50, max=50), dim=-1)
        strong_corr_prob = F.softmax(torch.clamp(selected_us_scores / temperature, min=-50, max=50), dim=-1)
        need_norm = False

    if need_norm:
        weak_corr_prob = weak_corr_prob / weak_corr_prob.sum(dim=-1, keepdim=True)
        strong_corr_prob = strong_corr_prob / strong_corr_prob.sum(dim=-1, keepdim=True)

    return weak_corr_prob, strong_corr_prob

##########################
def compute_batch_correlation(features, references, temperature):
    """
    批量计算特征与参考样本的相关性
    
    Args:
        features: [B, C]
        references: [B, num_refs, C]
    
    Returns:
        correlation: [B, num_refs]
    """
    # features: [B, C] -> [B, 1, C]
    # references: [B, num_refs, C]
    correlation = F.cosine_similarity(
        features.unsqueeze(1),  # [B, 1, C]
        references,             # [B, num_refs, C]
        dim=2
    )  # [B, num_refs]
    
    # 应用softmax转换为概率分布
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@")    
    # print(correlation)
    correlation_prob = F.softmax(correlation / temperature, dim=-1)
    # print(correlation_prob)
    return correlation_prob

##################
def compute_individual_consistency_loss(weak_features, strong_features, referenced_features):
    """计算基于个体参考的一致性损失"""
    
    # 为每个样本选择个性化参考
    # weak_references = select_reference_features_by_instance(weak_features, labeled_features)    # [B, 4, C]
    # strong_references = select_reference_features_by_instance(strong_features, labeled_features)  # [B, 4, C]
    
    # 计算相关性（需要修改相关性计算函数以支持批量处理）
    weak_corr = compute_batch_correlation(weak_features, referenced_features)    # [B, 4]
    strong_corr = compute_batch_correlation(strong_features, referenced_features)  # [B, 4]
    
    # 计算一致性损失
    consistency_loss = F.kl_div(
        (strong_corr + 1e-10).log(),
        weak_corr.detach(),
        reduction='batchmean'
    )
    
    return consistency_loss


def prob2rank_classification(prob, prob_s, k=None):
    """
    分类场景下的概率到排名转换
    
    Args:
        prob: 概率分布 [B, num_refs]
        k: 考虑的top-k
    
    Returns:
        rank: 排名分布 [B, k!]
    """
    from itertools import permutations
    
    # 生成所有排列
    full_permutation = [c for c in permutations(range(k))]
    full_permutation = torch.from_numpy(np.stack(full_permutation)).to(prob.device) # [k!, k]
    
    # 获取top-k索引 [b, k]
    _, prob_topk_index = prob.topk(k, dim=-1)
    
    # 正确的索引重排列方法
    batch_size = prob.shape[0]
    
    # 扩展索引以匹配排列
    prob_topk_index_expanded = prob_topk_index.unsqueeze(1)  # [b, 1, k]
    full_permutation_expanded = full_permutation.unsqueeze(0)  # [1, k!, k]
    
    # 为每个样本生成所有排列的索引
    # 使用高级索引
    A_list = []
    for i in range(batch_size):
        # 对每个样本，生成所有排列的索引
        sample_indices = prob_topk_index[i]  # [k]
        permuted_indices = sample_indices[full_permutation]  # [k!, k]
        A_list.append(permuted_indices)
    
    A = torch.stack(A_list, dim=0)  # [b, k!, k]
    
    # 扩展概率张量
    B = prob.unsqueeze(1).expand(-1, full_permutation.shape[0], -1)  # [b, k!, n]
    B_s = prob_s.unsqueeze(1).expand(-1, full_permutation.shape[0], -1)  # [b, k!, n]
    
    # 收集对应概率
    C = torch.gather(input=B, dim=-1, index=A)  # [b, k!, k]
    C_s = torch.gather(input=B_s, dim=-1, index=A)  # [b, k!, k]
    
    # 计算排名概率
    rank = C[:, :, 0] / (C[:, :, 0:].sum(dim=-1) + 1e-10)  # [b, k!]
    rank_s = C_s[:, :, 0] / (C_s[:, :, 0:].sum(dim=-1) + 1e-10)  # [b, k!]
    
    for i in range(1, k):
        rank *= C[:, :, i] / (C[:, :, i:].sum(dim=-1) + 1e-10)
        rank_s *= C_s[:, :, i] / (C_s[:, :, i:].sum(dim=-1) + 1e-10)
    
    return rank, rank_s

