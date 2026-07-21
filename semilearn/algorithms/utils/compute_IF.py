import os
import random
import time
import numpy as np
import torch
from captum.influence import TracInCPFast
from torch.utils.data import Dataset
import torch.nn.functional as F
# from models.resnet import resnet18 
from semilearn.nets.resnet.resnet import resnet18, ResNet


def load_db(db_path, class_to_idx):
    db = torch.load(db_path)

    images = []

    for key in sorted(db.keys()):

        for image_path in db[key]:
            ori_image_path = '../' + image_path
            images.append(ori_image_path)
    return images
def my_celoss(pred, label):
    if isinstance(pred, dict):
        pred = pred['logits']

    if pred.size() == label.size():
        pred = F.softmax(pred, dim=1)
        # 添加 1e-10 防止 log(0) 产生 NaN
        celoss = -torch.sum(label * torch.log(pred + 1e-8), dim=1).mean() # 相同位置元素相乘
    else:
        celoss = F.cross_entropy(pred, label, reduction='mean')
    return celoss


def computeIF(args, epoch, model, labeled_dataset, val_dataset, correct_list):
    

    print("==>computing IF score from checkpoint directory '{}'..".format(args.out))    
    checkpoints_dir = []
    # compute influence score for each validation image, and 
    # find the first k images with highest IF scores in labeled data 

    valimages_indices = [i for i in range(len(val_dataset))]
    benign_count = 0
    malignant_count = 0
    for i in range(len(val_dataset)):
        if val_dataset[i][1] == 0:
            benign_count += 1
        else:
            malignant_count += 1

    pathname = 'checkpoint'+str(epoch)+'.pth.tar'

    pathname = os.path.join(args.out, pathname)
    checkpoints_dir.append(pathname)

    # last_checkpoint = sorted(checkpoints_dir)[-1]
    for name in checkpoints_dir:
        checkpoint = torch.load(name)
        model.load_state_dict(checkpoint['state_dict'])

    # model = model.to('cpu')
    model.eval()

    # checkpoints_load_func(model, last_checkpoint)
    labeled_dataset_isreturn = labeled_dataset.get_return_idx()
    labeled_dataset.set_return_idx(False)

    val_examples_features = torch.stack([val_dataset[i][0] for i in valimages_indices])
    val_examples_true_labels = torch.Tensor([val_dataset[i][1] for i in valimages_indices]).long()

    if_before_time = time.time()
    print("before compute IF for val dataset in labeled dataset")
    k = 10
    proponents_indices, proponents_influence_scores, opponents_indices, opponents_influence_scores = compute_influence_scores(
    args, model, labeled_dataset, val_examples_features, val_examples_true_labels, checkpoints_dir, checkpoints_load_func, k)
    
    end_time = time.time() 
    total_minutes = (end_time - if_before_time) / 60.0

    print("Computed proponents / opponents from %d train candidates over validation dataset of %d examples in %d minutes"
            % (len(labeled_dataset), len(val_dataset), total_minutes))
    print(proponents_indices)
    print(proponents_influence_scores)
    print(opponents_indices)
    print(opponents_influence_scores)

     
    propid_list = proponents_indices.tolist()
    propscore_list = proponents_influence_scores.tolist()
    oppoid_list = opponents_indices.tolist()
    opposcore_list = opponents_influence_scores.tolist()

 
    correct_propid_list = [propid_list[index] for index in correct_list]
    correct_oppoid_list = [oppoid_list[index] for index in correct_list]

    correct_benign_propid_list = []
    correct_benign_oppoid_list = []
    correct_malignant_propid_list = []
    correct_malignant_oppoid_list = []

    for index in correct_list:
        if index < benign_count:
            correct_benign_propid_list.append(propid_list[index])
            correct_benign_oppoid_list.append(oppoid_list[index])
        else:
            correct_malignant_propid_list.append(propid_list[index])
            correct_malignant_oppoid_list.append(oppoid_list[index])
        
    print(correct_propid_list)
    print(correct_oppoid_list)
    print(correct_benign_propid_list)
    print(correct_benign_oppoid_list)
    print(correct_malignant_propid_list)
    print(correct_malignant_oppoid_list)

    #建立ID与filename的字典
    pth_dir = args.labeledpath

    train_data_dir = '../uda_data/train'
    classes, class_to_idx = find_classes(train_data_dir)
    imgs = load_db(pth_dir, class_to_idx) #list

    sif_imgid_imgname_dict = dict()

    test_filename_list = []
    testdir = '../uda_data/test'
    file_cls = os.listdir(testdir)
    for clsname in file_cls:
        cls_filename_list = []
        cls_dir = os.path.join(testdir, clsname)
        cls_filename_list = os.listdir(cls_dir)
       
        for name in cls_filename_list:
            test_filename_list.append(os.path.join(cls_dir,name))
    correct_test_filename_list = [name for index,name in enumerate(test_filename_list) if index in correct_list]

    dic_propif = dict()
    dic_oppoif = dict()
    print("test prop and oppo dict%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(correct_test_filename_list)
    print(len(correct_test_filename_list))
    print(len(correct_malignant_propid_list))

    # 
    for i in range(len(correct_malignant_propid_list)):
        dic_value = []
        # print(i)
        for j in range(len(correct_malignant_propid_list[0])):
            dic_value.append(imgs[correct_malignant_propid_list[i][j]])
            # dic_value.append(propscore_list[i][j])
        dic_propif[correct_test_filename_list[i]] = dic_value
    for i in range(len(correct_malignant_oppoid_list)):
        dic_value = []
        for j in range(len(correct_malignant_oppoid_list[0])):
            dic_value.append(imgs[correct_malignant_oppoid_list[i][j]])
            # dic_value.append(opposcore_list[i][j])
        dic_oppoif[correct_test_filename_list[i]] = dic_value

    print(dic_propif)
    print(dic_oppoif)
    
    alllists = [correct_benign_propid_list, correct_benign_oppoid_list, correct_malignant_propid_list, correct_malignant_oppoid_list]

    alllist_name = ['be_prop', 'be_oppo', 'ma_prop', 'ma_oppo']

    for i in range(len(alllists)):
            # newid_list = [item for list1 in alllists for item in list1]
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(i)
            name = alllists[i]
            if i == 0 or i == 3:
                benign_prop = []
                matrix = np.array(name)
                elements, counts = np.unique(matrix, return_counts=True)
                for element, count in zip(elements,counts):
                    print(f"{element}:{count}")
                    # del wrong prop or oppo index from labeled dataset
                    # if 'malignant' not in imgs[element]:
                    if count > 10:
                        benign_prop.append(element)
            else:
                malignant_prop = []
                matrix = np.array(name)
                elements, counts = np.unique(matrix, return_counts=True)
                for element, count in zip(elements,counts):
                    print(f"{element}:{count}")
                    # del wrong prop or oppo index from labeled dataset
                    # if 'benign' not in imgs[element]:
                    if count > 10:
                        malignant_prop.append(element)
    print(benign_prop)
    print(malignant_prop)
    
                
    labeled_dataset.set_return_idx(labeled_dataset_isreturn)

    model.to(args.device)
    return benign_prop, malignant_prop

# 梯度信息和阈值信息共同确定伪标签，其中梯度信息，即IF，可以看作是某种程度的不确定性，但是可能比不确定性的表示更简单，
# 或者更准确。可以辅助模型找到更好更准确的伪标签，同时在模型训练过程中引入更多无标签数据。可以从无标签数据中寻找对有标签影响最大的元素，
# 选定其伪标签，或者选定其梯度信息确定的标签，引入无标签的训练过程中
def computeIF_inbatch(args, state, x_lb, x_ulb_w, x_ulb_s, targets_x, ema_probuw, ema_probus):

    print("==>computing IF score in batch between weak labeled and weak unlabeled data")

    checkpoints_dir = []
    # model = model.clone()
    model = resnet18(num_classes=args.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(state)
    model.eval()

    ###
    inputs_x = x_lb
    inputs_u_w = x_ulb_w
    inputs_u_s = x_ulb_s

    

    # pathname = os.path.join(args.out, pathname)
    checkpoints_dir.append(state)
    # compute IF for u_w data in x dataset
    uwdataset = InfluencingDataset(inputs_x, targets_x)
    # compute IF for x data in u_w dataset
    # uwdataset = InfluencingDataset(inputs_u_w, logits_u_wd) 
    
    if_before_time = time.time()

    print("before computing IF for labeled dataset in u_w dataset")
    # k = args.k        
    # debias = args.debias

    # probability distribution over classes for unlabeled data
    # probs_uw = torch.softmax(logits_uw, dim=-1)
    # probs_us = torch.softmax(logits_us, dim=-1)

    # compute IF for u_w dataset in labeled dataset
    pindices_uw, pscores_uw, oindices_uw, oscores_uw = compute_influence_scores(
    args, model, uwdataset, inputs_u_w.clone(), ema_probuw, checkpoints_dir, checkpoints_load_func_inbatch)
    pindices_us, pscores_us, oindices_us, oscores_us = compute_influence_scores(
    args, model, uwdataset, inputs_u_s.clone(), ema_probus, checkpoints_dir, checkpoints_load_func_inbatch)
    # compute IF for x data in u_w dataset
    # pindices, pscores, oindices, oscores = compute_influence_scores(
    # args, model, uwdataset, inputs_x.clone(), targets_x, checkpoints_dir, checkpoints_load_func_inbatch)
    
    end_time = time.time() 
    total_minutes = end_time - if_before_time #/ 60.0

    print("Computed proponents / opponents from %d train candidates over validation dataset of %d examples in %d seconds"
            % (len(uwdataset), len(inputs_x), total_minutes))
    # print(indices)
    # print(scores)

    # 先实验k=3,把每个IF选对的加入进pseudo label中， 先看下效果；然后尝试放大k，因为最后选出来的k个会越来越接近
    model.zero_grad()
    model.train()

    return pindices_uw, oindices_uw, pindices_us, oindices_us, pscores_uw, oscores_uw, pscores_us, oscores_us

# def computeIF_inbatch2(args, model, inputs, targets_x, ema_probuw, ema_probus):

def compute_if_tracin_grad(logits_l, targets_l, logits_u, targets_u, k):
    """
    logits_l: [L, num_classes] 有标注样本logits
    targets_l: [L, ...]
    logits_u: [U, num_classes] 无标注样本logits
    targets_u: [U, ...]
    k: top-k
    返回: 支持分数、索引，反对分数、索引（每个无标注样本对每个有标注样本的影响）
    """
    L = logits_l.size(0)
    U = logits_u.size(0)
    # 1. 计算有标注样本的loss对参数的梯度
    loss_fn = my_celoss
    grads_labeled = []
    for i in range(L):
        loss = loss_fn(logits_l[i].unsqueeze(0), targets_l[i].unsqueeze(0))
        grad = torch.autograd.grad(loss, logits_l, retain_graph=True, create_graph=True, allow_unused=True)[0]
        grads_labeled.append(grad[i])  # 只取当前样本的梯度
    grads_labeled = torch.stack(grads_labeled)  # [L, num_classes]

    # 2. 计算无标注样本的loss对参数的梯度
    grads_ul = []
    for i in range(U):
        loss = loss_fn(logits_u[i].unsqueeze(0), targets_u[i].unsqueeze(0))
        grad = torch.autograd.grad(loss, logits_u, retain_graph=True, create_graph=True, allow_unused=True)[0]
        grads_ul.append(grad[i])
    grads_ul = torch.stack(grads_ul)  # [U, num_classes]

    # 3. influence分数 = 梯度内积
    influence_scores = torch.matmul(grads_ul, grads_labeled.t())  # [U, L]
    # 支持分数/索引: 取每行top-k最大
    proponents_scores, proponents_indices = torch.topk(influence_scores, k=k, dim=1)
    # 反对分数/索引: 取每行top-k最小
    opponents_scores, opponents_indices = torch.topk(influence_scores, k=k, dim=1, largest=False)

    return proponents_scores, proponents_indices, opponents_scores, opponents_indices, influence_scores

def compute_if_inbatch_grad(logits_l, targets_l, logits_uw, targets_uw, logits_us, targets_us, k):
    """
    计算有标注样本对弱/强增强无标注样本的IF分数和索引（支持/反对）
    返回：
      - 弱增强: 支持分数/索引, 反对分数/索引, influence矩阵
      - 强增强: 支持分数/索引, 反对分数/索引, influence矩阵
    """
    # 弱增强
    prop_scores_uw, prop_idx_uw, oppo_scores_uw, oppo_idx_uw, infmat_uw = compute_if_tracin_grad(
        logits_l, targets_l, logits_uw, targets_uw, k=k)
    # 强增强
    prop_scores_us, prop_idx_us, oppo_scores_us, oppo_idx_us, infmat_us = compute_if_tracin_grad(
        logits_l, targets_l, logits_us, targets_us, k=k)
    
    return prop_idx_uw, oppo_idx_uw, prop_idx_us, oppo_idx_us, prop_scores_uw, prop_scores_us      


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class InfluencingDataset(Dataset):
    def __init__(self, data, label):
        self.data = data.clone()
        self.label = label.clone()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def checkpoints_load_func(model, path):

    weights = torch.load(path)
    # Handle ModelWrapper
    target_model = model.model if hasattr(model, 'model') and not isinstance(model, ResNet) else model
    target_model.load_state_dict(weights["state_dict"])
    return 1.

def checkpoints_load_func_inbatch(model, state):

    # Handle ModelWrapper
    target_model = model.model if hasattr(model, 'model') and not isinstance(model, ResNet) else model
    target_model.load_state_dict(state)
    
    return 1.

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return out['logits']
        return out


def compute_influence_scores(
    args,
    model,
    labeled_dataset,
    test_examples_features,
    test_examples_true_labels,
    checkpoints_dir,
    checkpoints_load_func
    ):

    # Wrap model to ensure it returns a Tensor, not a dict
    wrapped_model = ModelWrapper(model)
    # Find the linear layer within the original model
    if hasattr(model, 'classifier'):
        final_fc_layer = model.classifier
    elif hasattr(model, 'fc'):
        final_fc_layer = model.fc
    else:
        # Fallback to last child if typical names aren't found
        final_fc_layer = list(model.children())[-1]

    tracin_cp_fast = TracInCPFast(
        model=wrapped_model,
        final_fc_layer=final_fc_layer,
        train_dataset=labeled_dataset,
        checkpoints=checkpoints_dir,
        checkpoints_load_func=checkpoints_load_func,
        loss_fn=my_celoss,
        batch_size=len(labeled_dataset) + len(test_examples_features),
        vectorize=True,
    )

    
    # if args.debias:
    #     opponents_indices, opponents_influence_scores = tracin_cp_fast.influence(
    #         (test_examples_features, test_examples_true_labels),
    #         k=args.k,
    #         proponents=False
    #     )
    #     # indices = opponents_indices
    #     # scores = opponents_influence_scores
    # else:
    proponents_indices, proponents_influence_scores = tracin_cp_fast.influence(
    (test_examples_features, test_examples_true_labels),
    k=args.k,
    proponents=True
    )
    opponents_indices, opponents_influence_scores = tracin_cp_fast.influence(
        (test_examples_features, test_examples_true_labels),
        k=args.k,
        proponents=False
    )


    return proponents_indices, proponents_influence_scores, opponents_indices, opponents_influence_scores