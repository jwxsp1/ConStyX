import torch
import time
import torch.nn.functional as F


def certainty_estimation(output):

    sigmoid_output = torch.sigmoid(output)

    disc_probabilities = sigmoid_output[:, 0, :, :].unsqueeze(1)
    cup_probabilities = sigmoid_output[:, 1, :, :].unsqueeze(1)
    #disc_uncertainties.append(disc_probabilities)
    #cup_uncertainties.append(cup_probabilities)

    #disc_uncertainty = torch.stack(disc_uncertainties).mean(dim=0)
    #cup_uncertainty = torch.stack(cup_uncertainties).mean(dim=0)
        #disc_entropy = -torch.sum(disc_uncertainty * torch.log(disc_uncertainty + 1e-10), dim=1)
        #cup_entropy = -torch.sum(cup_uncertainty * torch.log(cup_uncertainty + 1e-10), dim=1)
    disc_entropy=-disc_probabilities * torch.log(disc_probabilities + 1e-10)+ \
                -(1-disc_probabilities)*torch.log((1-disc_probabilities) + 1e-10)
        
    cup_entropy=-cup_probabilities * torch.log(cup_probabilities + 1e-10)+ \
                -(1-cup_probabilities)*torch.log((1-cup_probabilities) + 1e-10)

    min_value = 0
    max_value = 1
    for i in range(disc_entropy.size(0)):
        disc_entropy[i] = min_value + (disc_entropy[i] - disc_entropy[i].min()) * (max_value - min_value) / (
                disc_entropy[i].max() - disc_entropy[i].min())
        cup_entropy[i] = min_value + (cup_entropy[i] - cup_entropy[i].min()) * (max_value - min_value) / (
                cup_entropy[i].max() - cup_entropy[i].min())

    class_certainty = torch.cat([1-disc_entropy,1-cup_entropy],dim=1) #(b,2,h,w)

    return torch.exp(class_certainty)-1
    #return 4**class_certainty-1

def cal_weight_map(ori_features,aug_features,conf,sim_threhold):
    """
    ori_features: B*C*H*W
    aug_features: B*C*H*W
    conf:B*n*H*W
    """
    ori_features_reshaped = ori_features.permute(0, 2, 3, 1)
    aug_features_reshaped = aug_features.permute(0, 2, 3, 1)
    sim_map=F.cosine_similarity(ori_features_reshaped, aug_features_reshaped, dim=-1).unsqueeze(dim=1) #B*1*H*W
    low_sim_pos=(sim_map<sim_threhold).float()
    #print("low_sim_pos.sum():")
    #print(low_sim_pos.sum())
    certainty_map=certainty_estimation(conf)
    weight_map=certainty_map*low_sim_pos+torch.ones_like(certainty_map)*(1-low_sim_pos)

    return weight_map
    #return certainty_map

"""

def style_aug(grad,features,mask,class_num=3):
    #grad: B*C*H*W
    #features: B*C*H*W
    #maks:B*H*W
    B,C,H,W=grad.size()
    mask=mask.unsqueeze(1).repeat(1,features.size(1),1,1)
    ones=torch.ones_like(grad).cuda()
    aug_pos=torch.zeros_like(grad).cuda()
    for i in range(class_num):
        new_mask=(mask==i).float()
    #    if i==1:
    #        new_mask=((mask==i)&(mask!=i+1)).float()
        class_i_grad=(new_mask*grad).sum(dim=(2,3)).sum(0).unsqueeze(0)
        values, indices = torch.topk(class_i_grad.view(-1), grad.size(1)//6, largest=False)
        indices_mask = torch.zeros_like(class_i_grad, dtype=torch.float32).cuda()
        indices_mask[0, indices] = 1
        aug_pos_class_i=indices_mask.unsqueeze(2).unsqueeze(3)*new_mask
        aug_pos+=aug_pos_class_i
    style_sampled_direction=torch.rand((B,C,H,W)).cuda()*0.3*aug_pos
    features=features+style_sampled_direction
    return features
"""

def set_random_k_per_pixel(a, k):
    # a 是一个大小为 B*C*H*W 的张量
    B, C, H, W = a.size()

    # 创建一个与 a 相同形状的掩码张量，并初始化为0
    mask = torch.zeros_like(a, dtype=torch.float32).cuda()

    # 随机生成索引
    random_indices = torch.randint(0, C, (B, H, W, k)).cuda()

    # 将随机索引的值设置为1，使用 scatter_ 方法
    for i in range(k):
        mask.scatter_(1, random_indices[..., i].unsqueeze(1), 1.0)

    return mask

def style_aug(grad,features,mask,class_num=3):
    """
    grad: B*C*H*W
    features: B*C*H*W
    maks:B*H*W
    """
    B,C,H,W=grad.size()
    values, indices = torch.topk(grad, k=grad.size(1)//6, dim=1, largest=False)
    #values, indices = torch.topk(grad, k=grad.size(1)//6, dim=1, largest=True)
    mask = torch.zeros_like(grad, dtype=torch.float32)
    mask.scatter_(1, indices, 1.0)
#    mask=set_random_k_per_pixel(grad, grad.size(1)//6)
    style_sampled_direction=torch.rand((B,C,H,W)).cuda()*0.5*mask
    #style_sampled_direction=torch.randn((B,C,H,W)).cuda()*0.5*mask
    features=features+style_sampled_direction
    return features


def semantic_aug(features,mask,cov,class_num):
    """
    features: B*C*H*W
    mask:B*H*W
    """
    B,C,H,W=features.size()
    mask=mask.unsqueeze(1).repeat(1,features.size(1),1,1)
    aug_direction=torch.zeros_like(features).cuda()
    for i in range(class_num):
        std=cov[i]
        #print(std.size())
        new_mask=(mask==i).float()
    #   if i==1:
    #        new_mask=((mask==i)&(mask!=i+1)).float()
        sampled_direction_i=new_mask*std.repeat(features.size(0),1).unsqueeze(2).unsqueeze(3)
        aug_direction+=sampled_direction_i.detach()
    #features=features+aug_direction*torch.randn((B,C,H,W)).cuda()*1
    features=features+aug_direction*torch.randn((B,C,H,W)).cuda()
    #features=features+aug_direction*torch.rand((B,C,H,W)).cuda()
    return features



class EstimatorCV():
    def __init__(self, feature_num, class_num, device):
        super(EstimatorCV, self).__init__()

        self.class_num = class_num
        self.device = device
        self.CoVariance = torch.zeros(class_num, feature_num).to(device)
        self.Ave = torch.zeros(class_num, feature_num).to(device)
        self.Amount = torch.zeros(class_num).to(device)

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)
        #print(N,C,A)
        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )

        onehot = torch.zeros(N, C).to(self.device)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA
        #print(ave_CxA.size())
        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)
        #print(var_temp.size())
        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)
        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        #print(weight_CV.size())

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        self.Amount += onehot.sum(0)


