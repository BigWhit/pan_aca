import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EmbLoss_v2(nn.Module):
    def __init__(self, feature_dim=4, loss_weight=1.0):
        super(EmbLoss_v2, self).__init__()
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.delta_v = 0.5
        self.delta_d = 1.5
        self.weights = (1.0, 1.0)
        self.max_instances = 200

    def forward_single(self, emb, instance, kernel, training_mask, bboxes, max_distances):  # max_distances
        # print(max_distances)
        training_mask = (training_mask > 0.5).long()  # 0,1
        kernel = (kernel > 0.5).long()  # 0,1
        instance = instance * training_mask
        instance_kernel = (instance * kernel).view(-1)
        instance = instance.view(-1)  # [409600]
        emb = emb.view(self.feature_dim, -1)  # 四行[4,409600]

        unique_labels, unique_ids = torch.unique(instance_kernel,
                                                 sorted=True,
                                                 return_inverse=True)
        num_instance = unique_labels.size(0)
        # print(num_instance)
        # print(unique_labels)
        if num_instance <= 1:
            return 0
        # lagg
        emb_mean = emb.new_zeros((self.feature_dim, num_instance),
                                 dtype=torch.float32)  # 4*文本实例个数
        for i, lb in enumerate(unique_labels):
            if lb == 0:  # 文本为#的忽略
                continue
            ind_k = instance_kernel == lb  # 第i个kernel所有像素的索引
            emb_mean[:, i] = torch.mean(emb[:, ind_k], dim=1)  # 公式中的G(Ki)
        # print('emb_mean',emb_mean)
        l_agg = emb.new_zeros(num_instance, dtype=torch.float32)  # bug
        for i, lb in enumerate(unique_labels):  # 遍历每一个文本实例
            # print(len(unique_labels))
            # print(max_distances[i])
            # print('lb',lb)
            if lb == 0:  # 文本为#的忽略
                continue
            ind = instance == lb
            # print('ind',ind)
            emb_ = emb[:, ind]  # 公式中的F(p)
            dist = (emb_ - emb_mean[:, i:i + 1]).norm(p=2, dim=0)  # max_distances*
            # print(max_distances[lb])
            # print('lagg',math.exp(max_distances[lb]) )
            # dist = F.relu(dist - self.delta_v) ** 2 # 公式中的 D(p,Ki)
            dist = F.relu(math.exp(max_distances[lb]) * dist - self.delta_v) ** 2
            l_agg[i] = torch.mean(torch.log(dist + 1.0))  # 公式Lagg
        l_agg = torch.mean(l_agg[1:])
        # Ldis
        if num_instance > 2:
            # print(num_instance)#3
            emb_interleave = emb_mean.permute(1, 0).repeat(num_instance,
                                                           1)  # permute相当于transpose,维度转换，文本实例*4；repeat将numpy数组重复[文本实例*文本实例，4]
            # print('emb_interlleave', emb_interleave.size())#[3*3,4]
            # print('interleave',emb_interleave)
            emb_band = emb_mean.permute(1, 0).repeat(1, num_instance).view(
                -1, self.feature_dim)  # [文本实例*文本实例，4]
            # print('emb_band', emb_band.size())#9
            # print('band',emb_band)

            mask = (1 - torch.eye(num_instance, dtype=torch.int8)).view(
                -1, 1).repeat(1, self.feature_dim)
            mask = mask.view(num_instance, num_instance, -1)
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.view(num_instance * num_instance, -1)
            # print(mask)

            dist = emb_interleave - emb_band  # [3*3,4]
            dist = dist[mask > 0].view(-1, self.feature_dim).norm(p=2, dim=1)  # 2*1=2给定维dim (列)上每行的p范数
            # print(dist)
            # dist = F.relu(2 * self.delta_d - dist) ** 2  # D(Ki,Kj),ldis原本
            # print((1-math.exp(-(10/num_instance))))
            dist = F.relu(2 * self.delta_d - (1 - math.exp(-(10 / num_instance))) * dist) ** 2
            l_dis = [torch.log(dist + 1.0)]  # Ldis原本
            # print(type(l_dis))#list
            # ########## 比v1添加
            emb_bg = emb[:, instance == 0].view(self.feature_dim, -1)  # instance
            # print('bg', emb_bg.size(1))
            if emb_bg.size(1) > 100:
                rand_ind = np.random.permutation(emb_bg.size(1))[:100]  # 取前100
                emb_bg = emb_bg[:, rand_ind]  # [4,100]
            if emb_bg.size(1) > 0:  # 100
                for i, lb in enumerate(unique_labels):
                    # print('i',i)
                    # print('lb',lb)
                    if lb == 0:
                        continue
                    dist = (emb_bg - emb_mean[:, i:i + 1]).norm(p=2, dim=0)
                    # dist = F.relu(2 * self.delta_d - dist) ** 2 #原本
                    # print((1-math.exp(-(10/num_instance))))
                    dist = F.relu(2 * self.delta_d - (1 - math.exp(-(10 / num_instance))) * dist) ** 2
                    l_dis_bg = torch.mean(torch.log(dist + 1.0),
                                          0,
                                          keepdim=True)
                    l_dis.append(l_dis_bg)
            l_dis = torch.mean(torch.cat(l_dis))
        ###############
        else:
            l_dis = 0

        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = torch.mean(torch.log(torch.norm(emb_mean, 2, 0) + 1.0)) * 0.001
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self,
                emb,
                instance,
                kernel,
                training_mask,
                bboxes,
                max_distances,
                # min_distances,
                reduce=True):
        loss_batch = emb.new_zeros((emb.size(0)), dtype=torch.float32)

        for i in range(loss_batch.size(0)):  # loss_batch.size(0)=8,等于batch_size
            loss_batch[i] = self.forward_single(emb[i], instance[i], kernel[i],
                                                training_mask[i], bboxes[i], max_distances[i])  # max_distances[i]

        loss_batch = self.loss_weight * loss_batch

        if reduce:
            loss_batch = torch.mean(loss_batch)

        return loss_batch
