import torch
import torch.nn as nn


class KernelLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(KernelLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, input, target, mask,reduce=True):  # 加入miu
        # print(input.size())  # [8,640,640]
        batch_size = input.size(0)
        input = torch.sigmoid(input)
        # print(miu.size())#[8,640,640]
        input = input.contiguous().view(batch_size, -1)  # [8,409600]
        target = target.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()

        input = input * mask
        target = target * mask

        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1) + 0.001
        c = torch.sum(target * target, dim=1) + 0.001
        d = (2 * a) / (b + c)
        loss = 1 - d

        loss = self.loss_weight * loss

        if reduce:
            loss = torch.mean(loss)

        return loss
