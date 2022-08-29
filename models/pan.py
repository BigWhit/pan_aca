import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
from .utils import Conv_BN_ReLU
from .utils import ECAAttention


class PAN(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 detection_head):
        super(PAN, self).__init__()
        self.backbone = build_backbone(backbone)
        in_channels = neck.in_channels  # len(in_channels)=4,(64, 128, 256, 512)
        # eca
        self.eca1 = ECAAttention(in_channels[0])
        self.eca2 = ECAAttention(in_channels[1])
        self.eca3 = ECAAttention(in_channels[2])
        self.eca4 = ECAAttention(in_channels[3])
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)
        # eca
        # self.eca = ECAAttention(128)
        self.fpem1 = build_neck(neck)
        self.fpem2 = build_neck(neck)
        # self.gscnn = GSCNN(128)
        # self.eca = ECAAttention(kernel_size=3)
        self.det_head = build_head(detection_head)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                gt_instances=None,
                gt_bboxes=None,
                # miu=None,
                max_distances=None,
                # min_distances=None,
                img_metas=None,
                cfg=None):
        outputs = dict()
        # print(outputs)
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # backbone
        f = self.backbone(imgs)
        # print(f[0].size())#torch.Size([8, 64, 160, 160])
        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                backbone_time=time.time() - start
            ))
            start = time.time()
        # eca
        f1 = self.eca1(f[0])
        f2 = self.eca2(f[1])
        f3 = self.eca3(f[2])
        f4 = self.eca4(f[3])
        f1 = self.reduce_layer1(f1)
        f2 = self.reduce_layer2(f2)
        f3 = self.reduce_layer3(f3)
        f4 = self.reduce_layer4(f4)
        # reduce channel
        # f1 = self.reduce_layer1(f[0])  # [8,128,184,184]
        # f2 = self.reduce_layer2(f[1])  # [8,128,92,92]
        # f3 = self.reduce_layer3(f[2])  # [8,128,46,46]
        # f4 = self.reduce_layer4(f[3])  # [8,128,23,23]
        #
        # # eca attention
        # f1 = self.eca(f1)
        # f2 = self.eca(f2)
        # f3 = self.eca(f3)
        # f4 = self.eca(f4)
        # FPEM
        # f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        # f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)
        # FPEMV2
        f1, f2, f3, f4 = self.fpem1(f1, f2, f3, f4)
        f1, f2, f3, f4 = self.fpem2(f1, f2, f3, f4)

        # FFM
        # f1 = f1_1 + f1_2
        # f2 = f2_1 + f2_2
        # f3 = f3_1 + f3_2
        # f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())  # [8,128,160,160]
        f3 = self._upsample(f3, f1.size())  # [8,128,160,160]
        f4 = self._upsample(f4, f1.size())  # [8,128,160,160]


        f = torch.cat((f1, f2, f3, f4), 1)  # [8,512,160,160]

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                neck_time=time.time() - start
            ))
            start = time.time()

        # detection
        det_out = self.det_head(f)

        if not self.training and cfg.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                det_head_time=time.time() - start
            ))

        if self.training:
            det_out = self._upsample(det_out, imgs.size())
            det_loss = self.det_head.loss(det_out, gt_texts, gt_kernels, training_masks, gt_instances, gt_bboxes, max_distances)  # 加入最远坐标max_distances,miu
            outputs.update(det_loss)
        else:
            det_out = self._upsample(det_out, imgs.size(), 4)
            det_res = self.det_head.get_results(det_out, img_metas, cfg)
            outputs.update(det_res)

        return outputs
