import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS
from torch import nn
from torch.cuda.amp.autocast_mode import autocast


@HEADS.register_module()
class SegHead(BaseModule):
    def __init__(self,
                 in_channels,
                 num_classes,
                 down_sample,
                 semantic_threshold,
                 loss_semantic_weight=25):
        super(SegHead, self).__init__()

        self.down_sample = down_sample
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.semantic_threshold = semantic_threshold
        self.loss_semantic_weight = loss_semantic_weight

        self.final_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def get_downsampled_gt_semantic(self, gt_semantics):
        B, N, H, W = gt_semantics.shape
        gt_semantics = gt_semantics.view(
            B * N,
            H // self.down_sample,
            self.down_sample,
            W // self.down_sample,
            self.down_sample,
            1,
        )
        gt_semantics = gt_semantics.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantics = gt_semantics.view(
            -1, self.down_sample * self.down_sample)
        gt_semantics = torch.max(gt_semantics, dim=-1).values
        gt_semantics = gt_semantics.view(B * N, H // self.down_sample,
                                         W // self.down_sample)
        gt_semantics = F.one_hot(gt_semantics.long(),
                                 num_classes=2).view(-1, 2).float()
        return gt_semantics

    @force_fp32()
    def get_depth_and_semantic_loss(self, semantic_labels, semantic_preds):
        semantic_preds = semantic_preds.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        semantic_weight = torch.zeros_like(semantic_labels[:, 1:2])
        semantic_weight = torch.fill_(semantic_weight, 0.1)
        semantic_weight[semantic_labels[:, 1] > 0] = 0.9

        with autocast(enabled=False):
            pred = semantic_preds
            target = semantic_labels
            alpha = 0.25
            gamma = 2
            pt = (1 - pred) * target + pred * (1 - target)
            focal_weight = (alpha * target + (1 - alpha) *
                            (1 - target)) * pt.pow(gamma)
            semantic_loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
            semantic_loss = semantic_loss.sum() / max(1, len(semantic_loss))
        return self.loss_semantic_weight * semantic_loss

    def forward(self, mlvl_feats: torch.Tensor):
        semantic_digit = self.final_conv(mlvl_feats[0])

        semantic = semantic_digit.softmax(dim=1)
        mask = (semantic[:, 1:2] >= self.semantic_threshold)

        return mask, semantic

    def get_loss(self, pred_semantics, gt_semantics):
        semantic_labels = self.get_downsampled_gt_semantic(gt_semantics)

        loss_semantic = self.get_semantic_loss(semantic_labels, pred_semantics)

        return loss_semantic
