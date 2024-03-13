# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xiaoyu Tian
# ---------------------------------------------
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32
from mmcv.runner import auto_fp16
from mmdet.models import HEADS
from mmdet.models.builder import build_loss
from torch import nn


@HEADS.register_module()
class OccHead(BaseModule):
    """Head of Occupancy Network.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 pillar_h=16,
                 num_classes=18,
                 in_dims=256,
                 out_dim=256,
                 use_mask=False,  # True
                 loss_occ=None,  # CrossEntropyLoss
                 ):

        super(OccHead, self).__init__()

        self.fp16_enabled = False
        self.num_classes = num_classes
        self.use_mask = use_mask

        self.loss_occ = build_loss(loss_occ)

        self.bev_h, self.bev_w = bev_h, bev_w

        self.in_dims = in_dims
        self.out_dim = out_dim
        self.pillar_h = pillar_h  # 16

        self.final_conv = ConvModule(
            self.in_dims,  # 256
            self.out_dim,  # 256
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv2d')
        )

        self.predictor = nn.Sequential(  # 256->512->288
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, self.pillar_h * num_classes),
        )

    @auto_fp16(apply_to='bev_embed')
    def forward(self, bev_embed):
        """FlashOcc <https://arxiv.org/abs/2311.12058>
        Args:
            bev_embed: (1,256,100,100) - bs,c,dx,dy
        Returns:
            occ_pred: (1,200,200,16,18) - bs,dx,dy,dz,n_cls
        """

        # (bs,c,dx,dy)-->(bs,c,dx,dy)-->(bs,dx,dy,c)
        occ_pred = self.final_conv(bev_embed).permute(0, 2, 3, 1)
        bs, dx, dy = occ_pred.shape[:3]
        # (bs,dx,dy,c)-->(bs,dx,dy,2*c)-->(bs,dx,dy,dz*n_cls)
        occ_pred = self.predictor(occ_pred)
        occ_pred = occ_pred.view(bs, dx, dy, self.pillar_h, self.num_classes)  # (bs,200,200,16,18)

        return occ_pred

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             voxel_semantics,
             mask_camera,
             preds_dicts):

        loss_dict = dict()
        occ = preds_dicts
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 17
        loss_occ = self.loss_single(voxel_semantics, mask_camera, occ)
        loss_dict['loss_occ'] = loss_occ
        return loss_dict

    def loss_single(self, voxel_semantics, mask_camera, preds):
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)
            num_total_samples = mask_camera.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)

        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = preds.reshape(-1, self.num_classes)
            loss_occ = self.loss_occ(preds, voxel_semantics, )

        return loss_occ

    @force_fp32(apply_to=('preds'))
    def get_occ(self, preds):
        """Generate occupancy from occ head predictions.
        Args:
            preds : occ results.
        Returns:
            list[dict]: labels.
        """
        occ_out = preds
        occ_score = occ_out.softmax(-1)
        occ_score = occ_score.argmax(-1)

        return occ_score
