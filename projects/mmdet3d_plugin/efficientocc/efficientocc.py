# -*- coding: utf-8 -*-
import copy

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector
from mmdet3d.core import bbox3d2result
from mmseg.models import build_head as build_seg_head
from mmseg.ops import resize


@DETECTORS.register_module()
class EfficientOCC(BaseDetector):

    def __init__(
            self,
            backbone,
            neck,
            fpn_fuse,
            neck_fuse,
            view_transformer,
            voxel_encoder,  # VoxelEncoder
            bbox_head,
            seg_head,  # SegHead
            init_cfg=None,
            with_cp=False,
            **kwargs
    ):
        super().__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.view_transformer = build_neck(view_transformer)
        self.voxel_encoder = build_neck(voxel_encoder)

        self.fpn_fuse = fpn_fuse
        if self.fpn_fuse:
            for i, (in_channels, out_channels) in enumerate(
                    zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}',
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))

        self.bbox_head = build_head(bbox_head)
        self.seg_head = build_seg_head(seg_head) if seg_head is not None else None

        # checkpoint
        self.with_cp = with_cp

    def img_encoder(self, img):
        img = img.reshape([-1] + list(img.shape)[2:])  # (bs,num_cams,C,H,W)->(bs*num_cams,C,H,W)
        x = self.backbone(img)  # (bs*num_cams,256*i,64/i,176/i),i=1,2,4,8

        # fuse features
        def _inner_forward(x):
            out = self.neck(x)
            return out

        if self.with_cp and x.requires_grad:
            mlvl_feats = cp.checkpoint(_inner_forward, x)
        else:
            mlvl_feats = _inner_forward(x)  # (bs*num_cams,64,64/i,176/i),i=1,2,4,8
        mlvl_feats = list(mlvl_feats)

        features_2d = None
        if self.seg_head:
            features_2d = mlvl_feats

        if self.fpn_fuse:  # [0, 1, 2]
            mlvl_feats_ = []
            for msid in range(3):
                # fpn output fusion
                if getattr(self, f'neck_fuse_{msid}', None) is not None:
                    fuse_feats = [mlvl_feats[msid]]
                    for i in range(msid + 1, len(mlvl_feats)):
                        resized_feat = resize(
                            mlvl_feats[i],
                            size=mlvl_feats[msid].size()[2:],
                            mode="bilinear",
                            align_corners=False)
                        fuse_feats.append(resized_feat)

                    if len(fuse_feats) > 1:
                        fuse_feats = torch.cat(fuse_feats, dim=1)
                    else:
                        fuse_feats = fuse_feats[0]
                    fuse_feats = getattr(self, f'neck_fuse_{msid}')(fuse_feats)
                    mlvl_feats_.append(fuse_feats)
                else:
                    mlvl_feats_.append(mlvl_feats[msid])
            mlvl_feats = mlvl_feats_  # (bs*num_cams,64,64/i,176/i),i=1,2,4
        else:
            mlvl_feats = mlvl_feats[:3]

        return features_2d, mlvl_feats

    def extract_feat(self, img, img_metas=None):

        features_2d, mlvl_feats = self.img_encoder(img)

        # list[(bs,len(mlvl_feats)*seq*c,dx,dy,dz)]
        mlvl_volumes = self.view_transformer(img.shape, img_metas,
                                             mlvl_feats)

        def _inner_forward(x):
            out = self.voxel_encoder(x)
            return out

        if self.with_cp and mlvl_volumes.requires_grad:
            bev_feats = cp.checkpoint(_inner_forward, mlvl_volumes)
        else:
            bev_feats = _inner_forward(mlvl_volumes)  # (bs,256,200,200)

        return bev_feats, None, features_2d

    def forward_train(self,
                      img,
                      img_metas,
                      mask_lidar=None,  # (1,200,200,16)
                      mask_camera=None,  # (1,200,200,16)
                      voxel_semantics=None,  # (1,200,200,16)
                      gt_bev_seg=None,
                      **kwargs):

        feature_bev, valids, features_2d = self.extract_feat(img, img_metas)

        assert self.bbox_head is not None or self.seg_head is not None

        losses = dict()

        # occ loss
        x = self.bbox_head(feature_bev)
        loss_occ = self.bbox_head.loss(voxel_semantics, mask_camera, x)
        losses.update(loss_occ)

        if self.seg_head is not None:
            # semantic loss
            x_bev = self.seg_head(feature_bev)
            loss_seg = self.seg_head.losses(x_bev, gt_bev_seg)
            losses.update(loss_seg)

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas, **kwargs):

        feature_bev, _, features_2d = self.extract_feat(img, img_metas)

        # (1,200,200,16,18)
        x = self.bbox_head(feature_bev)

        occ = self.bbox_head.get_occ(x)

        return occ

    def aug_test(self, imgs, img_metas, **kwargs):
        img_shape_copy = copy.deepcopy(img_metas[0]['img_shape'])
        extrinsic_copy = copy.deepcopy(img_metas[0]['lidar2img']['extrinsic'])

        x_list = []
        img_metas_list = []
        for tta_id in range(2):
            img_metas[0]['img_shape'] = img_shape_copy[24 * tta_id:24 * (tta_id + 1)]
            img_metas[0]['lidar2img']['extrinsic'] = extrinsic_copy[24 * tta_id:24 * (tta_id + 1)]
            img_metas_list.append(img_metas)

            feature_bev, _, _ = self.extract_feat(imgs[:, 24 * tta_id:24 * (tta_id + 1)], img_metas, "test")
            x = self.bbox_head(feature_bev)
            x_list.append(x)

        bbox_list = self.bbox_head.get_tta_bboxes(x_list, img_metas_list, valid=None)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in [bbox_list]
        ]
        return bbox_results

    def show_results(self, *args, **kwargs):
        pass
