# -*- coding: utf-8 -*-
import copy
import math

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
            neck_fuse,
            neck_3d,  # M2BevNeck
            bbox_head,
            seg_head,
            n_voxels,
            voxel_size,
            multi_scale_id=None,
            init_cfg=None,
            with_cp=False,
            **kwargs
    ):
        super().__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        if isinstance(neck_fuse['in_channels'], list):
            for i, (in_channels, out_channels) in enumerate(zip(neck_fuse['in_channels'], neck_fuse['out_channels'])):
                self.add_module(
                    f'neck_fuse_{i}',
                    nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        else:
            self.neck_fuse = nn.Conv2d(neck_fuse["in_channels"], neck_fuse["out_channels"], 3, 1, 1)

        self.multi_scale_id = multi_scale_id

        self.bbox_head = build_head(bbox_head)
        self.seg_head = build_seg_head(seg_head) if seg_head is not None else None

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size

        # checkpoint
        self.with_cp = with_cp

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["ego2img"])
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)

    def extract_feat(self, img, img_metas=None):
        batch_size = img.shape[0]  # bs
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

        if self.multi_scale_id is not None:  # [0, 1, 2]
            mlvl_feats_ = []
            for msid in self.multi_scale_id:
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

        mlvl_volumes = []
        for lvl, mlvl_feat in enumerate(mlvl_feats):
            stride_i = math.ceil(img.shape[-1] / mlvl_feat.shape[-1])  # P4 880 / 32 = 27.5
            # [bs*seq*nv, c, h, w] -> [bs, seq*nv, c, h, w]
            mlvl_feat = mlvl_feat.reshape([batch_size, -1] + list(mlvl_feat.shape[1:]))
            # [bs, seq*nv, c, h, w] -> list([bs, nv, c, h, w])
            mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)

            volume_list = []
            for seq_id in range(len(mlvl_feat_split)):
                volumes = []
                for batch_id, seq_img_meta in enumerate(img_metas):
                    feat_i = mlvl_feat_split[seq_id][batch_id]  # [nv, c, h, w]
                    img_meta = copy.deepcopy(seq_img_meta)
                    img_meta["ego2img"] = img_meta["ego2img"][seq_id * 6:(seq_id + 1) * 6]
                    if isinstance(img_meta["img_shape"], list):
                        img_meta["img_shape"] = img_meta["img_shape"][seq_id * 6:(seq_id + 1) * 6]
                        img_meta["img_shape"] = img_meta["img_shape"][0]
                    height = math.ceil(img_meta["img_shape"][0] / stride_i)
                    width = math.ceil(img_meta["img_shape"][1] / stride_i)

                    projection = self._compute_projection(img_meta, stride_i).to(feat_i.device)
                    n_voxels, voxel_size = self.n_voxels[lvl], self.voxel_size[lvl]
                    points = get_points(  # [3, vx, vy, vz]
                        n_voxels=torch.tensor(n_voxels),
                        voxel_size=torch.tensor(voxel_size),
                        origin=torch.tensor(img_meta["origin"]),
                    ).to(feat_i.device)

                    volume, valid = backproject_vanilla(
                        feat_i[:, :, :height, :width], points, projection)
                    volume = volume.sum(dim=0)
                    valid = valid.sum(dim=0)
                    volume = volume / valid
                    valid = valid > 0
                    volume[:, ~valid[0]] = 0.0

                    volumes.append(volume)
                volume_list.append(torch.stack(volumes))  # list([bs, c, vx, vy, vz])

            mlvl_volumes.append(torch.cat(volume_list, dim=1))  # list([bs, seq*c, vx, vy, vz])

        # bev ms: multi-scale bev map (different x/y/z)
        for i in range(len(mlvl_volumes)):
            mlvl_volume = mlvl_volumes[i]
            bs, c, x, y, z = mlvl_volume.shape
            # collapse h, [bs, seq*c, vx, vy, vz] -> [bs, seq*c*vz, vx, vy]
            mlvl_volume = mlvl_volume.permute(0, 2, 3, 4, 1).reshape(bs, x, y, z * c).permute(0, 3, 1, 2)

            # different x/y, [bs, seq*c*vz, vx, vy] -> [bs, seq*c*vz, vx', vy']
            if i != 0:
                # upsampling to top level
                mlvl_volume = resize(
                    mlvl_volume,
                    mlvl_volumes[0].size()[2:4],
                    mode='bilinear',
                    align_corners=False)
            else:
                # same x/y
                pass

            # [bs, seq*c*vz, vx', vy'] -> [bs, seq*c*vz, vx, vy, 1]
            mlvl_volume = mlvl_volume.unsqueeze(-1)
            mlvl_volumes[i] = mlvl_volume
        mlvl_volumes = torch.cat(mlvl_volumes, dim=1)  # [bs, z1*c1+z2*c2+..., vx, vy, 1]

        x = mlvl_volumes

        def _inner_forward(x):
            out = self.neck_3d(x)
            return out

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)  # (bs,256,200,200)

        return x, None, features_2d

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
            loss_seg = self.seg_head.losses(x_bev, gt_bev)
            losses.update(loss_seg)

        return losses

    def forward_test(self, img, img_metas, **kwargs):
        return self.simple_test(img, img_metas)

    def simple_test(self, img, img_metas, **kwargs):

        feature_bev, _, features_2d = self.extract_feat(img, img_metas)

        # (1,200,200,16,18)
        x = self.bbox_head(feature_bev)

        occ = self.bbox_head.get_occ(x)

        return bbox_results

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


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0]),
                torch.arange(n_voxels[1]),
                torch.arange(n_voxels[2]),
            ]
        )
    )
    new_origin = origin - n_voxels / 2.0 * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points


def backproject_vanilla(features, points, projection):
    '''
    function: 2d feature + predefined point cloud -> 3d volume
    input:
        features: [6, 64, 225, 400]
        points: [3, 200, 200, 12]
        projection: [6, 3, 4]
    output:
        volume: [6, 64, 200, 200, 12]
        valid: [6, 1, 200, 200, 12]
    '''
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    # [3, 200, 200, 12] -> [1, 3, 480000] -> [6, 3, 480000]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    # [6, 3, 480000] -> [6, 4, 480000]
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    # ego_to_cam
    # [6, 3, 4] * [6, 4, 480000] -> [6, 3, 480000]
    points_2d_3 = torch.bmm(projection, points)  # lidar2img
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()  # [6, 480000]
    z = points_2d_3[:, 2]  # [6, 480000]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0)  # [6, 480000]
    volume = torch.zeros(
        (n_images, n_channels, points.shape[-1]), device=features.device
    ).type_as(features)  # [6, 64, 480000]
    for i in range(n_images):
        volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    # [6, 64, 480000] -> [6, 64, 200, 200, 12]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    # [6, 480000] -> [6, 1, 200, 200, 12]
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid
