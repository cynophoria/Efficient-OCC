import copy
import math

import torch
from mmcv.runner import BaseModule
from mmdet.models import NECKS


@NECKS.register_module()
class LSViewTransformer(BaseModule):

    def __init__(self,
                 n_voxels,
                 voxel_size,
                 linear_sample):
        super(LSViewTransformer, self).__init__()

        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.linear_sample = linear_sample

    @staticmethod
    def _compute_projection(img_meta, stride, noise=0):
        projection = []
        intrinsic = torch.eye(3, dtype=torch.float32)
        intrinsic[:2] /= stride
        extrinsics = map(torch.tensor, img_meta["ego2img"])
        for extrinsic in extrinsics:
            if noise > 0:
                projection.append(intrinsic @ extrinsic[:3] @ img_meta['bda_mat'].inverse() + noise)
            else:
                projection.append(intrinsic @ extrinsic[:3] @ img_meta['bda_mat'].inverse())
        return torch.stack(projection)

    def forward(self, img_shape, img_metas, mlvl_feats):

        mlvl_voxels = []
        for i in range(len(self.n_voxels)):

            n_voxels, voxel_size = self.n_voxels[i], self.voxel_size[i]
            voxels = get_voxels(  # [3,dx,dy,dz]
                n_voxels=torch.tensor(n_voxels),
                voxel_size=torch.tensor(voxel_size),
                origin=torch.tensor(img_metas[0]["origin"]),
                linear_sample=self.linear_sample
            ).to(mlvl_feats[0].device)

            mlvl_volumes = []
            for lvl, mlvl_feat in enumerate(mlvl_feats):
                stride_i = math.ceil(img_shape[-1] / mlvl_feat.shape[-1])  # 4,
                # (bs*seq*num_cams,c,h,w)->(bs,seq*num_cams,c,h,w)
                mlvl_feat = mlvl_feat.reshape([img_shape[0], -1] + list(mlvl_feat.shape[1:]))
                # (bs,seq*num_cams,c,h,w)->list(bs,num_cams,c,h,w)
                mlvl_feat_split = torch.split(mlvl_feat, 6, dim=1)

                volume_list = []
                for seq_id in range(len(mlvl_feat_split)):
                    volumes = []
                    for batch_id, seq_img_meta in enumerate(img_metas):
                        feat_i = mlvl_feat_split[seq_id][batch_id]  # (bs,c,h,w)
                        img_meta = copy.deepcopy(seq_img_meta)
                        img_meta["ego2img"] = img_meta["ego2img"][seq_id * 6:(seq_id + 1) * 6]
                        if isinstance(img_meta["img_shape"], list):
                            img_meta["img_shape"] = img_meta["img_shape"][seq_id * 6:(seq_id + 1) * 6]
                            img_meta["img_shape"] = img_meta["img_shape"][0]
                        height = math.ceil(img_meta["img_shape"][0] / stride_i)
                        width = math.ceil(img_meta["img_shape"][1] / stride_i)
                        # (num_cams,3,4)
                        projection = self._compute_projection(img_meta, stride_i).to(feat_i.device)

                        volume, valid = backproject_vanilla(  # (num_cams,c,dx,dy,dz)
                            feat_i[:, :, :height, :width], voxels, projection)
                        volume = volume.sum(dim=0)
                        valid = valid.sum(dim=0)
                        volume = volume / valid
                        valid = valid > 0
                        volume[:, ~valid[0]] = 0.0

                        volumes.append(volume)  # batch list[(num_cams,dx,dy,dz)]
                    volume_list.append(torch.stack(volumes))  # seq list[(bs,c,dx,dy,dz)]
                mlvl_volumes.append(torch.cat(volume_list, dim=1))  # len(mlvl_feats) list[(bs,seq*c,dx,dy,dz)]

            mlvl_voxels.append(torch.cat(mlvl_volumes, dim=1))  # len(voxel) list(bs,len(mlvl_feats)*seq*c,dx,dy,dz)

        return mlvl_voxels


@torch.no_grad()
def get_voxels(n_voxels, voxel_size, origin, linear_sample):
    if linear_sample:  # 线性采点
        dz = torch.arange(n_voxels[2], dtype=torch.float32)
    else:  # 非线性采点
        cum = torch.arange(n_voxels[2], dtype=torch.float32).cumsum(dim=0)
        dz = cum / cum.max() * (n_voxels[2] - 1)

    points = torch.stack(
        torch.meshgrid(
            [
                torch.arange(n_voxels[0], dtype=torch.float32),
                torch.arange(n_voxels[1], dtype=torch.float32),
                dz,
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
