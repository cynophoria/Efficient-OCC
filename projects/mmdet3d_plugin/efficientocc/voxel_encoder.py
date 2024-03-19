import torch
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from mmdet.models import NECKS
from mmseg.ops import resize
from torch import nn


class ResModule2D(nn.Module):
    def __init__(self, n_channels, norm_cfg=dict(type='BN2d'), groups=1):
        super().__init__()
        self.conv0 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True))
        self.conv1 = ConvModule(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.activation = nn.ReLU(inplace=True)

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): of shape (N, C, N_x, N_y, N_z).

        Returns:
            torch.Tensor: 5d feature map.
        """
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = identity + x
        x = self.activation(x)
        return x


@NECKS.register_module()
class VoxelEncoder(nn.Module):
    """voxel encoder
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers=2,
                 norm_cfg=dict(type='BN2d'),
                 stride=2,
                 fuse=None,
                 with_cp=False):
        super().__init__()

        self.with_cp = with_cp

        if fuse is not None:
            self.fuse = nn.Conv2d(fuse["in_channels"], fuse["out_channels"], kernel_size=1)
        else:
            self.fuse = None

        model = nn.ModuleList()
        model.append(ResModule2D(in_channels, norm_cfg))
        model.append(ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU', inplace=True)))
        for i in range(num_layers):
            model.append(ResModule2D(out_channels, norm_cfg))
            model.append(ConvModule(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU', inplace=True)))
        self.model = nn.Sequential(*model)

    @auto_fp16()
    def forward(self, mlvl_volumes):
        """Forward function.

        Args:
            mlvl_volumes (torch.Tensor): of shape (N, C_in, dx, dy, dz).

        Returns:
            torch.Tensor: of shape (N, C_out, dx, dy).
        """

        # collapse spatial to channel
        for i in range(len(mlvl_volumes)):
            mlvl_volume = mlvl_volumes[i]
            bs, c, x, y, z = mlvl_volume.shape
            # (bs,c,dx,dy,dz)->(bs,dx,dy,dz,c)->(bs,dx,dy,dz*c)->(bs,dz*c,dx,dy)
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
            mlvl_volumes[i] = mlvl_volume
        bev_feats = torch.cat(mlvl_volumes, dim=1)  # (bs,dz1*c1+dz2*c2+...,dx,dy)

        def _inner_forward(x):
            out = self.model.forward(x)
            return out

        if self.fuse is not None:
            bev_feats = self.fuse(bev_feats)

        if self.with_cp and bev_feats.requires_grad:
            bev_feats = cp.checkpoint(_inner_forward, bev_feats)
        else:
            bev_feats = _inner_forward(bev_feats)

        return bev_feats
