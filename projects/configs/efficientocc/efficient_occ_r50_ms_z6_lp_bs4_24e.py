_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

data_config = {
    'src_size': (900, 1600),
    'input_size': (256, 704),
    # train-aug
    'resize': (-0.06, 0.11),
    'crop': (-0.05, 0.05),
    'rot': (-5.4, 5.4),
    'flip': True,
    # test-aug
    'test_input_size': (256, 704),
    'test_resize': 0.0,
    'test_rotate': 0.0,
    'test_flip': False,
    # top, right, bottom, left
    'pad': (0, 0, 0, 0),
    'pad_divisor': 32,
    'pad_color': (0, 0, 0),
}

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

_dim_ = 256

# -*- coding: utf-8 -*-
multi_scale_id = [0, 1, 2]  # 4x/8x/16x

sequential = False
n_times = 1
samples_per_gpu = 4

model = dict(
    type='EfficientOCC',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        style='pytorch'
    ),
    neck=dict(
        type='FPN',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        in_channels=[256, 512, 1024, 2048],
        out_channels=64,
        num_outs=4),
    neck_fuse=dict(in_channels=[256, 192, 128], out_channels=[64, 64, 64]),
    neck_3d=dict(
        type='M2BevNeck',
        in_channels=_dim_,
        out_channels=_dim_,
        num_layers=6,
        stride=1,
        is_transpose=False,
        fuse=dict(in_channels=64 * n_times * 6 * 3, out_channels=_dim_),  # c*seq*h*fpn_lvl
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    seg_head=None,
    bbox_head=dict(
        type='OccHead',
        bev_h=200,
        bev_w=200,
        pillar_h=16,
        num_classes=18,
        in_dims=_dim_,
        out_dim=_dim_,
        use_mask=True,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
    ),
    multi_scale_id=multi_scale_id,  # 4x
    n_voxels=[
        [200, 200, 6],  # 4x
        [150, 150, 6],  # 8x
        [100, 100, 6],  # 16x
    ],
    voxel_size=[
        [0.5, 0.5, 1.0],  # 4x
        [2 / 3, 2 / 3, 1.0],  # 8x
        [1.0, 1.0, 1.0],  # 16x
    ],
)

dataset_type = 'InternalNuSceneOcc'
data_root = 'data/occ3d-nus/'
file_client_args = dict(backend='disk')
occ_gt_data_root = 'data/occ3d-nus'

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'voxel_semantics', 'mask_lidar', 'mask_camera'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False),
    dict(type='LoadOccGTFromFile', data_root=occ_gt_data_root),
    dict(type='KittiSetOrigin', point_cloud_range=point_cloud_range),
    dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=samples_per_gpu,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'pkl/fastocc_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        sequential=True,
        n_times=n_times,
        train_adj_ids=[1, 3, 5],
        max_interval=10,
        min_interval=0,
        prev_only=True,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'pkl/fastocc_infos_temporal_val.pkl',
             pipeline=test_pipeline,
             classes=class_names,
             modality=input_modality,
             samples_per_gpu=1,
             sequential=True,
             n_times=n_times,
             train_adj_ids=[1, 3, 5],
             max_interval=10,
             min_interval=0,
             test_adj='prev',
             test_adj_ids=[1, 3, 5],
             ),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'pkl/fastocc_infos_temporal_val.pkl',
              pipeline=test_pipeline,
              classes=class_names,
              modality=input_modality,
              sequential=sequential,
              n_times=n_times,
              train_adj_ids=[1, 3, 5],
              max_interval=10,
              min_interval=0,
              test_adj='prev',
              test_adj_ids=[1, 3, 5],
              ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1, decay_mult=1.0),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0,
    by_epoch=False
)

total_epochs = 24
evaluation = dict(start=20, interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/cascade_mask_rcnn_r50_fpn_coco-mstrain_3x_20e_nuim_bbox_mAP_0.5400_segm_mAP_0.4300.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=4)

# fp16 settings, the loss scale is specifically tuned to avoid Nan
fp16 = dict(loss_scale='dynamic')

# custom_hooks = [
#     dict(
#         type='MEGVIIEMAHook',
#         init_updates=10560,
#         priority='NORMAL',
#         interval=2,  # save only at epochs 2,4,6,...
#     ),
# ]


# [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019, 9.7 task/s, elapsed: 623s, ETA:     0s
# Starting Evaluation...
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 5.77
# ===> barrier - IoU = 37.96
# ===> bicycle - IoU = 3.19
# ===> bus - IoU = 41.19
# ===> car - IoU = 46.45
# ===> construction_vehicle - IoU = 15.92
# ===> motorcycle - IoU = 10.23
# ===> pedestrian - IoU = 15.15
# ===> traffic_cone - IoU = 9.63
# ===> trailer - IoU = 26.67
# ===> truck - IoU = 30.64
# ===> driveable_surface - IoU = 78.81
# ===> other_flat - IoU = 37.81
# ===> sidewalk - IoU = 47.34
# ===> terrain - IoU = 51.19
# ===> manmade - IoU = 38.33
# ===> vegetation - IoU = 33.11
# ===> mIoU of 6019 samples: 31.14
