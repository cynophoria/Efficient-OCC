import copy
import csv
import os
import random
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from tqdm import tqdm

from .occ_metrics import Metric_mIoU, Metric_FScore


@DATASETS.register_module()
class NuSceneOcc(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, eval_fscore=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_fscore = eval_fscore
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.data_infos = self.load_annotations(self.ann_file)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        # self.train_split=data['train_split']
        # self.val_split=data['val_split']
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        queue = []
        index_list = list(range(index - self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            queue.append(example)
        return self.union2one(queue)

    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )
        if 'occ_gt_path' in info:
            input_dict['occ_gt_path'] = info['occ_gt_path']
        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                     inverse=True)
        input_dict['ego2lidar'] = ego2lidar
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def evaluate_miou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        # if show_dir is not None:
        #     if not os.path.exists(show_dir):
        #         os.mkdir(show_dir)
        #     print('\nSaving output and gt in {} for visualization.'.format(show_dir))
        #     begin = eval_kwargs.get('begin', None)
        #     end = eval_kwargs.get('end', None)
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)
        if self.eval_fscore:
            self.fscore_eval_metrics = Metric_FScore(
                leaf_size=10,
                threshold_acc=0.4,
                threshold_complete=0.4,
                voxel_size=[0.4, 0.4, 0.4],
                range=[-40, -40, -1, 40, 40, 5.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=True,
            )
        print('\nStarting Evaluation...')
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]

            occ_gt = np.load(os.path.join(self.data_root, info['occ_gt_path']))
            # if show_dir is not None:
            #     if begin is not None and end is not None:
            #         if index >= begin and index < end:
            #             sample_token = info['token']
            #             save_path = os.path.join(show_dir, str(index).zfill(4))
            #             np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)
            #     else:
            #         sample_token = info['token']
            #         save_path = os.path.join(show_dir, str(index).zfill(4))
            #         np.savez_compressed(save_path, pred=occ_pred, gt=occ_gt, sample_token=sample_token)

            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)
            # occ_pred = occ_pred
            self.occ_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)
            if self.eval_fscore:
                self.fscore_eval_metrics.add_batch(occ_pred, gt_semantics, mask_lidar, mask_camera)

        res = self.occ_eval_metrics.count_miou(runner)
        if self.eval_fscore:
            self.fscore_eval_metrics.count_fscore()

        res.update(dict(epoch=eval_kwargs['epoch']))
        if not os.path.exists(osp.join(eval_kwargs['jsonfile_prefix'], 'results.csv')):
            os.makedirs(eval_kwargs['jsonfile_prefix'], exist_ok=True)
            with open(osp.join(eval_kwargs['jsonfile_prefix'], 'results.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(list(res.keys()))

        with open(osp.join(eval_kwargs['jsonfile_prefix'], 'results.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(res.values()))

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        print('\nFinished.')


@DATASETS.register_module()
class InternalNuSceneOcc(NuSceneOcc):

    def __init__(self,
                 sequential=False,
                 n_times=2,
                 prev_only=False,
                 next_only=False,
                 train_adj_ids=None,
                 test_adj='prev',
                 test_adj_ids=None,
                 max_interval=3,
                 min_interval=0,
                 verbose=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.sequential = sequential
        self.n_times = n_times
        self.prev_only = prev_only
        self.next_only = next_only
        self.train_adj_ids = train_adj_ids
        self.test_adj = test_adj
        self.test_adj_ids = test_adj_ids
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.verbose = verbose

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            index=index,
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if 'occ_gt_path' in info:
            input_dict['occ_gt_path'] = info['occ_gt_path']

        # ego2lidar
        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                     inverse=True)

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            ego2img_rts = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)

                ego2img_rt = lidar2img_rt @ ego2lidar
                ego2img_rts.append(ego2img_rt)

            if self.sequential:
                adjacent_type_list = []
                adjacent_id_list = []
                for time_id in range(1, self.n_times):
                    if info['prev'] is None or info['next'] is None:
                        adjacent = 'prev' if info['next'] is None else 'next'
                    else:
                        if self.prev_only or self.next_only:
                            adjacent = 'prev' if self.prev_only else 'next'
                        # stage: test
                        elif self.test_mode:
                            if self.test_adj_ids is not None:
                                assert len(self.test_adj_ids) == self.n_times - 1
                                select_id = self.test_adj_ids[time_id - 1]
                                assert self.min_interval <= select_id <= self.max_interval
                                adjacent = {True: 'prev', False: 'next'}[select_id > 0]
                            else:
                                adjacent = self.test_adj
                        # stage: train
                        elif self.train_adj_ids is not None:
                            assert len(self.train_adj_ids) == self.n_times - 1
                            select_id = self.train_adj_ids[time_id - 1]
                            assert self.min_interval <= select_id <= self.max_interval
                            adjacent = {True: 'prev', False: 'next'}[select_id > 0]
                        else:
                            adjacent = np.random.choice(['prev', 'next'])

                    if type(info[adjacent]) is list:
                        # stage: test
                        if self.test_mode:
                            if len(info[adjacent]) <= self.min_interval:
                                select_id = len(info[adjacent]) - 1
                            elif self.test_adj_ids is not None:
                                assert len(self.test_adj_ids) == self.n_times - 1
                                select_id = self.test_adj_ids[time_id - 1]
                                assert self.min_interval <= select_id <= self.max_interval
                                select_id = min(abs(select_id), len(info[adjacent]) - 1)
                            else:
                                assert self.min_interval >= 0 and self.max_interval >= 0, "single direction only here"
                                select_id_step = (self.max_interval + self.min_interval) // self.n_times
                                select_id = min(self.min_interval + select_id_step * time_id, len(info[adjacent]) - 1)
                        # stage: train
                        else:
                            if len(info[adjacent]) <= self.min_interval:
                                select_id = len(info[adjacent]) - 1
                            elif self.train_adj_ids is not None:
                                assert len(self.train_adj_ids) == self.n_times - 1
                                select_id = self.train_adj_ids[time_id - 1]
                                assert self.min_interval <= select_id <= self.max_interval
                                select_id = min(abs(select_id), len(info[adjacent]) - 1)
                            else:
                                assert self.min_interval >= 0 and self.max_interval >= 0, "single direction only here"
                                select_id = np.random.choice([adj_id for adj_id in range(
                                    min(self.min_interval, len(info[adjacent])),
                                    min(self.max_interval, len(info[adjacent])))])
                        info_adj = info[adjacent][select_id]
                        if self.verbose:
                            print(' get_data_info: ', 'time_id: ', time_id, adjacent, select_id)
                    else:
                        info_adj = info[adjacent]

                    adjacent_type_list.append(adjacent)
                    adjacent_id_list.append(select_id)

                    egocurr2global = np.eye(4, dtype=np.float32)
                    egocurr2global[:3, :3] = Quaternion(info['ego2global_rotation']).rotation_matrix
                    egocurr2global[:3, 3] = info['ego2global_translation']
                    egoadj2global = np.eye(4, dtype=np.float32)
                    egoadj2global[:3, :3] = Quaternion(info_adj['ego2global_rotation']).rotation_matrix
                    egoadj2global[:3, 3] = info_adj['ego2global_translation']
                    lidar2ego = np.eye(4, dtype=np.float32)
                    lidar2ego[:3, :3] = Quaternion(info['lidar2ego_rotation']).rotation_matrix
                    lidar2ego[:3, 3] = info['lidar2ego_translation']
                    lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                                         @ egoadj2global @ lidar2ego

                    for cam_id, (cam_type, cam_info) in enumerate(info_adj['cams'].items()):
                        image_paths.append(cam_info['data_path'])

                        # lidar curr -> lidar adj -> img adj
                        lidaradj2imgadj_rt = lidar2img_rts[cam_id]
                        lidarcurr2imgadj_rt = lidaradj2imgadj_rt @ np.linalg.inv(lidaradj2lidarcurr)
                        lidar2img_rts.append(lidarcurr2imgadj_rt)

                        # ego curr -> img adj
                        ego2imgadj_rt = lidarcurr2imgadj_rt @ ego2lidar
                        ego2img_rts.append(ego2imgadj_rt)

                if self.verbose:
                    time_list = [0.0]
                    for i in range(self.n_times - 1):
                        time = 1e-6 * (
                                info['timestamp'] - info[adjacent_type_list[i]][adjacent_id_list[i]]['timestamp'])
                        time_list.append(time)
                    print(' get_data_info: ', 'time: ', time_list)

                info['adjacent_type'] = adjacent_type_list
                info['adjacent_id'] = adjacent_id_list

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    ego2img=ego2img_rts,
                    lidar2img=lidar2img_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
