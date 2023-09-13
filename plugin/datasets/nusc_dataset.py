from.base_dataset import BaseMapDataset
from .map_utils.nuscmap_extractor import NuscMapExtractor
from mmdet.datasets import DATASETS
import numpy as np
from .visualize.renderer import Renderer
import mmcv
from time import time
from pyquaternion import Quaternion
import math
import os
import torch

@DATASETS.register_module()
class NuscDataset(BaseMapDataset):
    """NuScenes map dataset class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """
    
    def __init__(self, data_root, **kwargs):
        super().__init__(**kwargs)
        self.map_extractor = NuscMapExtractor(data_root, self.roi_size)
        self.renderer = Renderer(self.cat2id, self.roi_size, 'nusc')
        self.with_lidar = False
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        
        start_time = time()
        ann = mmcv.load(ann_file)
        samples = ann[::self.interval]

        if os.environ.get("DEBUG") is not None:
            samples = samples[:10]

        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples

    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        location = sample['location']
        lidar_path = sample['lidar_path']
        
        map_geoms = self.map_extractor.get_map_geom(location, sample['e2g_translation'], 
                sample['e2g_rotation'])

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
        
        ego2img_rts = []
        for cam_info in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                cam_info['extrinsics']), np.array(cam_info['intrinsics'])
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)

        # if sample['sample_idx'] == 0:
        #     is_first_frame = True
        # else:
        #     is_first_frame = self.flag[sample['sample_idx']] > self.flag[sample['sample_idx'] - 1]
        input_dict = {
            'pts_filename': lidar_path,
            'location': location,
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, **ego2cam**
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': sample['e2g_translation'], 
            'ego2global_rotation': Quaternion(sample['e2g_rotation']).rotation_matrix,
            # 'is_first_frame': is_first_frame, # deprecated
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name']
            # 'group_idx': self.flag[sample['sample_idx']]
        }
        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(sample["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = sample["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego
        if self.with_lidar:
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            input_dict["camera_intrinsics"] = []
            input_dict["camera2ego"] = []
            input_dict["camego2global"] = []
            for cam_info in sample['cams'].values():
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                lidar2cam_rt_t = lidar2cam_rt.T


                intrinsic = np.array(cam_info['intrinsics'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt_t)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt_t)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    cam_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = cam_info["sensor2ego_translation"]
                input_dict["camera2ego"].append(camera2ego)

                # camego to global transform
                camego2global = np.eye(4, dtype=np.float32)
                camego2global[:3, :3] = Quaternion(
                    cam_info['ego2global_rotation']).rotation_matrix
                camego2global[:3, 3] = cam_info['ego2global_translation']
                camego2global = torch.from_numpy(camego2global)
                input_dict["camego2global"].append(camego2global)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = intrinsic
                input_dict["camera_intrinsics"].append(camera_intrinsics)

            input_dict.update(
                dict(
                    lidar2img=lidar2img_rts,
                    lidar2cam=lidar2cam_rts,
                ))
        return input_dict