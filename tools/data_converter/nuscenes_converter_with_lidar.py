import mmcv
import numpy as np
import os
from os import path as osp
from pyquaternion import Quaternion
import argparse
from nusc_split import TRAIN_SCENES, VAL_SCENES

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

FAIL_SCENES = ['scene-0499', 'scene-0502', 'scene-0515', 'scene-0517']

def parse_args():
    parser = argparse.ArgumentParser(description='Data converter arg parser')
    parser.add_argument(
        '--data-root',
        type=str,
        help='specify the root path of dataset',
        default='datasets/nuScenes')
    parser.add_argument(
        '--newsplit',
        action='store_true')
    parser.add_argument(
        '-v','--version',
        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
        default='v1.0-trainval')
    
    args = parser.parse_args()
    return args

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }

    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep

def create_nuscenes_infos_map(root_path,
                            dest_path=None,
                            info_prefix='nuscenes',
                            version='v1.0-trainval',
                            new_split=False):
    """Create info file for map learning task on nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    else:
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    
    if new_split:
        train_scenes = TRAIN_SCENES
        val_scenes = VAL_SCENES

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    
    train_samples, val_samples, test_samples = [], [], []
    
    train_sample_idx = 0
    val_sample_idx = 0
    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)
        scene_record = nusc.get('scene', sample['scene_token'])
        log_record = nusc.get('log', scene_record['log_token'])
        location = log_record['location']
        scene_name = scene_record['name']
        if scene_name in FAIL_SCENES: continue
        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'cams': {},
            
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'e2g_translation': pose_record['translation'],
            'e2g_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
            'location': location,
            'scene_name': scene_name
        }


        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['e2g_rotation']
        e2g_t = info['e2g_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            sd_rec = nusc.get('sample_data', cam_token)
            cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])

            cam2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
            cam2ego_translation = np.array(cs_record['translation'])

            ego2cam_rotation = cam2ego_rotation.T
            ego2cam_translation = ego2cam_rotation.dot(-cam2ego_translation)

            transform_matrix = np.eye(4) #ego2cam
            transform_matrix[:3, :3] = ego2cam_rotation
            transform_matrix[:3, 3] = ego2cam_translation

            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(dict(
                extrinsics=transform_matrix, # ego2cam
                intrinsics=cs_record['camera_intrinsic'],
                img_fpath=str(nusc.get_sample_data_path(sd_rec['token']))
            ))
            info['cams'][cam] = cam_info
        
        if scene_name in train_scenes:
            info.update({
                'sample_idx': train_sample_idx,
                'prev': train_sample_idx - 1,
                'next': train_sample_idx + 1,
            })
            if sample['prev'] == '':
                info['prev'] = -1
            if sample['next'] == '':
                info['next'] = -1
            train_samples.append(info)
            train_sample_idx += 1
        elif scene_name in val_scenes:
            info.update({
                'sample_idx': val_sample_idx,
                'prev': val_sample_idx - 1,
                'next': val_sample_idx + 1,
            })
            if sample['prev'] == '':
                info['prev'] = -1
            if sample['next'] == '':
                info['next'] = -1
            val_sample_idx += 1
            val_samples.append(info)
        else:
            test_samples.append(info)
    
    if dest_path is None:
        dest_path = root_path
    
    if test:
        info_path = osp.join(dest_path, f'{info_prefix}_map_infos_test_w_lidar.pkl')
        print(f'saving test set to {info_path}')
        mmcv.dump(test_samples, info_path)

    else:
        # for training set
        if new_split:
            info_path = osp.join(dest_path, f'{info_prefix}_map_infos_train_newsplit_w_lidar.pkl')
        else:
            info_path = osp.join(dest_path, f'{info_prefix}_map_infos_train_w_lidar.pkl')
        print(f'saving training set to {info_path}')
        mmcv.dump(train_samples, info_path)

        # for val set
        if new_split:
            info_path = osp.join(dest_path, f'{info_prefix}_map_infos_val_newsplit_w_lidar.pkl')
        else:
            info_path = osp.join(dest_path, f'{info_prefix}_map_infos_val_w_lidar.pkl')
        print(f'saving validation set to {info_path}')
        mmcv.dump(val_samples, info_path)


if __name__ == '__main__':
    args = parse_args()

    create_nuscenes_infos_map(root_path=args.data_root, dest_path='datasets', version=args.version, new_split=args.newsplit)