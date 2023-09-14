import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES

import torch
from pyquaternion import Quaternion


@PIPELINES.register_module(force=True)
class LoadMultiViewImagesFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filenames']
        img = [mmcv.imread(name, self.color_type) for name in filename]
        if self.to_float32:
            img = [i.astype(np.float32) for i in img]
        results['img'] = img
        results['img_shape'] = [i.shape for i in img]
        results['ori_shape'] = [i.shape for i in img]
        # Set initial values for default meta_keys
        results['pad_shape'] = [i.shape for i in img]
        # results['scale_factor'] = 1.0
        num_channels = 1 if len(img[0].shape) < 3 else img[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__} (to_float32={self.to_float32}, '\
            f"color_type='{self.color_type}')"


@PIPELINES.register_module()
class CustomPointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs = np.stack(results['img'])
        img_aug_matrix  = results['img_aug_matrix']
        post_rots = [torch.tensor(single_aug_matrix[:3, :3]).to(torch.float) for single_aug_matrix in img_aug_matrix]
        post_trans = torch.stack([torch.tensor(single_aug_matrix[:3, 3]).to(torch.float) for single_aug_matrix in img_aug_matrix])
        # import pdb;pdb.set_trace()
        intrins = results['camera_intrinsics']
        depth_map_list = []
        
        for cid in range(len(imgs)):
            # import pdb;pdb.set_trace()
            lidar2lidarego = torch.tensor(results['lidar2ego']).to(torch.float32)
            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = results['ego2global_rotation']
            lidarego2global[:3, 3] = results['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)
            cam2camego = torch.tensor(results['camera2ego'][cid])

            camego2global = results['camego2global'][cid]

            cam2img = torch.tensor(intrins[cid]).to(torch.float32)
            
            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)

            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T.to(torch.float)) + lidar2img[:3, 3].to(torch.float).unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[1],
                                             imgs.shape[2])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)

        ##################################################################
        # i = 0
        # import os
        # import cv2
        # from PIL import Image
        # for image_id in range(imgs.shape[0]):
        #     i+=1
        #     image = imgs[image_id]
        #     gt_depth_image = depth_map[image_id].numpy()
            
        #     gt_depth_image = np.expand_dims(gt_depth_image,2).repeat(3,2)
            
        #     #apply colormap on depth image(image must be converted to 8-bit per pixel first)
        #     im_color=cv2.applyColorMap(cv2.convertScaleAbs(gt_depth_image,alpha=15),cv2.COLORMAP_JET)
        #     #convert to mat png
        #     image[gt_depth_image>0] = im_color[gt_depth_image>0]
        #     im=Image.fromarray(np.uint8(image))
        #     #save image
        #     os.makedirs("visualize_1", exist_ok=True)
        #     im.save('visualize_1/visualize_{}.png'.format(i))
        #################################################################

        results['gt_depth'] = depth_map
        return results
