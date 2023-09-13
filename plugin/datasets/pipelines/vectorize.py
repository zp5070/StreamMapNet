import cv2
import numpy as np
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString
from numpy.typing import NDArray
from typing import List, Tuple, Union, Dict
from IPython import embed
from shapely import affinity
@PIPELINES.register_module(force=True)
class VectorizeMap(object):
    """Generate vectoized map and put into `semantic_mask` key.
    Concretely, shapely geometry objects are converted into sample points (ndarray).
    We use args `sample_num`, `sample_dist`, `simplify` to specify sampling method.

    Args:
        roi_size (tuple or list): bev range .
        normalize (bool): whether to normalize points to range (0, 1).
        coords_dim (int): dimension of point coordinates.
        simplify (bool): whether to use simpily function. If true, `sample_num` \
            and `sample_dist` will be ignored.
        sample_num (int): number of points to interpolate from a polyline. Set to -1 to ignore.
        sample_dist (float): interpolate distance. Set to -1 to ignore.
    """

    def __init__(self, 
                 roi_size: Union[Tuple, List], 
                 normalize: bool,
                 coords_dim: int,
                 simplify: bool=False, 
                 sample_num: int=-1, 
                 sample_dist: float=-1, 
                 permute: bool=False,
                 aux_seg: dict=dict(
                    use_aux_seg=False,
                    ins_seg=False,
                    bev_seg=False,
                    pv_seg=False,
                    seg_classes=1,
                    feat_down_sample=32,
                    pv_thickness=1,
                    bev_thickness=1,
                    canvas_size=(400, 200) #(w, h)
                    )):

        self.coords_dim = coords_dim
        self.sample_num = sample_num
        self.sample_dist = sample_dist
        self.roi_size = np.array(roi_size)
        self.normalize = normalize
        self.simplify = simplify
        self.permute = permute
        self.aux_seg = aux_seg
        self.canvas_size = aux_seg['canvas_size']
        self.scale_x = self.canvas_size[0] / self.roi_size[0]
        self.scale_y = self.canvas_size[1] / self.roi_size[1]
        if sample_dist > 0:
            assert sample_num < 0 and not simplify
            self.sample_fn = self.interp_fixed_dist
        elif sample_num > 0:
            assert sample_dist < 0 and not simplify
            self.sample_fn = self.interp_fixed_num
        else:
            assert simplify

    def interp_fixed_num(self, line: LineString) -> NDArray:
        ''' Interpolate a line to fixed number of points.
        
        Args:
            line (LineString): line
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = np.linspace(0, line.length, self.sample_num)
        sampled_points = np.array([list(line.interpolate(distance).coords) 
            for distance in distances]).squeeze()

        return sampled_points

    def interp_fixed_dist(self, line: LineString) -> NDArray:
        ''' Interpolate a line at fixed interval.
        
        Args:
            line (LineString): line
        
        Returns:
            points (array): interpolated points, shape (N, 2)
        '''

        distances = list(np.arange(self.sample_dist, line.length, self.sample_dist))
        # make sure to sample at least two points when sample_dist > line.length
        distances = [0,] + distances + [line.length,] 
        
        sampled_points = np.array([list(line.interpolate(distance).coords)
                                for distance in distances]).squeeze()
        
        return sampled_points

    def get_vectorized_lines(self, map_geoms: Dict) -> Dict:
        ''' Vectorize map elements. Iterate over the input dict and apply the 
        specified sample funcion.
        
        Args:
            line (LineString): line
        
        Returns:
            vectors (array): dict of vectorized map elements.
        '''

        vectors = {}
        for label, geom_list in map_geoms.items():
            vectors[label] = []
            for geom in geom_list:
                if geom.geom_type == 'LineString':

                    if self.simplify:
                        line = geom.simplify(0.2, preserve_topology=True)
                        line = np.array(line.coords)
                    else:
                        line = self.sample_fn(geom)
                    line = line[:, :self.coords_dim]

                    if self.normalize:
                        line = self.normalize_line(line)
                    if self.permute:
                        if label == 3: # centerline label
                            line = self.permute_line_for_centerline(line)
                        else:
                            line = self.permute_line(line)
                    vectors[label].append(line)

                elif geom.geom_type == 'Polygon':
                    # polygon objects will not be vectorized
                    continue
                
                else:
                    raise ValueError('map geoms must be either LineString or Polygon!')
        return vectors

    def get_vectorized_lines_and_masks(self, map_geoms: Dict) -> Dict:
        ''' Vectorize map elements. Iterate over the input dict and apply the 
        specified sample funcion.
        
        Args:
            line (LineString): line
        
        Returns:
            vectors (array): dict of vectorized map elements.
        '''

        vectors, instance_masks = {}, {}
        for label, geom_list in map_geoms.items():
            vectors[label], instance_masks[label] = [], []
            for geom in geom_list:
                if geom.geom_type == 'LineString':
                    # get vectorized lines
                    if self.simplify:
                        line = geom.simplify(0.2, preserve_topology=True)
                        line = np.array(line.coords)
                    else:
                        line = self.sample_fn(geom)
                    line = line[:, :self.coords_dim]

                    if self.normalize:
                        line = self.normalize_line(line)
                    if self.permute:
                        if label == 3: # centerline label
                            line = self.permute_line_for_centerline(line)
                        else:
                            line = self.permute_line(line)
                    vectors[label].append(line)

                    # get instance masks
                    instance_mask = np.zeros((self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)

                    self.line_ego_to_mask(geom, instance_mask, color=1,
                        thickness=self.aux_seg['bev_thickness'])

                    instance_masks[label].append(instance_mask)

                elif geom.geom_type == 'Polygon':
                    # polygon objects will not be vectorized
                    continue
                
                else:
                    raise ValueError('map geoms must be either LineString or Polygon!')
        return vectors, instance_masks

    def normalize_line(self, line: NDArray) -> NDArray:
        ''' Convert points to range (0, 1).
        
        Args:
            line (LineString): line
        
        Returns:
            normalized (array): normalized points.
        '''

        origin = -np.array([self.roi_size[0]/2, self.roi_size[1]/2])

        line[:, :2] = line[:, :2] - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        line[:, :2] = line[:, :2] / (self.roi_size + eps)

        return line
    
    def permute_line(self, line: np.ndarray, padding=1e5):
        '''
        (num_pts, 2) -> (num_permute, num_pts, 2)
        where num_permute = 2 * (num_pts - 1)
        '''
        is_closed = np.allclose(line[0], line[-1], atol=1e-3)
        num_points = len(line)
        permute_num = num_points - 1
        permute_lines_list = []
        if is_closed:
            pts_to_permute = line[:-1, :] # throw away replicate start end pts
            for shift_i in range(permute_num):
                permute_lines_list.append(np.roll(pts_to_permute, shift_i, axis=0))
            flip_pts_to_permute = np.flip(pts_to_permute, axis=0)
            for shift_i in range(permute_num):
                permute_lines_list.append(np.roll(flip_pts_to_permute, shift_i, axis=0))
        else:
            permute_lines_list.append(line)
            permute_lines_list.append(np.flip(line, axis=0))

        permute_lines_array = np.stack(permute_lines_list, axis=0)

        if is_closed:
            tmp = np.zeros((permute_num * 2, num_points, self.coords_dim))
            tmp[:, :-1, :] = permute_lines_array
            tmp[:, -1, :] = permute_lines_array[:, 0, :] # add replicate start end pts
            permute_lines_array = tmp
        else:
            # padding
            padding = np.full([permute_num * 2 - 2, num_points, self.coords_dim], padding)
            permute_lines_array = np.concatenate((permute_lines_array, padding), axis=0)
        
        return permute_lines_array

    def permute_line_for_centerline(self, line: np.ndarray, padding=1e5):
        '''
        (num_pts, 2) -> (num_permute, num_pts, 2)
        where num_permute = 2 * (num_pts - 1)
        '''

        num_points = len(line)
        permute_num = num_points - 1
        permute_lines_list = []

        permute_lines_list.append(line)
        permute_lines_array = np.stack(permute_lines_list, axis=0)

        # padding
        padding = np.full([permute_num * 2 - 1, num_points, self.coords_dim], padding)
        permute_lines_array = np.concatenate((permute_lines_array, padding), axis=0)
        
        return permute_lines_array

    def line_ego_to_mask(self, 
                         line_ego: LineString, 
                         mask: NDArray, 
                         color: int=1, 
                         thickness: int=3) -> None:
        ''' Rasterize a single line to mask.
        
        Args:
            line_ego (LineString): line
            mask (array): semantic mask to paint on
            color (int): positive label, default: 1
            thickness (int): thickness of rasterized lines, default: 3
        '''

        trans_x = self.canvas_size[0] / 2
        trans_y = self.canvas_size[1] / 2
        line_ego = affinity.scale(line_ego, self.scale_x, self.scale_y, origin=(0, 0))
        line_ego = affinity.affine_transform(line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
        
        coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
        coords = coords.reshape((-1, 2))
        # coords = np.flip(coords, axis=1)
        coords[:, 1:2] = self.canvas_size[1] - coords[:, 1:2]
        assert len(coords) >= 2
        
        cv2.polylines(mask, np.int32([coords]), False, color=color, thickness=thickness)

    def __call__(self, input_dict):
        map_geoms = input_dict['map_geoms']

        if self.aux_seg.get('use_aux_seg'):
            vectors, instance_masks = self.get_vectorized_lines_and_masks(map_geoms)
            input_dict['vectors'] = vectors

            input_dict['instance_masks'] = None
            input_dict['semantic_masks'] = None
            if self.aux_seg.get('ins_seg'):
                input_dict['instance_masks'] = instance_masks
            if self.aux_seg.get('bev_seg'):
                semantic_masks = np.zeros((len(instance_masks), self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)
                merged_semantic_masks = np.zeros((1, self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)
                for label, instance_masks_list in instance_masks.items():
                    for instance_mask in instance_masks_list:
                        semantic_masks[label] += instance_mask
                    semantic_masks[label] = (semantic_masks[label] > 0).astype(np.uint8)
                    merged_semantic_masks[0] += semantic_masks[label]
                merged_semantic_masks = (merged_semantic_masks > 0).astype(np.uint8)
                if self.aux_seg.get('seg_classes') == 1:
                    input_dict['semantic_masks'] = merged_semantic_masks
                else:
                    input_dict['semantic_masks'] = semantic_masks
        else:
            input_dict['vectors'] = self.get_vectorized_lines(map_geoms)
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(simplify={self.simplify}, '
        repr_str += f'sample_num={self.sample_num}), '
        repr_str += f'sample_dist={self.sample_dist}), ' 
        repr_str += f'roi_size={self.roi_size})'
        repr_str += f'normalize={self.normalize})'
        repr_str += f'coords_dim={self.coords_dim})'
        repr_str += f'canvas_size={self.canvas_size})'

        return repr_str