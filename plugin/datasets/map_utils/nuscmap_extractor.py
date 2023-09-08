from shapely.geometry import LineString, box, Polygon
from shapely import affinity, ops, strtree

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from .utils import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union

import networkx as nx

def to_patch_coord(new_polygon, patch_angle, patch_x, patch_y):
    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                  origin=(patch_x, patch_y), use_radians=False)
    new_polygon = affinity.affine_transform(new_polygon,
                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
    return new_polygon

class CNuScenesMapExplorer(NuScenesMapExplorer):
    def __ini__(self, *args, **kwargs):
        super(self, CNuScenesMapExplorer).__init__(*args, **kwargs)

    def _get_centerline(self,
                           patch_box: Tuple[float, float, float, float],
                           patch_angle: float,
                           layer_name: str,
                           return_token: bool = False) -> dict:
        """
         Retrieve the centerline of a particular layer within the specified patch.
         :param patch_box: Patch box defined as [x_center, y_center, height, width].
         :param patch_angle: Patch orientation in degrees.
         :param layer_name: name of map layer to be extracted.
         :return: dict(token:record_dict, token:record_dict,...)
         """
        if layer_name not in ['lane','lane_connector']:
            raise ValueError('{} is not a centerline layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_api, layer_name)

        centerline_dict = dict()
        for record in records:
            if record['polygon_token'] is None:
                continue
            polygon = self.map_api.extract_polygon(record['polygon_token'])

            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)

                if not new_polygon.is_empty:
                    centerline = self.map_api.discretize_lanes(
                            record, 0.5)
                    centerline = list(self.map_api.discretize_lanes([record['token']], 0.5).values())[0]
                    centerline = LineString(np.array(centerline)[:,:2].round(3))
                    if centerline.is_empty:
                        continue
                    centerline = centerline.intersection(patch)
                    if not centerline.is_empty:
                        centerline = \
                            to_patch_coord(centerline, patch_angle, patch_x, patch_y)

                        record_dict = dict(
                            centerline=centerline,
                            token=record['token'],
                            incoming_tokens=self.map_api.get_incoming_lane_ids(record['token']),
                            outgoing_tokens=self.map_api.get_outgoing_lane_ids(record['token']),
                        )
                        centerline_dict.update({record['token']: record_dict})
        return centerline_dict

class NuscMapExtractor(object):
    """NuScenes map ground-truth extractor.

    Args:
        data_root (str): path to nuScenes dataset
        roi_size (tuple or list): bev range
    """
    def __init__(self, data_root: str, roi_size: Union[List, Tuple]) -> None:
        self.roi_size = roi_size
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=data_root, map_name=loc)
            self.map_explorer[loc] = CNuScenesMapExplorer(self.nusc_maps[loc])
        
        # local patch in nuScenes format
        self.local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2, 
                roi_size[0] / 2, roi_size[1] / 2)

        self.centerline_classes = ['lane_connector','lane']
    
    def _union_ped(self, ped_geoms: List[Polygon]) -> List[Polygon]:
        ''' merge close ped crossings.
        
        Args:
            ped_geoms (list): list of Polygon
        
        Returns:
            union_ped_geoms (Dict): merged ped crossings 
        '''

        def get_rec_direction(geom):
            rect = geom.minimum_rotated_rectangle
            rect_v_p = np.array(rect.exterior.coords)[:3]
            rect_v = rect_v_p[1:]-rect_v_p[:-1]
            v_len = np.linalg.norm(rect_v, axis=-1)
            longest_v_i = v_len.argmax()

            return rect_v[longest_v_i], v_len[longest_v_i]

        tree = strtree.STRtree(ped_geoms)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = []
        remain_idx = [i for i in range(len(ped_geoms))]
        for i, pgeom in enumerate(ped_geoms):

            if i not in remain_idx:
                continue
            # update
            remain_idx.pop(remain_idx.index(i))
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            final_pgeom.append(pgeom)

            for o in tree.query(pgeom):
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue

                o_v, o_v_norm = get_rec_direction(o)
                cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
                if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                    final_pgeom[-1] =\
                        final_pgeom[-1].union(o)
                    # update
                    remain_idx.pop(remain_idx.index(o_idx))

        results = []
        for p in final_pgeom:
            results.extend(split_collections(p))
        return results
        
    def get_map_geom(self, 
                     location: str, 
                     e2g_translation: Union[List, NDArray],
                     e2g_rotation: Union[List, NDArray]) -> Dict[str, List[Union[LineString, Polygon]]]:
        ''' Extract geometries given `location` and ego pose.
        
        Args:
            location (str): city name
            e2g_translation (array): ego2global translation, shape (3,)
            e2g_rotation (array): ego2global quaternion, shape (4, )
            
        Returns:
            geometries (Dict): extracted geometries by category.
        '''

        # (center_x, center_y, len_y, len_x) in nuscenes format
        patch_box = (e2g_translation[0], e2g_translation[1], 
                self.roi_size[1], self.roi_size[0])
        rotation = Quaternion(e2g_rotation)
        yaw = quaternion_yaw(rotation) / np.pi * 180

        # get dividers
        lane_dividers = self.map_explorer[location]._get_layer_line(
                    patch_box, yaw, 'lane_divider')
        
        road_dividers = self.map_explorer[location]._get_layer_line(
                    patch_box, yaw, 'road_divider')
        
        all_dividers = []
        for line in lane_dividers + road_dividers:
            all_dividers += split_collections(line)

        # get ped crossings
        ped_crossings = []
        ped = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'ped_crossing')
        
        for p in ped:
            ped_crossings += split_collections(p)
        # some ped crossings are split into several small parts
        # we need to merge them
        ped_crossings = self._union_ped(ped_crossings)
        
        ped_crossing_lines = []
        for p in ped_crossings:
            # extract exteriors to get a closed polyline
            line = get_ped_crossing_contour(p, self.local_patch)
            if line is not None:
                ped_crossing_lines.append(line)

        # get boundaries
        # we take the union of road segments and lanes as drivable areas
        # we don't take drivable area layer in nuScenes since its definition may be ambiguous
        road_segments = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'road_segment')
        lanes = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'lane')
        union_roads = ops.unary_union(road_segments)
        union_lanes = ops.unary_union(lanes)
        drivable_areas = ops.unary_union([union_roads, union_lanes])
        
        drivable_areas = split_collections(drivable_areas)
        
        # boundaries are defined as the contour of drivable areas
        boundaries = get_drivable_area_contour(drivable_areas, self.roi_size)

        # get centerline
        centerline_geom = self.get_centerline_geom(location, patch_box, yaw, self.centerline_classes)
        centerline_geoms_list, _ = self.union_centerline(centerline_geom)

        return dict(
            divider=all_dividers, # List[LineString]
            ped_crossing=ped_crossing_lines, # List[LineString]
            boundary=boundaries, # List[LineString]
            drivable_area=drivable_areas, # List[Polygon],
            centerline = centerline_geoms_list
        )

    def get_centerline_geom(self, location, patch_box, patch_angle, layer_names):
        map_geom = {}
        for layer_name in layer_names:
            if layer_name in self.centerline_classes:
                return_token = False
                layer_centerline_dict = self.map_explorer[location]._get_centerline(
                patch_box, patch_angle, layer_name, return_token=return_token)
                if len(layer_centerline_dict.keys()) == 0:
                    continue
                # import ipdb;ipdb.set_trace()
                map_geom.update(layer_centerline_dict)
        return map_geom

    def union_centerline(self, centerline_geoms):
        # import ipdb;ipdb.set_trace()
        pts_G = nx.DiGraph()
        junction_pts_list = []
        for key, value in centerline_geoms.items():
            centerline_geom = value['centerline']
            if centerline_geom.geom_type == 'MultiLineString':
                start_pt = np.array(centerline_geom.geoms[0].coords).round(3)[0]
                end_pt = np.array(centerline_geom.geoms[-1].coords).round(3)[-1]
                for single_geom in centerline_geom.geoms:
                    single_geom_pts = np.array(single_geom.coords).round(3)
                    for idx, pt in enumerate(single_geom_pts[:-1]):
                        pts_G.add_edge(tuple(single_geom_pts[idx]),tuple(single_geom_pts[idx+1]))
            elif centerline_geom.geom_type == 'LineString':
                centerline_pts = np.array(centerline_geom.coords).round(3)
                start_pt = centerline_pts[0]
                end_pt = centerline_pts[-1]
                for idx, pts in enumerate(centerline_pts[:-1]):
                    pts_G.add_edge(tuple(centerline_pts[idx]),tuple(centerline_pts[idx+1]))
            else:
                raise NotImplementedError
            valid_incoming_num = 0
            for idx, pred in enumerate(value['incoming_tokens']):
                if pred in centerline_geoms.keys():
                    valid_incoming_num += 1
                    pred_geom = centerline_geoms[pred]['centerline']
                    if pred_geom.geom_type == 'MultiLineString':
                        pred_pt = np.array(pred_geom.geoms[-1].coords).round(3)[-1]
        #                 if pred_pt != centerline_pts[0]:
                        pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
                    else:
                        pred_pt = np.array(pred_geom.coords).round(3)[-1]
                        pts_G.add_edge(tuple(pred_pt), tuple(start_pt))
            if valid_incoming_num > 1:
                junction_pts_list.append(tuple(start_pt))
            
            valid_outgoing_num = 0
            for idx, succ in enumerate(value['outgoing_tokens']):
                if succ in centerline_geoms.keys():
                    valid_outgoing_num += 1
                    succ_geom = centerline_geoms[succ]['centerline']
                    if succ_geom.geom_type == 'MultiLineString':
                        succ_pt = np.array(succ_geom.geoms[0].coords).round(3)[0]
        #                 if pred_pt != centerline_pts[0]:
                        pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
                    else:
                        succ_pt = np.array(succ_geom.coords).round(3)[0]
                        pts_G.add_edge(tuple(end_pt), tuple(succ_pt))
            if valid_outgoing_num > 1:
                junction_pts_list.append(tuple(end_pt))

        roots = (v for v, d in pts_G.in_degree() if d == 0)
        leaves = [v for v, d in pts_G.out_degree() if d == 0]
        all_paths = []
        for root in roots:
            paths = nx.all_simple_paths(pts_G, root, leaves)
            all_paths.extend(paths)
            # for leave in leaves:
            #     paths = nx.all_simple_paths(pts_G, root, leave)
            #     all_paths.extend(paths)
        final_centerline_paths = []
        for path in all_paths:
            merged_line = LineString(path)
            merged_line = merged_line.simplify(0.2, preserve_topology=True)
            final_centerline_paths.append(merged_line)
        return final_centerline_paths, pts_G