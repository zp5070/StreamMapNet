import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchvision.models.resnet import resnet18, resnet50

from mmdet3d.models.builder import (build_backbone, build_head,
                                    build_neck)

from .base_mapper import BaseMapper, MAPPERS
from copy import deepcopy
from ..utils.memory_buffer import StreamTensorMemory
from mmcv.cnn.utils import constant_init, kaiming_init
from mmcv.utils import TORCH_VERSION, digit_version

@MAPPERS.register_module()
class StreamMapNet(BaseMapper):

    def __init__(self,
                 bev_h,
                 bev_w,
                 roi_size,
                 backbone_cfg=dict(),
                 head_cfg=dict(),
                 neck_cfg=None,
                 model_name=None, 
                 streaming_cfg=dict(),
                 pretrained=None,
                 aux_seg=dict(),
                 **kwargs):
        super().__init__()

        #Attribute
        self.model_name = model_name
        self.last_epoch = None
  
        self.backbone = build_backbone(backbone_cfg)

        if neck_cfg is not None:
            self.neck = build_head(neck_cfg)
        else:
            self.neck = nn.Identity()

        self.head = build_head(head_cfg)
        self.num_decoder_layers = self.head.transformer.decoder.num_layers
        self.k_one2many = self.head.k_one2many

        # aux seg
        self.aux_seg = aux_seg
        if self.aux_seg['use_aux_seg']:
            embed_dims = self.backbone.transformer.embed_dims
            if not (self.aux_seg['bev_seg'] or self.aux_seg['pv_seg']):
                raise ValueError('aux_seg must have bev_seg or pv_seg')
            if self.aux_seg['bev_seg']:
                self.seg_head = nn.Sequential(
                    nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dims, aux_seg['seg_classes'], kernel_size=1, padding=0)
                )
            if self.aux_seg['pv_seg']:            
                self.pv_seg_head = nn.Sequential(
                    nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dims, self.aux_seg['seg_classes'], kernel_size=1, padding=0)
                )

        # BEV 
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.roi_size = roi_size

        if streaming_cfg:
            self.streaming_bev = streaming_cfg['streaming_bev']
        else:
            self.streaming_bev = False
        if self.streaming_bev:
            self.stream_fusion_neck = build_neck(streaming_cfg['fusion_cfg'])
            self.batch_size = streaming_cfg['batch_size']
            self.bev_memory = StreamTensorMemory(
                self.batch_size,
            )
            
            xmin, xmax = -roi_size[0]/2, roi_size[0]/2
            ymin, ymax = -roi_size[1]/2, roi_size[1]/2
            x = torch.linspace(xmin, xmax, bev_w)
            y = torch.linspace(ymax, ymin, bev_h)
            y, x = torch.meshgrid(y, x)
            z = torch.zeros_like(x)
            ones = torch.ones_like(x)
            plane = torch.stack([x, y, z, ones], dim=-1)

            self.register_buffer('plane', plane.double())
        
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if pretrained:
            import logging
            logger = logging.getLogger()
            from mmcv.runner import load_checkpoint
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            try:
                self.neck.init_weights()
            except AttributeError:
                pass
            if self.streaming_bev:
                self.stream_fusion_neck.init_weights()

    def update_bev_feature(self, curr_bev_feats, img_metas):
        '''
        Args:
            curr_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
            img_metas: current image metas (List of #bs samples)
            bev_memory: where to load and store (training and testing use different buffer)
            pose_memory: where to load and store (training and testing use different buffer)

        Out:
            fused_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
        '''

        bs = curr_bev_feats.size(0)
        fused_feats_list = []

        memory = self.bev_memory.get(img_metas)
        bev_memory, pose_memory = memory['tensor'], memory['img_metas']
        is_first_frame_list = memory['is_first_frame']

        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                fused_feats_list.append(new_feat)
            else:
                # else, warp buffered bev feature to current pose
                prev_e2g_trans = self.plane.new_tensor(pose_memory[i]['ego2global_translation'], dtype=torch.float64)
                prev_e2g_rot = self.plane.new_tensor(pose_memory[i]['ego2global_rotation'], dtype=torch.float64)
                curr_e2g_trans = self.plane.new_tensor(img_metas[i]['ego2global_translation'], dtype=torch.float64)
                curr_e2g_rot = self.plane.new_tensor(img_metas[i]['ego2global_rotation'], dtype=torch.float64)
                
                prev_g2e_matrix = torch.eye(4, dtype=torch.float64, device=prev_e2g_trans.device)
                prev_g2e_matrix[:3, :3] = prev_e2g_rot.T
                prev_g2e_matrix[:3, 3] = -(prev_e2g_rot.T @ prev_e2g_trans)

                curr_e2g_matrix = torch.eye(4, dtype=torch.float64, device=prev_e2g_trans.device)
                curr_e2g_matrix[:3, :3] = curr_e2g_rot
                curr_e2g_matrix[:3, 3] = curr_e2g_trans

                curr2prev_matrix = prev_g2e_matrix @ curr_e2g_matrix
                prev_coord = torch.einsum('lk,ijk->ijl', curr2prev_matrix, self.plane).float()[..., :2]

                # from (-30, 30) or (-15, 15) to (-1, 1)
                prev_coord[..., 0] = prev_coord[..., 0] / (self.roi_size[0]/2)
                prev_coord[..., 1] = -prev_coord[..., 1] / (self.roi_size[1]/2)

                warped_feat = F.grid_sample(bev_memory[i].unsqueeze(0), 
                                prev_coord.unsqueeze(0), 
                                padding_mode='zeros', align_corners=False).squeeze(0)
                new_feat = self.stream_fusion_neck(warped_feat, curr_bev_feats[i])
                fused_feats_list.append(new_feat)

        fused_feats = torch.stack(fused_feats_list, dim=0)

        self.bev_memory.update(fused_feats, img_metas)
        
        return fused_feats

    def forward_train(self,
                      img,
                      vectors,
                      points=None,
                      img_metas=None,
                      semantic_masks=None,
                      instance_masks=None,
                      pv_masks=None,
                      gt_depth=None,
                      **kwargs):
        '''
        Args:
            img: torch.Tensor of shape [B, N, 3, H, W]
                N: number of cams
            vectors: list[list[Tuple(lines, length, label)]]
                - lines: np.array of shape [num_points, 2]. 
                - length: int
                - label: int
                len(vectors) = batch_size
                len(vectors[_b]) = num of lines in sample _b
            img_metas: 
                img_metas['lidar2img']: [B, N, 4, 4]
        Out:
            loss, log_vars, num_sample
        '''
        #  prepare labels and images

        gts_one2many, gts, img, img_metas, valid_idx, points = self.batch_data(
            vectors, img, img_metas, img.device, points, instance_masks)
        
        bs = img.shape[0]

        # Backbone
        ret_dict = self.backbone(img, img_metas=img_metas, points=points)
        _bev_feats = ret_dict['bev']
        if self.streaming_bev:
            self.bev_memory.train()
            _bev_feats = self.update_bev_feature(_bev_feats, img_metas)
        
        # Neck
        bev_feats = self.neck(_bev_feats)

        preds_list, loss_dict, det_match_idxs, det_match_gt_idxs = self.head(
            bev_features=bev_feats, 
            img_metas=img_metas, 
            gts=gts,
            gts_one2many=gts_one2many,
            return_loss=True)

        # calculate depth loss
        if gt_depth is not None:
            depth = ret_dict['depth']
            loss_depth = self.backbone.transformer.encoder.get_depth_loss(gt_depth, depth)
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                loss_depth = torch.nan_to_num(loss_depth)
            loss_dict.update(loss_depth=loss_depth)

        # calculate seg loss
        if self.aux_seg['use_aux_seg']:
            if self.aux_seg['bev_seg']:
                seg_bev_embed = bev_feats
                outputs_seg = self.seg_head(seg_bev_embed)
                loss_seg = self.head.loss_seg(outputs_seg, semantic_masks.float())
                loss_dict['loss_seg'] = loss_seg
            
            if self.aux_seg['pv_seg']:
                mlvl_feats = ret_dict['mlvl_feats']
                bs, num_cam, _, feat_h, feat_w = mlvl_feats[-1].shape
                outputs_pv_seg = self.pv_seg_head(mlvl_feats[-1].flatten(0,1))
                outputs_pv_seg = outputs_pv_seg.view(bs, num_cam, -1, feat_h, feat_w)
                loss_pv_seg = self.head.loss_seg(outputs_pv_seg, pv_masks.float())
                loss_dict['loss_pv_seg'] = loss_pv_seg

        # format loss
        loss = 0
        for name, var in loss_dict.items():
            loss = loss + var

        # update the log
        log_vars = {k: v.item() for k, v in loss_dict.items()}
        log_vars.update({'total': loss.item()})

        num_sample = img.size(0)

        return loss, log_vars, num_sample

    @torch.no_grad()
    def forward_test(self, img, points=None, img_metas=None, **kwargs):
        '''
            inference pipeline
        '''

        #  prepare labels and images
        
        tokens = []
        for img_meta in img_metas:
            tokens.append(img_meta['token'])

        result_dict = self.backbone(img, img_metas, points=points)
        _bev_feats = result_dict['bev']
        img_shape = [_bev_feats.shape[2:] for i in range(_bev_feats.shape[0])]

        if self.streaming_bev:
            self.bev_memory.eval()
            _bev_feats = self.update_bev_feature(_bev_feats, img_metas)
            
        # Neck
        bev_feats = self.neck(_bev_feats)

        preds_list = self.head(bev_feats, img_metas=img_metas, return_loss=False)
        
        # take predictions from the last layer
        preds_dict = preds_list[-1]

        results_list = self.head.post_process(preds_dict, tokens)

        return results_list

    def batch_data(self, vectors, imgs, img_metas, device, points=None, instance_masks=None):
        bs = len(vectors)
        # filter none vector's case
        num_gts = []
        for idx in range(bs):
            num_gts.append(sum([len(v) for k, v in vectors[idx].items()]))
        valid_idx = [i for i in range(bs) if num_gts[i] > 0]
        assert len(valid_idx) == bs # make sure every sample has gts

        gts = []
        all_labels_list = []
        all_lines_list = []
        all_masks_list = []
        gts_one2many = []
        all_labels_list_one2many = []
        all_lines_list_one2many = []
        all_masks_list_one2many = []
        for idx in range(bs):
            labels = []
            lines = []
            for label, _lines in vectors[idx].items():
                for _line in _lines:
                    labels.append(label)
                    if len(_line.shape) == 3: # permutation
                        num_permute, num_points, coords_dim = _line.shape
                        lines.append(torch.tensor(_line).reshape(num_permute, -1)) # (38, 40)
                    elif len(_line.shape) == 2:
                        lines.append(torch.tensor(_line).reshape(-1)) # (40, )
                    else:
                        assert False
            mask_labels = []
            mask_lines = []
            for label, _lines in instance_masks[idx].items():
                for _line in _lines:
                    mask_labels.append(label)
                    mask_lines.append(torch.tensor(_line, dtype=float))
            assert mask_labels == labels
            all_labels_list.append(torch.tensor(labels, dtype=torch.long).to(device))
            all_lines_list.append(torch.stack(lines).float().to(device))
            all_masks_list.append(torch.stack(mask_lines).float().to(device))
            all_labels_list_one2many.append(torch.tensor(labels * self.k_one2many, dtype=torch.long).to(device))
            all_lines_list_one2many.append(torch.stack(lines * self.k_one2many).float().to(device))
            all_masks_list_one2many.append(torch.stack(mask_lines * self.k_one2many).float().to(device))
        gts = {
            'labels': all_labels_list,
            'lines': all_lines_list,
            'masks': all_masks_list
        }
        gts_one2many = {
            'labels': all_labels_list_one2many,
            'lines': all_lines_list_one2many,
            'masks': all_masks_list_one2many
        }
        gts = [deepcopy(gts) for _ in range(self.num_decoder_layers)]
        gts_one2many = [deepcopy(gts_one2many) for _ in range(self.num_decoder_layers)]

        return gts_one2many, gts, imgs, img_metas, valid_idx, points

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.streaming_bev:
            self.bev_memory.train(*args, **kwargs)
    
    def eval(self):
        super().eval()
        if self.streaming_bev:
            self.bev_memory.eval()

