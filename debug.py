import mmcv
ann_file='datasets/streammapnet/nuscenes_map_infos_val_w_lidar.pkl'
ann = mmcv.load(ann_file)
for idx, sample in enumerate(ann):
    if sample['token'] == '30e55a3ec6184d8cb1944b39ba19d622':
        print(idx, sample['token'])