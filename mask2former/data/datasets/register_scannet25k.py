# Copyright (c) Facebook, Inc. and its affiliates.
import os
import yaml
import numpy as np
import pickle

from detectron2.data import DatasetCatalog, MetadataCatalog

SCANNET_SEM_SEG_CATEGORIES = [
    {"name": 'undefined', "id" : 0, "trainId" : 0},
    {"name": 'wall', "id" : 1, "trainId" : 1},
    {"name": 'floor', "id" : 2, "trainId" : 2},
    {"name": 'cabinet', "id" : 3, "trainId" : 3},
    {"name": 'bed', "id" : 4, "trainId" : 4},
    {"name": 'chair', "id" : 5, "trainId" : 5},
    {"name": 'sofa', "id" : 6, "trainId" : 6},
    {"name": 'table', "id" : 7, "trainId" : 7},
    {"name": 'door', "id" : 8, "trainId" : 8},
    {"name": 'window', "id" : 9, "trainId" : 9},
    {"name": 'bookshelf', "id" : 10, "trainId" : 10},
    {"name": 'picture', "id" : 11, "trainId" : 11},
    {"name": 'counter', "id" : 12, "trainId" : 12},
    {"name": 'desk', "id" : 13, "trainId" : 13},
    {"name": 'curtain', "id" : 14, "trainId" : 14},
    {"name": 'refrigerator', "id" : 15, "trainId" : 15},
    {"name": 'shower curtain', "id" : 16, "trainId" : 16},
    {"name": 'toilet', "id" : 17, "trainId" : 17},
    {"name": 'sink', "id" : 18, "trainId" : 18},
    {"name": 'bathtub', "id" : 19, "trainId" : 19},
    {"name": 'otherfurniture', "id" : 20, "trainId" : 20},
]

SCANNET_COLOR_PALETTE = list(map(tuple, [
    (0, 0, 0),           # undefined
    (174, 199, 232),     # wall
    (152, 223, 138),     # floor
    (31, 119, 180),      # cabinet
    (255, 187, 120),     # bed
    (188, 189, 34),      # chair
    (140, 86, 75),       # sofa
    (255, 152, 150),     # table
    (214, 39, 40),       # door
    (197, 176, 213),     # window
    (148, 103, 189),     # bookshelf
    (196, 156, 148),     # picture
    (23, 190, 207),      # counter
    (247, 182, 210),     # desk
    (219, 219, 141),     # curtain
    (255, 127, 14),      # refrigerator
    (158, 218, 229),     # shower curtain
    (44, 160, 44),       # toilet
    (112, 128, 144),     # sink
    (227, 119, 194),     # bathtub
    (82, 84, 163),       # otherfurn
]))

def _get_scannet25k_meta():
    # Id 0 is reserved for ignore_label, we do not change ignore_label for 0
    # to 255 in our pre-processing ANYMORE. Feng modified it
    stuff_ids = [k["id"] for k in SCANNET_SEM_SEG_CATEGORIES]
    assert len(stuff_ids) == 21, len(stuff_ids)

    stuff_classes = [k["name"] for k in SCANNET_SEM_SEG_CATEGORIES]

    ret = {
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_scannet25k_recursive(root):
    meta = _get_scannet25k_meta()
    for mode, dirname in [("train", "training"), ("val", "validation")]:
        name = f"scannet25k_sem_seg_{mode}"
        DatasetCatalog.register(name, lambda mode=mode: get_scannet25k_dicts(mode))
        
        MetadataCatalog.get(name).set(
        stuff_classes=meta["stuff_classes"][:],
        evaluator_type="sem_seg",
        ignore_label=0,  # NOTE: gt is saved in uint8 = 255 bytes. 
        image_root=root,
        sem_seg_root=root,
        stuff_colors=SCANNET_COLOR_PALETTE,
        )

def get_scannet25k_dicts(mode): 
    root_dir = _root
    
    if mode in ['train', 'val']:
        source_path = 'scans'
    elif mode == 'test':
        source_path = 'scans_test'    
        
    train_subsample_every = 1
    val_subsample_every = 1

#     # Uncomment when running eval_scannet.job on training data instead of validation data
#     if mode == 'train':
#         mode = 'val'
#     elif mode == 'val':
#         mode = 'train'

    fragment_foldername = 'axis_aligned_metas'
        
    with open(os.path.join(root_dir, fragment_foldername, 'fragments_{}.pkl'.format(mode)), 'rb') as f:
        metas = np.array(pickle.load(f))
       
        if mode == 'train':
            subsample_every = train_subsample_every
        elif mode == 'val':
            subsample_every = val_subsample_every
        else:
            raise ValueError('unrecognised dataset split')

        if subsample_every > 1:
            subsample_idx = np.arange(0, len(metas), subsample_every)
            metas = metas[subsample_idx]
        metas = list(metas)
    
    
    dataset_dicts = []
    for meta in metas:
        scene_id = meta['scene']
        
        rgb_img_dir = os.path.join(root_dir, source_path, scene_id, 'color_resized')
        gt_img_dir = os.path.join(root_dir, source_path, scene_id, 'label-filt-scannet20')


        for i in meta['image_ids']:
            record = {}                
            record["file_name"] = os.path.join(rgb_img_dir, "{}.png".format(i))
            record["sem_seg_file_name"] = os.path.join(gt_img_dir, "{}.png".format(i))
            record["image_id"] = scene_id + '_' + str(meta['fragment_id']) + '_' + str(i)
            record["height"] = 480
            record["width"] = 640
            dataset_dicts.append(record)
            
    print(mode)
    print('n images: ', len(dataset_dicts))
            
    return dataset_dicts


_root = "/project/fsun/data/scannet"
register_all_scannet25k_recursive(_root)
