# Copyright (c) Facebook, Inc. and its affiliates.
import os
import yaml

from detectron2.data import DatasetCatalog, MetadataCatalog

REPLICA_SEM_SEG_CATEGORIES = [
    {"name": 'undefined', "id" : 0, "trainId" : 0},
    {"name": 'beanbag', "id" : 1, "trainId" : 1},
    {"name": 'bed', "id" : 2, "trainId" : 2},
    {"name": 'bike', "id" : 3, "trainId" : 3},
    {"name": 'book', "id" : 4, "trainId" : 4},
    {"name": 'cabinet', "id" : 5, "trainId" : 5},
    {"name": 'ceiling', "id" : 6, "trainId" : 6},
    {"name": 'chair', "id" : 7, "trainId" : 7},
    {"name": 'clothing', "id" : 8, "trainId" : 8},
    {"name": 'container', "id" : 9, "trainId" : 9},
    {"name": 'curtain', "id" : 10, "trainId" : 10},
    {"name": 'cushion', "id" : 11, "trainId" : 11},
    {"name": 'door', "id" : 12, "trainId" : 12},
    {"name": 'floor', "id" : 13, "trainId" : 13},
    {"name": 'indoor-plant', "id" : 14, "trainId" : 14},
    {"name": 'lamp', "id" : 15, "trainId" : 15},
    {"name": 'refrigerator', "id" : 16, "trainId" : 16},
    {"name": 'rug', "id" : 17, "trainId" : 17},
    {"name": 'shelf', "id" : 18, "trainId" : 18},
    {"name": 'sink', "id" : 19, "trainId" : 19},
    {"name": 'sofa', "id" : 20, "trainId" : 20},
    {"name": 'stair', "id" : 21, "trainId" : 21},
    {"name": 'structure', "id" : 22, "trainId" : 22},
    {"name": 'table', "id" : 23, "trainId" : 23},
    {"name": 'tv-screen', "id" : 24, "trainId" : 24},
    {"name": 'tv-stand', "id" : 25, "trainId" : 25},
    {"name": 'wall', "id" : 26, "trainId" : 26},
    {"name": 'wall-cabinet', "id" : 27, "trainId" : 27},
    {"name": 'wall-decoration', "id" : 28, "trainId" : 28},
    {"name": 'window', "id" : 29, "trainId" : 29}
]

REPLICA_COLOR_PALETTE = list(map(tuple, [
    [0, 0, 0],
    [174, 199, 232],
    [255, 127, 14],
    [255, 187, 120],
    [44, 160, 60],
    [152, 223, 138],
    [214, 39, 40],
    [255, 152, 150],
    [148, 103, 189],
    [197, 176, 213],
    [140, 86, 75],
    [196, 156, 148],
    [227, 119, 194],
    [247, 182, 210],
    [123, 126, 129],
    [195, 200, 205],
    [188, 189, 34],
    [215, 219, 141],
    [23, 190, 207],
    [158, 218, 229],
    [57, 59, 121],
    [82, 84, 163],
    [107, 110, 207],
    [140, 162, 82],
    [181, 207, 107],
    [206, 219, 156],
    [140, 109, 49],
    [189, 158, 57],
    [231, 186, 82],
    [231, 203, 148],
])) # len = 30

def _get_replica_meta():
    # Id 0 is reserved for ignore_label, we do not change ignore_label for 0
    # to 255 in our pre-processing ANYMORE. Feng modified it
    stuff_ids = [k["id"] for k in REPLICA_SEM_SEG_CATEGORIES]
    assert len(stuff_ids) == 30, len(stuff_ids)

    stuff_classes = [k["name"] for k in REPLICA_SEM_SEG_CATEGORIES]

    ret = {
        "stuff_classes": stuff_classes,
    }
    return ret


def register_all_replica_recursive(root):
    meta = _get_replica_meta()
    for mode, dirname in [("train", "training"), ("val", "validation")]:
        name = f"replica_sem_seg_{mode}"
        DatasetCatalog.register(name, lambda mode=mode: get_replica_dicts(mode))
        
        MetadataCatalog.get(name).set(
        stuff_classes=meta["stuff_classes"][:],
        evaluator_type="sem_seg",
        ignore_label=0,  # NOTE: gt is saved in uint8 = 255 bytes. 
        image_root=root,
        sem_seg_root=root,
        stuff_colors=REPLICA_COLOR_PALETTE,
        )

def get_replica_dicts(mode, 
                      replica_baseline_config='/home/fsun/segfusion/configs/fusion/replica_accuracy.yaml'):
    segfusion_dir = "/home/fsun/segfusion"
    replica_dir = os.path.join(segfusion_dir, 'replica')
    
    with open(replica_baseline_config) as file:
        config = yaml.full_load(file)
        scene_list = os.path.join(segfusion_dir, config["DATA"][mode + "_scene_list"])
        print('train_list', scene_list)
        
        
    dataset_dicts = []
    with open(scene_list) as f:
        lines = f.readlines()
        
        for l in lines:
            scene_id, sequence_id = l.split("/")[:2]
            rgb_img_dir = os.path.join(replica_dir, scene_id, sequence_id, 'left_rgb')
            gt_img_dir = os.path.join(replica_dir, scene_id, sequence_id, 'left_class30')
            
            n_imgs = len([name for name in os.listdir(rgb_img_dir) if os.path.isfile(os.path.join(rgb_img_dir, name))])

            for i in range(n_imgs):
                record = {}                
                record["file_name"] = os.path.join(rgb_img_dir, "{}.png".format(i))
                record["sem_seg_file_name"] = os.path.join(gt_img_dir, "{}.png".format(i))
                record["image_id"] = scene_id + '_' + sequence_id + '_' + str(i)
                record["height"] = 512
                record["width"] = 512
                dataset_dicts.append(record)
    return dataset_dicts


_root = "~/segfusion/replica"
register_all_replica_recursive(_root)
