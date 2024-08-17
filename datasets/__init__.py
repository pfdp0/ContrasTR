# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco
from .bdd100k_detection import build as build_bdd
from .crowdh_detection import build as build_crowdh
from .mot import build as build_mot, MOTDataset


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco
    if isinstance(dataset, MOTDataset):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    elif args.dataset_file == 'bdd100k':
        return build_bdd(image_set, args)
    elif args.dataset_file == 'crowdh':
        return build_crowdh(image_set, args)
    elif args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


def build_mot_dataset(image_set, args):
    if args.dataset_file in['bdd100k', 'mot17']:
        return build_mot(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
