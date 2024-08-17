from pathlib import Path

import datasets.transforms as T
from datasets.coco import CocoDetection
from util.misc import get_local_rank, get_local_size


def make_bdd_transforms(image_set, smaller_scales: bool = False):
    # imagenet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    if smaller_scales:
        print("Using smaller augmentation scales (to lower memory usage)")
        scales = [520, 560, 600, 640, 672, 704, 736]
    else:
        scales = [520, 560, 600, 640, 672, 704, 736, 768, 800, 840, 900]

    if image_set == 'train' or image_set == 'train_full':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales),
                T.Compose([
                    T.RandomResize([640, 720, 800]),
                    T.RandomSizeCrop(620, 800),
                    T.RandomResize(scales),
                ])
            ),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
    if image_set == 'val':
        return T.Compose([normalize])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "annotations" / f'{mode}_train_half.json'),
        "val": (root / "val" / "images", root / "val" / "annotations" / f'{mode}_val_half.json'),
    }

    img_folder, ann_file = PATHS[image_set]

    n_views = 2 if (image_set in ["train", "train_full"]) and args.contrastive_pretraining else 1
    dataset = CocoDetection(img_folder, ann_file, transforms=make_bdd_transforms(image_set, smaller_scales=args.bdd_smaller_scales),
                            return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(),
                            local_size=get_local_size(), n_views=n_views)

    return dataset
