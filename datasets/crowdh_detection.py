from pathlib import Path

import torch
import torch.utils.data

from datasets.coco import CocoDetection, convert_coco_poly_to_mask, make_coco_transforms
from util.misc import get_local_rank, get_local_size


class ConvertCHPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        # area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        # target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        # give unique fixed id to each object of an image
        target["instance_id"] = torch.as_tensor([i for i in range(keep.sum().item())], dtype=torch.long)

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided CrowdHuman path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "annotations" / f'{mode}_train_half.json'),
        "val": (root / "val" / "images", root / "val" / "annotations" / f'{mode}_val_half.json'),
    }

    img_folder, ann_file = PATHS[image_set]

    n_views = 2 if (image_set in ["train", "train_full"]) and args.contrastive_pretraining else 1
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set),
                            return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(),
                            local_size=get_local_size(), n_views=n_views, prepare_fn=ConvertCHPolysToMask)

    return dataset
