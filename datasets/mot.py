import os
import os.path
from io import BytesIO
from itertools import cycle, islice
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from PIL import Image
from torch.utils.data import Sampler
from torchvision.datasets.vision import VisionDataset

import datasets.transforms as T
from datasets.coco import convert_coco_poly_to_mask
from datasets.coco_video import COCOVideo
from util.misc import get_local_rank, get_local_size


class ConvertCocoVideoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, video_id):
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
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["instance_id"] = torch.as_tensor([obj["instance_id"] for obj in anno], dtype=torch.long)[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        target["video_id"] = torch.as_tensor([video_id])

        return image, target


class ConvertMOT17VideoPolysToMask(object):
    """
    Convert coco-style annotations to a format that can be consumed by the model.
    The only difference with the coco version is that the bounding boxes are not clamped to the image dimensions.
    """
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, video_id):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        # compute the area of "clamped" boxes to filter out invisible boxes
        clamped_boxes = boxes.clone()
        clamped_boxes[:, 0::2].clamp_(min=0, max=w)
        clamped_boxes[:, 1::2].clamp_(min=0, max=h)

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

        # Check if boxes are inside the image
        keep = (clamped_boxes[:, 3] > clamped_boxes[:, 1]) & (clamped_boxes[:, 2] > clamped_boxes[:, 0])
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
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]
        target["instance_id"] = torch.as_tensor([obj["instance_id"] for obj in anno], dtype=torch.long)[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        target["video_id"] = torch.as_tensor([video_id])

        return image, target


def make_bdd_transforms(image_set, smaller_scales: bool = False):
    # imagenet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

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
            T.Normalize(mean, std),
        ])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    raise ValueError(f'unknown {image_set}')


def make_mot17_transforms(image_set):
    # imagenet mean and std
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # transformations meant to be uniformly applied to a sequence of images

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train' or image_set == 'train_full':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    # MOT17 specific: bounding boxes can have values outside the image, so no need to clamp
                    T.RandomSizeCropMOT17(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    raise ValueError(f'unknown {image_set}')


class MOTDataset(VisionDataset):

    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(MOTDataset, self).__init__(img_folder, transforms, None,
                                         None)  # transform and target_transform are set to None
        self.coco = COCOVideo(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

        self._transforms = transforms
        self.prepare = ConvertCocoVideoPolysToMask(return_masks)

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')
        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx_infos):
        idx, is_first = idx_infos

        def coco_getitem(index):
            coco = self.coco
            img_id = self.ids[index]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            path = coco.loadImgs(img_id)[0]['file_name']
            video_id = coco.loadImgs(img_id)[0]['video_id']
            img = self.get_image(path)
            return img, target, video_id

        img, target, video_id = coco_getitem(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target, video_id)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, video_id, is_first


class BddMOTDataset(MOTDataset):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(BddMOTDataset, self).__init__(img_folder, ann_file, transforms, return_masks, cache_mode, local_rank, local_size)
        self.prepare = ConvertCocoVideoPolysToMask(return_masks)


class MOT17MOTDataset(MOTDataset):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(MOT17MOTDataset, self).__init__(img_folder, ann_file, transforms, return_masks, cache_mode, local_rank, local_size)
        self.prepare = ConvertMOT17VideoPolysToMask(return_masks)


class DistributedTrainBatchSamplerMOT(Sampler):
    """
    Distributed batch sampler for MOT datasets.
    The sampler will sample video_samples frames per video
    so the number of different videos per batch is batch_size // video_samples.

    :param dataset: the dataset to sample from
    :param shuffle: whether to shuffle the dataset
    :param batch_size: the batch size
    :param video_samples: the number of frames to sample per video
    :param num_replicas: the number of distributed replicas
    :param rank: the rank of the current replica
    :param seed: the seed for the random number generator
    """
    def __init__(self, dataset, shuffle: bool = True, batch_size: int = 1,
                 video_samples: int = 10, num_replicas: int = None, rank: int = None, seed: int = 0) -> None:
        super().__init__(dataset)

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        assert (batch_size * num_replicas) % video_samples == 0, "Full batch size and video samples have to match"

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas
        self.samples_are_set = False
        self.video_samples = video_samples
        self.dataset = dataset

        self.batches = None

        self.videos = self.dataset.coco.vidToImgs

        self.imgIDsToDatasetIDs = {k: i for i, k in enumerate(sorted(dataset.coco.imgs.keys()))}
        self.imgIDsToDatasetIDs[-1] = -1

        # compute the number of batches
        self.num_videos_per_minibatch = self.batch_size * self.num_replicas // self.video_samples
        num_groups_per_video = [len(frames) // self.video_samples for frames in self.videos.values()]
        self.num_groups = sum(num_groups_per_video)
        self.num_batches = self.num_groups // self.num_videos_per_minibatch
        print(f"Training with {self.num_batches} mini-batches of "
              f"{self.num_videos_per_minibatch} * {self.video_samples} frames")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        np.random.seed(self.seed + self.epoch)

        frame_groups = list()
        for video_id, frames in self.videos.items():
            frames = np.asarray(frames)
            np.random.shuffle(frames)
            # remove frames so that the number of frames is a multiple of video_samples
            if (len(frames) % self.video_samples) != 0:
                frames = frames[:-(len(frames) % self.video_samples)]

            # goes from image_id to dataset id
            for i in range(len(frames)):
                frames[i] = self.imgIDsToDatasetIDs[frames[i]]

            num_frames = len(frames)
            frame_groups.extend(np.split(frames, num_frames // self.video_samples))

        # merge the groups to form the different videos and shuffle them
        frame_groups = np.asarray(frame_groups)  # [num_groups, video_samples]
        np.random.shuffle(frame_groups)

        # remove the last groups so that the number of groups is a multiple of the batch size
        stop_at = (self.num_groups % self.num_videos_per_minibatch)
        if stop_at != 0:
            frame_groups = frame_groups[:-stop_at]  # [num_groups, video_samples]

        # merge the groups to form the batches
        self.batches = frame_groups.reshape(-1, (self.batch_size * self.num_replicas))  # [num_batches, full_batch_size]
        if not self.batches.shape[0] == self.num_batches:
            raise ValueError("An error occurred while creating the batches.")
        self.samples_are_set = True

    def __iter__(self):
        assert self.samples_are_set, "The set_epoch() method must be called before iterating"

        batches = self.batches.reshape((self.num_batches, self.num_replicas, self.batch_size))
        batches = batches[:, self.rank, :]  # only keep the batches for the current rank

        for i in range(self.num_batches):
            yield [(batches[i, b], True)
                   for b in range(self.batch_size)]

    def __len__(self):
        return self.num_batches


class DistributedTrainBatchSubSamplerMOT(DistributedTrainBatchSamplerMOT):
    def __init__(self, dataset, shuffle: bool = True, batch_size: int = 1,
                 subsample_videos: float = 0.25, subsample_frames: float = 0.5,
                 video_samples: int = 10, num_replicas: int = None, rank: int = None, seed: int = 0) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        assert (batch_size * num_replicas) % video_samples == 0, "Full batch size and video samples have to match"

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas
        self.samples_are_set = False
        self.video_samples = video_samples
        self.dataset = dataset
        self.subsample_frames = subsample_frames
        self.subsample_videos = subsample_videos

        self.batches = None

        self.imgIDsToDatasetIDs = {k: i for i, k in enumerate(sorted(dataset.coco.imgs.keys()))}
        self.imgIDsToDatasetIDs[-1] = -1

        # subsample the videos and the frames
        np.random.seed(0)  # use a fixed seed for always subsampling in the same way
        video_keys = np.asarray(list(self.dataset.coco.vidToImgs.keys()))
        np.random.shuffle(video_keys)
        video_keys = video_keys[:round(len(video_keys) * subsample_videos)]   # subsample the videos
        kept_videos = {}
        for video_id in video_keys:
            frames = self.dataset.coco.vidToImgs[video_id]
            frames = np.asarray(frames)
            np.random.shuffle(frames)
            frames = frames[:round(len(frames) * subsample_frames)]  # subsample the frames of the video
            kept_videos[video_id] = frames

        self.videos = kept_videos

        # compute the number of batches
        self.num_videos_per_minibatch = self.batch_size * self.num_replicas // self.video_samples
        num_groups_per_video = [len(frames) // self.video_samples for frames in self.videos.values()]
        self.num_groups = sum(num_groups_per_video)
        self.num_batches = self.num_groups // self.num_videos_per_minibatch
        print(f"Training with {self.num_batches} mini-batches of "
              f"{self.num_videos_per_minibatch} * {self.video_samples} frames")


class DistributedValidBatchSamplerMOT(Sampler):
    def __init__(self, dataset, batch_size: int = 1, num_replicas: int = None, rank: int = None) -> None:

        super().__init__(dataset)

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.batch_size = batch_size
        self.rank = rank
        self.num_replicas = num_replicas
        self.samples_are_set = False
        self.dataset = dataset

        self.imgIDsToDatasetIDs = {k: i for i, k in enumerate(sorted(dataset.coco.imgs.keys()))}

        # compute the length of the videos and sort them in descending order
        self.videos_length = {k: len(v) for k, v in dataset.coco.vidToImgs.items()}
        self.videos_length = dict(sorted(self.videos_length.items(), key=lambda item: item[1], reverse=True))

        videos_images_gpus = [[] for i in range(self.num_replicas)]

        # distribute the videos to the different GPUs
        direction = 1
        gpu_index = 0
        for v, l in self.videos_length.items():
            videos_images_gpus[gpu_index].append(
                list(map(self.imgIDsToDatasetIDs.get, dataset.coco.vidToImgs[v])))  # directly append dataset indices
            gpu_index += direction
            # if the gpu index is out of bounds, change the direction
            if gpu_index > self.num_replicas - 1 or gpu_index < 0:
                direction = direction * -1
                gpu_index += direction

        self.frames = [[[] for i in range(self.batch_size)] for j in range(self.num_replicas)]
        reset_flags = [[[] for i in range(self.batch_size)] for j in range(self.num_replicas)]

        for r in range(self.num_replicas):
            videos_images_gpu = sorted(videos_images_gpus[r], key=len, reverse=True)
            direction = 1
            batch_item_index = 0
            for video_images_gpu in videos_images_gpu:
                self.frames[r][batch_item_index].extend(video_images_gpu)
                flags = [False if i != 0 else True for i in range(len(video_images_gpu))]
                reset_flags[r][batch_item_index].extend(flags)
                batch_item_index += direction
                if batch_item_index > self.batch_size - 1 or batch_item_index < 0:
                    direction = direction * -1
                    batch_item_index += direction

        # cycle the videos to align the number of frames per batch item
        wasted_frames = 0
        for r in range(self.num_replicas):
            max_batch_item_len = len(max(self.frames[r], key=len))
            for bi, (batch_item, batch_flag) in enumerate(zip(self.frames[r], reset_flags[r])):
                if len(batch_item) < max_batch_item_len:
                    wasted_frames += (max_batch_item_len - len(batch_item))
                    self.frames[r][bi] = list(islice(cycle(batch_item), max_batch_item_len))
                    reset_flags[r][bi] = list(islice(cycle(batch_flag), max_batch_item_len))

        # repeat frames to have the same number of samples per GPU
        max_gpu_len = 0
        for r in range(self.num_replicas):
            gpu_items_num = len(self.frames[r][0])
            if gpu_items_num > max_gpu_len:
                max_gpu_len = gpu_items_num

        for r in range(self.num_replicas):
            if len(self.frames[r][0]) < max_gpu_len:
                wasted_frames += (max_gpu_len - len(self.frames[r][0])) * self.batch_size
                for bi, (batch_item, batch_flag) in enumerate(zip(self.frames[r], reset_flags[r])):
                    self.frames[r][bi] = list(islice(cycle(batch_item), max_gpu_len))
                    reset_flags[r][bi] = list(islice(cycle(batch_flag), max_gpu_len))

        # to numpy array
        total_frames = 0
        for r in range(self.num_replicas):
            assert all(self.frames[
                           r]), "Insufficient number of videos for the selected number of GPUs and batch size, try to reduce at least one of them."
            self.frames[r] = np.array(self.frames[r])
            reset_flags[r] = np.array(reset_flags[r])
            total_frames += np.sum(self.frames[r].size)

        self.reset_flags = reset_flags

        print(
            "Fraction of wasted frames due to batch size and GPUs n combination: {}".format(
                round(wasted_frames / total_frames, 2)))

    def __iter__(self):
        for i in range(self.frames[self.rank].shape[-1]):
            yield [(self.frames[self.rank][b, i],
                    self.reset_flags[self.rank][b, i])
                   for b in (range(self.batch_size))]

    def __len__(self):
        return len(self.frames[self.rank][0])


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train" / "images", root / "train" / "annotations" / f'{mode}_train_half.json'),
        "val": (root / "val" / "images", root / "val" / "annotations" / f'{mode}_val_half.json'),
        "train_full": (root / "train_full" / "images", root / "train_full" / "annotations" / f'{mode}_train_full.json'),
        "test": (root / "test" / "images", root / "test" / "annotations" / f'{mode}_test.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    if args.dataset_file == 'bdd100k':
        dataset = BddMOTDataset(img_folder, ann_file, transforms=make_bdd_transforms(image_set, smaller_scales=args.bdd_smaller_scales),
                                return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(),
                                local_size=get_local_size())
    elif args.dataset_file == 'mot17':
        dataset = MOT17MOTDataset(img_folder, ann_file, transforms=make_mot17_transforms(image_set),
                                  return_masks=args.masks, cache_mode=args.cache_mode, local_rank=get_local_rank(),
                                  local_size=get_local_size())
    else:
        raise ValueError(f'dataset {args.dataset_file} not supported')

    return dataset
