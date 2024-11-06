# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import os
import sys
from argparse import Namespace
from typing import Iterable, Optional

import math
import torch
import wandb
from torch.cuda.amp import GradScaler

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.mot_eval import MOTEvaluator, build_official_mot_eval
from datasets.panoptic_eval import PanopticEvaluator
from models.contrastr import BatchedEmbeddings


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    scaler: GradScaler, device: torch.device, epoch: int,
                    max_norm: float = 0, args: Optional[Namespace] = None):

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    num_steps = len(data_loader)
    world_size = utils.get_world_size()

    for loader_idx, (samples, targets, video_ids, reset_flags) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        samples = samples.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.mixed_precision):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            """
            the contrastive loss is computed over the full set of embeddings on each GPU but the backward pass only 
            back-propagates gradients for the GPU's embeddings; pytorch dist package averages gradients 
            during synchronization, so we have to multiply by the world size to obtain the correct gradients.
            """
            losses = sum(
                loss_dict[k] * weight_dict[k] * (world_size if "contrastive" in k else 1.)
                for k in loss_dict.keys() if k in weight_dict
            )

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)  # unscale the gradients for clipping
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, class_error=loss_dict_reduced['class_error'],
                             lr=optimizer.param_groups[0]["lr"], grad_norm=grad_total_norm, **loss_dict_reduced_unscaled)

        if loader_idx % args.print_freq == 0:
            if wandb.run is not None:
                log_dict = {
                    "Loss/train": loss_value,
                    "Loss_dict/train/unscaled": loss_dict_reduced_unscaled,
                    "Loss_dict/train/scaled": loss_dict_reduced_scaled,
                    "epoch": epoch
                }
                wandb.log(log_dict, step=epoch * num_steps + loader_idx)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, dataset, base_ds, device, epoch, output_dir, args: Namespace):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)  # COCO detection metrics
    mot_evaluator = MOTEvaluator(objectness_threshold=args.objectness_threshold)  # MOT metrics

    if epoch == -1 or epoch == args.epochs - 1:
        # evaluate with official dataset-dependent MOT metrics
        official_mot_eval = build_official_mot_eval(args.dataset_file, args.data_path, args.objectness_threshold,
                                                    args.output_dir, dataset.coco.cats)
    else:
        official_mot_eval = None

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    prev_tracking_embeddings = BatchedEmbeddings(args.batch_size_val, args.num_queries, args.hidden_dim,
                                                 max_prev_frames=args.max_prev_frames,
                                                 objectness_threshold=args.objectness_threshold,
                                                 mixed_precision=args.mixed_precision)
    prev_tracking_embeddings.to(device)
    current_frames = torch.zeros(args.batch_size_val, dtype=torch.long, device=device)

    for loader_idx, (samples, targets, video_ids, reset_flags) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if torch.any(reset_flags):
            current_frames[reset_flags] = 0  # resets frame numbers to 0 if we start a new video
            prev_tracking_embeddings.reset_memory_at(reset_flags)

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.mixed_precision):
            outputs = model(samples, prev_tracking_embeddings, current_frames)
            loss_dict = criterion(outputs, targets, skip_contrastive_loss=True)
            weight_dict = criterion.weight_dict

        # cast outputs back to fp32 for metrics and visualization
        outputs = cast_to_fp32(outputs)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        mot_results, mot_targets = postprocessors['mot'](outputs, targets)
        if official_mot_eval is not None:
            off_mot_results, _ = postprocessors['mot_official'](outputs, targets)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
        if mot_evaluator is not None:
            image_ids = [target['image_id'].item() for target in targets]
            mot_evaluator.update(video_ids, image_ids, current_frames, mot_results, mot_targets)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if official_mot_eval is not None:
            image_ids = [t['image_id'].item() for t in targets]
            video_names = [dataset.coco.videos[vi.item()]['name'] for vi in video_ids]
            frame_names = [os.path.basename(dataset.coco.imgs[ii]['file_name']) for ii in image_ids]
            official_mot_eval.add_predictions(video_names, current_frames, frame_names, off_mot_results)

        # increase frame numbers
        current_frames += 1

    # Reset memory of tracking head
    if args.distributed:
        model.module.tracking_head.reset_id_count()
    else:
        model.tracking_head.reset_id_count()

    # compute MOT metrics for all videos
    if mot_evaluator is not None:
        mot_evaluator.compute()
    if official_mot_eval is not None:
        official_mot_eval.store_subprocess_predictions()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if mot_evaluator is not None:
        mot_evaluator.synchronize_between_processes()
    if official_mot_eval is not None:
        official_mot_eval.merge_from_processes_predictions()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    if mot_evaluator is not None:
        mot_evaluator.summarize()
    if official_mot_eval is not None:
        official_mot_eval.evaluate_with_official_code()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    loss_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metrics_stats = dict()
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            metrics_stats['coco_eval_bbox'] = dict(zip(utils.DETECTION_METRICS_NAMES, coco_evaluator.coco_eval['bbox'].stats.tolist()))
        if 'segm' in postprocessors.keys():
            metrics_stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if mot_evaluator is not None:
        metrics_stats['mota'] = {k: v.item() for k, v in mot_evaluator.mota.items()}
        metrics_stats['motp'] = {k: v.item() for k, v in mot_evaluator.motp.items()}
        metrics_stats['sub_metrics'] = {k1: {k2: v2.item() for k2, v2 in v1.items()} for k1, v1 in mot_evaluator.sub_metrics_sum.items()}
    if panoptic_res is not None:
        metrics_stats['PQ_all'] = panoptic_res["All"]
        metrics_stats['PQ_th'] = panoptic_res["Things"]
        metrics_stats['PQ_st'] = panoptic_res["Stuff"]
    return loss_stats, metrics_stats, coco_evaluator


@torch.no_grad()
def test(model, postprocessors, data_loader, dataset, device, args):
    """
    :targets: include only the original target sizes
    """
    model.eval()
    num_steps = len(data_loader)

    # evaluate with official MOT metrics
    official_mot_eval = build_official_mot_eval(args.dataset_file,
                                                args.data_path,
                                                args.objectness_threshold,
                                                args.output_dir,
                                                dataset.coco.cats)

    prev_tracking_embeddings = BatchedEmbeddings(args.batch_size, args.num_queries, args.hidden_dim,
                                                 max_prev_frames=args.max_prev_frames,
                                                 objectness_threshold=args.objectness_threshold,
                                                 mixed_precision=args.mixed_precision)
    prev_tracking_embeddings.to(device)
    current_frames = torch.zeros(args.batch_size, dtype=torch.long, device=device)

    for loader_idx, (samples, targets, video_ids, reset_flags) in enumerate(data_loader):
        if torch.any(reset_flags):
            current_frames[reset_flags] = 0  # resets frame numbers to 0 if we start a new video
            prev_tracking_embeddings.reset_memory_at(reset_flags)

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples, prev_tracking_embeddings, current_frames)

        off_mot_results, _ = postprocessors['mot_official'](outputs, targets)

        # evaluating with official metrics
        if official_mot_eval is not None:
            image_ids = [t['image_id'].item() for t in targets]
            video_names = [dataset.coco.videos[vi.item()]['name'] for vi in video_ids]
            frame_names = [os.path.basename(dataset.coco.imgs[ii]['file_name']) for ii in image_ids]
            official_mot_eval.add_predictions(video_names, current_frames, frame_names, off_mot_results)

        # increase frame numbers
        current_frames += 1

        if loader_idx % args.print_freq == 0 or loader_idx == num_steps - 1:
            print(f"Official submission: [{loader_idx}/{num_steps}]")


    # Reset memory of criterion and tracking head
    if args.distributed:
        model.module.tracking_head.reset_id_count()
    else:
        model.tracking_head.reset_id_count()

    if official_mot_eval is not None:
        official_mot_eval.store_subprocess_predictions()
        official_mot_eval.merge_from_processes_predictions()


def cast_to_fp32(v_in):
    """Recursively cast tensors in a nested structure to fp32"""
    if isinstance(v_in, torch.Tensor):
        return v_in.float() if v_in.dtype == torch.float16 else v_in
    elif isinstance(v_in, (list, tuple)):
        l_2 = list()
        for v_2 in v_in:
            l_2.append(cast_to_fp32(v_2))
        return l_2
    elif isinstance(v_in, dict):
        d_2 = dict()
        for k_2, v_2 in v_in.items():
            d_2.update({k_2: cast_to_fp32(v_2)})
        return d_2
    else:
        raise ValueError("Unhandled type {}".format(type(v_in)))
