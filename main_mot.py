# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

import datasets
import util.misc as utils
from datasets import build_mot_dataset, get_coco_api_from_dataset
from datasets.mot import DistributedTrainBatchSamplerMOT, DistributedValidBatchSamplerMOT, DistributedTrainBatchSubSamplerMOT
from engine_mot import train_one_epoch, evaluate, test
from models import build_mot_model


def get_args_parser():
    parser = argparse.ArgumentParser('ContrasTR method for MOT', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--batch_size_val', default=2, type=int)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='accumulate gradients for multiple steps before updating weights, set to 1 to disable')

    parser.add_argument('--sgd', action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--diff_classes', default=False, action='store_true',
                        help='Load a model trained on a dataset with a different number of classes')

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the backbone to use")
    parser.add_argument('--swin_checkpoint', default='./data/coco', type=str)
    parser.add_argument('--swin_checkpointing', action='store_true', default=False,
                        help="Use checkpoint on the swin backbone")
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help="Freeze the backbone weights")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=True)
    parser.add_argument('--no_box_refine', dest='with_box_refine', action='store_false')
    parser.add_argument('--two_stage', default=True)
    parser.add_argument('--no_two_stage', dest='two_stage', action='store_false')

    # Deformable DETR improvements (from DINO)
    parser.add_argument("--mixed_selection", default=True)
    parser.add_argument("--no_mixed_selection", dest='mixed_selection', action='store_false')
    parser.add_argument("--look_forward_twice", default=True)
    parser.add_argument("--no_look_forward_twice", dest='look_forward_twice', action='store_false')

    # * Tracking
    parser.add_argument('--video_samples', default=10, type=int,
                        help="Number of images sampled per video per batch")
    parser.add_argument('--tracking_threshold', default=0.5, type=float,
                        help="Minimum cosine similarity to match a detected object with a previously tracked object")
    parser.add_argument('--objectness_threshold', default=0.5, type=float,
                        help="Minimum detection score threshold for predictions to be added to model memory")
    parser.add_argument('--max_prev_frames', default=5, type=int,
                        help="Maximum number of frames for which previous embeddings are kept into the model memory")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--no_contrastive_loss', dest='contrastive_loss', action='store_false',
                        help="Disables contrastive loss")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--cont_loss_coef', default=1, type=float)
    parser.add_argument('--cont_loss_temp', default=0.1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--bdd_smaller_scales', default=False, action='store_true',
                        help="use only the smaller scales for BDD100k train transforms (to lower memory usage)")
    parser.add_argument('--train_on_full_dataset', default=False, action='store_true',
                        help="train on 'train_full' set (for MOT17 official submission)")
    parser.add_argument('--subsample_train_videos', default=1, type=float)
    parser.add_argument('--subsample_train_frames', default=1, type=float)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--mixed_precision', default=True,
                        help='mixed precision training and evaluation (default: True)')
    parser.add_argument('--no_mixed_precision', dest='mixed_precision', action='store_false')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--from_pretrained', default=False, action='store_true',
                        help='start from a pretrained model (resets epoch and lr)')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--validation_freq', default=1, type=int)

    # logging
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='')
    parser.add_argument('--wandb_entity', type=str, default='')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # initialize wandb for logging
    if args.wandb and utils.get_rank() == 0:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name,
            config=vars(args),
            dir=args.output_dir,
            settings=wandb.Settings(start_method="fork")
        )

    model, criterion, postprocessors = build_mot_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.train_on_full_dataset:
        dataset_train = build_mot_dataset(image_set='train_full', args=args)
    else:
        dataset_train = build_mot_dataset(image_set='train', args=args)
    dataset_val = build_mot_dataset(image_set='val', args=args)
    if args.test:
        dataset_test = build_mot_dataset(image_set='test', args=args)

    if args.distributed:
        if args.cache_mode:
            NotImplementedError("Sampler is only implemented without cache mode")
        else:
            if args.subsample_train_videos == 1 and args.subsample_train_frames == 1:  # don't subsample (default behaviour)
                batch_sampler_train = DistributedTrainBatchSamplerMOT(dataset_train, batch_size=args.batch_size,
                                                                      video_samples=args.video_samples)
            else:
                batch_sampler_train = DistributedTrainBatchSubSamplerMOT(dataset_train, batch_size=args.batch_size,
                                                                      video_samples=args.video_samples,
                                                                      subsample_frames=args.subsample_train_frames,
                                                                      subsample_videos=args.subsample_train_videos)
            batch_sampler_val = DistributedValidBatchSamplerMOT(dataset_val, batch_size=args.batch_size_val)
            if args.test:
                batch_sampler_test = DistributedValidBatchSamplerMOT(dataset_test, batch_size=args.batch_size_val)
    else:
        if args.subsample_train_videos == 1 and args.subsample_train_frames == 1:  # don't subsample (default behaviour)
            batch_sampler_train = DistributedTrainBatchSamplerMOT(dataset_train, batch_size=args.batch_size,
                                                                  video_samples=args.video_samples,
                                                                  rank=0, num_replicas=1)
        else:
            batch_sampler_train = DistributedTrainBatchSubSamplerMOT(dataset_train, batch_size=args.batch_size,
                                                                  video_samples=args.video_samples,
                                                                  rank=0, num_replicas=1,
                                                                  subsample_frames=args.subsample_train_frames,
                                                                  subsample_videos=args.subsample_train_videos)
        batch_sampler_val = DistributedValidBatchSamplerMOT(dataset_val, batch_size=args.batch_size_val, rank=0, num_replicas=1)
        if args.test:
            batch_sampler_test = DistributedValidBatchSamplerMOT(dataset_test, batch_size=args.batch_size_val, rank=0, num_replicas=1)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_mot, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, batch_sampler=batch_sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn_mot, num_workers=args.num_workers,
                                 pin_memory=True)
    if args.test:
        data_loader_test = DataLoader(dataset_test, batch_sampler=batch_sampler_test,
                                     drop_last=False, collate_fn=utils.collate_fn_mot, num_workers=args.num_workers,
                                     pin_memory=True)

    num_training_steps = len(data_loader_train)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print("Trainable param: {}".format(n))

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names)
                 and not match_name_keywords(n, args.lr_linear_proj_names)
                 and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names)
                       and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
    ]

    if args.freeze_backbone:
        # set requires_grad=False for parameters in the backbone
        for n, p in model_without_ddp.named_parameters():
            if match_name_keywords(n, args.lr_backbone_names):
                p.requires_grad = False
    else:
        param_dicts.append({
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names)
                       and p.requires_grad],
            "lr": args.lr_backbone,
        })

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        if args.diff_classes:
            # Delete weights related to classification
            keys = list(checkpoint['model'].keys())
            for key in keys:
                if 'class_embed' in key:
                    del checkpoint['model'][key]

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if (not args.eval and not args.test and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint
                and 'epoch' in checkpoint and not args.from_pretrained):
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            warnings.warn("Overriding the resumed lr_drop with the one in the arguments")
            lr_scheduler.step_size = args.lr_drop
            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
            scaler.load_state_dict(checkpoint['scaler'])

        """# evaluate the resumed model
        if not args.eval and not args.test:
            _ = evaluate(
                model, criterion, postprocessors, data_loader_val, dataset_val, base_ds, device, -2, args.output_dir, args
            )"""

    if args.test:
        # Generate the test results
        test(model, postprocessors, data_loader_test, dataset_test, device, args)
        print("Test is done")
        return
    elif args.eval:
        valid_loss_stats, valid_metrics_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, dataset_val, base_ds, device, -1, args.output_dir, args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        batch_sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, scaler, device, epoch, args.clip_max_norm, args)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            model_checkpoint = model_without_ddp.state_dict()

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_checkpoint,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'scaler': scaler.state_dict(),
                }, checkpoint_path)

        # Validate only every "validation_freq" epochs and at the last epoch
        if epoch % args.validation_freq == (args.validation_freq - 1) or epoch == (args.epochs - 1):
            valid_loss_stats, valid_metrics_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, dataset_val, base_ds, device, epoch, args.output_dir, args
            )

            if args.distributed:
                torch.distributed.barrier()

            # logs
            test_detection_precisions = {k: v for k, v in valid_metrics_stats['coco_eval_bbox'].items() if k[0:2] == 'AP'}
            test_detection_recall = {k: v for k, v in valid_metrics_stats['coco_eval_bbox'].items() if k[0:2] == 'AR'}

            if wandb.run is not None:
                log_dict = {
                    "Loss/val": valid_loss_stats["loss"],
                    "Loss_dict/val": valid_loss_stats,
                    "Detection/Precisions": test_detection_precisions,
                    "Detection/Recalls": test_detection_recall,
                    "MOT/MOTA": valid_metrics_stats['mota'],
                    "MOT/MOTP": valid_metrics_stats['motp'],
                    "MOT/sub_metrics": valid_metrics_stats['sub_metrics'],
                    "epoch": epoch
                }
                wandb.log(log_dict, step=(epoch + 1) * num_training_steps)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in valid_loss_stats.items()},
                         **{f'val_metrics_{k}': v for k, v in valid_metrics_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}  # **{f'test_{k}': v for k, v in test_stats.items()},

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:  # TODO: check again
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
