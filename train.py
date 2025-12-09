#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from utils.utils import (
    Loss,
    preprocess,
    test,
    combined_test,
    visualize,
    SplitDistributedSampler,
)

import numa
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
import os
import gc
import random
import time
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import traceback
import uuid
import warnings

from contextlib import nullcontext
from pathlib import Path
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from models.model import UNet2D, CombinedModel
from utils.args import get_args
from utils.dataset import get_dataset

gc.collect()
torch.cuda.empty_cache()
EPS = 1e-8

def set_device(device) -> None:
    torch.cuda.set_device(device)
    num_sockets = numa.get_max_node() + 1
    socket_id = torch.cuda.current_device() // (
        max(torch.cuda.device_count() // num_sockets, 1)
    )
    node_mask = {socket_id}
    numa.bind(node_mask)

os.environ["NCCL_DEBUG"] = "WARN"

TEST_IDS = [
    "102293",
    "101746",
    "100755", 
    "100470",
    "104875",
    "106407",
    "106390",
    "106383", 
    "100000", 
    "100006",
    "100008",
    "100018", 
    "100024",
    "100038",
    "100051",
    "100053", 
    "100095", 
    "100098",
    "100099",
    "100119",  
]
test_ids = None


def main():
    args = get_args()
    args.output_dir = os.path.join(args.output_dir, "{}/{}".format(args.num_lights, args.task))
    test_ids = TEST_IDS
    args.in_channels = args.bins

    if args.task in ["depth", "specular", "combined"]:
        args.input = "hist" # lidar histograms are used as input
        args.in_channels = args.bins
        args.out_channels = 1
    elif args.task == "shadows":
        args.input = "hist_tofhist" # lidar histograms and two-bounce ToF are concatenated as input
        args.in_channels = args.bins * 2
        args.out_channels = 1
    else:
        raise("Invalid Task {}.".format(args.task))

    args.laser_kernel = None
    args.convolve = None

    print("Running with the following arguments:\n{}".format(args))

    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except: 
        local_rank = 0
    main_worker(local_rank, args, test_ids)


def main_worker(local_rank, args, test_ids):
    print("Use GPU: {} for training".format(local_rank))
    set_device(local_rank)
    dist.init_process_group()

    # create model
    model = None
    if args.task == "combined":
        model = CombinedModel(
            depth_in=args.in_channels,
            depth_out=args.out_channels,
            shadow_in=args.bins * 2,
            shadow_out=args.out_channels,
            f_maps=args.f_maps,
            num_levels=args.num_levels,
            bins=args.bins,
        )

        # Load & Freeze Pre-trained Depth Model
        if args.checkpoint_depth != "" and os.path.isfile(args.checkpoint_depth):
            print("=> loading checkpoint '{}'".format(args.checkpoint_depth))
            checkpoint = torch.load(
                args.checkpoint_depth,
                map_location="cpu",
            )
            model.depth_model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}'".format(
                    args.checkpoint_depth,
                )
            )
            del checkpoint
            torch.cuda.empty_cache()

            # Freeze all layers in model.depth_model
            for param in model.depth_model.parameters():
                param.requires_grad = False

        # Load Pre-trained Shadow model
        if args.checkpoint_shadow != "" and os.path.isfile(args.checkpoint_shadow):
            print("=> loading checkpoint '{}'".format(args.checkpoint_shadow))
            checkpoint = torch.load(
                args.checkpoint_shadow,
                map_location="cpu",
            )
            model.shadow_model.load_state_dict(checkpoint["state_dict"])
            print(
                "=> loaded checkpoint '{}'".format(
                    args.checkpoint_shadow,
                )
            )
            del checkpoint
            torch.cuda.empty_cache()

    else:
        model = UNet2D(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            final_sigmoid=False,
            is3d=False,
            layer_order="cbr",
            f_maps=args.f_maps,
            num_levels=args.num_levels,
            upsample_image=args.spat_down,
        )
    print(model)
    print("Total parameters: {}".format(sum(p.numel() for p in model.parameters())))

    model.cuda()
    print(
        "torch.cuda.memory_allocated: %fGB"
        % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.memory_reserved: %fGB"
        % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
    )
    print(
        "torch.cuda.max_memory_reserved: %fGB"
        % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
    )

    for param in model.parameters():
        param.data = param.data.contiguous()

    # optimizer, schedulers, loss
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=False
    )
    criterion = Loss(args)

    if sum([args.amp, args.bf16, args.tf32]) > 1:
        raise NotImplementedError
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=1000_000,
        pct_start=1e-2,
    )
    log_dir = args.output_dir
    print("Logging to {}".format(log_dir))
    ckpt_dir = os.path.join(log_dir, "data")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "test_output"), exist_ok=True)
    tb_writer = SummaryWriter(log_dir=log_dir) if dist.get_rank() == 0 else None

    start_epoch = 0
    step = 0
    model = torch.nn.parallel.DistributedDataParallel(
        model, gradient_as_bucket_view=True, static_graph=True
    )

    train_dataset = get_dataset(
        data_path=args.data_root,
        file_list=args.file_list,
        mode="train",
        max_dataset_size=args.dataset_size,
        task=args.task,
        remove=test_ids,
        num_lights=args.num_lights,
        temp_res=args.temp_res,
    )
    val_dataset = get_dataset(
        data_path=args.data_root,
        file_list=args.file_list,
        mode="val",
        max_dataset_size=args.dataset_size,
        task=args.task,
        remove=test_ids,
        num_lights=args.num_lights,
        temp_res=args.temp_res,
    )
    test_dataset = get_dataset(
        data_path=args.data_root,
        file_list=args.file_list,
        mode="val",
        max_dataset_size=args.dataset_size,
        task=args.task,
        ids=test_ids,
        num_lights=args.num_lights,
        temp_res=args.temp_res,
    )

    print("Train: {}, Val: {}".format(len(train_dataset), len(val_dataset)))
    train_sampler = SplitDistributedSampler(train_dataset, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    best_val_loss = 1e10
    best_val_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        end = time.time()
        count = 0
        for i, data in enumerate(train_loader):
            # measure data loading time
            data_time = time.time() - end

            if args.task == "combined":
                images = data[0].cuda(non_blocking=True).float()
                lights = data[1].cuda(non_blocking=True).float()
                depth = data[2].cuda(non_blocking=True).float()
                ground_truth = data[3].cuda(non_blocking=True).float()
            else:
                images = data[0].cuda(non_blocking=True).float()
                ground_truth = data[1].cuda(non_blocking=True).float()

                try:
                    images, ground_truth = preprocess(
                        images,
                        ground_truth,
                        args.input,
                        args.bins,
                        convolve=args.convolve,
                        model_type=args.model,
                        temp_down=args.temp_down,
                        spat_down=args.spat_down,
                        kernel=args.laser_kernel,
                    )
                except Exception as e:
                    print(traceback.format_exc())
                    continue

            with (
                nullcontext() if (step + 1) % args.grad_steps == 0 else model.no_sync()
            ):
                with torch.cuda.amp.autocast(
                    enabled=args.amp or args.bf16,
                    dtype=torch.bfloat16 if args.bf16 else torch.float16,
                ):
                    if args.task == "combined":
                        x_depth, x_recon = model(images, lights)
                        x_recon = x_recon.squeeze(1)
                    else:
                        x_recon = model(images).squeeze(1)

                # Loss
                loss = criterion.compute_loss(
                    x_recon, ground_truth, torch.sum(images, dim=1).squeeze()
                )

                if (step + 1) % args.log_steps == 0 and tb_writer:
                    loss_tb = loss.item()

                # compute gradient and do SGD step
                loss = loss / args.grad_steps
                loss = grad_scaler.scale(loss)
                loss.backward()

            if (step + 1) % args.grad_steps == 0:
                grad_scaler.step(optimizer)
                grad_scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            if (step + 1) % args.log_steps == 0 and tb_writer:
                tb_writer.add_scalar(
                    "train/loss", loss_tb, global_step=step, new_style=True
                )
                print(
                    f"Step: {step}, Loss: {loss_tb:.4f}, "
                    f"Data time: {data_time:.4f}, Batch time: {batch_time:.4f}"
                )
            if count == 0 and dist.get_rank() == 0:
                visualize(
                    args,
                    images,
                    [x_depth, x_recon] if args.task == "combined" else x_recon,
                    [depth, ground_truth] if args.task == "combined" else ground_truth,
                    os.path.join(
                        ckpt_dir,
                        "train_epoch_{}_{}".format(
                            str(epoch + 1).zfill(4), dist.get_rank()
                        ),
                    ),
                )

            count += 1

            scheduler.step()
            step += 1

        if dist.get_rank() == 0:
            # validation
            model.eval()
            with torch.no_grad():
                metrics_l1 = []
                metrics_iou = []
                metrics_f1 = []
                for i, vdata in enumerate(val_loader):

                    if args.task == "combined":
                        vimages = vdata[0].cuda(non_blocking=True).float()
                        vlights = vdata[1].cuda(non_blocking=True).float()
                        vdepth = vdata[2].cuda(non_blocking=True).float()
                        v_gt = vdata[3].cuda(non_blocking=True).float()
                    else:
                        vimages = vdata[0].cuda(non_blocking=True).float()
                        v_gt = vdata[1].cuda(non_blocking=True).float()

                        try:
                            vimages, v_gt = preprocess(
                                vimages,
                                v_gt,
                                args.input,
                                args.bins,
                                convolve=args.convolve,
                                model_type=args.model,
                                temp_down=args.temp_down,
                                spat_down=args.spat_down,
                                kernel=args.laser_kernel,
                            )
                        except Exception as e:
                            print(traceback.format_exc())
                            continue

                    # Forward pass
                    with torch.cuda.amp.autocast(
                        enabled=args.amp or args.bf16,
                        dtype=torch.bfloat16 if args.bf16 else torch.float16,
                    ):
                        if args.task == "combined":
                            v_depth, v_recon = model(images, lights)
                            v_recon = v_recon.squeeze(1)
                        else:
                            v_recon = model(vimages).squeeze(1)

                    # L1 Metrics
                    metric_l1 = criterion.compute_val_loss(v_recon, v_gt)
                    if type(metric_l1) is tuple:
                        metrics_l1.append(metric_l1[0])
                        metrics_iou.append(metric_l1[1].cuda())
                        metrics_f1.append(metric_l1[2].cuda())
                    else:
                        metrics_l1.append(metric_l1)

                    # Save image
                    if i == 0:
                        visualize(
                            args,
                            vimages,
                            [v_depth, v_recon] if args.task == "combined" else v_recon,
                            [vdepth, v_gt] if args.task == "combined" else v_gt,
                            os.path.join(
                                ckpt_dir,
                                "val_epoch_{}_{}".format(
                                    str(epoch + 1).zfill(4), dist.get_rank()
                                ),
                            ),
                        )
            metrics_l1 = torch.stack(metrics_l1).mean()
            tb_writer.add_scalar(
                "val/l1_loss", metrics_l1, global_step=step, new_style=True
            )
            if len(metrics_iou) > 0:
                metrics_iou = torch.stack(metrics_iou).mean()
                tb_writer.add_scalar(
                    "val/iou", metrics_iou, global_step=step, new_style=True
                )
            if len(metrics_f1) > 0:
                metrics_f1 = torch.stack(metrics_f1).mean()
                tb_writer.add_scalar(
                    "val/f1", metrics_f1, global_step=step, new_style=True
                )
            comparison = (
                best_val_loss if type(best_val_loss) is not list else best_val_loss[0]
            )
            if metrics_l1 < comparison:
                if metrics_iou != [] and metrics_f1 != []:
                    best_val_loss = [
                        metrics_l1.item(),
                        metrics_iou.item(),
                        metrics_f1.item(),
                    ]
                else:
                    best_val_loss = metrics_l1
                best_val_epoch = epoch

            # test model and save outputs
            if args.task != "combined":
                test(
                    model,
                    test_loader,
                    args,
                    dist.get_rank(),
                    os.path.join(ckpt_dir, "test_output"),
                    ckpt_dir,
                    test_ids,
                )
            else:
                combined_test(
                    model,
                    test_loader,
                    args,
                    dist.get_rank(),
                    os.path.join(ckpt_dir, "test_output"),
                    ckpt_dir,
                    test_ids,
                )

            # switch to train mode
            model.train()

            # # save checkpoint
            if epoch % 50 == 0 or epoch == args.epochs - 1:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "state_dict": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join(
                        ckpt_dir, "model_{}.pth".format(str(epoch + 1).zfill(4))
                    ),
                )

            # Save config after the first epoch only
            if epoch == 0:
                fp = open(os.path.join(ckpt_dir, "config.txt"), "w")
                fp.write("{}".format(str(args)))
                fp.close()

            fp = open(os.path.join(ckpt_dir, "val_loss.txt"), "w")
            fp.write("best_val_epoch, best_val_loss, current_val_loss\n")
            if metrics_iou != [] and metrics_f1 != []:
                fp.write(
                    "{}, {}, {}".format(
                        best_val_epoch,
                        best_val_loss,
                        [
                            metrics_l1.item(),
                            metrics_iou.item(),
                            metrics_f1.item(),
                        ],
                    )
                )
            else:
                fp.write("{}, {}, {}".format(best_val_epoch, best_val_loss, metrics_l1))
            fp.close()
        dist.barrier()


if __name__ == "__main__":
    main()
