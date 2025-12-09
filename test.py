#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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

from utils.args import get_args
from utils.dataset import get_dataset
from utils.models.model import UNet2D, CombinedModel
from utils.utils import test_batch

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
    "112986",
    "113440",
    "112996",
    "112176",
    "112493",
    "113458",
    "113565",
    "113668",
    "113850",
    "113485",
    "113235",
    "113718",
    "113499",
    "113472",
    "113762",
    "113525",
    "113532",
    "113692",
    "113353",
    "113236",
    "113175",
    "113211",
    "113326",
    "113409",
    "113442",
    "113307",
    "113302",
    "113629",
    "113739",
    "113346",
    "113550",
    "113628",
    "113582",
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

     # optionally resume from a checkpoint
    if os.path.isfile(args.checkpoint_path):
        print("=> loading checkpoint '{}'".format(args.checkpoint_path))
        checkpoint = torch.load(
            args.checkpoint_path,
            map_location="cpu",
        )

        state_dict = checkpoint["state_dict"]

        if args.task == "shadows":
            shadow_model_state_dict = {
                k: v for k, v in state_dict.items() if k.startswith("shadow_model.")
            }
            shadow_model_state_dict = {
                k.replace("shadow_model.", ""): v
                for k, v in shadow_model_state_dict.items()
            }
            model.load_state_dict(shadow_model_state_dict)
        else:
            model.load_state_dict(state_dict)

        print(
            "=> loaded checkpoint '{}'".format(
                args.checkpoint_path,
            )
        )
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint_path))
        exit()

    model = torch.nn.DataParallel(model)

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

    log_dir = args.output_dir
    print("Logging to {}".format(log_dir))
    ckpt_dir = os.path.join(log_dir, "data")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, "eval"), exist_ok=True)
    tb_writer = SummaryWriter(log_dir=log_dir) if dist.get_rank() == 0 else None

    start_epoch = 0
    step = 0

    test_dataset = get_dataset(
        data_path=args.data_root,
        file_list=args.file_list,
        mode="train",
        max_dataset_size=args.dataset_size,
        task=args.task,
        ids=test_ids,
        num_lights=args.num_lights,
        temp_res=args.temp_res,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # validate
    model.eval()

    try:
        test_batch(
            model,
            test_loader,
            args,
            dist.get_rank(),
            os.path.join(ckpt_dir, "eval"),
            os.path.join(ckpt_dir, "eval"),
            test_ids,
            plot_grid=True,
        )
    except Exception as e:
        print("Exception: {}".format(e))
        print(traceback.format_exc())

    dist.barrier()


if __name__ == "__main__":
    main()
