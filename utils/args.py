#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch ToF VAE Training")
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        metavar="N",
        help="number of total steps to run",
    )
    parser.add_argument(
        "--grad-steps",
        default=1,
        type=int,
        metavar="N",
        help="number of steps to accumulate gradient",
    )
    parser.add_argument(
        "-bn",
        "--bottleneck",
        default=16,
        type=int,
        metavar="N",
        help="bottleneck size",
    )
    parser.add_argument(
        "--log-steps",
        default=100,
        type=int,
        metavar="N",
        help="number of steps to log",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-2,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-3,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-3)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--data_root",
        help="Data path.",
        default="./data",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="model type.",
        default="unet2d",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--loss",
        help="loss type.",
        default="l1",
        type=str,
    )
    parser.add_argument(
        "--file_list",
        help="List of file identifiers.",
        default="./data/dataset.txt",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--task",
        help="Name of task, e.g. depth, shadows, tof, bins, correspondence.",
        default="depth",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        default="./output",
        type=str,
    )
    parser.add_argument(
        "--bins",
        default=637,
        type=int,
        help="number of histogram bins",
    )
    parser.add_argument(
        "-s",
        "--dataset_size",
        default=100000,
        type=int,
        help="number of elements in total dataset",
    )
    parser.add_argument(
        "-nl",
        "--num_lights",
        default=25,
        type=int,
        help="number of illumination spots",
    )
    parser.add_argument(
        "--out_channels",
        default=1,
        type=int,
        help="number of network out channels",
    )
    parser.add_argument(
        "--in_channels",
        default=637,
        type=int,
        help="number of network in channels",
    )
    parser.add_argument(
        "-fm",
        "--f_maps",
        default=128,
        type=int,
        help="feature maps",
    )
    parser.add_argument(
        "-lvl",
        "--num_levels",
        default=6,
        type=int,
        help="num levels in encoder/decoder",
    )
    parser.add_argument(
        "--temp_res",
        default=0.0384,
        type=float,
        help="temporal resolution of the measurements",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--amp", action="store_true", help="use amp and float16")
    parser.add_argument("--bf16", action="store_true", help="use amp and bfloat16")
    parser.add_argument("--tf32", action="store_true", help="use tf32")
    parser.add_argument(
        "--ema", action="store_true", help="use EMA (Exponential Moving Average)"
    )
    parser.add_argument(
        "--checkpoint_depth",
        help="",
        default="./checkpoints/25/depth.pth",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_shadow",
        help="",
        default="./checkpoints/25/shadows.pth",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_path",
        help="",
        default="./checkpoints/25/depth.pth",
        type=str,
    )
    args = parser.parse_args()
    return args
