#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import traceback

# from kornia.losses import ssim_loss
from torch import nn
from torch.nn import functional as F
from utils.boundary_metric import SI_boundary_F1

EPS = 1e-8

class SplitDistributedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.drop_last:
            raise NotImplementedError

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + self.rank)
            indices = torch.randperm(self.num_samples, generator=g)  # type: ignore[arg-type]
        else:
            indices = torch.arange(self.num_samples)  # type: ignore[arg-type]

        indices = (indices * self.num_replicas + self.rank).tolist()

        return iter(indices)


class LaserKernel(torch.nn.Module):
    def __init__(self, initial_laser, device="cuda"):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            1,
            1,
            initial_laser.shape[0],
            padding=(initial_laser.shape[0] - 1) // 2,
            padding_mode="zeros",
            device=device,
        )
        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = False
        self.conv.bias.zero_()
        self.update_laser(initial_laser)

    def update_laser(self, laser):
        self.conv.weight.data = laser[None, None, ...].to(self.conv.weight.device)

    def forward(self, x):
        return self.conv(x)


def test(model, loader, args, rank, savedir, temp_dir, test_ids, plot_grid=False):
    log = open(os.path.join(savedir, "loss.txt"), "w")
    log.write("Identifier, Mean Loss\n")
    with torch.no_grad():
        for i, data in enumerate(loader):
            tof = None
            v_condition = None
            if (
                len(data) == 3
                and "cond" not in args.input
                and not (args.input == "intensity" and args.task == "shadows")
                and args.task != "depth_spec"
            ):
                val_images, val_ground_truth, tof = data
            elif len(data) == 3 and (
                "cond" in args.input
                or (args.input == "intensity" and args.task == "shadows")
            ):
                val_images, v_condition, val_ground_truth = data
            elif len(data) == 3 and args.task == "depth_spec":
                val_images, val_ground_truth, _ = data
            else:
                val_images, val_ground_truth = data
            fname = test_ids[i][:-4]
            vimages = val_images.cuda(non_blocking=True).float()
            v_gt = val_ground_truth.cuda(non_blocking=True).float()
            if v_condition is not None:
                v_condition = v_condition.cuda(non_blocking=True).float()

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
                v_recon = None
                if v_condition is None:
                    v_recon = model(vimages).squeeze()
                else:
                    v_recon = model(vimages, v_condition).squeeze()

            loss = compute_val_loss(args, v_recon, v_gt)
            if type(loss) is tuple:
                loss = [item.item() for item in loss]
            log.write("{}, {}\n".format(fname, loss))

            results = v_recon.detach().cpu().numpy()

            # Save image
            if temp_dir is not None:
                visualize(
                    args,
                    vimages,
                    v_recon,
                    v_gt,
                    os.path.join(
                        savedir,
                        "val_epoch_{}_{}".format(fname, rank),
                    ),
                    plot_grid=plot_grid,
                )
            np.save(
                os.path.join(
                    savedir,
                    "outputs_{}_{}.npy".format(fname, rank),
                ),
                results,
            )

            if tof is not None:
                np.save(
                    os.path.join(
                        savedir,
                        "2b_tof_{}_{}.npy".format(fname, rank),
                    ),
                    tof.detach().cpu().numpy().squeeze(),
                )

    log.close()


def test_batch(model, loader, args, rank, savedir, temp_dir, test_ids, plot_grid=False):
    log = open(os.path.join(savedir, "loss.txt"), "w")
    log.write("Identifier, Mean Loss\n")
    tof_total = None
    results_total = None
    images_total = None
    gt_total = None
    save_tof = False
    with torch.no_grad():
        for i, data in enumerate(loader):
            idx = int(np.floor(i * args.batch_size) / args.num_lights)
            tof = None
            v_condition = None
            if (
                len(data) == 3
                and "cond" not in args.input
                and not (args.input == "intensity" and args.task == "shadows")
            ):
                val_images, val_ground_truth, tof = data
                save_tof = True
            elif len(data) == 3 and (
                "cond" in args.input
                or (args.input == "intensity" and args.task == "shadows")
            ):
                val_images, v_condition, val_ground_truth = data
            else:
                val_images, val_ground_truth = data
            fname = test_ids[idx][:-4]
            vimages = val_images.cuda(non_blocking=True).float()
            v_gt = val_ground_truth.cuda(non_blocking=True).float()
            if v_condition is not None:
                v_condition = v_condition.cuda(non_blocking=True).float()

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
                )
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                continue

            # Forward pass
            with torch.cuda.amp.autocast(
                enabled=args.amp or args.bf16,
                dtype=torch.bfloat16 if args.bf16 else torch.float16,
            ):
                v_recon = None
                if v_condition is None:
                    v_recon = model(vimages).squeeze()
                else:
                    v_recon = model(vimages, v_condition).squeeze()

            loss = compute_val_loss(args, v_recon, v_gt)
            if type(loss) is tuple:
                loss = [item.item() for item in loss]
            log.write("{}, {}\n".format(fname, loss))

            results = v_recon.detach().cpu().numpy()

            if save_tof:
                if tof_total is None:
                    tof_total = tof.detach().cpu().numpy().squeeze()
                else:
                    tof_total = np.concatenate(
                        (tof_total, tof.detach().cpu().numpy().squeeze()), axis=0
                    )
            if results_total is None:
                results_total = results
            else:
                results_total = np.concatenate((results_total, results), axis=0)
            if images_total is None:
                images_total = vimages.detach().cpu().numpy().squeeze()
            else:
                images_total = np.concatenate(
                    (images_total, vimages.detach().cpu().numpy().squeeze()), axis=0
                )
            if gt_total is None:
                gt_total = v_gt.detach().cpu().numpy().squeeze()
            else:
                gt_total = np.concatenate(
                    (gt_total, v_gt.detach().cpu().numpy().squeeze()), axis=0
                )

            if args.task == "shadows":
                if int(np.floor((i + 1) * args.batch_size)) % args.num_lights == 0:
                    np.save(
                        os.path.join(
                            savedir,
                            "outputs_{}_{}.npy".format(fname, rank),
                        ),
                        results_total,
                    )

                    if save_tof:
                        if tof is not None:
                            np.save(
                                os.path.join(
                                    savedir,
                                    "2b_tof_{}_{}.npy".format(fname, rank),
                                ),
                                tof_total,
                            )

                        print(
                            "Saved ToF: {}, Results: {}.".format(
                                tof_total.shape, results_total.shape
                            )
                        )
                    else:
                        print("Saved Results: {}.".format(results_total.shape))

                    results_total = None
                    images_total = None
                    gt_total = None
                    tof_total = None

        if args.task != "shadows":
            np.save(
                os.path.join(
                    savedir,
                    "outputs_{}_{}.npy".format(fname, rank),
                ),
                results_total,
            )

    log.close()


def depth_test_metrics(
    model, loader, args, rank, savedir, temp_dir, test_ids, plot_grid=False
):
    log = open(os.path.join(savedir, "loss.txt"), "w")
    log.write("Identifier, Mean Loss\n")
    l1s = []
    f1s = []
    save_tof = False
    with torch.no_grad():
        for i, data in enumerate(loader):
            idx = int(np.floor(i * args.batch_size) / args.num_lights)
            tof = None
            v_condition = None
            if (
                len(data) == 3
                and "cond" not in args.input
                and not (args.input == "intensity" and args.task == "shadows")
            ):
                val_images, val_ground_truth, tof = data
                save_tof = True
            elif len(data) == 3 and (
                "cond" in args.input
                or (args.input == "intensity" and args.task == "shadows")
            ):
                val_images, v_condition, val_ground_truth = data
            else:
                val_images, val_ground_truth = data
            vimages = val_images.cuda(non_blocking=True).float()
            v_gt = val_ground_truth.cuda(non_blocking=True).float()
            if v_condition is not None:
                v_condition = v_condition.cuda(non_blocking=True).float()

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
                )
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                continue

            # Forward pass
            with torch.cuda.amp.autocast(
                enabled=args.amp or args.bf16,
                dtype=torch.bfloat16 if args.bf16 else torch.float16,
            ):
                v_recon = None
                if v_condition is None:
                    v_recon = model(vimages).squeeze()
                else:
                    v_recon = model(vimages, v_condition).squeeze()

            l1 = F.l1_loss(v_recon.squeeze(), v_gt)
            for j in range(v_recon.shape[0]):
                f1 = SI_boundary_F1(
                    v_recon.squeeze()[j].detach().cpu().numpy() * 4.5,
                    v_gt[j].detach().cpu().numpy() * 4.5,
                )
                f1s.append(f1)
            l1s.append(l1.item() * 4.5)

            print("Loss {}, L1: {}, F1: {}".format(i + 1, np.mean(l1s), np.mean(f1s)))
            log.write("{}, {}\n".format(str(i + 1), l1, f1))

    log.write("MAE: {}, Boundary F1: {}\n".format(np.mean(l1s), np.mean(f1s)))

    log.close()


def test_joint(model, loader, args, rank, savedir, temp_dir, test_ids):
    log = open(os.path.join(savedir, "loss.txt"), "w")
    log.write("Identifier, Mean Loss\n")

    with torch.no_grad():
        for i, vdata in enumerate(loader):
            fname = test_ids[i][:-4]

            # B x 256 x 256 x 235
            vimages = vdata[0].cuda(non_blocking=True).float()
            vcondition = vdata[1].cuda(non_blocking=True).float()
            v_gt_shad = vdata[2].cuda(non_blocking=True).float()
            v_gt_tof = vdata[3].cuda(non_blocking=True).float()

            try:
                vimages = preprocess(vimages, args.input, args.bins)
            except Exception as e:
                print(e)
                continue

            # Forward pass
            with torch.cuda.amp.autocast(
                enabled=args.amp or args.bf16,
                dtype=torch.bfloat16 if args.bf16 else torch.float16,
            ):
                v_recon_shad, v_recon_tof = model(vimages, vcondition)
                v_recon_shad = v_recon_shad.squeeze()
                v_recon_tof = v_recon_tof.squeeze()

            # Save image
            if i == 0 and temp_dir is not None:
                visualize(
                    args,
                    vimages,
                    v_recon_shad,
                    v_gt_shad,
                    os.path.join(
                        savedir,
                        "test_epoch_{}_{}_shad".format(fname, rank),
                    ),
                    task="shadows",
                )
                visualize(
                    args,
                    vimages,
                    v_recon_tof,
                    v_gt_tof,
                    os.path.join(
                        savedir,
                        "test_epoch_{}_{}_tof".format(fname, rank),
                    ),
                    task="tof",
                )

            loss_shad = compute_val_loss(args, v_recon_shad, v_gt_shad, task="shadows")
            loss_tof = compute_val_loss(args, v_recon_tof, v_gt_tof, task="tof")
            log.write("{}, {}, {}\n".format(fname, loss_shad, loss_tof))

            results_shad = v_recon_shad.detach().cpu().numpy()
            results_tof = v_recon_tof.detach().cpu().numpy()

            np.save(
                os.path.join(
                    savedir,
                    "shadows_{}_{}.npy".format(fname, rank),
                ),
                results_shad,
            )

            np.save(
                os.path.join(
                    savedir,
                    "tof_{}_{}.npy".format(fname, rank),
                ),
                results_tof,
            )

    log.close()


def combined_test(model, loader, args, rank, savedir, temp_dir, test_ids):
    with torch.no_grad():
        for i, (vimages, vlights, vdepth, vshadow) in enumerate(loader):
            fname = test_ids[i][:-4]

            # B x 256 x 256 x 235
            vimages = vimages.cuda(non_blocking=True).float()
            vlights = vlights.cuda(non_blocking=True).float()
            vdepth = vdepth.cuda(non_blocking=True).float()
            vshadow = vshadow.cuda(non_blocking=True).float()

            # Forward pass
            with torch.cuda.amp.autocast(
                enabled=args.amp or args.bf16,
                dtype=torch.bfloat16 if args.bf16 else torch.float16,
            ):
                vx_depth, vx_shadow = model(vimages, vlights)

            results_depth = vx_depth.squeeze().detach().cpu().numpy()
            results_shadow = vx_shadow.squeeze().detach().cpu().numpy()

            # Save image
            if temp_dir is not None:
                visualize(
                    args,
                    vimages,
                    [vx_depth, vx_shadow],
                    [vdepth, vshadow],
                    os.path.join(
                        savedir,
                        "val_epoch_{}_{}".format(fname, rank),
                    ),
                )

            np.save(
                os.path.join(
                    savedir,
                    "depth_{}_{}.npy".format(fname, rank),
                ),
                results_depth,
            )

            np.save(
                os.path.join(
                    savedir,
                    "shadow_{}_{}.npy".format(fname, rank),
                ),
                results_shadow,
            )


def edge_aware_smoothness_loss(depth, image_gray):
    def gradient(x):
        # Compute gradients in x and y directions
        dx = x[:, :, 1:] - x[:, :, :-1]
        dy = x[:, 1:, :] - x[:, :-1, :]
        return dx, dy

    depth_dx, depth_dy = gradient(depth)
    image_dx, image_dy = gradient(image_gray)

    # Weight depth gradients by image gradients
    weights_x = torch.exp(-torch.abs(image_dx))
    weights_y = torch.exp(-torch.abs(image_dy))

    smoothness_x = torch.abs(depth_dx) * weights_x
    smoothness_y = torch.abs(depth_dy) * weights_y

    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


class Loss:
    def __init__(self, args, task=None):
        self.loss_type = args.loss
        if task is None:
            task = args.task
        self.args = args
        self.task = task
        if "depth" in task:
            if args.loss == "l1":
                self.criterion = nn.L1Loss()
            elif args.loss == "mse":
                self.criterion = nn.MSELoss()
            elif args.loss == "ssim":
                self.criterion = ssim_loss
            elif args.loss == "combine":
                self.criterion = ssim_loss
                self.criterion2 = nn.L1Loss()
            elif args.loss == "smooth":
                self.criterion = ssim_loss
                self.criterion2 = nn.L1Loss()
        elif task == "shadows" or task == "specular" or task == "combined":
            self.criterion = nn.BCEWithLogitsLoss()

    def compute_loss(self, pred, target, intensity):
        pred = pred.squeeze()
        target = target.squeeze()
        task = self.task

        if "depth" in task:
            if self.loss_type == "ssim":
                return self.criterion(
                    pred.unsqueeze(1), target.unsqueeze(1), window_size=11
                )
            elif self.loss_type == "combine":
                ssim_loss = self.criterion(
                    pred.unsqueeze(1), target.unsqueeze(1), window_size=11
                )
                l1_loss = self.criterion2(pred, target)
                return 0.85 * ssim_loss + 0.15 * l1_loss
            elif self.loss_type == "smooth":
                ssim_loss = self.criterion(
                    pred.unsqueeze(1), target.unsqueeze(1), window_size=11
                )
                l1_loss = self.criterion2(pred, target)
                smooth = edge_aware_smoothness_loss(pred, intensity)
                return (0.85 * ssim_loss + 0.15 * l1_loss) + (1e-3 * smooth)
            return self.criterion(pred, target)
        elif task == "shadows" or task == "specular" or task == "combined":
            return self.criterion(pred, target)
    
    def compute_val_loss(self, pred, target):
        pred = pred.squeeze()
        target = target.squeeze()
        task = self.task

        if "depth" in task:
            return F.l1_loss(pred, target)
        elif task == "shadows" or task == "specular" or task == "combined":
            pred = nn.Sigmoid()(pred)
            thresh = 0.5
            pred[pred >= thresh] = 1
            pred[pred < thresh] = 0
            pred = pred.squeeze()
            target = target.squeeze()
            l1 = F.l1_loss(pred, target)
            iou, f1 = calculate_segmentation_metrics(target, pred)
            return l1, iou, f1


def compute_val_loss(args, pred, target, task=None):
    pred = pred.squeeze()
    target = target.squeeze()
    if task is None:
        task = args.task

    if "depth" in task:
        return F.l1_loss(pred, target)
    elif task == "shadows" or task == "specular":
        pred = nn.Sigmoid()(pred)
        thresh = 0.5
        pred[pred >= thresh] = 1
        pred[pred < thresh] = 0
        pred = pred.squeeze()
        target = target.squeeze()
        l1 = F.l1_loss(pred, target)
        iou, f1 = calculate_segmentation_metrics(target, pred)
        return l1, iou, f1


def calculate_segmentation_metrics(gt_mask, pred_mask):
    # Ensure inputs are PyTorch tensors
    if not isinstance(gt_mask, torch.Tensor):
        gt_mask = torch.tensor(gt_mask)
    if not isinstance(pred_mask, torch.Tensor):
        pred_mask = torch.tensor(pred_mask)

    # Ensure masks are binary
    gt_mask = gt_mask.bool()
    pred_mask = pred_mask.bool()

    # Calculate IoU
    intersection = torch.logical_and(gt_mask, pred_mask).sum().float()
    union = torch.logical_or(gt_mask, pred_mask).sum().float()
    iou = intersection / union if union != 0 else torch.tensor(0.0).cuda()

    # Calculate F1 score
    true_positives = intersection
    false_positives = pred_mask.sum().float() - true_positives
    false_negatives = gt_mask.sum().float() - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) != 0
        else torch.tensor(0.0).cuda()
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) != 0
        else torch.tensor(0.0).cuda()
    )

    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else torch.tensor(0.0)
    )

    return iou, f1_score


def convolve_tof(tof, kernel, n_bins):
    tof = tof.transpose(1, 2).reshape(-1, n_bins)
    tof = kernel(tof[:, None, :]).squeeze()
    tof = tof.reshape(-1, 1, n_bins).transpose(1, 2)
    return tof


def torch_laser_kernel(laser, device="cuda"):
    m = torch.nn.Conv1d(
        1,
        1,
        laser.shape[0],
        padding=(laser.shape[0] - 1) // 2,
        padding_mode="zeros",
        device=device,
    )
    m.weight.requires_grad = False
    m.bias.requires_grad = False
    m.bias *= 0
    m.weight = torch.nn.Parameter(laser[None, None, ...]).cuda()
    return m


def preprocess(
    images,
    ground_truth,
    input_type,
    bins,
    convolve=None,
    model_type="unet2d",
    temp_down=0,
    spat_down=0,
    kernel=None,
):

    if convolve is not None and input_type != "hist_tofhist":
        # min_vals, _ = torch.min(images, dim=-1, keepdim=True)
        # max_vals, _ = torch.max(images, dim=-1, keepdim=True)
        # images = (images - min_vals) / (max_vals - min_vals)
        # images = torch.nan_to_num(images)

        random = 0
        if isinstance(convolve, list):
            random = np.random.randint(len(convolve) + 1) - 1
            convolve = convolve[random]
            if random > 0:
                x = torch.zeros(convolve.shape[0] + 1).cuda()
                x[0] = convolve[0]
                x[1:] = convolve
                convolve = x
        else:
            scale = (50 / 2.35482004503) / 8
            shift_amount = torch.normal(
                mean=torch.tensor([0.0]),
                std=torch.tensor([scale]),
            )
            shift_amount = int(torch.floor(shift_amount).item())
            convolve = torch.roll(convolve, shifts=shift_amount, dims=-1)
        if random != -1:
            shape = images.shape
            images = torch.reshape(images, (-1, 1, shape[-1]))
            if kernel is None:
                kernel = torch_laser_kernel(convolve, device="cuda")
            else:
                kernel.update_laser(convolve)
            images = convolve_tof(images, kernel, shape[-1])
            images = torch.reshape(
                images.squeeze(), (shape[0], shape[1], shape[2], shape[3])
            )

            # Noise
            noise_level = np.random.uniform(200, 3000)
            images = torch.clamp(images, min=1e-10)
            images = torch.poisson(images * noise_level)  # / noise_level

    elif input_type == "hist" or input_type == "hist_cond":
        # images = torch.log10(images + 1e-8)
        # images = (images - np.log10(EPS)) / (np.log10(5000) - np.log10(EPS))
        # images = torch.clip(images, 0, 1)

        # images = images.permute(0, 3, 1, 2)
        # images = torch.log10(images + 1e-8)

        # Noise Floor
        # min_vals, _ = torch.min(images, dim=-1, keepdim=True)
        # max_vals, _ = torch.max(images, dim=-1, keepdim=True)
        # images = (images - min_vals) / (max_vals - min_vals)
        # images = torch.nan_to_num(images)
        # div = np.random.randint(1000, 50000)  # 100000)  # 0.000005, 0.1)
        # noise_level = 1000
        # noise = torch.ones(*images.shape).cuda() * noise_level
        # noise = torch.poisson(torch.clamp(noise, min=1e-10))
        # images = images + (noise / float(div))

        if temp_down != 0:
            if images.shape[-1] % temp_down != 0:
                # images = images[:, :, :, :-1]
                images = images[:, :, :, :1500]
            images = images.reshape(
                images.shape[0],
                images.shape[1],
                images.shape[2],
                int(images.shape[3] / temp_down),
                temp_down,
            )
            images = images.sum(dim=-1).squeeze()
        if spat_down != 0:
            images = images[:, ::spat_down, ::spat_down, :]

        min_vals, _ = torch.min(images, dim=-1, keepdim=True)
        max_vals, _ = torch.max(images, dim=-1, keepdim=True)
        images = (images - min_vals) / (max_vals - min_vals)
        images = torch.nan_to_num(images)
        images = images.permute(0, 3, 1, 2)

    elif input_type == "hist_tofhist":
        lidar = images[:, :, :, :bins]
        lidar_shape = lidar.shape

        if convolve is not None:
            min_vals, _ = torch.min(lidar, dim=-1, keepdim=True)
            max_vals, _ = torch.max(lidar, dim=-1, keepdim=True)
            lidar = (lidar - min_vals) / (max_vals - min_vals)
            lidar = torch.nan_to_num(lidar)

            random = 0
            if convolve is not None:
                if isinstance(convolve, list):
                    random = np.random.randint(len(convolve) + 1) - 1
                    convolve = convolve[random]
                    if random > 0:
                        x = torch.zeros(convolve.shape[0] + 1).cuda()
                        x[0] = convolve[0]
                        x[1:] = convolve
                        convolve = x
                else:
                    scale = (50 / 2.35482004503) / 8
                    shift_amount = torch.normal(
                        mean=torch.tensor([0.0]),
                        std=torch.tensor([scale]),
                    )
                    shift_amount = int(torch.floor(shift_amount).item())
                    convolve = torch.roll(convolve, shifts=shift_amount, dims=-1)
                if random != -1:
                    shape = lidar.shape
                    lidar = torch.reshape(lidar, (-1, 1, shape[-1]))
                    if kernel is None:
                        kernel = torch_laser_kernel(convolve, device="cuda")
                    else:
                        kernel.update_laser(convolve)
                    lidar = convolve_tof(lidar, kernel, shape[-1])
                    lidar = torch.reshape(
                        lidar.squeeze(), (shape[0], shape[1], shape[2], shape[3])
                    )
                    # Generate Poisson noise
                    noise_level = np.random.uniform(10, 1000)
                    lidar = torch.clamp(lidar, min=1e-10)
                    lidar = torch.poisson(lidar * noise_level)

        if temp_down != 0:
            if lidar.shape[-1] % temp_down != 0:
                # lidar = lidar[:, :, :, :-1]
                lidar = lidar[:, :, :, :1500]
            lidar = lidar.reshape(
                lidar.shape[0],
                lidar.shape[1],
                lidar.shape[2],
                int(lidar.shape[3] / temp_down),
                temp_down,
            )
            lidar = lidar.sum(dim=-1).squeeze()
        if spat_down != 0:
            lidar = lidar[:, ::spat_down, ::spat_down, :]

        min_vals, _ = torch.min(lidar, dim=-1, keepdim=True)
        max_vals, _ = torch.max(lidar, dim=-1, keepdim=True)
        lidar = (lidar - min_vals) / (max_vals - min_vals)
        lidar = torch.nan_to_num(lidar)

        bin_image = images[:, :, :, bins:].squeeze()
        if len(bin_image.shape) == 2:
            bin_image = bin_image.unsqueeze(0)
        bin_image = bin_image.long()
        B, H, W = bin_image.shape
        batch_indices = torch.arange(B).view(-1, 1, 1).expand(B, H, W)
        row_indices = torch.arange(H).view(1, -1, 1).expand(B, H, W)
        col_indices = torch.arange(W).view(1, 1, -1).expand(B, H, W)
        hist = torch.zeros([*lidar_shape]).to(lidar.device)
        bin_image = torch.clip(bin_image, 0, bins - 1)
        hist[batch_indices, row_indices, col_indices, bin_image] = 1.0

        if temp_down != 0:
            if hist.shape[-1] % temp_down != 0:
                # hist = hist[:, :, :, :-1]
                hist = hist[:, :, :, :1500]
            hist = hist.reshape(
                hist.shape[0],
                hist.shape[1],
                hist.shape[2],
                int(hist.shape[3] / temp_down),
                temp_down,
            )
            hist = hist.sum(dim=-1).squeeze()
            hist = torch.clip(hist, 0, 1)
        if spat_down != 0:
            hist = hist[:, ::spat_down, ::spat_down, :]

        if model_type == "unet25":
            # B x 256 x 256 x bins --> B x 2 x 256 x 256 x bins
            images = torch.stack((lidar, hist), axis=1)
            images = images.permute(0, 1, 4, 2, 3)
        else:
            images = torch.concatenate((lidar, hist), axis=-1)
            images = images.permute(0, 3, 1, 2)
    else:
        shape = images.shape
        images = images.reshape(shape[0], shape[1] * shape[2], shape[3], shape[4])

    return images, ground_truth


def visualize(args, images, pred, target, path, task=None, plot_grid=False):
    if task is None:
        task = args.task
    if "depth" in task:
        viz_depth(pred, target, path)
    elif task == "shadows":
        viz_shadows(
            pred,
            target,
            path,
            args.num_lights,
            args.input,
            args.task,
            plot_grid=plot_grid,
        )
    elif task == "specular":
        viz_specular(images, pred, target, path)
    elif "combined" in task:
        viz_depth(pred[0].squeeze(), target[0].squeeze(), path + "_depth")
        viz_shadows(
            pred[1].squeeze(),
            target[1].squeeze(),
            path + "_shadow",
            args.num_lights,
            args.input,
            args.task,
        )


def viz_depth(pred, target, path):
    gt_depth = target[0].detach().cpu().numpy().squeeze() * 255.0
    pred_d = pred[0].detach().cpu().numpy().squeeze() * 255.0
    output = np.concatenate((gt_depth, pred_d), axis=0)
    plt.imshow(output)
    plt.savefig(path)
    plt.close()


def viz_shadows(pred, target, path, num_lights, input_type, task, plot_grid=False):
    thresh = 0.5
    shads = nn.Sigmoid()(pred)
    preds = shads.detach().cpu().numpy().squeeze()
    preds[preds >= thresh] = 1.0
    preds[preds < thresh] = 0.0
    preds = preds * 255.0
    gt = target.detach().cpu().numpy().squeeze() * 255.0
    if plot_grid:
        fig, axes = plt.subplots(
            int(np.sqrt(gt.shape[0])), int(np.sqrt(gt.shape[0])), figsize=(10, 10)
        )
        grid = int(np.sqrt(gt.shape[0]))
        count = 0
        for i in range(grid):
            for j in range(grid):
                output = np.concatenate((gt[count], preds[count]), axis=0)
                axes[i, j].imshow(output)
                axes[i, j].axis("off")
                count += 1
        plt.tight_layout()
        plt.savefig(
            path,
            dpi=1000,
        )
        plt.clf()

    else:
        output = None
        if (
            "hist_tofhist" in input_type
            or "combined" in task
        ):
            output = np.concatenate((gt[0], preds[0]), axis=0)
        else:
            output_top = []
            output_bottom = []
            for s in range(num_lights):
                output_top.append(gt[0, s])
                output_bottom.append(preds[0, s])
            output_top = np.concatenate(output_top, axis=1)
            output_bottom = np.concatenate(output_bottom, axis=1)
            output = np.concatenate((output_top, output_bottom), axis=0)
        plt.imshow(output)
        plt.savefig(
            path,
            dpi=1000,
        )
        plt.close()


def viz_specular(images, pred, target, path):
    spec = nn.Sigmoid()(pred)
    preds = spec.detach().cpu().numpy().squeeze()
    thresh = 0.5
    preds[preds >= thresh] = 1.0
    preds[preds < thresh] = 0.0
    preds = preds * 255.0
    gt = target.detach().cpu().numpy().squeeze() * 255.0
    input = None
    if images.shape[1] == 1:
        input = np.concatenate(
            (
                images[0].detach().cpu().numpy().squeeze(),
                images[0].detach().cpu().numpy().squeeze(),
            ),
            axis=0,
        )
    else:
        input = np.concatenate(
            (
                np.sum(images[0].detach().cpu().numpy().squeeze(), axis=0).squeeze(),
                np.sum(images[0].detach().cpu().numpy().squeeze(), axis=0).squeeze(),
            ),
            axis=0,
        )
    output_seg = np.concatenate((gt[0], preds[0]), axis=0)
    output = np.concatenate((input, output_seg), axis=1)
    plt.imshow(output)
    plt.savefig(
        path,
        dpi=1000,
    )
    plt.close()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
