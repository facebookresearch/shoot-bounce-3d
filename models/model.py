#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
# Code from the 3D UNet implementation:
# https://github.com/wolny/pytorch-3dunet/

import math
import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from models.buildingblocks import (
    create_decoders,
    create_encoders,
    DoubleConv,
    ResNetBlock,
    ResNetBlockSE,
)


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported dataset class: {class_name}")


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2**k for k in range(num_levels)]


EPS = 1e-8


class Joint(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=False,
        basic_module=DoubleConv,
        f_maps=256,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        is_segmentation=True,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        is3d=False,
    ):
        super(Joint, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if "g" in layer_order:
            assert (
                num_groups is not None
            ), "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(
            in_channels,
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            conv_upscale,
            dropout_prob,
            layer_order,
            num_groups,
            pool_kernel_size,
            is3d,
        )

        # create decoder path
        self.decoders_shadow = create_decoders(
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            upsample,
            dropout_prob,
            is3d,
        )

        self.decoders_tof = create_decoders(
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            upsample,
            dropout_prob,
            is3d,
        )

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        # if is3d:
        #     self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        # else:
        self.final_conv_shadow = nn.Conv2d(f_maps[0], out_channels, 1)
        self.final_conv_tof = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x, condition=None):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x, condition)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        i = 0
        x_shadow = None
        x_tof = None
        for decoder_shadow, decoder_tof, encoder_features in zip(
            self.decoders_shadow, self.decoders_tof, encoders_features
        ):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            if i == 0:
                x_shadow = decoder_shadow(encoder_features, x)
                x_tof = decoder_tof(encoder_features, x)
            else:
                x_shadow = decoder_shadow(encoder_features, x_shadow)
                x_tof = decoder_tof(encoder_features, x_tof)

            i = i + 1

        x_shadow = self.final_conv_shadow(x_shadow)
        x_tof = self.final_conv_tof(x_tof)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        # if not self.training and self.final_activation is not None:
        #     print("final activation")
        #     x = self.final_activation(x)

        return x_shadow, x_tof


def convolve_tof(color, kernel, n_bins):
    color = color.transpose(1, 2).reshape(-1, n_bins)
    color = kernel(color[:, None, :]).squeeze()
    color = color.reshape(-1, 1, n_bins).transpose(1, 2)
    return color


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


class CombinedNoise(nn.Module):
    def __init__(
        self,
        depth_in,
        depth_out,
        shadow_in,
        shadow_out,
        f_maps,
        num_levels,
        temp_res=0.0384,  # 0.03837343462,
        tmin=0.9,  # 2,
        max_depth=1.3716,  # 6, 1.27,  #
        bins=1502,  # 469,
        convolve=None,
        kernel=None,
        temp_down=0,
    ):
        super(CombinedNoise, self).__init__()
        # self.depth_model = UNet2D(
        #     in_channels=depth_in,
        #     out_channels=depth_out,
        #     final_sigmoid=False,
        #     is3d=False,
        #     layer_order="cbr",
        #     f_maps=f_maps,
        #     num_levels=num_levels,
        # )

        # self.shadow_model = UNet2D(
        #     in_channels=shadow_in,
        #     out_channels=shadow_out,
        #     final_sigmoid=False,
        #     is3d=False,
        #     layer_order="cbr",
        #     f_maps=f_maps,
        #     num_levels=num_levels,
        # )

        bins = 1502

        self.depth_model = UNet25(
            channels_in=1,
            channels_out=1,
            f_maps=f_maps,
            num_levels=num_levels,
            bins=bins,
        )

        self.shadow_model = UNet25(
            channels_in=2,
            channels_out=1,
            f_maps=f_maps,
            num_levels=num_levels,
            bins=bins,
        )

        self.temp_res = temp_res
        self.tmin = tmin  # 2
        self.bins = bins
        self.max_depth = max_depth  # 4.5  # 6
        self.convolve = convolve
        self.kernel = kernel
        self.temp_down = temp_down

    def forward(self, hist, lights):
        # Compute depth
        # images = torch.log10(hist + 1e-8)
        # images = (images - np.log10(EPS)) / (np.log10(5000) - np.log10(EPS))
        # images = torch.clip(images, 0, 1)
        # images = images.permute(0, 3, 1, 2)

        input_shape = hist.shape

        # print("images: {}, {}, {}".format(torch.min(hist), torch.max(hist), hist.shape))

        min_vals, _ = torch.min(hist, dim=-1, keepdim=True)
        max_vals, _ = torch.max(hist, dim=-1, keepdim=True)
        images = (hist - min_vals) / (max_vals - min_vals)
        images = torch.nan_to_num(images)

        # import matplotlib

        # matplotlib.use("Agg")
        # import matplotlib.pyplot as plt

        # plt.plot(images[0, 1, 12, 128, :].detach().cpu().numpy())
        # plt.savefig("debug0.png")
        # plt.close()

        # random = 0
        # convolve = None
        # if self.convolve is not None:
        #     if isinstance(self.convolve, list):
        #         random = np.random.randint(len(self.convolve) + 1) - 1
        #         convolve = self.convolve[random]
        #         if random > 0:
        #             x = torch.zeros(convolve.shape[0] + 1).cuda()
        #             x[0] = convolve[0]
        #             x[1:] = convolve
        #             convolve = x
        #         # print("Random! {} convolve: {}".format(random, convolve.shape))
        #     else:
        #         scale = (50 / 2.35482004503) / 8
        #         shift_amount = torch.normal(
        #             mean=torch.tensor([0.0]),
        #             std=torch.tensor([scale]),
        #         )
        #         shift_amount = int(torch.floor(shift_amount).item())
        #         # shift_amount = torch.randint(-4, 4 + 1, (1,)).item()
        #         convolve = torch.roll(self.convolve, shifts=shift_amount, dims=-1)
        #         # plt.plot(images[1, 12, 128, :].detach().cpu().numpy())
        #         # plt.savefig("shift.png")
        #         # plt.close()
        #         # print("Jittered")
        #     if random != -1:
        #         shape = images.shape
        #         images = torch.reshape(images, (-1, 1, shape[-1]))
        #         if self.kernel is None:
        #             self.kernel = torch_laser_kernel(convolve, device="cuda")
        #         else:
        #             self.kernel.update_laser(convolve)
        #         images = convolve_tof(images, self.kernel, shape[-1])
        #         images = torch.reshape(
        #             images.squeeze(), (shape[0], shape[1], shape[2], shape[3])
        #         )
        #         # Generate Poisson noise
        #         noise_level = np.random.uniform(10, 1000)
        #         # noise_level = 1000
        #         images = torch.clamp(images, min=1e-10)
        #         images = torch.poisson(images * noise_level)  # / noise_level

        #         # print("Added noise")

        if self.temp_down != 0:
            if images.shape[-1] % self.temp_down != 0:
                # lidar = lidar[:, :, :, :-1]
                images = images[:, :, :, :1500]
            images = images.reshape(
                images.shape[0],
                images.shape[1],
                images.shape[2],
                int(images.shape[3] / self.temp_down),
                self.temp_down,
            )
            images = images.sum(dim=-1).squeeze()

            # print("Temporal downsample")

        min_vals, _ = torch.min(images, dim=-1, keepdim=True)
        max_vals, _ = torch.max(images, dim=-1, keepdim=True)
        images = (images - min_vals) / (max_vals - min_vals)
        images = torch.nan_to_num(images)

        images = images.permute(0, 3, 1, 2)

        # print(
        #     "Input: {}, {}, {}".format(
        #         images.shape, torch.min(images), torch.max(images)
        #     )
        # )

        depth = self.depth_model(images)

        print("Depth: {}, {}".format(torch.min(depth), torch.max(depth)))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.imshow(depth[0].squeeze().detach().cpu().numpy())
        plt.savefig("depth_pred.png")
        plt.close()
        exit()

        # Convert depth into 2B ToF histograms
        tof = self.depth_to_tof(
            torch.clip(depth.squeeze(), 0, 1) * self.max_depth, lights
        )
        # tof = self.depth_to_tof(gt_depth, lights)
        bin_image = torch.floor(tof / 0.00239833966) - np.ceil(
            self.tmin / 0.00239833966
        )

        # min_vals, _ = torch.min(hist, dim=-1, keepdim=True)
        # max_vals, _ = torch.max(hist, dim=-1, keepdim=True)
        # lidar = (hist - min_vals) / (max_vals - min_vals)
        # lidar = torch.nan_to_num(lidar)

        if len(bin_image.shape) == 2:
            bin_image = bin_image.unsqueeze(0)
        bin_image = bin_image.long()
        B, H, W = bin_image.shape
        batch_indices = torch.arange(B).view(-1, 1, 1).expand(B, H, W)
        row_indices = torch.arange(H).view(1, -1, 1).expand(B, H, W)
        col_indices = torch.arange(W).view(1, 1, -1).expand(B, H, W)
        hist = torch.zeros([*input_shape]).to(images.device)
        bin_image = torch.clip(bin_image, 0, self.bins - 1)

        # Differentiable version:
        # hist2 = torch.zeros([*lidar.shape]).to(lidar.device)
        # bin_image = torch.clip(bin_image, 0, bins - 1)
        # hist2.scatter_(
        #     3,
        #     bin_image.unsqueeze(3),
        #     torch.ones_like(bin_image, dtype=hist.dtype).unsqueeze(3).to(lidar.device),
        # )

        # print(
        #     "Bin image: {}, {}, Hist: {}, Col/Row: {}-{}/{}-{}".format(
        #         torch.min(bin_image),
        #         torch.max(bin_image),
        #         hist.shape,
        #         torch.min(row_indices),
        #         torch.max(row_indices),
        #         torch.min(col_indices),
        #         torch.max(row_indices),
        #     )
        # )
        hist[batch_indices, row_indices, col_indices, bin_image] = 1.0

        if self.temp_down != 0:
            if hist.shape[-1] % self.temp_down != 0:
                # hist = hist[:, :, :, :-1]
                hist = hist[:, :, :, :1500]
            hist = hist.reshape(
                hist.shape[0],
                hist.shape[1],
                hist.shape[2],
                int(hist.shape[3] / self.temp_down),
                self.temp_down,
            )
            hist = hist.sum(dim=-1).squeeze()
            hist = torch.clip(hist, 0, 1)

        # Compute shadows
        hist = hist.permute(0, 3, 1, 2)
        images = torch.stack((images, hist), dim=1)

        # plt.plot(images[0, 0, :, 12, 128].detach().cpu().numpy())
        # plt.savefig("raw_hist.png")
        # plt.close()
        # plt.plot(images[0, 1, :, 12, 128].detach().cpu().numpy())
        # plt.savefig("tof_hist.png")
        # plt.close()
        # exit()

        shadow = self.shadow_model(images)

        return depth, shadow

    def depth_to_tof(
        self,
        depth,
        virtual_lights,
        num_lights=1,
        temp_res=0.0384,
        debug=False,
    ):
        if len(depth.shape) == 2:
            depth = depth.unsqueeze(0)

        batch_size = depth.shape[0]
        f = 256.0 / (2 * math.tan(np.radians(90.0) / 2))
        K = torch.tensor([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]]).to(
            depth.device
        )

        u, v = torch.meshgrid(
            torch.arange(depth.shape[2], device=depth.device),
            torch.arange(depth.shape[1], device=depth.device),
            indexing="xy",
        )
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]

        X = x.unsqueeze(0) * depth
        Y = y.unsqueeze(0) * depth
        Z = depth
        pc_0 = torch.stack((X, Y, Z), dim=3)

        # Compute 2 bounce ToF
        camera_position = torch.zeros(batch_size, num_lights, 1, 3, device=depth.device)
        lights = camera_position.clone()
        virtual_lights = virtual_lights.unsqueeze(2)  # N x 16 x 1 x 3

        pc = pc_0.view(batch_size, 1, -1, 3).expand(-1, num_lights, -1, -1)

        d1 = torch.norm(virtual_lights - lights, dim=-1)  # N x 16 x 1
        d2 = torch.norm(pc - virtual_lights, dim=-1)  # N x 16 x (256*256)
        d3 = torch.norm(camera_position - pc, dim=-1)  # N x 16 x (256*256)

        total = d1 + d2 + d3
        total_torch_batch = total.view(batch_size, num_lights, 256, 256)

        # # NumPy Version
        # depth = depth.detach().cpu().numpy()[0].squeeze()
        # virtual_lights = virtual_lights.detach().cpu().numpy()[0]
        # print("Input", depth.shape, virtual_lights.shape)
        # f = 256.0 / (2 * math.tan(np.radians(90.0) / 2))
        # K = np.array([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]])

        # u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        # x = (u - K[0, 2]) / K[0, 0]
        # y = (v - K[1, 2]) / K[1, 1]
        # X = x * depth
        # Y = y * depth
        # Z = depth
        # pc_0 = np.stack((X, Y, Z), axis=2)

        # # Compute 2 bounce ToF
        # camera_position = np.expand_dims(
        #     np.stack([np.array([0, 0, 0]) for _ in range(num_lights)]), 1
        # )  # 4 x 1 x 3
        # lights = deepcopy(camera_position)  # 4 x 1 x 3
        # virtual_lights = np.expand_dims(virtual_lights, 1)  # 4 x 1 x 3
        # pc = np.stack([pc_0.reshape(-1, 3) for _ in range(num_lights)])  # 4 x N x 3

        # d1 = np.linalg.norm(
        #     virtual_lights - lights, axis=-1
        # )  # 4 x N -- light to virtual light
        # d2 = np.linalg.norm(
        #     pc - virtual_lights, axis=-1
        # )  # 4 x N -- virtual light to point
        # d3 = np.linalg.norm(camera_position - pc, axis=-1)  # 4 x N -- point to camera
        # total = d1 + d2 + d3
        # total_numpy = total.reshape(num_lights, 256, 256)

        # # PyTorch Version
        # depth = torch.tensor(depth)
        # virtual_lights = torch.tensor(virtual_lights)
        # K = torch.tensor([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]])
        # u, v = torch.meshgrid(
        #     torch.arange(depth.shape[1]), torch.arange(depth.shape[0]), indexing="xy"
        # )
        # x = (u - K[0, 2]) / K[0, 0]
        # y = (v - K[1, 2]) / K[1, 1]
        # X = x * depth
        # Y = y * depth
        # Z = depth
        # pc_0 = torch.stack((X, Y, Z), dim=2)

        # # Compute 2 bounce ToF
        # camera_position = torch.stack(
        #     [torch.zeros(3) for _ in range(num_lights)]
        # ).unsqueeze(1)  # 4 x 1 x 3
        # lights = camera_position.clone()  # 4 x 1 x 3
        # virtual_lights = virtual_lights.unsqueeze(1)  # 4 x 1 x 3
        # pc = torch.stack([pc_0.reshape(-1, 3) for _ in range(num_lights)])  # 4 x N x 3

        # d1 = torch.norm(
        #     virtual_lights - lights, dim=-1
        # )  # 4 x N -- light to virtual light
        # d2 = torch.norm(pc - virtual_lights, dim=-1)  # 4 x N -- virtual light to point
        # d3 = torch.norm(camera_position - pc, dim=-1)  # 4 x N -- point to camera
        # total = d1 + d2 + d3
        # total_torch = total.reshape(num_lights, 256, 256)

        # print(total_numpy.shape, total_torch.shape, total_torch_batch.shape)
        # print(np.array_equal(total_numpy, total_torch.detach().cpu().numpy()))
        # print(np.allclose(total_numpy, total_torch.detach().cpu().numpy()))
        # print(np.array_equal(total_numpy, total_torch_batch.detach().cpu().numpy()))

        # import matplotlib
        # import matplotlib.pyplot as plt

        # plt.imshow(total_numpy.squeeze())
        # plt.colorbar()
        # plt.savefig(
        #     "./2b_tof_numpy.png",
        #     dpi=1000,
        # )
        # plt.close()
        # plt.imshow(total_torch.detach().cpu().numpy().squeeze())
        # plt.colorbar()
        # plt.savefig(
        #     "./2b_tof_torch.png",
        #     dpi=1000,
        # )
        # plt.close()

        # plt.imshow(total_torch_batch[0].detach().cpu().numpy().squeeze())
        # plt.colorbar()
        # plt.savefig(
        #     "./2b_tof_torch_batch.png",
        #     dpi=1000,
        # )
        # plt.close()

        # exit()

        return total_torch_batch.squeeze()


class CombinedNoise2D(nn.Module):
    def __init__(
        self,
        depth_in,
        depth_out,
        shadow_in,
        shadow_out,
        f_maps,
        num_levels,
        temp_res=0.0384,  # 0.03837343462,
        tmin=0.9,  # 2,
        max_depth=1.27,  # 1.3716,  # 6,
        bins=1502,  # 469,
        convolve=None,
        kernel=None,
        temp_down=0,
    ):
        super(CombinedNoise2D, self).__init__()
        self.depth_model = UNet2D(
            in_channels=depth_in,
            out_channels=depth_out,
            final_sigmoid=False,
            is3d=False,
            layer_order="cbr",
            f_maps=f_maps,
            num_levels=num_levels,
        )

        self.shadow_model = UNet2D(
            in_channels=shadow_in,
            out_channels=shadow_out,
            final_sigmoid=False,
            is3d=False,
            layer_order="cbr",
            f_maps=f_maps,
            num_levels=num_levels,
        )

        bins = 1502

        # self.depth_model = UNet25(
        #     channels_in=1,
        #     channels_out=1,
        #     f_maps=f_maps,
        #     num_levels=num_levels,
        #     bins=bins,
        # )

        # self.shadow_model = UNet25(
        #     channels_in=2,
        #     channels_out=1,
        #     f_maps=f_maps,
        #     num_levels=num_levels,
        #     bins=bins,
        # )

        self.temp_res = temp_res
        self.tmin = tmin  # 2
        self.bins = bins
        self.max_depth = max_depth  # 4.5  # 6
        self.convolve = convolve
        self.kernel = kernel
        self.temp_down = temp_down

    def forward(self, hist, lights):
        # Compute depth
        # images = torch.log10(hist + 1e-8)
        # images = (images - np.log10(EPS)) / (np.log10(5000) - np.log10(EPS))
        # images = torch.clip(images, 0, 1)
        # images = images.permute(0, 3, 1, 2)

        input_shape = hist.shape

        # print("images: {}, {}, {}".format(torch.min(hist), torch.max(hist), hist.shape))

        min_vals, _ = torch.min(hist, dim=-1, keepdim=True)
        max_vals, _ = torch.max(hist, dim=-1, keepdim=True)
        images = (hist - min_vals) / (max_vals - min_vals)
        images = torch.nan_to_num(images)

        # import matplotlib

        # matplotlib.use("Agg")
        # import matplotlib.pyplot as plt

        # plt.plot(images[0, 1, 12, 128, :].detach().cpu().numpy())
        # plt.savefig("debug0.png")
        # plt.close()

        # random = 0
        # convolve = None
        # if self.convolve is not None:
        #     if isinstance(self.convolve, list):
        #         random = np.random.randint(len(self.convolve) + 1) - 1
        #         convolve = self.convolve[random]
        #         if random > 0:
        #             x = torch.zeros(convolve.shape[0] + 1).cuda()
        #             x[0] = convolve[0]
        #             x[1:] = convolve
        #             convolve = x
        #         # print("Random! {} convolve: {}".format(random, convolve.shape))
        #     else:
        #         scale = (50 / 2.35482004503) / 8
        #         shift_amount = torch.normal(
        #             mean=torch.tensor([0.0]),
        #             std=torch.tensor([scale]),
        #         )
        #         shift_amount = int(torch.floor(shift_amount).item())
        #         # shift_amount = torch.randint(-4, 4 + 1, (1,)).item()
        #         convolve = torch.roll(self.convolve, shifts=shift_amount, dims=-1)
        #         # plt.plot(images[1, 12, 128, :].detach().cpu().numpy())
        #         # plt.savefig("shift.png")
        #         # plt.close()
        #         # print("Jittered")
        #     if random != -1:
        #         shape = images.shape
        #         images = torch.reshape(images, (-1, 1, shape[-1]))
        #         if self.kernel is None:
        #             self.kernel = torch_laser_kernel(convolve, device="cuda")
        #         else:
        #             self.kernel.update_laser(convolve)
        #         images = convolve_tof(images, self.kernel, shape[-1])
        #         images = torch.reshape(
        #             images.squeeze(), (shape[0], shape[1], shape[2], shape[3])
        #         )
        #         # Generate Poisson noise
        #         noise_level = np.random.uniform(10, 1000)
        #         # noise_level = 1000
        #         images = torch.clamp(images, min=1e-10)
        #         images = torch.poisson(images * noise_level)  # / noise_level

        #         # print("Added noise")

        if self.temp_down != 0:
            if images.shape[-1] % self.temp_down != 0:
                # lidar = lidar[:, :, :, :-1]
                images = images[:, :, :, :1500]
            images = images.reshape(
                images.shape[0],
                images.shape[1],
                images.shape[2],
                int(images.shape[3] / self.temp_down),
                self.temp_down,
            )
            images = images.sum(dim=-1).squeeze()

            # print("Temporal downsample")

        min_vals, _ = torch.min(images, dim=-1, keepdim=True)
        max_vals, _ = torch.max(images, dim=-1, keepdim=True)
        images = (images - min_vals) / (max_vals - min_vals)
        images = torch.nan_to_num(images)

        images = images.permute(0, 3, 1, 2)

        # print(
        #     "Input: {}, {}, {}".format(
        #         images.shape, torch.min(images), torch.max(images)
        #     )
        # )

        depth = self.depth_model(images)

        # print("Depth: {}, {}".format(torch.min(depth), torch.max(depth)))
        # import matplotlib

        # matplotlib.use("Agg")
        # import matplotlib.pyplot as plt

        # plt.imshow(depth[0].squeeze().detach().cpu().numpy())
        # plt.savefig("depth_pred.png")
        # plt.close()
        # exit()

        # Convert depth into 2B ToF histograms
        tof = self.depth_to_tof(
            torch.clip(depth.squeeze(), 0, 1) * self.max_depth, lights
        )
        # tof = self.depth_to_tof(gt_depth, lights)
        bin_image = torch.floor(tof / 0.00239833966) - np.ceil(
            self.tmin / 0.00239833966
        )

        # min_vals, _ = torch.min(hist, dim=-1, keepdim=True)
        # max_vals, _ = torch.max(hist, dim=-1, keepdim=True)
        # lidar = (hist - min_vals) / (max_vals - min_vals)
        # lidar = torch.nan_to_num(lidar)

        if len(bin_image.shape) == 2:
            bin_image = bin_image.unsqueeze(0)
        bin_image = bin_image.long()
        B, H, W = bin_image.shape
        batch_indices = torch.arange(B).view(-1, 1, 1).expand(B, H, W)
        row_indices = torch.arange(H).view(1, -1, 1).expand(B, H, W)
        col_indices = torch.arange(W).view(1, 1, -1).expand(B, H, W)
        hist = torch.zeros([*input_shape]).to(images.device)
        bin_image = torch.clip(bin_image, 0, self.bins - 1)

        # Differentiable version:
        # hist2 = torch.zeros([*lidar.shape]).to(lidar.device)
        # bin_image = torch.clip(bin_image, 0, bins - 1)
        # hist2.scatter_(
        #     3,
        #     bin_image.unsqueeze(3),
        #     torch.ones_like(bin_image, dtype=hist.dtype).unsqueeze(3).to(lidar.device),
        # )

        # print(
        #     "Bin image: {}, {}, Hist: {}, Col/Row: {}-{}/{}-{}".format(
        #         torch.min(bin_image),
        #         torch.max(bin_image),
        #         hist.shape,
        #         torch.min(row_indices),
        #         torch.max(row_indices),
        #         torch.min(col_indices),
        #         torch.max(row_indices),
        #     )
        # )
        hist[batch_indices, row_indices, col_indices, bin_image] = 1.0

        if self.temp_down != 0:
            if hist.shape[-1] % self.temp_down != 0:
                # hist = hist[:, :, :, :-1]
                hist = hist[:, :, :, :1500]
            hist = hist.reshape(
                hist.shape[0],
                hist.shape[1],
                hist.shape[2],
                int(hist.shape[3] / self.temp_down),
                self.temp_down,
            )
            hist = hist.sum(dim=-1).squeeze()
            hist = torch.clip(hist, 0, 1)

        # Compute shadows
        hist = hist.permute(0, 3, 1, 2)
        images = torch.concatenate((images, hist), axis=1)

        # plt.plot(images[0, 0, :, 12, 128].detach().cpu().numpy())
        # plt.savefig("raw_hist.png")
        # plt.close()
        # plt.plot(images[0, 1, :, 12, 128].detach().cpu().numpy())
        # plt.savefig("tof_hist.png")
        # plt.close()
        # exit()

        shadow = self.shadow_model(images)

        return depth, shadow

    def depth_to_tof(
        self,
        depth,
        virtual_lights,
        num_lights=1,
        temp_res=0.0384,
        debug=False,
    ):
        if len(depth.shape) == 2:
            depth = depth.unsqueeze(0)

        batch_size = depth.shape[0]
        f = 256.0 / (2 * math.tan(np.radians(90.0) / 2))
        K = torch.tensor([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]]).to(
            depth.device
        )

        u, v = torch.meshgrid(
            torch.arange(depth.shape[2], device=depth.device),
            torch.arange(depth.shape[1], device=depth.device),
            indexing="xy",
        )
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]

        X = x.unsqueeze(0) * depth
        Y = y.unsqueeze(0) * depth
        Z = depth
        pc_0 = torch.stack((X, Y, Z), dim=3)

        # Compute 2 bounce ToF
        camera_position = torch.zeros(batch_size, num_lights, 1, 3, device=depth.device)
        lights = camera_position.clone()
        virtual_lights = virtual_lights.unsqueeze(2)  # N x 16 x 1 x 3

        pc = pc_0.view(batch_size, 1, -1, 3).expand(-1, num_lights, -1, -1)

        d1 = torch.norm(virtual_lights - lights, dim=-1)  # N x 16 x 1
        d2 = torch.norm(pc - virtual_lights, dim=-1)  # N x 16 x (256*256)
        d3 = torch.norm(camera_position - pc, dim=-1)  # N x 16 x (256*256)

        total = d1 + d2 + d3
        total_torch_batch = total.view(batch_size, num_lights, 256, 256)

        # # NumPy Version
        # depth = depth.detach().cpu().numpy()[0].squeeze()
        # virtual_lights = virtual_lights.detach().cpu().numpy()[0]
        # print("Input", depth.shape, virtual_lights.shape)
        # f = 256.0 / (2 * math.tan(np.radians(90.0) / 2))
        # K = np.array([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]])

        # u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        # x = (u - K[0, 2]) / K[0, 0]
        # y = (v - K[1, 2]) / K[1, 1]
        # X = x * depth
        # Y = y * depth
        # Z = depth
        # pc_0 = np.stack((X, Y, Z), axis=2)

        # # Compute 2 bounce ToF
        # camera_position = np.expand_dims(
        #     np.stack([np.array([0, 0, 0]) for _ in range(num_lights)]), 1
        # )  # 4 x 1 x 3
        # lights = deepcopy(camera_position)  # 4 x 1 x 3
        # virtual_lights = np.expand_dims(virtual_lights, 1)  # 4 x 1 x 3
        # pc = np.stack([pc_0.reshape(-1, 3) for _ in range(num_lights)])  # 4 x N x 3

        # d1 = np.linalg.norm(
        #     virtual_lights - lights, axis=-1
        # )  # 4 x N -- light to virtual light
        # d2 = np.linalg.norm(
        #     pc - virtual_lights, axis=-1
        # )  # 4 x N -- virtual light to point
        # d3 = np.linalg.norm(camera_position - pc, axis=-1)  # 4 x N -- point to camera
        # total = d1 + d2 + d3
        # total_numpy = total.reshape(num_lights, 256, 256)

        # # PyTorch Version
        # depth = torch.tensor(depth)
        # virtual_lights = torch.tensor(virtual_lights)
        # K = torch.tensor([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]])
        # u, v = torch.meshgrid(
        #     torch.arange(depth.shape[1]), torch.arange(depth.shape[0]), indexing="xy"
        # )
        # x = (u - K[0, 2]) / K[0, 0]
        # y = (v - K[1, 2]) / K[1, 1]
        # X = x * depth
        # Y = y * depth
        # Z = depth
        # pc_0 = torch.stack((X, Y, Z), dim=2)

        # # Compute 2 bounce ToF
        # camera_position = torch.stack(
        #     [torch.zeros(3) for _ in range(num_lights)]
        # ).unsqueeze(1)  # 4 x 1 x 3
        # lights = camera_position.clone()  # 4 x 1 x 3
        # virtual_lights = virtual_lights.unsqueeze(1)  # 4 x 1 x 3
        # pc = torch.stack([pc_0.reshape(-1, 3) for _ in range(num_lights)])  # 4 x N x 3

        # d1 = torch.norm(
        #     virtual_lights - lights, dim=-1
        # )  # 4 x N -- light to virtual light
        # d2 = torch.norm(pc - virtual_lights, dim=-1)  # 4 x N -- virtual light to point
        # d3 = torch.norm(camera_position - pc, dim=-1)  # 4 x N -- point to camera
        # total = d1 + d2 + d3
        # total_torch = total.reshape(num_lights, 256, 256)

        # print(total_numpy.shape, total_torch.shape, total_torch_batch.shape)
        # print(np.array_equal(total_numpy, total_torch.detach().cpu().numpy()))
        # print(np.allclose(total_numpy, total_torch.detach().cpu().numpy()))
        # print(np.array_equal(total_numpy, total_torch_batch.detach().cpu().numpy()))

        # import matplotlib
        # import matplotlib.pyplot as plt

        # plt.imshow(total_numpy.squeeze())
        # plt.colorbar()
        # plt.savefig(
        #     "./2b_tof_numpy.png",
        #     dpi=1000,
        # )
        # plt.close()
        # plt.imshow(total_torch.detach().cpu().numpy().squeeze())
        # plt.colorbar()
        # plt.savefig(
        #     "./2b_tof_torch.png",
        #     dpi=1000,
        # )
        # plt.close()

        # plt.imshow(total_torch_batch[0].detach().cpu().numpy().squeeze())
        # plt.colorbar()
        # plt.savefig(
        #     "./2b_tof_torch_batch.png",
        #     dpi=1000,
        # )
        # plt.close()

        # exit()

        return total_torch_batch.squeeze()


class CombinedModel(nn.Module):
    def __init__(
        self,
        depth_in,
        depth_out,
        shadow_in,
        shadow_out,
        f_maps,
        num_levels,
        temp_res=0.0384,
        tmin=1,
        max_depth=4.5,
        bins=637,
    ):
        super(CombinedModel, self).__init__()
        self.depth_model = UNet2D(
            in_channels=depth_in,
            out_channels=depth_out,
            final_sigmoid=False,
            is3d=False,
            layer_order="cbr",
            f_maps=f_maps,
            num_levels=num_levels,
        )

        self.shadow_model = UNet2D(
            in_channels=shadow_in,
            out_channels=shadow_out,
            final_sigmoid=False,
            is3d=False,
            layer_order="cbr",
            f_maps=f_maps,
            num_levels=num_levels,
        )

        self.temp_res = temp_res
        self.tmin = tmin  # 2
        self.bins = bins
        self.max_depth = max_depth  # 4.5  # 6

    def forward(self, hist, lights):
        # Compute depth
        # images = torch.log10(hist + 1e-8)
        # images = (images - np.log10(EPS)) / (np.log10(5000) - np.log10(EPS))
        # images = torch.clip(images, 0, 1)
        # images = images.permute(0, 3, 1, 2)

        min_vals, _ = torch.min(hist, dim=-1, keepdim=True)
        max_vals, _ = torch.max(hist, dim=-1, keepdim=True)
        images = (hist - min_vals) / (max_vals - min_vals)
        images = torch.nan_to_num(images)
        images = images.permute(0, 3, 1, 2)

        depth = self.depth_model(images)

        # Convert depth into 2B ToF histograms
        tof = self.depth_to_tof(
            torch.clip(depth.squeeze(), 0, 1) * self.max_depth, lights
        )
        bin_image = torch.floor(tof / self.temp_res) - np.ceil(
            self.tmin / self.temp_res
        )

        min_vals, _ = torch.min(hist, dim=-1, keepdim=True)
        max_vals, _ = torch.max(hist, dim=-1, keepdim=True)
        lidar = (hist - min_vals) / (max_vals - min_vals)
        lidar = torch.nan_to_num(lidar)

        if len(bin_image.shape) == 2:
            bin_image = bin_image.unsqueeze(0)
        bin_image = bin_image.long()
        B, H, W = bin_image.shape
        batch_indices = torch.arange(B).view(-1, 1, 1).expand(B, H, W)
        row_indices = torch.arange(H).view(1, -1, 1).expand(B, H, W)
        col_indices = torch.arange(W).view(1, 1, -1).expand(B, H, W)
        hist = torch.zeros([*lidar.shape]).to(lidar.device)
        bin_image = torch.clip(bin_image, 0, self.bins - 1)

        # Differentiable version:
        # hist2 = torch.zeros([*lidar.shape]).to(lidar.device)
        # bin_image = torch.clip(bin_image, 0, bins - 1)
        # hist2.scatter_(
        #     3,
        #     bin_image.unsqueeze(3),
        #     torch.ones_like(bin_image, dtype=hist.dtype).unsqueeze(3).to(lidar.device),
        # )

        # print(
        #     "Bin image: {}, {}, Hist: {}, Col/Row: {}-{}/{}-{}".format(
        #         torch.min(bin_image),
        #         torch.max(bin_image),
        #         hist.shape,
        #         torch.min(row_indices),
        #         torch.max(row_indices),
        #         torch.min(col_indices),
        #         torch.max(row_indices),
        #     )
        # )
        hist[batch_indices, row_indices, col_indices, bin_image] = 1.0

        # Compute shadows
        images = torch.concatenate((lidar, hist), axis=-1)
        images = images.permute(0, 3, 1, 2)
        shadow = self.shadow_model(images)

        return depth, shadow

    def depth_to_tof(
        self,
        depth,
        virtual_lights,
        num_lights=1,
        temp_res=0.0384,
        debug=False,
    ):
        if len(depth.shape) == 2:
            depth = depth.unsqueeze(0)

        batch_size = depth.shape[0]
        f = 256.0 / (2 * math.tan(np.radians(90.0) / 2))
        K = torch.tensor([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]]).to(
            depth.device
        )

        u, v = torch.meshgrid(
            torch.arange(depth.shape[2], device=depth.device),
            torch.arange(depth.shape[1], device=depth.device),
            indexing="xy",
        )
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]

        X = x.unsqueeze(0) * depth
        Y = y.unsqueeze(0) * depth
        Z = depth
        pc_0 = torch.stack((X, Y, Z), dim=3)

        # Compute 2 bounce ToF
        camera_position = torch.zeros(batch_size, num_lights, 1, 3, device=depth.device)
        lights = camera_position.clone()
        virtual_lights = virtual_lights.unsqueeze(2)  # N x 16 x 1 x 3

        pc = pc_0.view(batch_size, 1, -1, 3).expand(-1, num_lights, -1, -1)

        d1 = torch.norm(virtual_lights - lights, dim=-1)  # N x 16 x 1
        d2 = torch.norm(pc - virtual_lights, dim=-1)  # N x 16 x (256*256)
        d3 = torch.norm(camera_position - pc, dim=-1)  # N x 16 x (256*256)

        total = d1 + d2 + d3
        total_torch_batch = total.view(batch_size, num_lights, 256, 256)

        # # NumPy Version
        # depth = depth.detach().cpu().numpy()[0].squeeze()
        # virtual_lights = virtual_lights.detach().cpu().numpy()[0]
        # print("Input", depth.shape, virtual_lights.shape)
        # f = 256.0 / (2 * math.tan(np.radians(90.0) / 2))
        # K = np.array([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]])

        # u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        # x = (u - K[0, 2]) / K[0, 0]
        # y = (v - K[1, 2]) / K[1, 1]
        # X = x * depth
        # Y = y * depth
        # Z = depth
        # pc_0 = np.stack((X, Y, Z), axis=2)

        # # Compute 2 bounce ToF
        # camera_position = np.expand_dims(
        #     np.stack([np.array([0, 0, 0]) for _ in range(num_lights)]), 1
        # )  # 4 x 1 x 3
        # lights = deepcopy(camera_position)  # 4 x 1 x 3
        # virtual_lights = np.expand_dims(virtual_lights, 1)  # 4 x 1 x 3
        # pc = np.stack([pc_0.reshape(-1, 3) for _ in range(num_lights)])  # 4 x N x 3

        # d1 = np.linalg.norm(
        #     virtual_lights - lights, axis=-1
        # )  # 4 x N -- light to virtual light
        # d2 = np.linalg.norm(
        #     pc - virtual_lights, axis=-1
        # )  # 4 x N -- virtual light to point
        # d3 = np.linalg.norm(camera_position - pc, axis=-1)  # 4 x N -- point to camera
        # total = d1 + d2 + d3
        # total_numpy = total.reshape(num_lights, 256, 256)

        # # PyTorch Version
        # depth = torch.tensor(depth)
        # virtual_lights = torch.tensor(virtual_lights)
        # K = torch.tensor([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]])
        # u, v = torch.meshgrid(
        #     torch.arange(depth.shape[1]), torch.arange(depth.shape[0]), indexing="xy"
        # )
        # x = (u - K[0, 2]) / K[0, 0]
        # y = (v - K[1, 2]) / K[1, 1]
        # X = x * depth
        # Y = y * depth
        # Z = depth
        # pc_0 = torch.stack((X, Y, Z), dim=2)

        # # Compute 2 bounce ToF
        # camera_position = torch.stack(
        #     [torch.zeros(3) for _ in range(num_lights)]
        # ).unsqueeze(1)  # 4 x 1 x 3
        # lights = camera_position.clone()  # 4 x 1 x 3
        # virtual_lights = virtual_lights.unsqueeze(1)  # 4 x 1 x 3
        # pc = torch.stack([pc_0.reshape(-1, 3) for _ in range(num_lights)])  # 4 x N x 3

        # d1 = torch.norm(
        #     virtual_lights - lights, dim=-1
        # )  # 4 x N -- light to virtual light
        # d2 = torch.norm(pc - virtual_lights, dim=-1)  # 4 x N -- virtual light to point
        # d3 = torch.norm(camera_position - pc, dim=-1)  # 4 x N -- point to camera
        # total = d1 + d2 + d3
        # total_torch = total.reshape(num_lights, 256, 256)

        # print(total_numpy.shape, total_torch.shape, total_torch_batch.shape)
        # print(np.array_equal(total_numpy, total_torch.detach().cpu().numpy()))
        # print(np.allclose(total_numpy, total_torch.detach().cpu().numpy()))
        # print(np.array_equal(total_numpy, total_torch_batch.detach().cpu().numpy()))

        # import matplotlib
        # import matplotlib.pyplot as plt

        # plt.imshow(total_numpy.squeeze())
        # plt.colorbar()
        # plt.savefig(
        #     "./2b_tof_numpy.png",
        #     dpi=1000,
        # )
        # plt.close()
        # plt.imshow(total_torch.detach().cpu().numpy().squeeze())
        # plt.colorbar()
        # plt.savefig(
        #     "./2b_tof_torch.png",
        #     dpi=1000,
        # )
        # plt.close()

        # plt.imshow(total_torch_batch[0].detach().cpu().numpy().squeeze())
        # plt.colorbar()
        # plt.savefig(
        #     "./2b_tof_torch_batch.png",
        #     dpi=1000,
        # )
        # plt.close()

        # exit()

        return total_torch_batch.squeeze()


class CombinedE2E(nn.Module):
    def __init__(
        self,
        depth_in,
        depth_out,
        shadow_in,
        shadow_out,
        f_maps=256,
        num_levels=5,
        basic_module=DoubleConv,
        layer_order="cbr",
        num_groups=8,
        is_segmentation=True,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        is3d=False,
        temp_res=0.03837343462,
        tmin=2,
        bins=469,
    ):
        super(CombinedE2E, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # create encoder path
        self.encoders = create_encoders(
            depth_in,
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            conv_upscale,
            dropout_prob,
            layer_order,
            num_groups,
            pool_kernel_size,
            is3d,
        )

        # create decoder path
        self.decoders_depth = create_decoders(
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            upsample,
            dropout_prob,
            is3d,
        )

        self.decoders_shadow = create_decoders(
            f_maps,  #  * 2,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            upsample,
            dropout_prob,
            is3d,
        )

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        self.final_conv_depth = nn.Conv2d(f_maps[0], depth_out, 1)
        self.final_conv_shadow = nn.Conv2d(f_maps[0], shadow_out, 1)

        self.temp_res = temp_res
        self.tmin = 2
        self.bins = bins
        self.max_depth = 6

    def forward(self, hist, lights):
        # Compute depth
        images = torch.log10(hist + 1e-8)
        images = (images - np.log10(EPS)) / (np.log10(5000) - np.log10(EPS))
        images = torch.clip(images, 0, 1)
        x_hist = images.permute(0, 3, 1, 2)

        # print("Input", x_hist.shape)

        # encoder part - raw histogram
        encoders_features = []
        for encoder in self.encoders:
            x_hist = encoder(x_hist)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x_hist)
        encoders_features = encoders_features[1:]

        # decoder part - depth
        x_depth = None
        for i, (decoder, encoder_features) in enumerate(
            zip(self.decoders_depth, encoders_features)
        ):
            if i == 0:
                # print(i, encoder_features.shape, x_hist.shape)
                x_depth = decoder(encoder_features, x_hist)
            else:
                # print(i, encoder_features.shape, x_depth.shape)
                x_depth = decoder(encoder_features, x_depth)
        depth = self.final_conv_depth(x_depth)

        # print("Depth", depth.shape)

        # Convert depth into 2B ToF histograms
        tof = self.depth_to_tof(
            torch.clip(depth.squeeze(), 0, 1) * self.max_depth, lights
        )
        bin_image = torch.floor(tof / self.temp_res) - np.ceil(
            self.tmin / self.temp_res
        )

        min_vals, _ = torch.min(hist, dim=-1, keepdim=True)
        max_vals, _ = torch.max(hist, dim=-1, keepdim=True)
        lidar = (hist - min_vals) / (max_vals - min_vals)
        lidar = torch.nan_to_num(lidar)

        if len(bin_image.shape) == 2:
            bin_image = bin_image.unsqueeze(0)
        bin_image = bin_image.long()
        hist = torch.zeros([*lidar.shape]).to(lidar.device)
        bin_image = torch.clip(bin_image, 0, self.bins - 1)
        hist.scatter_(
            3,
            bin_image.unsqueeze(3),
            torch.ones_like(bin_image, dtype=hist.dtype).unsqueeze(3).to(lidar.device),
        )
        x_tof = hist.permute(0, 3, 1, 2)
        # Non differentiable version:
        # hist[batch_indices, row_indices, col_indices, bin_image] = 1.0

        # encoder part - visible histogram
        encoders_features_tof = []
        for encoder in self.encoders:
            x_tof = encoder(x_tof)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features_tof.insert(0, x_tof)
        encoders_features_tof = encoders_features_tof[1:]

        # decoder part - shadows
        curr_feats = None
        for i, (decoder, encoder_features) in enumerate(
            zip(self.decoders_shadow, encoders_features_tof)
        ):
            skip_feats = (
                encoders_features[i] + encoder_features
            )  # torch.cat((encoders_features[i], encoder_features), dim=1)
            if i == 0:
                # print(
                #     skip_feats.shape,
                #     encoders_features[i].shape,
                #     encoder_features.shape,
                #     x_tof.shape,
                # )
                curr_feats = x_hist + x_tof  # torch.cat((x_hist, x_tof), dim=1)
            # print(
            #     skip_feats.shape,
            #     encoders_features[i].shape,
            #     encoder_features.shape,
            #     curr_feats.shape,
            # )
            curr_feats = decoder(skip_feats, curr_feats)
        shadow = self.final_conv_shadow(curr_feats)

        return depth, shadow

    def depth_to_tof(
        self,
        depth,
        virtual_lights,
        num_lights=1,
    ):
        batch_size = depth.shape[0]
        f = 256.0 / (2 * math.tan(np.radians(90.0) / 2))
        K = torch.tensor([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]]).to(
            depth.device
        )

        u, v = torch.meshgrid(
            torch.arange(depth.shape[2], device=depth.device),
            torch.arange(depth.shape[1], device=depth.device),
            indexing="xy",
        )
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]

        X = x.unsqueeze(0) * depth
        Y = y.unsqueeze(0) * depth
        Z = depth
        pc_0 = torch.stack((X, Y, Z), dim=3)

        # Compute 2 bounce ToF
        camera_position = torch.zeros(batch_size, num_lights, 1, 3, device=depth.device)
        lights = camera_position.clone()
        virtual_lights = virtual_lights.unsqueeze(2)  # N x 16 x 1 x 3

        pc = pc_0.view(batch_size, 1, -1, 3).expand(-1, num_lights, -1, -1)

        d1 = torch.norm(virtual_lights - lights, dim=-1)  # N x 16 x 1
        d2 = torch.norm(pc - virtual_lights, dim=-1)  # N x 16 x (256*256)
        d3 = torch.norm(camera_position - pc, dim=-1)  # N x 16 x (256*256)

        total = d1 + d2 + d3
        total_torch_batch = total.view(batch_size, num_lights, 256, 256)

        return total_torch_batch.squeeze()


class UNet25(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        f_maps=256,
        num_levels=5,
        basic_module=DoubleConv,
        layer_order="cbr",
        num_groups=8,
        is_segmentation=True,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        is3d=False,
        temp_res=0.03837343462,
        tmin=2,
        bins=469,
    ):
        super(UNet25, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # print("Fmaps", f_maps)

        # create encoder path
        # self.encoders_2d = create_encoders(
        #     depth_in,
        #     f_maps,
        #     basic_module,
        #     conv_kernel_size,
        #     conv_padding,
        #     conv_upscale,
        #     dropout_prob,
        #     layer_order,
        #     num_groups,
        #     pool_kernel_size,
        #     False,
        # )

        # create encoder path
        self.encoders_3d = create_encoders(
            channels_in,
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            conv_upscale,
            dropout_prob,
            layer_order,
            num_groups,
            pool_kernel_size,
            True,
        )

        # create decoder path
        self.decoders = create_decoders(
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            upsample,
            dropout_prob,
            False,
        )

        # skips1 = nn.ModuleList(
        #     [
        #         nn.Linear(469, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1),
        #     ],
        # )
        # skips2 = nn.ModuleList(
        #     [
        #         nn.Linear(234, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1),
        #     ],
        # )
        # skips3 = nn.ModuleList(
        #     [
        #         nn.Linear(117, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1),
        #     ],
        # )
        # self.skips = [skips1.cuda(), skips2.cuda(), skips3.cuda()]

        # # No ft extraction - aria v3 (469 bins)
        # skips1 = nn.ModuleList(
        #     [
        #         nn.Linear(469, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1),
        #     ],
        # )
        # skips2 = nn.ModuleList(
        #     [
        #         nn.Linear(234, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1),
        #     ],
        # )
        # skips3 = nn.ModuleList(
        #     [
        #         nn.Linear(117, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1),
        #     ],
        # )
        # skips4 = nn.ModuleList(
        #     [
        #         nn.Linear(58, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1),
        #     ],
        # )
        # skips5 = nn.ModuleList(
        #     [
        #         nn.Linear(29, 1),
        #     ],
        # )
        # skips6 = nn.ModuleList(
        #     [
        #         nn.Linear(14, 1),
        #     ],
        # )

        # No ft extraction - aria final (637 bins)
        # skips1 = nn.ModuleList(
        #     [
        #         nn.Linear(637, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1),
        #     ],
        # )
        # skips2 = nn.ModuleList(
        #     [
        #         nn.Linear(318, 128),
        #         nn.ReLU(),
        #         nn.Linear(128, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1),
        #     ],
        # )
        # skips3 = nn.ModuleList(
        #     [
        #         nn.Linear(159, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1),
        #     ],
        # )
        # skips4 = nn.ModuleList(
        #     [
        #         nn.Linear(79, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 1),
        #     ],
        # )
        # skips5 = nn.ModuleList(
        #     [
        #         nn.Linear(39, 1),
        #     ],
        # )
        # skips6 = nn.ModuleList(
        #     [
        #         nn.Linear(19, 1),
        #     ],
        # )

        skips1 = nn.ModuleList(
            [
                nn.Linear(160, 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            ],
        )
        skips2 = nn.ModuleList(
            [
                nn.Linear(80, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            ],
        )
        skips3 = nn.ModuleList(
            [
                nn.Linear(40, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            ],
        )
        skips4 = nn.ModuleList(
            [
                nn.Linear(20, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            ],
        )
        # skips5 = nn.ModuleList(
        #     [
        #         nn.Linear(10, 1),
        #     ],
        # )
        # skips6 = nn.ModuleList(
        #     [
        #         nn.Linear(5, 1),
        #     ],
        # )
        self.skips = [
            skips1.cuda(),
            skips2.cuda(),
            skips3.cuda(),
            skips4.cuda(),
            # skips5.cuda(),
            # skips6.cuda(),
        ]

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        self.final_conv = nn.Conv2d(f_maps[0], channels_out, 1)

        # self.feature_extractor = nn.Conv2d(637, 160, 1) # Aria

        self.feature_extractor = nn.Conv2d(375, 160, 1)  # Real Temp 4
        # self.feature_extractor = nn.Conv2d(751, 160, 1)  # Real Temp 2

        self.temp_res = temp_res
        self.tmin = 2
        self.bins = bins
        self.max_depth = 6

    def forward(self, x):
        # x_hist_2d = x
        # print(x.shape)
        x_hist_3d = x
        if len(x.shape) == 4:
            # print("input", x.shape)
            x_hist_3d = self.feature_extractor(x)
            # print("features", x_hist_3d.shape)
            x_hist_3d = x_hist_3d.unsqueeze(1)
        else:
            shape = x.shape
            x_hist_3d = torch.reshape(x, (-1, shape[2], shape[3], shape[4]))
            x_hist_3d = self.feature_extractor(x_hist_3d)
            x_hist_3d = torch.reshape(
                x_hist_3d, (shape[0], shape[1], 160, shape[3], shape[4])
            )

        # encoder part - raw histogram
        # encoders_features = []
        # for encoder in self.encoders_2d:
        #     x_hist_2d = encoder(x_hist_2d)
        #     print(x_hist_2d.shape)
        #     # reverse the encoder outputs to be aligned with the decoder
        #     encoders_features.insert(0, x_hist_2d)
        # encoders_features = encoders_features[1:]

        # print("3d...")

        # encoder part - raw histogram
        encoders_features_3d = []
        count = 0
        for encoder in self.encoders_3d:
            x_hist_3d = encoder(x_hist_3d)
            # print(x_hist_3d.shape)
            x_hist_skip = torch.reshape(
                torch.permute(x_hist_3d, (0, 1, 3, 4, 2)), (-1, x_hist_3d.shape[2])
            )
            for layer in self.skips[count]:
                # print(layer.device, x_hist_skip.device)
                x_hist_skip = layer(x_hist_skip)
            # print(
            #     x_hist_3d.shape,
            #     x_hist_skip.reshape(
            #         x_hist_3d.shape[0],
            #         x_hist_3d.shape[1],
            #         1,
            #         x_hist_3d.shape[3],
            #         x_hist_3d.shape[4],
            #     ).shape,
            # )
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features_3d.insert(
                0,
                x_hist_skip.reshape(
                    x_hist_3d.shape[0],
                    x_hist_3d.shape[1],
                    x_hist_3d.shape[3],
                    x_hist_3d.shape[4],
                ),
            )
            count += 1
        encoders_features_3d = encoders_features_3d[1:]

        # exit()

        x_hist_skip = x_hist_skip.reshape(
            x_hist_3d.shape[0],
            x_hist_3d.shape[1],
            x_hist_3d.shape[3],
            x_hist_3d.shape[4],
        )

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features_3d):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x_hist_skip = decoder(encoder_features, x_hist_skip)

        x = self.final_conv(x_hist_skip)

        return x


class LearnableUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(LearnableUpsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        conv_upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        upsample (str): algorithm used for decoder upsampling:
            InterpolateUpsampling:   'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
            TransposeConvUpsampling: 'deconv'
            No upsampling:           None
            Default: 'default' (chooses automatically)
        dropout_prob (float or tuple): dropout probability, default: 0.1
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid,
        basic_module,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        is3d=True,
        upsample_image=0,
    ):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if "g" in layer_order:
            assert (
                num_groups is not None
            ), "num_groups must be specified if GroupNorm is used"

        # create encoder path
        self.encoders = create_encoders(
            in_channels,
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            conv_upscale,
            dropout_prob,
            layer_order,
            num_groups,
            pool_kernel_size,
            is3d,
        )

        # create decoder path
        self.decoders = create_decoders(
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            upsample,
            dropout_prob,
            is3d,
        )

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

        self.upsample = None
        if upsample_image != 0:
            self.upsample = LearnableUpsample(
                out_channels, out_channels * (upsample_image**2), upsample_image
            )

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        if self.upsample is not None:
            x = self.upsample(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction.
        # During training the network outputs logits
        # if not self.training and self.final_activation is not None:
        #     print("final activation")
        #     x = self.final_activation(x)

        return x


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=False,
        f_maps=128,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        is_segmentation=False,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        super(UNet3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=False,
        )


class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=128,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        super(ResidualUNet3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ResNetBlock,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=False,
        )


class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        super(ResidualUNetSE3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ResNetBlockSE,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=True,
        )


class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=False,
        f_maps=256,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        upsample_image=0,
        **kwargs,
    ):
        super(UNet2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=False,
            upsample_image=upsample_image,
        )


class ResidualUNet2D(AbstractUNet):
    """
    Residual 2DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        super(ResidualUNet2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ResNetBlock,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=False,
        )
