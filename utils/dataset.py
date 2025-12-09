#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import h5py
import Imath
import math
import matplotlib.pyplot as plt
import numpy as np
import OpenEXR
import os
import time
import torch

from copy import deepcopy
from torch.utils.data import Dataset

EPS = 1e-8
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def lidar_to_exr(data, out_path):
    # Prepare the header
    # Prepare the header
    # data = data.astype(np.float32)
    header = OpenEXR.Header(637, 256 * 256)
    header["compression"] = Imath.Compression(
        Imath.Compression.DWAA_COMPRESSION
    )  # ZIP_COMPRESSION)
    header["channels"] = {
        "Y": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF), 1, 1)
    }

    # Create and write the EXR file
    exr = OpenEXR.OutputFile(out_path, header)
    exr.writePixels({"Y": data.tobytes()})
    exr.close()


def exr_to_numpy_lidar(file_path):
    # Open the EXR file
    file = OpenEXR.InputFile(file_path)

    # Get the header and extract data window (size)
    header = file.header()
    dw = header["dataWindow"]
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    # Read the data
    y_str = file.channel("Y", Imath.PixelType(Imath.PixelType.HALF))

    # Convert to numpy array
    y_arr = np.frombuffer(y_str, dtype=np.float16)

    # Reshape the array
    y_arr = y_arr.reshape(256, 256, 637)

    return y_arr


def exr_to_numpy_depth(exr_file_path):
    # IMREAD_UNCHANGED loads the raw floating point data
    image = cv2.imread(exr_file_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"Could not read {exr_file_path}. File might be corrupt or path is wrong.")

    # Handle multi-channel EXRs
    if image.ndim == 3:
        # Standard convention: Depth is usually in the first channel (R or Z)
        # OpenCV loads as BGR, so index 0 is Blue, 1 is Green, 2 is Red.
        # However, for single-channel data saved as RGB, all 3 are usually identical.
        return image[:, :, 0]

    return image


class DepthDataset(Dataset):
    def __init__(
        self,
        data_path,
        file_list,
        mode,
        max_dataset_size=50000,
        remove=None,
        ids=None,
        num_lights=25,
    ):
        self.data_path = data_path
        self.file_list = file_list
        self.mode = mode
        self.files = []
        self.num_lights = num_lights

        if remove is None:
            remove = []

        self.train_ratio = 0.9
        self.val_ratio = 0.1
        self.test_ratio = 0
        self.max_tof = 25.45584
        self.max_depth = 4.5
        self.tmin = 1
        self.min_tof = 1
        with open(file_list, "r") as f:
            for i, line in enumerate(f):
                if i == 0: continue
                line = line.split(".tar.gz")[0].strip()
                self.files.append(line)

        dataset_size = len(self.files)
        if max_dataset_size is not None:
            dataset_size = min(dataset_size, max_dataset_size)
        self.files = self.files[:dataset_size]

        # Split data into train and val
        if mode == "train":
            self.files = self.files[: int(self.train_ratio * len(self.files))]
        elif mode == "val":
            self.files = self.files[
                int(self.train_ratio * len(self.files)) : int(
                    (self.train_ratio + self.val_ratio) * len(self.files)
                )
            ]
        elif mode == "test":
            self.files = self.files[
                int((self.train_ratio + self.val_ratio) * len(self.files)) :
            ]
        else:
            raise ValueError("Invalid mode: {}".format(mode))

        for item in remove:
            if item in self.files:
                self.files.remove(item)
        if ids is not None:
            self.files = ids

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Get file paths
        identifier = self.files[index]
        file_path_lidar = os.path.join(
            self.data_path,
            identifier,
            "lidar_{}.exr".format(self.num_lights),
        )
        file_path_gt = os.path.join(
            self.data_path, identifier, "depth.exr"
        )        

        # Load data
        lidar = exr_to_numpy_lidar(file_path_lidar)
        gt = exr_to_numpy_depth(file_path_gt)

        # Normalize depth
        gt = np.clip(gt / self.max_depth, 0.0, 1.0)

        return (
            lidar,
            gt,
        )


class SpecularSegmentationDataset(Dataset):
    def __init__(
        self,
        data_path,
        file_list,
        mode,
        max_dataset_size=50000,
        remove=None,
        ids=None,
        num_lights=25,
    ):
        self.data_path = data_path
        self.file_list = file_list
        self.mode = mode
        self.files = []
        self.num_lights = num_lights

        if remove is None:
            remove = []

        self.train_ratio = 0.9
        self.val_ratio = 0.1
        self.test_ratio = 0
        self.max_tof = 25.45584
        self.max_depth = 4.5
        self.tmin = 1
        self.min_tof = 1
        with open(file_list, "r") as f:
            for i, line in enumerate(f):
                if i == 0: continue
                line = line.split(".tar.gz")[0].strip()
                self.files.append(line)

        dataset_size = len(self.files)
        if max_dataset_size is not None:
            dataset_size = min(dataset_size, max_dataset_size)
        self.files = self.files[:dataset_size]

        # Split data into train and val
        if mode == "train":
            self.files = self.files[: int(self.train_ratio * len(self.files))]
        elif mode == "val":
            self.files = self.files[
                int(self.train_ratio * len(self.files)) : int(
                    (self.train_ratio + self.val_ratio) * len(self.files)
                )
            ]
        elif mode == "test":
            self.files = self.files[
                int((self.train_ratio + self.val_ratio) * len(self.files)) :
            ]
        else:
            raise ValueError("Invalid mode: {}".format(mode))

        for item in remove:
            if item in self.files:
                self.files.remove(item)
        if ids is not None:
            self.files = ids

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Get file paths
        identifier = self.files[index]
        file_path_lidar = os.path.join(
            self.data_path,
            identifier,
            "lidar_{}.exr".format(self.num_lights),
        )
        file_path_mask = os.path.join(self.data_path, identifier, "specular_mask.png")

        # Load data
        lidar = exr_to_numpy_lidar(file_path_lidar)
        mask = cv2.imread(file_path_mask, cv2.IMREAD_GRAYSCALE) / 255.0
        mask[mask >= 0.1] = 1.0
        mask[mask < 0.1] = 0.0

        # # Randomly turn mirrors into holes
        # if np.sum(mask.flatten()) > 0:
        #     hole = np.random.randint(0, 3)  # 1/3 chance of hole
        #     if hole == 2:
        #         lidar[mask == 1] = 0
        #         mask[mask == 1] = 0

        return (
            lidar,
            mask,
        )


class CombinedDataset(Dataset):
    def __init__(
        self,
        data_path,
        file_list,
        mode,
        max_dataset_size=100000,
        predownload=False,
        max_depth=5.0,
        temp_res=0.0384,
        remove=None,
        ids=None,
        override="",
        num_lights=25,
    ):
        self.data_path = data_path
        self.file_list = file_list
        self.mode = mode
        self.files = []
        self.temp_res = temp_res
        self.override = override
        self.num_lights = num_lights

        if remove is None:
            remove = []

        self.train_ratio = 0.9
        self.val_ratio = 0.1
        self.test_ratio = 0
        self.max_tof = 25.45584
        self.max_depth = 4.5
        self.tmin = 1
        self.min_tof = 1
        with open(file_list, "r") as f:
            for i, line in enumerate(f):
                if i == 0: continue
                line = line.split(".tar.gz")[0].strip()
                self.files.append(line)
        if self.num_lights == 25:
            self.light_idxs = [
                1,
                3,
                5,
                7,
                9,
                21,
                23,
                25,
                27,
                29,
                41,
                43,
                45,
                47,
                49,
                61,
                63,
                65,
                67,
                69,
                81,
                83,
                85,
                87,
                89,
            ]
        elif self.num_lights == 4:
            self.light_idxs = [22, 27, 72, 77]

        dataset_size = len(self.files)
        if max_dataset_size is not None:
            dataset_size = min(dataset_size, max_dataset_size)
        self.files = self.files[:dataset_size]

        # Split data into train and val
        if mode == "train":
            self.files = self.files[: int(self.train_ratio * len(self.files))]
        elif mode == "val":
            self.files = self.files[
                int(self.train_ratio * len(self.files)) : int(
                    (self.train_ratio + self.val_ratio) * len(self.files)
                )
            ]
        elif mode == "test":
            self.files = self.files[
                int((self.train_ratio + self.val_ratio) * len(self.files)) :
            ]
        else:
            raise ValueError("Invalid mode: {}".format(mode))

        for item in remove:
            if item in self.files:
                self.files.remove(item)
        if ids is not None:
            self.files = ids

        if self.override != "":
            files = []
            for f in self.files:
                files.extend([f] * self.num_lights)
            self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Paths
        identifier = self.files[index]
        file_path_lidar = os.path.join(
            self.data_path,
            identifier,
            "lidar_{}.exr".format(self.num_lights),
        )
        file_path_shadows = os.path.join(
            self.data_path,
            identifier,
            "shadows_v1.npz",
        )
        file_path_lights = os.path.join(
            self.data_path, identifier, "lights_camera_frame.npy"
        )
        file_path_depth = os.path.join(
            self.data_path, identifier, "depth.exr"
        )

        # Load data
        lidar = exr_to_numpy_lidar(file_path_lidar)
        depth = exr_to_numpy_depth(file_path_depth) / self.max_depth
        lights = np.load(file_path_lights)
        shadows = np.load(file_path_shadows)["shadows"]
        if shadows.shape[-1] == 3:
            shadows = np.mean(shadows, axis=-1, keepdims=True).squeeze()

        if self.num_lights != 100:
            shadows = shadows[self.light_idxs]
            lights = lights[self.light_idxs]

        # Choose random illumination
        idx = np.random.randint(0, lights.shape[0])
        if self.override != "":
            # logic: this gives us index into light
            idx = index % self.num_lights

        depth = torch.tensor(depth)
        light = torch.tensor(lights[idx]).unsqueeze(0)
        shadow = torch.tensor(shadows[idx, :, :].squeeze())
        lidar = torch.tensor(lidar)

        return (lidar, light, depth, shadow)


class ShadowDataset(Dataset):
    def __init__(
        self,
        data_path,
        file_list,
        mode,
        max_dataset_size=50000,
        temp_res=0.0384,  # 0.03837343462,
        remove=None,
        ids=None,
        override="",
        num_lights=25,
    ):
        self.data_path = data_path
        self.file_list = file_list
        self.mode = mode
        self.files = []
        self.temp_res = temp_res
        self.override = override
        self.num_lights = num_lights

        if remove is None:
            remove = []

        self.train_ratio = 0.9
        self.val_ratio = 0.1
        self.test_ratio = 0
        self.max_tof = 25.45584
        self.max_depth = 4.5
        self.tmin = 1
        self.min_tof = 1
        with open(file_list, "r") as f:
            for i, line in enumerate(f):
                if i == 0: continue
                line = line.split(".tar.gz")[0].strip()
                self.files.append(line)
        if self.num_lights == 25:
            self.light_idxs = [
                1,
                3,
                5,
                7,
                9,
                21,
                23,
                25,
                27,
                29,
                41,
                43,
                45,
                47,
                49,
                61,
                63,
                65,
                67,
                69,
                81,
                83,
                85,
                87,
                89,
            ]
        elif self.num_lights == 4:
            self.light_idxs = [22, 27, 72, 77]

        dataset_size = len(self.files)
        if max_dataset_size is not None:
            dataset_size = min(dataset_size, max_dataset_size)
        self.files = self.files[:dataset_size]

        # Split data into train and val
        if mode == "train":
            self.files = self.files[: int(self.train_ratio * len(self.files))]
        elif mode == "val":
            self.files = self.files[
                int(self.train_ratio * len(self.files)) : int(
                    (self.train_ratio + self.val_ratio) * len(self.files)
                )
            ]
            self.files = self.files[:2500]
        elif mode == "test":
            self.files = self.files[
                int((self.train_ratio + self.val_ratio) * len(self.files)) :
            ]
        else:
            raise ValueError("Invalid mode: {}".format(mode))

        for item in remove:
            if item in self.files:
                self.files.remove(item)
        if ids is not None:
            self.files = ids

        if self.override != "":
            files = []
            for f in self.files:
                files.extend([f] * self.num_lights)
            self.files = files


    def __len__(self):
            return len(self.files)

    def __getitem__(self, index):
        # Paths
        identifier = self.files[index]
        file_path_lidar = os.path.join(
            self.data_path,
            identifier,
            "lidar_{}.exr".format(self.num_lights),
        )

        file_path_shadows = os.path.join(
            self.data_path,
            identifier,
            "shadows_v1.npz",
        )

        if self.override != "":
            file_path_tof = self.override
            file_path_lights = os.path.join(
                self.data_path, identifier, "lights_camera_frame.npy"
            )
            file_path_depth = os.path.join(
                self.data_path, identifier, "depth.exr"
            )
        else:
            file_path_depth = os.path.join(
                self.data_path,
                identifier,
                "depth.exr",
            )

            file_path_lights = os.path.join(
                self.data_path, identifier, "lights_camera_frame.npy"
            )

        lidar = exr_to_numpy_lidar(file_path_lidar)
        lights = np.load(file_path_lights)
        if self.num_lights != 100:
            lights = lights[self.light_idxs]
        depth = exr_to_numpy_depth(file_path_depth)
        tof = self.depth_to_tof(depth, lights, num_lights=self.num_lights)

        # tof_gt = None
        if self.override != "":
            tof = np.load(file_path_tof)
            depth_idx = int(
                index / self.num_lights
            )  # --> gives us the idx for the depth map
            depth_im = tof[depth_idx] * self.max_depth
            # lights = np.load(file_path_lights)
            tof = self.depth_to_tof(depth_im, lights, num_lights=self.num_lights)
            # depth_gt = exr_to_numpy_depth(file_path_depth)
            # tof = self.depth_to_tof(depth_gt, lights)
            # tof_gt = np.load(file_path_tof2)
            # tof = tof_gt

            # exit()

            # tof2 = np.load(file_path_tof2)
            # # diff = np.abs(tof[0] - tof2[0])
            # diff = np.abs(depth_gt - depth_im)
            # plt.imshow(diff)
            # plt.colorbar()
            # plt.savefig(
            #     "./depth_diff.png",
            #     dpi=1000,
            # )
            # plt.close()
            # plt.imshow(depth_im)
            # plt.colorbar()
            # plt.savefig(
            #     "./depth_ours.png",
            #     dpi=1000,
            # )
            # plt.close()
            # plt.imshow(depth_gt)
            # plt.colorbar()
            # plt.savefig(
            #     "./depth_gt.png",
            #     dpi=1000,
            # )
            # plt.close()
            # print("Diff:", np.mean(diff), np.min(diff), np.max(diff))
            # exit()
        shadows = np.load(file_path_shadows)["shadows"]
        if shadows.shape[-1] == 3:
            shadows = np.mean(shadows, axis=-1, keepdims=True).squeeze()
        if self.num_lights != 100:
            shadows = shadows[self.light_idxs]

            # if self.override == "":
            #     print(tof.shape)
            #     tof = tof[self.light_idxs]
            # tof = tof[self.light_idxs]

        # Choose random illumination
        idx = np.random.randint(0, tof.shape[0])

        if self.override != "":
            # logic: this gives us index into light
            idx = index % self.num_lights
        # idx = index % self.num_lights  #

        tof = tof[idx, :, :].squeeze()

        # This is because of our tmin and tmax being slightly off for some scenes
        # if self.data_source == "aria":
        #     if np.max(tof) > 20.0 or np.min(tof) < 2.0:
        #         print("ToF is out of range: {}, {}.".format(np.min(tof), np.max(tof)))
        #         raise ValueError("ToF is out of range.")
        # elif self.data_source == "aria_100" or "aria_final" in self.data_source:
        #     if np.max(tof) > self.max_tof or np.min(tof) < self.min_tof:
        #         print("ToF is out of range: {}, {}.".format(np.min(tof), np.max(tof)))
        #         raise ValueError("ToF is out of range.")

        # tof = tof + np.random.normal(
        #     loc=0, scale=self.temp_res * 2, size=(tof.shape[0], tof.shape[1])
        # )
        tof = torch.tensor(tof)
        shadows = torch.tensor(shadows[idx, :, :].squeeze())
        lidar = torch.tensor(lidar)
        # print(file_path_lidar, idx)

        bins = torch.floor(tof / self.temp_res) - np.ceil(self.tmin / self.temp_res)
        bins = bins.long()
        inp = torch.concatenate((lidar, bins[:, :, None]), dim=-1)
        # print("Input", inp.shape)
        # print("Time", time.time() - start)

        # # Compute histogram (no pulse template)
        # bins = torch.floor(tof / self.temp_res) - np.ceil(self.tmin / self.temp_res)
        # bins = bins.long()
        # H, W = bins.shape
        # row_indices = torch.arange(H).view(-1, 1).expand(H, W)
        # col_indices = torch.arange(W).view(1, -1).expand(H, W)
        # hist = torch.zeros([*lidar.shape])
        # hist[row_indices, col_indices, bins] = 1.0
        # # print("Hist", hist.shape, "Lidar", lidar.shape)

        # inp = None
        # if "diff" in self.input_type:
        #     inp = hist - lidar
        #     inp[inp < 0] = 0
        # else:
        #     inp = np.concatenate((lidar, hist), axis=-1)

        # print("Input", inp.shape)
        # print("Time", time.time() - start)

        # bins = np.floor(tof / self.temp_res) - np.ceil(self.tmin / self.temp_res)
        # bins = bins.astype(np.int8)
        # print("Bins", bins.shape)
        # row_indices, col_indices = np.indices(bins.shape)
        # print("Row/Col", row_indices.shape, col_indices.shape)
        # hist = np.zeros([*lidar.shape])
        # print("Hist", hist.shape)
        # hist[row_indices, col_indices, bins] = 1.0

        # inp = hist - lidar
        # inp[inp < 0] = 0

        if self.override != "":
            return (
                inp,
                shadows,
                tof,
            )

        return (
            inp,
            shadows,
        )

    def depth_to_tof(
        self,
        depth,
        virtual_lights,
        num_lights=25,  # 100,  #
        temp_res=0.0384,  # 0.03837343462,
        debug=False,
    ):
        f = 256.0 / (2 * math.tan(np.radians(90.0) / 2))
        K = np.array([[f, 0.0, 128.0], [0.0, f, 128.0], [0.0, 0.0, 1.0]])

        u, v = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        x = (u - K[0, 2]) / K[0, 0]
        y = (v - K[1, 2]) / K[1, 1]
        X = x * depth
        Y = y * depth
        Z = depth
        pc_0 = np.stack((X, Y, Z), axis=2)

        # Compute 2 bounce ToF
        camera_position = np.expand_dims(
            np.stack([np.array([0, 0, 0]) for _ in range(num_lights)]), 1
        )  # 4 x 1 x 3
        lights = deepcopy(camera_position)  # 4 x 1 x 3
        virtual_lights = np.expand_dims(virtual_lights, 1)  # 4 x 1 x 3
        pc = np.stack([pc_0.reshape(-1, 3) for _ in range(num_lights)])  # 4 x N x 3

        d1 = np.linalg.norm(
            virtual_lights - lights, axis=-1
        )  # 4 x N -- light to virtual light
        d2 = np.linalg.norm(
            pc - virtual_lights, axis=-1
        )  # 4 x N -- virtual light to point
        d3 = np.linalg.norm(camera_position - pc, axis=-1)  # 4 x N -- point to camera
        total = d1 + d2 + d3
        total = total.reshape(num_lights, 256, 256)

        return total


def get_dataset(
    data_path,
    file_list,
    mode,
    max_dataset_size,
    task,
    remove=None,
    ids=None,
    override="",
    num_lights=25,
    temp_res=0.0384,
):
    if task == "depth":
        return DepthDataset(
            data_path=data_path,
            file_list=file_list,
            mode=mode,
            max_dataset_size=max_dataset_size,
            remove=remove,
            ids=ids,
            num_lights=num_lights,
        )
    elif task == "shadows":
        return ShadowDataset(
            data_path=data_path,
            file_list=file_list,
            mode=mode,
            max_dataset_size=max_dataset_size,
            remove=remove,
            ids=ids,
            override=override,
            num_lights=num_lights,
            temp_res=temp_res,
        )
    elif task == "specular":
        return SpecularSegmentationDataset(
            data_path=data_path,
            file_list=file_list,
            mode=mode,
            max_dataset_size=max_dataset_size,
            remove=remove,
            ids=ids,
            num_lights=num_lights,
        )
    elif task == "combined":
        return CombinedDataset(
            data_path=data_path,
            file_list=file_list,
            mode=mode,
            max_dataset_size=max_dataset_size,
            remove=remove,
            ids=ids,
            override=override,
            num_lights=num_lights,
            temp_res=temp_res,
        )


if __name__ == "__main__":
    ### DEPTH DATASET ###
    print("Testing depth dataset...")
    dataset = DepthDataset(
        data_path="./data",
        file_list="./data/dataset.txt",
        mode="val",
        max_dataset_size=100000,
        num_lights=25,
    )
    print("Dataset size: {}".format(len(dataset)))
    lidar, depth = dataset[0]
    print("Lidar shape: {}, Depth shape: {}\n".format(lidar.shape, depth.shape))


    ### SHADOW DATASET ###
    print("Testing shadow dataset...")
    dataset = ShadowDataset(
        data_path="./data",
        file_list="./data/dataset.txt",
        mode="val",
        max_dataset_size=100000,
        num_lights=25,
    )
    print("Dataset size: {}".format(len(dataset)))
    shadow_transient, shadow = dataset[0]
    print("Shadow Transient shape: {}, Shadow shape: {}\n".format(shadow_transient.shape, shadow.shape))


    ### SPECULAR SEGMENTATION DATASET ###
    print("Testing specular segmentation dataset...")
    dataset = SpecularSegmentationDataset(
        data_path="./data",
        file_list="./data/dataset.txt",
        mode="val",
        max_dataset_size=100000,
        num_lights=25,
    )
    print("Dataset size: {}".format(len(dataset)))
    lidar, mask = dataset[0]
    print("Lidar shape: {}, Mask shape: {}\n".format(lidar.shape, mask.shape))


    ### COMBINED DATASET ###
    print("Testing combined dataset...")
    dataset = CombinedDataset(
        data_path="./data",
        file_list="./data/dataset.txt",
        mode="val",
        max_dataset_size=100000,
        num_lights=25,
    )
    print("Dataset size: {}".format(len(dataset)))
    lidar, light, depth, shadow = dataset[0]
    print("Lidar shape: {}, Lights shape: {}, Depth shape: {}, Shadow shape: {}".format(
        lidar.shape, light.shape, depth.shape, shadow.shape)
    )