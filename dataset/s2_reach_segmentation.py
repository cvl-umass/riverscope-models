from torch.utils.data.dataset import Dataset

import rasterio
import os
import torch
import fnmatch
import numpy as np
import pandas as pd
import pdb
import torchvision.transforms as transforms
from PIL import Image
import random
import torch.nn.functional as F
from loguru import logger

import h5py
import json
import pickle
import ast

class SentinelReachSegmentation(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(
        self, 
        root="src_eval_s2_widths/",  # path for data splits
        split='test', 
        return_fp = False,
        resize_size=None,
        is_all_bands=False,
    ):
        assert split=="test"
        # NOTE for label: 0=background, 1=river, 2=other water
        
        # labels only in channel 0 of the label img i.e., [:,:,0]
        self.input_col_name = "tile_fp"  # tif
        # self.label_col_name = "original_label_path"  #png

        self.num_channels = 6   # 2,3,4,8,11,12
        
        self.split = split
        self.return_fp = return_fp
        self.root = root
        
        # all_fns_fp = os.path.join(root, f"test_tiles_for_s2.csv")
        all_fns_fp = os.path.join(root, f"test_tiles_for_s2_unfiltered.csv")
        all_fns = pd.read_csv(all_fns_fp)
        self.fns = all_fns

        self.data_len = len(self.fns)
        final_size = 512
        if resize_size is not None:
            final_size = resize_size
        self.transforms_list = [transforms.ToTensor()]
        self.transforms_list += [transforms.Resize(size=(final_size,final_size))]
        if "train" in self.split:
            self.transforms_list += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        self.is_all_bands = is_all_bands

        # if split == "train":
        #     self.data_len = 45

    def __getitem__(self, index):
        data_row = self.fns.iloc[index]
        input_fp = data_row[self.input_col_name]
        
        if self.is_all_bands:   # means all 12 bands are loaded
            image = rasterio.open(input_fp).read()  # (12,h,w)
            image = image[[1,2,3,7,10,11],:,:]      # bands 2,3,4,8,11,12
            image = np.transpose(image, (1,2,0))    # (500,500,6)
        else:
            input_fp = os.path.join("src_eval_s2_widths", input_fp)
            image = rasterio.open(input_fp).read()  # (6,h,w)
            image = np.transpose(image, (1,2,0))    # (500,500,4)

        label = np.zeros_like(image)    # NOTE: no label available
        
        data_transforms = transforms.Compose(self.transforms_list)
        data = np.concatenate((image,label), axis=-1)
        trans_data = data_transforms(data)  # output of transforms: (5, 512, 512)
        image = trans_data[:self.num_channels, :, :].float()
        label = trans_data[-1, :, :].float()


        if self.split == "train":
            # Add Random channel mixing
            ccm = torch.eye(self.num_channels)[None,None,:,:]
            r = torch.rand(3,)*0.25 + torch.Tensor([0,1,0])
            filter = r[None, None, :, None]
            ccm = torch.nn.functional.conv2d(ccm, filter, stride=1, padding="same")
            ccm = torch.squeeze(ccm)
            try:
                image = torch.tensordot(ccm, image, dims=([1],[0])) # not exactly the same perhaps
            except:
                logger.warning("Error introducing random channel mixing")
                pass    # NOTE: Error for multispectral images
            
            # Add Gaussian noise
            r = torch.rand(1,1)*0.04
            image = image + torch.normal(mean=0.0, std=r[0][0], size=(self.num_channels,image.shape[1],image.shape[2]))
        
        # Min-max Normalization
        # Normalize data
        # """
        if (torch.max(image)-torch.min(image)):
            image = image - torch.min(image)
            image = image / torch.maximum(torch.max(image),torch.tensor(1))
        else:
            logger.warning(f"all zero image. setting all labels to zero. index: {index}. {self.split} {input_fp}")
            image = torch.zeros_like(image).float()
            label = torch.zeros_like(label).float()
            # all_data = torch.zeros_like(all_data)
        # """
        labels = {
            "water_mask": label.type(torch.FloatTensor),
        }
        if self.return_fp:
            return (
                image,    # shape: (3, 512, 512)
                labels,
                input_fp,
            )

        else:
            return (
                image,    # shape: (3, 512, 512)
                labels,
            )


    def __len__(self):
        return self.data_len

