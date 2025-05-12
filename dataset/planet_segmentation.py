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


class PlanetSegmentation(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(
        self, 
        root="data_dir/",  # path for data splits
        split='train', 
        adaptor="linear",
        return_fp = False,
        resize_size=None,
    ):
        assert split in ["train", "valid", "test"]
        # NOTE for label: 0=background, 1=river, 2=other water
        self.num_outputs = 1    # river vs not river
        # labels only in channel 0 of the label img i.e., [:,:,0]
        self.input_col_name = "normalized_planetscope_path"  # tif
        self.label_col_name = "label_path"  #png

        self.num_channels = 4   # RGB+NIR
        self.split = split
        self.return_fp = return_fp
        self.root = root
        
        all_fns_fp = os.path.join("data", f"{split}.csv")
        all_fns = pd.read_csv(all_fns_fp)
        self.fns = all_fns

        self.data_len = len(self.fns)
        final_size = 512
        if resize_size is not None:
            final_size = resize_size
        self.transforms_list = [transforms.ToTensor()]
        
        if "train" in self.split:
            self.transforms_list += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        self.transforms_list += [transforms.Resize(size=(final_size,final_size))]

        self.adaptor = adaptor
        logger.debug(f"Using adaptor: {self.adaptor}")
        
    def __getitem__(self, index):
        data_row = self.fns.iloc[index]
        input_fp = os.path.join(self.root, data_row[self.input_col_name])
        label_fp = os.path.join(self.root, data_row[self.label_col_name])
        
        image = rasterio.open(input_fp).read()
        image = np.transpose(image, (1,2,0))    # (500,500,4)

        label = rasterio.open(label_fp).read()
        label = np.transpose(label, (1,2,0))    # (500,500,1)
        # label = np.where(label==1,1,0)  # NOTE: only detecting RIVER WATER
        label = np.where(label!=0,1,0)  # NOTE: detecting ALL water
        
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
            # logger.debug(f"image: {type(image)}. ccm: {type(ccm)}")
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
            # logger.warning(f"all zero image. setting all labels to zero. index: {index}. {self.split} {input_fp}")
            image = torch.zeros_like(image).float()
            label = torch.zeros_like(label).float()
            
        if self.adaptor=="drop":
            image = image[:3, :, :]         # only keep first 3 channels
            image = image[[2,1,0], :, :]    # RGB from BGR
        labels = {
            "water_mask": label.float(),
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

