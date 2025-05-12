from torch.utils.data.dataset import Dataset

import os
import torch
import fnmatch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import random
import torch.nn.functional as F
from loguru import logger


class Sentinel2Data(Dataset):
    """
    This file is directly modified from https://pytorch.org/docs/stable/torchvision/datasets.html
    """
    def __init__(
        self, 
        root="/datasets/ai/allenai/satlas_pretrain/sentinel2", 
        split='train',
        nodata_val=-9999,
        return_fp = False,
        resize_size=512,
        adaptor="linear",
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.root = root
        self.bands = ["tci", "b08", "b11", "b12"]   # NOTE: tci is RGB. need to flip to get BGR (b2,b3,b4)
        self.return_fp = return_fp


        # Read the data file
        self.data_path = f"data_splits_sentinel2/{split}_data_sampled.csv"
        logger.debug(f"self.data_path: {self.data_path}")
        self.reference_df = pd.read_csv(self.data_path)
        logger.debug(f"Using reference_df: {self.reference_df.shape}")
        
        self.data_len = len(self.reference_df)
        self.resize_size = resize_size
        self.adaptor = adaptor
        logger.debug(f"Using adaptor: {self.adaptor}")
        logger.debug(f"Using resize_size: {self.resize_size}")
        # if split == "train":
        # self.data_len = 64

    def __getitem__(self, index):
        sample = self.reference_df.iloc[index]
        col, row = sample["col"], sample["row"]
        parent_dir = os.path.join(self.root, sample["tile_name"])
        landcover_path = sample["landcover_path"]
        tci_path = sample["tci_path"]

        input_data = []
        for band_name in self.bands:
            fp = os.path.join(parent_dir, band_name, f"{col}_{row}.png")
            band_data = Image.open(fp)
            band_data = np.array(band_data)
            if band_name=="tci":    #flip RGB to BGR
                band_data = band_data[:,:,::-1]
            else:
                band_data = band_data[:,:,None]
            input_data.append(band_data)
        input_data = np.concatenate(input_data, axis=-1)    # shape: (512,512,6)

        landcover_data = Image.open(landcover_path)
        water_mask = (np.array(landcover_data)==1).astype(np.uint8)  # shape: (512, 512)
        water_mask = water_mask[:,:,None]*255  # shape: (512, 512, 1); NOTE: multiplied to 255 since image data is 0 to 255

        # Random crop
        if self.split == "train":
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(size=(400,400)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.Resize((self.resize_size, self.resize_size)),
            ])
        else:
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.resize_size, self.resize_size)),
            ])
            
        all_data = np.concatenate((input_data, water_mask), axis=-1)
        all_data = data_transforms(all_data)
        image = all_data[:6,:,:]

        
        # Data Augmentation
        if self.split == "train":
            # Add Random channel mixing
            ccm = torch.eye(6)[None,None,:,:]
            r = torch.rand(3,)*0.25 + torch.Tensor([0,1,0])
            filter = r[None, None, :, None]
            ccm = torch.nn.functional.conv2d(ccm, filter, stride=1, padding="same")
            ccm = torch.squeeze(ccm)
            image = torch.tensordot(ccm, image, dims=([1],[0])) # not exactly the same perhaps

            # Add Gaussian noise
            r = torch.rand(1,1)*0.04
            image = image + torch.normal(mean=0.0, std=r[0][0], size=(len(self.bands)+2,self.resize_size,self.resize_size))   # add two since tci is counted as 1

        # Min-max Normalization
        # Normalize data
        if (torch.max(image)-torch.min(image)):
            image = image - torch.min(image)
            image = image / torch.maximum(torch.max(image),torch.tensor(1))
        else:
            logger.warning(f"all zero image. setting all labels to zero. index: {index}. {self.split} {fp}")
            image = torch.zeros_like(image)
            all_data = torch.zeros_like(all_data)
        
        water_mask = all_data[-1,:,:]
        water_mask = torch.where(water_mask>0.5, 1.0, 0)


        if self.adaptor=="drop":
            image = image[:3, :, :]         # only keep first 3 channels
            image = image[[2,1,0], :, :]    # RGB from BGR (NOTE: it was flipped above to bgr)
        labels = {
            "water_mask": water_mask.type(torch.FloatTensor),
        }
        if self.return_fp:
            return (
                image.float(),    # shape: (6, 512, 512)
                labels,
                tci_path,
            )

        else:
            return (
                image.float(),    # shape: (6, 512, 512)
                labels,
            )

    def __len__(self):
        return self.data_len


