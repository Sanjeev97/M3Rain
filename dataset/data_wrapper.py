import torch
from sklearn import preprocessing
from torch.utils.data import Dataset
from einops import rearrange
import torchvision.transforms as transforms

import numpy as np


class DataWrapper(object):
    def __init__(self, img_size=128, s=1, kernel_size=9, train=True):
        self.img_size = img_size
        self.s = s
        self.kernel_size = kernel_size

        if train:
            self.transform = self.get_simclr_pipeline_transform()
        else:
            self.transform = self.get_transform_val()

    def __call__(self, x):
        x = x.to(torch.float32)
        
        # Check the shape of the input tensor
        if x.size(1) == 1:
            # Handle the case where there's only one channel
            # Instead of trying to extract multiple layers, we'll just use the same layer
            # for all positions and apply different augmentations
            single_layer = x[:, 0, :, :]
            
            # # Create a three-channel tensor by duplicating the single layer
            # x = torch.stack((single_layer, single_layer, single_layer), dim=0)
            
            # do nothing
            x = x
            
        else:
            # Original code for when we have multiple channels
            # Select the top layer (first layer)
            top_layer = x[:, 0, :, :]
            
            # If we have at least 2 channels, use the second; otherwise duplicate first
            mid_layer = x[:, min(1, x.size(1)-1), :, :]
            
            # If we have at least 3 channels, use the third; otherwise use the last available
            bottom_layer = x[:, min(2, x.size(1)-1), :, :]
            
            # Stack the selected layers along the new third dimension
            x = torch.stack((top_layer, mid_layer, bottom_layer), dim=0)
        
        # x = rearrange(x, 'c t h w -> t c h w')
        print(x.shape)
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj

    def get_simclr_pipeline_transform(self):
        # Simple resize transform for now
        data_transforms = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])
        return data_transforms

    def get_transform_val(self):
        # Simple resize transform for validation
        data_transforms = transforms.Compose([transforms.Resize((self.img_size, self.img_size))])
        return data_transforms


class ScalarNorm(object):
    def __init__(self):
        self.norm = preprocessing.StandardScaler()

    def __call__(self, x, reverse=False):
        if not reverse:
            x = self.norm.fit_transform(x)
        else:
            x = self.norm.inverse_transform(x)

        x = torch.from_numpy(x)
        x = x.to(torch.float32)
        return x


if __name__ == '__main__':
    # Test the data wrapper with a single-channel input
    x = torch.randn(4, 1, 24, 24)  # Batch of 4, 1 channel, 24x24 images
    wrapper = DataWrapper()
    xi, xj = wrapper(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape xi: {xi.shape}")
    print(f"Output shape xj: {xj.shape}")
    
    # Test with a multi-channel input
    x = torch.randn(4, 3, 24, 24)  # Batch of 4, 3 channels, 24x24 images
    xi, xj = wrapper(x)
    print(f"Input shape (multi-channel): {x.shape}")
    print(f"Output shape xi: {xi.shape}")
    print(f"Output shape xj: {xj.shape}")