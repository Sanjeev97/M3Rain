import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os

class AlignedH5Dataset(Dataset):
    """
    Dataset class for loading aligned radar and PWS data from HDF5 file.
    This combines functionality of both Radar_Dataset and PWS_Dataset.
    """
    def __init__(self, h5_file, flag='train'):
        """
        Initialize the dataset.
        
        Args:
            h5_file (str): Path to the HDF5 file containing aligned data
            flag (str): 'train' or 'test' to specify which dataset to use
        """
        self.h5_file = h5_file
        self.flag = flag
        
        # Verify flag validity
        assert flag in ['train', 'test'], "flag must be either 'train' or 'test'"
        
        # Open the HDF5 file to get metadata
        with h5py.File(self.h5_file, 'r') as h5f:
            # Get the number of sequences
            self.num_samples = h5f[f'aligned_{flag}'].attrs['all_len']
            print(f"Number of samples in {flag} dataset: {self.num_samples}")
            
            # Check the structure of the first item to determine dimensions
            first_group = h5f[f'aligned_{flag}/0']
            self.radar_shape = first_group['radar'].shape
            self.pws_shape = first_group['pws'].shape
            print(f"Radar shape: {self.radar_shape}, PWS shape: {self.pws_shape}")
        
        self.seq_len = 4  # Input sequence length (consistent with original code)
        self.label_len = 4  # Label/output sequence length
        self.pred_len = 4  # Prediction length
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, index):
        """
        Get a single item from the dataset.
        
        Args:
            index (int): Index of the item to fetch
            
        Returns:
            tuple: (radar_data, pws_data, timestamp_idx)
                - radar_data: Radar image sequence (shape matches original dataset)
                - pws_data: PWS data corresponding to the radar sequence
                - timestamp_idx: Index of the sample (used for debugging/logging)
        """
        with h5py.File(self.h5_file, 'r') as h5f:
            group_path = f'aligned_{self.flag}/{index}'
            
            # Get the radar and PWS data for this sequence
            radar_data = h5f[f'{group_path}/radar'][()]
            pws_data = h5f[f'{group_path}/pws'][()]
            
            # Convert radar data to tensor and add channel dimension if needed
            radar_tensor = torch.from_numpy(radar_data).float()
            if len(radar_tensor.shape) == 3:  # If missing channel dimension
                radar_tensor = radar_tensor.unsqueeze(1)
                
            # Convert PWS data to tensor and ensure proper format
            pws_tensor = torch.from_numpy(pws_data).float()
            pws_tensor = pws_tensor.unsqueeze(1)  # Add channel dimension to match original code
            
            # Apply log transformation to PWS data with small epsilon to avoid log(0)
            # epsilon = 1e-6
            # pws_tensor = torch.log(pws_tensor + epsilon)
            
            # Return the data in the format expected by the model
            # Instead of returning timestamps (which can cause collation issues), return the index
            return radar_tensor, pws_tensor, index

def create_data_loaders(h5_file_path, batch_size, num_workers, pin_memory=True):
    """
    Create data loaders for training and testing from the H5 file.
    
    Args:
        h5_file_path (str): Path to the HDF5 file
        batch_size (int): Batch size for the data loaders
        num_workers (int): Number of worker processes for loading data
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = AlignedH5Dataset(h5_file_path, flag='train')
    test_dataset = AlignedH5Dataset(h5_file_path, flag='test')
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # drop_last=True,
        drop_last=False,  # Keep all samples in test set for evaluation
    )
    
    return train_loader, test_loader


# Adding this function for debugging purposes
def get_sample_with_timestamps(dataset, index):
    """
    Get a sample with full timestamp information (for debugging/inspection)
    
    Args:
        dataset: AlignedH5Dataset instance
        index: Sample index to retrieve
        
    Returns:
        tuple: (radar_data, pws_data, timestamps)
    """
    with h5py.File(dataset.h5_file, 'r') as h5f:
        group_path = f'aligned_{dataset.flag}/{index}'
        
        # Get the radar and PWS data for this sequence
        radar_data = h5f[f'{group_path}/radar'][()]
        pws_data = h5f[f'{group_path}/pws'][()]
        
        # Get timestamps
        timestamps = h5f[f'{group_path}'].attrs['dates']
        
        # Convert to tensors
        radar_tensor = torch.from_numpy(radar_data).float()
        if len(radar_tensor.shape) == 3:
            radar_tensor = radar_tensor.unsqueeze(1)
            
        pws_tensor = torch.from_numpy(pws_data).float()
        pws_tensor = pws_tensor.unsqueeze(1)
        
        return radar_tensor, pws_tensor, timestamps

# Example usage
if __name__ == '__main__':
    h5_file_path = "/home/C00535626/Radar-Rainfall/MViT/data/klch_radar_pws_aligned.h5"
    
    # Create datasets
    train_dataset = AlignedH5Dataset(h5_file_path, flag='train')
    test_dataset = AlignedH5Dataset(h5_file_path, flag='test')
    
    # Print dataset information
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Get a sample
    radar_sample, pws_sample, idx = train_dataset[0]
    print(f"Radar sample shape: {radar_sample.shape}")
    print(f"PWS sample shape: {pws_sample.shape}")
    print(f"Sample index: {idx}")
    
    # Get sample with timestamps for debugging
    radar_data, pws_data, timestamps = get_sample_with_timestamps(train_dataset, 0)
    print(f"Sample timestamps: {timestamps[0]} to {timestamps[-1]}")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(h5_file_path, batch_size=4, num_workers=2)
    
    # Test a batch
    for radar_batch, pws_batch, idx_batch in train_loader:
        print(f"Radar batch shape: {radar_batch.shape}")
        print(f"PWS batch shape: {pws_batch.shape}")
        print(f"Batch indices: {idx_batch}")
        break