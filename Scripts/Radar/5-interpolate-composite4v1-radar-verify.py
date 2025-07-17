import os
import numpy as np
import pickle
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc  # For garbage collection
import psutil  # For memory monitoring

def get_memory_usage():
    """Return the current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    return mem

def log_memory(message):
    """Log a message with current memory usage"""
    mem = get_memory_usage()
    print(f"{message} - Memory usage: {mem:.2f} MB")

def read_files_and_create_composite(sorted_files, read_path, num_layers=4, batch_size=50):
    """Read files in batches and create composite reflectivity on-the-fly"""
    all_timestamps = []
    all_composites = []
    
    # Process files in batches
    num_batches = (len(sorted_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(sorted_files))
        
        batch_files = sorted_files[start_idx:end_idx]
        print(f"Processing batch {batch_idx+1}/{num_batches} (files {start_idx} to {end_idx-1})")
        
        batch_timestamps = []
        batch_data = []
        
        # Read files in this batch
        for f in tqdm(batch_files, desc=f"Batch {batch_idx+1}"):
            file_path = read_path + f
            try:
                with open(file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                    for x in loaded_data:
                        # Get timestamp and radar data
                        timestamp = x["time"]
                        data = x["REF"].filled(0.0).squeeze()
                        
                        # Print shape of the first data sample to understand structure
                        if len(batch_data) == 0:
                            print(f"First data sample shape: {data.shape}")
                        
                        # Create composite immediately
                        if data.shape[0] >= num_layers:
                            # Take max across specified number of layers (elevation angles)
                            # Use axis=0 to take max across the first dimension (layers/elevation angles)
                            composite = np.nanmax(data[:num_layers], axis=0)
                        else:
                            # If we don't have enough layers, take max of all available
                            composite = np.nanmax(data, axis=0)
                        
                        # Add a channel dimension to match expected 4D shape (time, channel, height, width)
                        composite = np.expand_dims(composite, axis=0)
                        
                        # Print shape of the first composite to confirm
                        if len(batch_data) == 0:
                            print(f"First composite shape: {composite.shape}")
                        
                        batch_timestamps.append(timestamp)
                        batch_data.append(composite)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        
        # Add to overall results
        all_timestamps.extend(batch_timestamps)
        all_composites.extend(batch_data)
        
        # Force garbage collection to free memory
        gc.collect()
        
        log_memory(f"Completed batch {batch_idx+1}. Records so far: {len(all_timestamps)}")
    
    # Convert composites to array at the end
    # This should create an array with shape (time, channel, height, width)
    composite_array = np.array(all_composites)
    print(f"Final composite array shape: {composite_array.shape}")

    
    return all_timestamps, composite_array

def interpolate_composite_data(composite_data, timestamps, regular_time_index):
    """
    Interpolate composite reflectivity data to regular time intervals.
    
    Parameters:
    -----------
    composite_data : np.ndarray
        Composite reflectivity data with shape (time, channel, height, width)
    timestamps : list
        List of datetime objects for the original data
    regular_time_index : list
        List of datetime objects for the regular time grid
    
    Returns:
    --------
    interpolated_data : np.ndarray
        Interpolated composite reflectivity with shape (len(regular_time_index), channel, height, width)
    """
    # Check shape and print detailed information
    print(f"Composite data shape: {composite_data.shape}")
    if len(composite_data.shape) != 4:
        raise ValueError(f"Expected 4D shape (time, channel, height, width), got {composite_data.shape}")
    
    b, c, h, w = composite_data.shape
    print(f"Time steps: {b}, Channels: {c}, Height: {h}, Width: {w}")
    
    # Convert timestamps to Unix timestamps
    timestamps_unix = np.array([dt.timestamp() for dt in timestamps])
    regular_time_index_unix = np.array([dt.timestamp() for dt in regular_time_index])
    
    # Reshape spatial dimensions into a single dimension for vectorized computation
    data_flat = composite_data.reshape(b, c, h*w)
    
    # Create output array
    interpolated_data_flat = np.zeros((len(regular_time_index), c, h * w), dtype=np.float32)

    print(f"Interpolating composite reflectivity to regular time grid...")
    # Process in small batches to save memory
    batch_size = 1000  # Number of pixels to process at once
    for channel in range(c):
        print(f"Interpolating channel {channel+1}/{c}")
        values = data_flat[:, channel, :]
        
        for p_start in range(0, h*w, batch_size):
            p_end = min(p_start + batch_size, h*w)
            
            # Create interpolation function for this batch of pixels
            interp_func = interp1d(
                timestamps_unix, values[:, p_start:p_end], 
                axis=0, kind='linear', fill_value='extrapolate'
            )
            
            # Apply interpolation
            interpolated_data_flat[:, channel, p_start:p_end] = interp_func(regular_time_index_unix)
            
            # Force garbage collection periodically
            if p_start % (5 * batch_size) == 0:
                gc.collect()
                log_memory(f"Channel {channel+1}, pixels {p_start}-{p_end}")

    # Reshape back to original spatial dimensions
    interpolated_data = interpolated_data_flat.reshape(len(regular_time_index), c, h, w)
    return interpolated_data

def verify_interpolation(original_composite, original_timestamps, 
                         interpolated_data, regular_time_index, 
                         verification_dir='verification_plots/'):
    """
    Verify the quality of interpolated radar data by performing multiple checks
    and generating visualization plots.
    """
    print("\n" + "="*50)
    print("INTERPOLATION VERIFICATION")
    print("="*50)
    
    # Create verification directory if it doesn't exist
    os.makedirs(verification_dir, exist_ok=True)
    
    # 1. Basic Data Integrity Checks
    print("\n1. BASIC DATA INTEGRITY CHECKS")
    print(f"Original composite data shape: {original_composite.shape}")
    print(f"Interpolated data shape: {interpolated_data.shape}")
    
    # Check for NaN or infinity values
    orig_nan_count = np.isnan(original_composite).sum()
    interp_nan_count = np.isnan(interpolated_data).sum()
    orig_inf_count = np.isinf(original_composite).sum()
    interp_inf_count = np.isinf(interpolated_data).sum()
    
    print(f"Original data NaN count: {orig_nan_count}")
    print(f"Interpolated data NaN count: {interp_nan_count}")
    print(f"Original data Inf count: {orig_inf_count}")
    print(f"Interpolated data Inf count: {interp_inf_count}")
    
    # Check value ranges
    orig_min, orig_max = np.nanmin(original_composite), np.nanmax(original_composite)
    interp_min, interp_max = np.nanmin(interpolated_data), np.nanmax(interpolated_data)
    
    print(f"Original data range: [{orig_min:.2f}, {orig_max:.2f}]")
    print(f"Interpolated data range: [{interp_min:.2f}, {interp_max:.2f}]")
    
    # 2. Statistical Distribution Comparison
    print("\n2. STATISTICAL DISTRIBUTION COMPARISON")
    
    # Sample the data for histograms to reduce memory usage
    orig_sample_size = min(100000, original_composite.size)
    interp_sample_size = min(100000, interpolated_data.size)
    
    orig_indices = np.random.choice(original_composite.size, orig_sample_size, replace=False)
    interp_indices = np.random.choice(interpolated_data.size, interp_sample_size, replace=False)
    
    orig_flat = original_composite.flatten()[orig_indices]
    interp_flat = interpolated_data.flatten()[interp_indices]
    
    # Basic statistics
    print(f"Original data (sampled) - Mean: {np.nanmean(orig_flat):.2f}, Median: {np.nanmedian(orig_flat):.2f}, Std: {np.nanstd(orig_flat):.2f}")
    print(f"Interpolated data (sampled) - Mean: {np.nanmean(interp_flat):.2f}, Median: {np.nanmedian(interp_flat):.2f}, Std: {np.nanstd(interp_flat):.2f}")
    
    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(orig_flat[~np.isnan(orig_flat)], bins=50, alpha=0.5, label='Original (sampled)')
    plt.hist(interp_flat[~np.isnan(interp_flat)], bins=50, alpha=0.5, label='Interpolated (sampled)')
    plt.title('Reflectivity Value Distribution Comparison')
    plt.xlabel('Reflectivity (dBZ)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f"{verification_dir}value_distribution.png")
    plt.close()
    print(f"Value distribution plot saved to {verification_dir}value_distribution.png")
    
    # 3. Temporal Verification
    print("\n3. TEMPORAL VERIFICATION")
    
    # Check time intervals
    orig_intervals = [(original_timestamps[i+1] - original_timestamps[i]).total_seconds() 
                      for i in range(len(original_timestamps)-1)]
    interp_intervals = [(regular_time_index[i+1] - regular_time_index[i]).total_seconds() 
                        for i in range(len(regular_time_index)-1)]
    
    print(f"Original time intervals - Min: {min(orig_intervals):.2f}s, Max: {max(orig_intervals):.2f}s, Mean: {np.mean(orig_intervals):.2f}s")
    
    if len(set(interp_intervals)) == 1:
        print(f"Interpolated data has regular intervals of {interp_intervals[0]:.2f}s")
    else:
        print(f"Warning: Interpolated data does not have constant intervals")
        print(f"Interval range: [{min(interp_intervals):.2f}s, {max(interp_intervals):.2f}s]")
    
    # 4. Spatial Structure Preservation - Sample a few frames for comparison
    print("\n4. SPATIAL STRUCTURE PRESERVATION")
    
    # Sample a few times from original and interpolated data for comparison
    num_samples = min(5, len(original_timestamps), len(regular_time_index))
    orig_indices = np.linspace(0, len(original_timestamps)-1, num_samples).astype(int)
    interp_indices = np.linspace(0, len(regular_time_index)-1, num_samples).astype(int)
    
    # Create comparison plots for each sampled time
    for i in range(num_samples):
        orig_idx = orig_indices[i]
        interp_idx = interp_indices[i]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original data
        im0 = axes[0].imshow(original_composite[orig_idx, 0], cmap='jet', vmin=0, vmax=60)
        axes[0].set_title(f"Original Composite - {original_timestamps[orig_idx]}")
        fig.colorbar(im0, ax=axes[0])
        
        # Interpolated data
        im1 = axes[1].imshow(interpolated_data[interp_idx, 0], cmap='jet', vmin=0, vmax=60)
        axes[1].set_title(f"Interpolated Composite - {regular_time_index[interp_idx]}")
        fig.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(f"{verification_dir}spatial_comparison_{i}.png")
        plt.close()
    
    print(f"Spatial structure comparison plots saved to {verification_dir}")
    
    # Final assessment
    print("\n5. FINAL ASSESSMENT")
    issues = []
    
    if orig_nan_count == 0 and interp_nan_count > 0:
        issues.append("Interpolation introduced NaN values")
    
    if orig_inf_count == 0 and interp_inf_count > 0:
        issues.append("Interpolation introduced infinity values")
    
    if interp_min < orig_min - 5 or interp_max > orig_max + 5:
        issues.append("Interpolation produced values outside the original range")
    
    if len(set(interp_intervals)) > 1:
        issues.append("Interpolated timestamps are not at regular intervals")
    
    if not issues:
        print("✓ Interpolation verification passed all checks!")
    else:
        print("⚠ Verification found these issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "verification_dir": verification_dir
    }

# Main script
read_path = "/opt/sanjeev/NOAA/python3106/lakecharles-100km/"
out_path_file = '/opt/sanjeev/NOAA/python3106/klch_radar_composite4v1layers_interpolated-100km.pkl'
verification_dir = '/opt/sanjeev/NOAA/python3106/radar-100/verification_plots_composite4v1/'
print(f"Output file: {out_path_file}")

if not os.path.exists(out_path_file):
    # Initial memory check
    log_memory("Starting script")
    
    # Get list of files
    files = os.listdir(read_path)
    sorted_files = sorted(files)
    print(f"Total files to process: {len(sorted_files)}")
    
    # Process parameters
    num_layers_for_composite = 4  # Change this value if you want to use a different number of layers
    batch_size = 25  # Reduced batch size for lower memory usage
    
    # Read files in batches and create composites on the fly
    print("Reading files and creating composites on the fly...")
    timestamps, composite_data = read_files_and_create_composite(
        sorted_files, read_path, 
        num_layers=num_layers_for_composite, 
        batch_size=batch_size
    )
    
    print(f"Successfully read {len(timestamps)} timestamps")
    log_memory("After creating composites array")
    print(f"Composite data shape: {composite_data.shape}")
    
    # Process timestamps
    print("Converting timestamps to datetime objects...")
    dt_timestamps = [datetime.strptime(ts, '%Y%m%d_%H%M%S') for ts in timestamps]
    
    # Generate a regular time series with 15-minute intervals
    start_time = dt_timestamps[0]
    end_time = dt_timestamps[-1]
    print(f"Time range: {start_time} to {end_time}")
    
    # 15 minutes = 900 seconds
    regular_time_index = [start_time + timedelta(minutes=15) * i for i in range(int((end_time - start_time).total_seconds() / 900) + 1)]
    regular_timestamps = [dt.strftime('%Y%m%d_%H%M%S') for dt in regular_time_index]
    
    print(f"Regular timestamps from {regular_timestamps[0]} to {regular_timestamps[-1]} ({len(regular_timestamps)} timestamps)")

    # Step 2: Interpolate the composite reflectivity data to regular time intervals
    print('----- Interpolating composite reflectivity data to regular time intervals -----')
    interpolated_data = interpolate_composite_data(composite_data, dt_timestamps, regular_time_index)
    
    # Verify the interpolation results
    print('----- Verifying interpolation results -----')
    verification_results = verify_interpolation(
        composite_data, dt_timestamps, 
        interpolated_data, regular_time_index, 
        verification_dir=verification_dir
    )
    
    if verification_results["passed"]:
        print("Interpolation verification passed. Saving data...")
        # Save to output file
        tmp_dic = {'data': interpolated_data, 'timestamp': regular_timestamps}
        
        with open(out_path_file, 'wb') as file:
            pickle.dump(tmp_dic, file)
        print("Data saved successfully!")
    else:
        print("Interpolation verification failed with the following issues:")
        for issue in verification_results["issues"]:
            print(f"  - {issue}")
        
        user_input = input("Do you want to save the data anyway? (y/n): ")
        if user_input.lower() == 'y':
            # Save to output file
            tmp_dic = {'data': interpolated_data, 'timestamp': regular_timestamps}
            
            with open(out_path_file, 'wb') as file:
                pickle.dump(tmp_dic, file)
            print("Data saved despite verification issues!")
        else:
            print("Data not saved. Please review the verification results.")
else:
    print(f"Output file already exists: {out_path_file}")
    
    # Option to verify existing data
    user_input = input("Do you want to verify the existing data? (y/n): ")
    if user_input.lower() == 'y':
        print("Loading existing data for verification...")
        with open(out_path_file, 'rb') as file:
            loaded_data = pickle.load(file)
            
        interpolated_data = loaded_data['data']
        timestamps = loaded_data['timestamp']
        
        # You would need the original data for comparison
        # This is just a placeholder - you'd need to reload the original data
        print("Note: Full verification requires original data, which needs to be reloaded.")
        print("Performing basic checks on the interpolated data...")
        
        # Basic checks that don't require original data
        print(f"Interpolated data shape: {interpolated_data.shape}")
        print(f"Number of timestamps: {len(timestamps)}")
        print(f"First timestamp: {timestamps[0]}")
        print(f"Last timestamp: {timestamps[-1]}")
        
        # Check for NaN or infinity values
        nan_count = np.isnan(interpolated_data).sum()
        inf_count = np.isinf(interpolated_data).sum()
        print(f"NaN count: {nan_count}")
        print(f"Infinity count: {inf_count}")
        
        # Check value range
        data_min, data_max = np.nanmin(interpolated_data), np.nanmax(interpolated_data)
        print(f"Value range: [{data_min:.2f}, {data_max:.2f}]")