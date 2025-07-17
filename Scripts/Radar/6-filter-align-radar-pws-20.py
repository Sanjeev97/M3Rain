import numpy as np
import h5py 
from tqdm import tqdm
from PIL import Image
import pickle
import pandas as pd
from scipy.interpolate import interp1d

# Constants
TRAIN_TEST_SPLIT = 0.85  # 85% train, 15% test
THRESHOLD = 3.0
# data_path = "/opt/sanjeev/NOAA/MVT/data/klch_radar_interpolated.pkl"
data_path = "/opt/sanjeev/NOAA/python3106/klch_radar_composite4v1layers_interpolated-100km.pkl"
h5_data = "/opt/sanjeev/NOAA/MVT/data/klch_radar_pws_aligned_100km_20P_composite4.h5"
pws_path = "/opt/sanjeev/NOAA/MVT/data/PWS/filled_weather_data_15T_20P.csv"

print("Starting data processing...")

# Create HDF5 file
h5_file = h5py.File(h5_data, 'w')
h5_file.create_group('train')
h5_file.create_group('test')
h5_file.create_group('pws')
h5_file.create_group('mapping')

# ---------------------------
# 1. PROCESS RADAR DATA
# ---------------------------

dataDic = {}   
with open(data_path, 'rb') as file:
    dataDic = pickle.load(file)
    data = dataDic['data'][24:]
    timestamps = dataDic['timestamp'][24:]

print("----------Radar data shape-----------")
print(data.shape)
print("----------Radar start date-----------")
print(timestamps[0])
print("----------Radar end date-----------")
print(timestamps[-1])

def dbz_to_uint8(Z):
    """
    Convert reflectivity values to data values following these rules:
    Level 0: Z < 8 dBZ -> 0
    Level 1: 8 ≤ Z < 16 dBZ -> 8
    Level 2: 16 ≤ Z < 20 dBZ -> 16
    Level N: 17+N < Z < 18+N dBZ -> 17+N
    Level 53: Z ≥ 70 dBZ -> 70
    Missing data -> 255
    """
    result = np.zeros_like(Z, dtype=np.uint8)
    
    # Base levels
    result[(Z < 8)] = 0
    result[(Z >= 8) & (Z < 16)] = 8
    result[(Z >= 16) & (Z < 20)] = 16
    
    # Variable levels (N)
    for n in range(53):  # Goes up to level 53 (before 70 dBZ threshold)
        level_value = 17 + n
        result[(Z > level_value) & (Z < level_value + 1)] = level_value
    
    # Highest level
    result[Z >= 70] = 70
    
    # Missing data
    result[np.isnan(Z)] = 255
    
    return result

# Process radar data
print("Processing radar data...")
long_seq = []
for x, y in tqdm(zip(data, timestamps)):
    # image = Image.fromarray(dbz_to_uint8(x[1,:,:]))  # taking the 1st layer of radar plot or the only layer for composite interpolated radar
    image = Image.fromarray(dbz_to_uint8(x[0,:,:]))  # taking the only layer for composite interpolated radar
    grayscale_image = image.convert('L')
    image_array = np.array(grayscale_image)
    long_seq.append((image_array, y))

seq_len = 0
min_vals = 30000
min_key = ""

# Extract sequences with sufficient reflectivity
print("Extracting significant radar sequences...")
i = 4
final_seq = []
final_dates = []
while i + 4 < len(long_seq):
    if long_seq[i][0].mean() > THRESHOLD:  # mean of each frame
        seq = long_seq[i-4:i+4]
        all_means = sum([frame.mean() for frame, _ in seq])  # just taking the frame for means
        key = str(seq_len)
        if all_means < min_vals:
            min_vals = all_means
            min_key = key
        
        final_seq.append([frame for frame, _ in seq])  # Store just the frames
        final_dates.append([date for _, date in seq])  # Store the corresponding dates
        
        i = i + 4
        seq_len += 1
        continue
    i = i + 4

print(f"Total radar sequences extracted: {seq_len}")
print(f"Minimum mean value: {min_vals}")
print(f"Key with minimum value: {min_key}")

# Check if data is chronologically ordered
is_ordered = all(final_dates[i][0] <= final_dates[i+1][0] for i in range(len(final_dates)-1))
print(f"Radar data is chronologically ordered: {is_ordered}")

# Split into train and test without sorting
split_idx = int(seq_len * TRAIN_TEST_SPLIT)

train_seq = final_seq[:split_idx]
train_dates = final_dates[:split_idx]
test_seq = final_seq[split_idx:]
test_dates = final_dates[split_idx:]

# ---------------------------
# 2. PROCESS PWS DATA
# ---------------------------

print("Processing weather station data...")

# Read PWS data
wu_laf = pd.read_csv(pws_path)
wu_laf = wu_laf[2:]  # Skip header rows

select_cols = column_names = [
    "winddirAvg",
    "humidityHigh",
    "humidityLow",
    "humidityAvg",
    "imperial.tempHigh",
    "imperial.tempLow",
    "imperial.tempAvg",
    "imperial.windspeedHigh",
    "imperial.windspeedLow",
    "imperial.windspeedAvg",
    "imperial.windgustHigh",
    "imperial.windgustLow",
    "imperial.windgustAvg",
    "imperial.dewptHigh",
    "imperial.dewptLow",
    "imperial.dewptAvg",
    "imperial.pressureMax",
    "imperial.pressureMin",
    "imperial.pressureTrend",
    "imperial.precipRate"
]

pws_np = wu_laf[select_cols].values
print(pws_np)
pws_column_names = select_cols # Exclude the first column (timestamp)
shifted_timestamps_15min = pd.to_datetime(wu_laf['obsTimeUtc']).values
pws_ids = wu_laf['obsTimeUtc'].values

# ---------------------------
# 3. ALIGN AND SAVE DATA
# ---------------------------

print("Aligning radar and PWS data...")
# Create groups for aligned data
h5_file.create_group('aligned_train')
h5_file.create_group('aligned_test')

# Create a mapping between radar timestamps and PWS timestamps
radar_to_pws_mapping = {}

for seq_idx, seq_dates in enumerate(tqdm(final_dates, desc="Creating timestamp mappings")):
    radar_to_pws_mapping[seq_idx] = []
    
    for frame_idx, radar_date in enumerate(seq_dates):
        # Convert to pd.Timestamp for comparison if needed
        if not isinstance(radar_date, pd.Timestamp):
            radar_date = pd.to_datetime(radar_date, format="%Y%m%d_%H%M%S")
        
        # Find closest PWS timestamp
        time_diffs = np.abs([(radar_date - ts).total_seconds() for ts in shifted_timestamps_15min])
        closest_pws_idx = np.argmin(time_diffs)
        
        # Store mapping (radar frame index, PWS timestamp index)
        radar_to_pws_mapping[seq_idx].append((frame_idx, closest_pws_idx))
        
        # Optional: print time difference for the first few sequences
        if seq_idx < 2:
            print(f"Seq {seq_idx}, Frame {frame_idx}: Radar time {radar_date}, closest PWS time {shifted_timestamps_15min[closest_pws_idx]}, diff: {time_diffs[closest_pws_idx]:.1f} seconds")

# Save radar sequences to HDF5 file and create aligned data
print("Saving radar training data and creating aligned sequences...")
for i, seq in enumerate(tqdm(train_seq)):
    # Save original radar sequence
    h5_file['train'].create_dataset(str(i), data=np.array(seq), dtype='uint8', compression='lzf')
    
    # Get corresponding PWS data for this sequence
    pws_indices = [mapping[1] for mapping in radar_to_pws_mapping[i]]
    aligned_pws_data = pws_np[pws_indices]
    
    # Create aligned data group and save both radar and PWS data
    aligned_group = h5_file.create_group(f'aligned_train/{i}')
    aligned_group.create_dataset('radar', data=np.array(seq), dtype='uint8', compression='lzf')
    aligned_group.create_dataset('pws', data=aligned_pws_data, dtype='float32', compression='lzf')
    
    # Store timestamp information as attributes
    date_strings = [str(date) for date in train_dates[i]]
    aligned_group.attrs['dates'] = date_strings
    aligned_group.attrs['pws_indices'] = pws_indices

print("Saving radar testing data and creating aligned sequences...")
for i, seq in enumerate(tqdm(test_seq)):
    # Save original radar sequence
    h5_file['test'].create_dataset(str(i), data=np.array(seq), dtype='uint8', compression='lzf')
    
    # Get the original index in final_seq
    seq_idx = i + split_idx
    
    # Get corresponding PWS data for this sequence
    pws_indices = [mapping[1] for mapping in radar_to_pws_mapping[seq_idx]]
    aligned_pws_data = pws_np[pws_indices]
    
    # Create aligned data group and save both radar and PWS data
    aligned_group = h5_file.create_group(f'aligned_test/{i}')
    aligned_group.create_dataset('radar', data=np.array(seq), dtype='uint8', compression='lzf')
    aligned_group.create_dataset('pws', data=aligned_pws_data, dtype='float32', compression='lzf')
    
    # Store timestamp information as attributes
    date_strings = [str(date) for date in test_dates[i]]
    aligned_group.attrs['dates'] = date_strings
    aligned_group.attrs['pws_indices'] = pws_indices

# Store radar metadata
h5_file['train'].create_dataset('all_len', data=len(train_seq))
h5_file['test'].create_dataset('all_len', data=len(test_seq))
h5_file['aligned_train'].attrs['all_len'] = len(train_seq)
h5_file['aligned_test'].attrs['all_len'] = len(test_seq)

# Store complete PWS data
print("Saving complete PWS data...")
h5_file['pws'].create_dataset('data', data=pws_np, dtype='float32', compression='lzf')
h5_file['pws'].create_dataset('timestamps', data=np.array(pws_ids, dtype=h5py.string_dtype()), compression='lzf')

# Store the mappings for reference
print("Saving data mappings...")
for seq_idx, mappings in radar_to_pws_mapping.items():
    # Convert to array for storage
    mapping_array = np.array(mappings, dtype=np.int32)
    
    # Store in train or test based on sequence index
    if seq_idx < split_idx:
        h5_file['mapping'].create_dataset(f'train_{seq_idx}', data=mapping_array, compression='lzf')
    else:
        h5_file['mapping'].create_dataset(f'test_{seq_idx-split_idx}', data=mapping_array, compression='lzf')

# Store radar dates as attributes (keeping this for backward compatibility)
print("Saving timestamp attributes...")
for i, dates in enumerate(train_dates):
    date_strings = [str(date) for date in dates]
    h5_file['train'][str(i)].attrs['dates'] = date_strings

for i, dates in enumerate(test_dates):
    date_strings = [str(date) for date in dates]
    h5_file['test'][str(i)].attrs['dates'] = date_strings

h5_file.close()
print("Data successfully saved to HDF5 file")

# ---------------------------
# 4. VERIFICATION FUNCTION
# ---------------------------

def verify_alignment(h5_file_path, num_samples=3):
    """
    Function to verify the alignment by displaying a few samples from the aligned data
    """
    with h5py.File(h5_file_path, 'r') as h5f:
        pws_data = h5f['pws']['data'][()]
        pws_timestamps = h5f['pws']['timestamps'][()]
        
        print("\nVerification of Data Alignment:")
        print("=" * 50)
        
        # Check a few aligned train samples
        train_len = h5f['aligned_train'].attrs['all_len']
        for i in range(min(num_samples, train_len)):
            group_path = f'aligned_train/{i}'
            if group_path in h5f:
                aligned_group = h5f[group_path]
                radar_data = aligned_group['radar'][()]
                pws_data_aligned = aligned_group['pws'][()]
                radar_dates = aligned_group.attrs['dates']
                pws_indices = aligned_group.attrs['pws_indices']
                
                print(f"\nAligned Train Sample {i}:")
                print(f"  Radar sequence shape: {radar_data.shape}")
                print(f"  PWS data shape: {pws_data_aligned.shape}")
                print(f"  First radar timestamp: {radar_dates[0]}")
                print(f"  Last radar timestamp: {radar_dates[-1]}")
                
                # Show aligned PWS data for first and last frame
                first_pws_idx = pws_indices[0]
                last_pws_idx = pws_indices[-1]
                
                print(f"  First frame radar reflectivity mean: {np.mean(radar_data[0]):.2f}")
                print(f"  First frame PWS data (rate, total): {pws_data_aligned[0]} at {pws_timestamps[first_pws_idx]}")
                print(f"  Last frame radar reflectivity mean: {np.mean(radar_data[-1]):.2f}")
                print(f"  Last frame PWS data (rate, total): {pws_data_aligned[-1]} at {pws_timestamps[last_pws_idx]}")
        
        # Check a few aligned test samples
        test_len = h5f['aligned_test'].attrs['all_len']
        for i in range(min(num_samples, test_len)):
            group_path = f'aligned_test/{i}'
            if group_path in h5f:
                aligned_group = h5f[group_path]
                radar_data = aligned_group['radar'][()]
                pws_data_aligned = aligned_group['pws'][()]
                radar_dates = aligned_group.attrs['dates']
                pws_indices = aligned_group.attrs['pws_indices']
                
                print(f"\nAligned Test Sample {i}:")
                print(f"  Radar sequence shape: {radar_data.shape}")
                print(f"  PWS data shape: {pws_data_aligned.shape}")
                print(f"  First radar timestamp: {radar_dates[0]}")
                print(f"  Last radar timestamp: {radar_dates[-1]}")
                
                # Show aligned PWS data for first and last frame
                first_pws_idx = pws_indices[0]
                last_pws_idx = pws_indices[-1]
                
                print(f"  First frame radar reflectivity mean: {np.mean(radar_data[0]):.2f}")
                print(f"  First frame PWS data (rate, total): {pws_data_aligned[0]} at {pws_timestamps[first_pws_idx]}")
                print(f"  Last frame radar reflectivity mean: {np.mean(radar_data[-1]):.2f}")
                print(f"  Last frame PWS data (rate, total): {pws_data_aligned[-1]} at {pws_timestamps[last_pws_idx]}")
                
        print("\nAlignment verification complete.")

# Run verification
verify_alignment(h5_data, num_samples=2)

print("\nComplete processing finished successfully!")