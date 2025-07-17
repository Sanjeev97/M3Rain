import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from datetime import datetime, timedelta


# ---------------------------
# 2. PROCESS PWS DATA
# ---------------------------
pws_path = "/opt/sanjeev/NOAA/MVT/data/PWS/lkch_rainv1.csv"
print("Processing weather station data...")


select_cols = [
    'imperial.tempHigh', 'imperial.tempLow', 'imperial.tempAvg',
    'humidityHigh', 'humidityLow', 'humidityAvg',
    'imperial.dewptHigh', 'imperial.dewptLow', 'imperial.dewptAvg',
    'imperial.pressureMax', 'imperial.pressureMin', 'imperial.pressureTrend',
    'imperial.windspeedHigh', 'imperial.windspeedLow', 'imperial.windspeedAvg',
    'imperial.windgustHigh', 'imperial.windgustLow', 'imperial.windgustAvg',
    'winddirAvg', "imperial.precipRate"]

# Read PWS data
wu_laf = pd.read_csv(pws_path)
df = wu_laf[2:]  # Skip two rows

# Convert timestamp to datetime
df['obsTimeUtc'] = pd.to_datetime(df['obsTimeUtc'])

# Convert timezone-aware timestamps to naive timestamps
df['obsTimeUtc'] = df['obsTimeUtc'].dt.tz_localize(None)

# Remove seconds while preserving exact minute values
df['obsTimeUtc'] = df['obsTimeUtc'].dt.floor('1min')

# Print a few examples to verify
print("\nExample timestamps after removing seconds (first 5 rows):")
for i in range(min(5, len(df))):
    print(f"Modified: {df['obsTimeUtc'].iloc[i]}")

# Check for duplicates after removing seconds
duplicates = df.duplicated(subset='obsTimeUtc', keep='first')
if duplicates.any():
    print(f"\nFound {duplicates.sum()} duplicate timestamps after removing seconds")
    print("Keeping only the first occurrence of each duplicate")
    df = df[~duplicates].copy()

# Store original timestamps for later
original_timestamps = df['obsTimeUtc'].copy()

# Extract actual timestamp minutes from the data
actual_minutes = sorted(df['obsTimeUtc'].dt.minute.unique())
print(f"Observed minute values in the data: {actual_minutes}")

# Determine if the data follows a regular 5-minute pattern (0, 5, 10, 15...)
regular_5min_pattern = sorted([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])

if set(actual_minutes).issubset(set(regular_5min_pattern)):
    print("Data follows the standard 5-minute pattern (0, 5, 10, 15...)")
    minute_pattern = regular_5min_pattern
else:
    # Find the most common minute values that appear to follow a 5-minute pattern
    minute_counts = df['obsTimeUtc'].dt.minute.value_counts()
    common_minutes = minute_counts[minute_counts > 100].index.tolist()
    print(f"Most common minute values: {sorted(common_minutes)}")
    
    # Try to identify the pattern
    if len(common_minutes) >= 10:  # If we have enough data points
        common_minutes = sorted(common_minutes)
        diffs = [common_minutes[i+1] - common_minutes[i] for i in range(len(common_minutes)-1)]
        
        if all(diff == 5 for diff in diffs) or all(diff == 5 or diff == 55 for diff in diffs):
            print(f"Data follows a 5-minute pattern with minutes: {common_minutes}")
            minute_pattern = common_minutes
        else:
            print("Unable to identify a clear 5-minute pattern. Using the standard pattern.")
            minute_pattern = regular_5min_pattern
    else:
        print("Not enough data to identify a pattern. Using the standard 5-minute pattern.")
        minute_pattern = regular_5min_pattern

# Get start and end dates
start_date = df['obsTimeUtc'].min()
end_date = df['obsTimeUtc'].max()

print(f"\nData spans from {start_date} to {end_date}")



# Generate expected timestamps based on the identified pattern
print("Generating expected timestamps based on the identified minute pattern...")
expected_timestamps = []
current = start_date

while current <= end_date:
    # Only include timestamps whose minutes match the observed pattern
    if current.minute in minute_pattern:
        # expected_timestamps.append(current)
        
        # Add 2 minutes to each timestamp
        shifted_timestamp = current + timedelta(minutes=2)
        expected_timestamps.append(shifted_timestamp)        
    
    # Move to next 5-minute interval
    current += timedelta(minutes=15)

# Create a dataframe with all expected shifted timestamps
expected_df = pd.DataFrame({'shifted_obsTimeUtc': expected_timestamps})

# Also create original timestamps for merging with original data
original_timestamps = [ts - timedelta(minutes=2) for ts in expected_timestamps]
expected_df['original_obsTimeUtc'] = original_timestamps

# Merge with original data using the original timestamps
temp_df = pd.merge(expected_df, df, left_on='original_obsTimeUtc', right_on='obsTimeUtc', how='left')

# Create the final dataframe with shifted timestamps
complete_df = temp_df.drop(['original_obsTimeUtc', 'obsTimeUtc'], axis=1)
complete_df.rename(columns={'shifted_obsTimeUtc': 'obsTimeUtc'}, inplace=True)

print(f"\nOriginal data: {len(df)} rows")
print(f"Complete time series: {len(complete_df)} rows")
print(f"Missing intervals: {len(complete_df) - len(df)} rows")

# Identify gaps (NaN values)
print("\nIdentifying gaps...")
gap_mask = complete_df.isna().any(axis=1)
gap_rows = complete_df[gap_mask]
print(f"Found {len(gap_rows)} rows with missing values")

# Identify columns that need to be filled
non_timestamp_cols = [col for col in complete_df.columns if col != 'obsTimeUtc']
columns_to_fill = [col for col in non_timestamp_cols if complete_df[col].isna().any()]
print(f"Columns to fill: {len(columns_to_fill)}")

# Group columns by filling method
# 1. Numerical continuous variables that should be interpolated
interpolate_cols = [
    'imperial.tempAvg', 'imperial.tempHigh', 'imperial.tempLow',
    'imperial.dewptAvg', 'imperial.dewptHigh', 'imperial.dewptLow',
    'imperial.pressureMax', 'imperial.pressureMin', 'imperial.pressureTrend',
    'humidityAvg', 'humidityHigh', 'humidityLow'
]

# 2. Wind-related variables that need vector decomposition
wind_cols = [
    'winddirAvg', 'imperial.windspeedAvg', 'imperial.windspeedHigh', 'imperial.windspeedLow',
    'imperial.windgustAvg', 'imperial.windgustHigh', 'imperial.windgustLow'
]

# 3. Precipitation variables that need special handling
precip_cols = [
    'imperial.precipRate', 'imperial.precipTotal'
]

# 4. Other variables that can be forward-filled or backfilled
other_cols = [col for col in columns_to_fill if col not in interpolate_cols + wind_cols + precip_cols]
print(f"Other columns to fill: {other_cols}")

print("\nFilling gaps in different variable types...")

# 1. Fill continuous numerical variables using spline interpolation
if set(interpolate_cols).intersection(columns_to_fill):
    print("Filling meteorological variables with interpolation...")
    interp_cols = [col for col in interpolate_cols if col in columns_to_fill]
    
    for col in interp_cols:
        # Use pandas interpolation with cubic spline for smooth transitions
        complete_df[col] = complete_df[col].interpolate(method='cubic', limit_direction='both')
        
        # Check if any NaN values remain (at edges)
        if complete_df[col].isna().any():
            # Fill remaining NaNs with nearest valid values
            complete_df[col] = complete_df[col].interpolate(method='nearest', limit_direction='both')
            
# 2. Handle wind direction and speed
if set(wind_cols).intersection(columns_to_fill):
    print("Filling wind variables with vector decomposition and interpolation...")
    
    # Check if we have wind direction and at least one wind speed
    if 'winddirAvg' in columns_to_fill and any(col in columns_to_fill for col in wind_cols if 'windspeed' in col):
        # Convert wind direction to radians
        theta = np.deg2rad(complete_df['winddirAvg'])
        
        # Create temporary columns for U and V components
        # Only use records where both direction and speed are available
        wind_speed_col = next((col for col in wind_cols if 'windspeed' in col and col in complete_df.columns), None)
        
        if wind_speed_col:
            # Calculate U and V only where both values are available
            mask = (~complete_df['winddirAvg'].isna()) & (~complete_df[wind_speed_col].isna())
            
            # Initialize U and V columns with NaN
            complete_df['U'] = np.nan
            complete_df['V'] = np.nan
            
            complete_df.loc[mask, 'U'] = -complete_df.loc[mask, wind_speed_col] * np.sin(theta[mask])
            complete_df.loc[mask, 'V'] = -complete_df.loc[mask, wind_speed_col] * np.cos(theta[mask])
            
            # Interpolate the U and V components
            complete_df['U'] = complete_df['U'].interpolate(method='linear', limit_direction='both')
            complete_df['V'] = complete_df['V'].interpolate(method='linear', limit_direction='both')
            
            # Convert back to speed and direction
            speed = np.sqrt(complete_df['U']**2 + complete_df['V']**2)
            direction = (np.rad2deg(np.arctan2(-complete_df['U'], -complete_df['V'])) + 360) % 360
            
            # Fill in the wind columns
            if 'winddirAvg' in columns_to_fill:
                complete_df['winddirAvg'] = np.where(complete_df['winddirAvg'].isna(), direction, complete_df['winddirAvg'])
            
            for col in wind_cols:
                if 'windspeed' in col and col in columns_to_fill:
                    # Scale the speed for different wind measures
                    if 'High' in col:
                        factor = 1.5  # Higher for gusts
                    elif 'Low' in col:
                        factor = 0.5  # Lower for minimum
                    else:
                        factor = 1.0  # Average
                    
                    complete_df[col] = np.where(complete_df[col].isna(), speed * factor, complete_df[col])
            
            # Drop temporary columns
            complete_df.drop(['U', 'V'], axis=1, inplace=True)
    
    # Handle remaining wind columns
    for col in wind_cols:
        if col in columns_to_fill and complete_df[col].isna().any():
            # For any remaining NaN values, use linear interpolation
            complete_df[col] = complete_df[col].interpolate(method='linear', limit_direction='both')
            
            # If still have NaNs at the edges, use backfill/forward fill
            if complete_df[col].isna().any():
                complete_df[col] = complete_df[col].fillna(method='ffill').fillna(method='bfill')            
                
                
# 3. Handle precipitation data
if set(precip_cols).intersection(columns_to_fill):
    print("Filling precipitation variables...")
    
    # For precipitation rate, we use nearest neighbor interpolation or zeros
    if 'imperial.precipRate' in columns_to_fill:
        # Fill with zeros for small gaps (no rain is more likely than rain)
        gap_size = complete_df['imperial.precipRate'].isna().sum()
        if gap_size < 100:  # For smaller gaps, assume no rain
            complete_df['imperial.precipRate'] = complete_df['imperial.precipRate'].fillna(0)
        else:
            # For larger gaps, use a more sophisticated approach
            # First, find periods where it was raining
            rain_periods = complete_df['imperial.precipRate'] > 0
            
            # If there are rain periods around a gap, interpolate
            if rain_periods.any():
                # Create a temporary series for targeted interpolation
                temp_series = complete_df['imperial.precipRate'].copy()
                
                # Find gaps within rain periods or close to rain
                rolling = rain_periods.rolling(10, min_periods=1).sum().fillna(0)
                rain_gaps = complete_df['imperial.precipRate'].isna() & (rolling > 0)
                
                # Interpolate only within rain periods, keep other gaps as 0
                if rain_gaps.any():
                    temp_series.loc[~rain_gaps] = complete_df.loc[~rain_gaps, 'imperial.precipRate']
                    temp_series = temp_series.interpolate(method='linear', limit=6)
                
                # Fill remaining with zeros (no rain)
                complete_df['imperial.precipRate'] = temp_series.fillna(0)
            else:
                # If no rain periods, fill all with zeros
                complete_df['imperial.precipRate'] = complete_df['imperial.precipRate'].fillna(0)
    
    complete_df['imperial.precipRate'] = complete_df['imperial.precipRate'] * 25.4       
                
                
def validate_filled_data(df):
    """
    Validate and correct weather data based on realistic value ranges
    and logical consistency rules, specifically for a simplified schema.
    
    Parameters:
    df (pd.DataFrame): The filled weather data
    """
    # 1. Define realistic value ranges for this dataset
    valid_ranges = {
        'imperial.tempHigh': (-30, 120),
        'imperial.tempLow': (-30, 120),
        'imperial.tempAvg': (-30, 120),
        'humidityHigh': (0, 100),
        'humidityLow': (0, 100),
        'humidityAvg': (0, 100),
        'imperial.dewptHigh': (-30, 100),
        'imperial.dewptLow': (-30, 100),
        'imperial.dewptAvg': (-30, 100),
        'imperial.pressureMax': (27, 32),
        'imperial.pressureMin': (27, 32),
        'imperial.pressureTrend': (-1, 1),
        'imperial.windspeedHigh': (0, 150),
        'imperial.windspeedLow': (0, 150),
        'imperial.windspeedAvg': (0, 150),
        'imperial.windgustHigh': (0, 200),
        'imperial.windgustLow': (0, 200),
        'imperial.windgustAvg': (0, 200),
        'winddirAvg': (0, 360)
    }

    # 2. Clip out-of-range values
    for col, (min_val, max_val) in valid_ranges.items():
        
        if col in df.columns:
            print(f"Checking {col} for out-of-range values...")
            too_low = df[col] < min_val
            too_high = df[col] > max_val

            if too_low.any() or too_high.any():
                print(f"Correcting out-of-range values in {col}:")
                print(f"  Values < {min_val}: {too_low.sum()}")
                print(f"  Values > {max_val}: {too_high.sum()}")
                df[col] = df[col].clip(min_val, max_val)

    # 3. Logical consistency: tempHigh ≥ tempAvg ≥ tempLow
    temp_cols = ['imperial.tempHigh', 'imperial.tempAvg', 'imperial.tempLow']
    if all(col in df.columns for col in temp_cols):
        inconsistent = ((df['imperial.tempHigh'] < df['imperial.tempAvg']) |
                        (df['imperial.tempAvg'] < df['imperial.tempLow']))
        if inconsistent.any():
            print(f"Fixing {inconsistent.sum()} inconsistent temperature relationships")
            for idx in df[inconsistent].index:
                high = df.loc[idx, 'imperial.tempHigh']
                low = df.loc[idx, 'imperial.tempLow']
                if high < low:
                    high, low = low, high
                    df.loc[idx, 'imperial.tempHigh'] = high
                    df.loc[idx, 'imperial.tempLow'] = low
                df.loc[idx, 'imperial.tempAvg'] = (high + low) / 2

    # 4. Logical consistency: dewptHigh ≥ dewptAvg ≥ dewptLow
    dewpt_cols = ['imperial.dewptHigh', 'imperial.dewptAvg', 'imperial.dewptLow']
    if all(col in df.columns for col in dewpt_cols):
        inconsistent = ((df['imperial.dewptHigh'] < df['imperial.dewptAvg']) |
                        (df['imperial.dewptAvg'] < df['imperial.dewptLow']))
        if inconsistent.any():
            print(f"Fixing {inconsistent.sum()} inconsistent dew point relationships")
            for idx in df[inconsistent].index:
                high = df.loc[idx, 'imperial.dewptHigh']
                low = df.loc[idx, 'imperial.dewptLow']
                if high < low:
                    high, low = low, high
                    df.loc[idx, 'imperial.dewptHigh'] = high
                    df.loc[idx, 'imperial.dewptLow'] = low
                df.loc[idx, 'imperial.dewptAvg'] = (high + low) / 2

    # 5. Wind gust must be ≥ wind speed
    if all(col in df.columns for col in ['imperial.windgustAvg', 'imperial.windspeedAvg']):
        inconsistent = df['imperial.windgustAvg'] < df['imperial.windspeedAvg']
        if inconsistent.any():
            print(f"Fixing {inconsistent.sum()} cases where gustAvg < windspeedAvg")
            df.loc[inconsistent, 'imperial.windgustAvg'] = df.loc[inconsistent, 'imperial.windspeedAvg']

    return df

complete_df.drop(columns=other_cols, inplace=True)
complete_df.drop(columns=['imperial.precipTotal'], inplace=True)

# Verify no NaN values remain
remaining_nans = complete_df.isna().sum().sum()
if remaining_nans > 0:
    print(f"Warning: {remaining_nans} NaN values remain after filling")
    print("Columns with remaining NaNs:")
    for col in complete_df.columns:
        nan_count = complete_df[col].isna().sum()
        if nan_count > 0:
            print(f"  {col}: {nan_count} NaNs")
    
    # Fill any remaining NaNs with appropriate defaults
    complete_df = complete_df.fillna(0)
else:
    print("All gaps successfully filled!")                         
    
print(complete_df.columns)    

# Save to file if output path is provided
output_path = '/opt/sanjeev/NOAA/MVT/data/PWS/filled_weather_data_15T_20P.csv'
print(f"Saving filled data to {output_path}")
# complete_df[:48180].to_csv(output_path, index=False)
complete_df[:96359].to_csv(output_path, index=False)