# extraction of data for lafayette and saving to file

# from matplotlib import pyplot as plt
import numpy as np
# import pandas as pd
import netCDF4
import os
import pickle


from datetime import date, timedelta

def calculate_index(lon_m, lat_m, lon_p, lat_p):
    '''
    give the longitude and latitude postion, (lon_p, lat_p)
    calculate the nearest point in an area defined by lon_m and lat_m
    return 
        a 25x25 area, where the nearest point is the center
    '''
    distances = np.sqrt((lon_m - lon_p)**2 + (lat_m - lat_p)**2)
    min_distance_index = np.unravel_index(np.argmin(distances), distances.shape)
    return min_distance_index

def fetch_area_data(matrix, center, n=25):
    '''
    extract a sub-area around center, size = nxn
    :param matrix: original matrix
    :param center: the index of the center
    :param n: the size of the sub area on one side
    :return: the subarea
    '''

    x, y = center
    half_size = n // 2
    x_start = max(x - half_size, 0)
    y_start = max(y - half_size, 0)
    x_end = min(x + half_size + 1, matrix.shape[2])
    y_end = min(y + half_size + 1, matrix.shape[3])
    return matrix[:, :, x_start:x_end, y_start:y_end]

def fetch_lon_lat_data(lon_m, lat_m, center, n=25):
    '''
    extract longitude and latitude of sub-area around center, size = nxn
    :param lon_m: longitude matrix
    :param lat_m: latitude matrix
    :param center: the index of the center
    :param n: the size of the sub area on one side
    :return: the grid longitude and latitude
    '''
    x, y = center
    half_size = n // 2
    x_start = max(x - half_size, 0)
    y_start = max(y - half_size, 0)
    x_end = min(x + half_size + 1, lon_m.shape[0])
    y_end = min(y + half_size + 1, lon_m.shape[1])
    return lon_m[x_start:x_end, y_start:y_end], lat_m[x_start:x_end, y_start:y_end]



loc = [30.12608, -93.22342]

params = ["REF", "VEL"]
# params = ["REF", "VEL", "SW", "ZDR", "PHI", "RHO","CFP", "PURPLE_HAZE"]

# size = 32 # by default netcdf4 grid is 1km*1km -> 25km*25km
size = 100 # by default netcdf4 grid is 1km*1km -> 25km*25km

# month = "08"
year = "2024"
for month in range(1, 13):
    month = '0' + str(month) if int(month) < 10 else month
    
    for day in range(1, 32):  # Iterate from 1 to 31 inclusive
        day_str = f"{day:02d}"  # Format day as a two-digit string
        inputPath = f"./grid/{year}{month}{day_str}/"
        print("Input Path:", inputPath)

        out_path = "./python3106/lakecharles-32km/" + year + "/" + year + str(month) + day_str
        print("Output Path:", out_path)

        if not os.path.exists(inputPath):
            print('read path does not exist')
            break
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        file_names = os.listdir(inputPath)
        filtered_files = [file_name for file_name in file_names if not file_name.startswith(".")]
        print(len(filtered_files))

        dayData = []
        for f in sorted(filtered_files):
            tmp_dic = {}
            file_path = inputPath+f
            nc = netCDF4.Dataset(file_path)

            loc_index = calculate_index(nc["lat0"][:], nc["lon0"][:], loc[0], loc[1])
            
            for param in params:
                subarea = fetch_area_data(nc[param][:], loc_index, n=size)
                tmp_dic[param] = subarea
            subarea_lon, subarea_lat = fetch_lon_lat_data(nc["lon0"][:],nc["lat0"][:], loc_index, n=size)
            tmp_dic['lon'] = subarea_lon
            tmp_dic['lat'] = subarea_lat
            # print(tmp_dic["REF"])
            time = f.split('_')[1] + '_' + f.split('_')[2].split('.')[0]
            tmp_dic['time'] = time
            print(tmp_dic['time'])

            dayData.append(tmp_dic)
            # break
        out_path_file = out_path + '.pkl'
        with open(out_path_file, 'wb') as file:
            pickle.dump(dayData, file)
            # break
        print(len(dayData))
            # current_date += timedelta(days=1)
            # break    
