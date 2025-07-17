# this script is for downloading the radar image files

import subprocess
import os

# year = 2024
year = 2022


try:
    # for month in range(1, 13):
    for month in range(7, 8):
        
        month = '0' + str(month) if int(month) < 10 else month
        print(month)
        # os.mkdir('./' + month)
        for day in range(1, 32): # 1 to 31 folders will be empty for days not there in the month
            day = '0' + str(day) if int(day) < 10 else day
            print(day)
            # os.mkdir('./' + month + '/' + day)
            
            bash_command = 'aws s3 cp s3://noaa-nexrad-level2/' + str(year) + '/' + str(month) +'/' + str(day) + '/' + 'KLCH' + ' ./' + str(year) + '/' + str(month) +'/' + str(day) + '/' + '  --recursive --no-sign-request'
            result = subprocess.run(bash_command, shell=True, check=True, capture_output=True, text=True)
            output = result.stdout.strip()
            print(f"Bash command executed successfully. Output: {output}")
except subprocess.CalledProcessError as e:
    print(f"Error running Bash command: {e}")