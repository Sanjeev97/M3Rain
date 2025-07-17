# this converts the radar data of nc files to carterian format

import subprocess
import os

year ="2024"
# month = "07"
outputPath ="./grid"

for month in range(1, 10):
    month = '0' + str(month) if int(month) < 10 else month
    
    for day in range(1, 32):  # Iterate from 1 to 31 inclusive
        day_str = f"{day:02d}"  # Format day as a two-digit string
        inputPath = f"./output/2024{month}{day_str}/"
        print("Input Path:", inputPath)

        # get all the file names in the directory
        try:
            file_names = os.listdir(inputPath)
            filtered_files = [file_name for file_name in file_names if not file_name.startswith(".")]

            print(filtered_files)
            i = 0
            for file_name in filtered_files:
                finalInputPath = inputPath + file_name

                # Define your bash command
                bash_convert = "Radx2Grid -f " + finalInputPath + " -outdir " + outputPath 

                # Run the bash command
                process_convert = subprocess.Popen(bash_convert.split(), stdout=subprocess.PIPE)

                # Get the output of the command
                output, error = process_convert.communicate()

                # Print the output
                print(output.decode())
                i = i + 1
                print(file_name + ' ' + str(i))
        except:    
            continue # skipping the days that are not in the month eg 29 for feb
