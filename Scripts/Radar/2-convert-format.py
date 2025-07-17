import subprocess
import os

year = "2024"
for month in range(1, 10):
    month = '0' + str(month) if int(month) < 10 else month
    # month = "01"
    outputPath = "./output"

    for day in range(1, 32):  # Iterate from 1 to 31 inclusive
        day_str = f"{day:02d}"  # Format day as a two-digit string
        inputPath = f"./{year}/{month}/{day_str}/"
        print("Input Path:", inputPath)

        # get all the file names in the directory
        file_names = os.listdir(inputPath)

        # Remove the files with "_MDM" at the end
        filtered_files = [file_name for file_name in file_names if not file_name.endswith("_MDM")]

        print(filtered_files)

        i = 0
        for file_name in filtered_files:
            finalInputPath = "./" + year + "/" + str(month) + "/" + day_str + "/" + file_name

            # Define your bash command
            bash_convert = "RadxConvert -f " + finalInputPath + " -outdir " + outputPath 

            # Run the bash command
            process_convert = subprocess.Popen(bash_convert.split(), stdout=subprocess.PIPE)

            # Get the output of the command
            output, error = process_convert.communicate()

            # Print the output
            print(output.decode())
            i = i + 1
            print(file_name + ' ' + str(i))

