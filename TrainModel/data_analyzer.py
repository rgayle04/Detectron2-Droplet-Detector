import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import glob
import numpy as np
import sys
from pathlib import Path 
from collections import defaultdict

extensions = ['*.csv','*.xl','*.xlsx', '*.xlsm']

def analyze(csv_path, opath):
    print(f'{csv_path}')
    os.makedirs(opath, exist_ok=True)

    directories = opath.split("\\")
    print(directories[3])

    output = os.path.join(opath, f"Summarized {directories[3]} .csv")

    csvFiles = []
    for ext in extensions:
        csvFiles.extend(glob.glob(os.path.join(csv_path, ext)))
    csvFiles = sorted(csvFiles)

    # Group files by name (excluding last 3 characters)
    grouped_files = defaultdict(list)
    
    for csv_file in csvFiles:
        name = Path(csv_file).stem
        # Group by name without last 3 characters
        group_key = name[:-3] if len(name) > 3 else name
        grouped_files[group_key].append(csv_file)

    header = True
    if os.path.exists(output) and os.path.getsize(output) > 0:
        header = False
    
    with open(output, 'a', newline='') as outfile:
        if header:
            outfile.write('Video Name,Linear Permeability,3rd Deg Permeability,Linear Std,3rd Deg Poly Std,Contact Angle, File Count\n')

        # Process each group
        for group_name, file_list in sorted(grouped_files.items()):
            print(f'Processing group: {group_name} with {len(file_list)} files')
            
            linear_values = []
            poly_values = []
            CA_values = []
            i = 0
            for csv_file in file_list:
                name = Path(csv_file).stem
                print(f'  - {csv_file}')
                df = pd.read_csv(csv_file)
                linear = df.at[0, 'Permeability (avg DIB Rad)']
                poly = df.at[0, 'Permeability (slope)']
                linear = abs(linear)
                poly = abs(poly)
                linear_values.append(linear)
                poly_values.append(poly)
                CA = df['Contact Angle'].mean()
                CA_values.append(CA)
                outfile.write(f'{name}, {linear},{poly},{""}, {""}, {CA}\n')
                i+=1
                
            if len(file_list) > 1:
            # Calculate mean and standard deviation
                linear_mean = np.mean(linear_values)
                linear_std = np.std(linear_values, ddof= 1)
                poly_mean = np.mean(poly_values)
                poly_std = np.std(poly_values, ddof= 1)
                CA_mean = np.mean(CA_values)
                file_count = len(file_list)
                
                outfile.write(f'{group_name},{linear_mean},{poly_mean},{linear_std},{poly_std},{CA_mean},{file_count}\n')
                if i == len(file_list):
                    outfile.write(f'\n')
            else:
                outfile.write(f'\n')


def main(csv_path, opath):
    analyze(csv_path, opath)


if __name__ == "__main__":
    csv_path = sys.argv[1]
    opath = sys.argv[2]

    main(csv_path, opath)
