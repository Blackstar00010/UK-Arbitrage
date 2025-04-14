import os
import pandas as pd

base_directory = '../files/Clustering_adj/'
output_directory = '../files/position_LS/equal_weight_adj/'

# Get all subdirectories in the base directory
subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

for subdir in subdirectories:
    directory = os.path.join(base_directory, subdir)
    long_short = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

    merged_df = pd.DataFrame() 

    for file in long_short:
        data = pd.read_csv(os.path.join(directory, file))

        if data.empty:
            continue

        # Keep only the 'Firm Name' and 'Long Short' columns
        data = data[['Firm Name', 'Long Short']]

        # Change the column name into file name (ex: 1990-01)
        file_column_name = os.path.splitext(file)[0]
        data = data.rename(columns={'Long Short': file_column_name})

        if merged_df.empty:
            merged_df = data
        else:
            merged_df = pd.merge(merged_df, data, on='Firm Name', how='outer')

    merged_df = merged_df.sort_values('Firm Name')

    # Save the merged DataFrame to a CSV file in the output directory
    merged_df.to_csv(os.path.join(output_directory, f'{subdir}_combined_LS.csv'), index=False)
