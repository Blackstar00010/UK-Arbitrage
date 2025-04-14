import os
import pandas as pd

directory = '../files/Clustering/K-Means'
long_short = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

merged_df = pd.DataFrame()

for file in long_short:
    data = pd.read_csv(os.path.join(directory, file))

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

merged_df.to_csv('../files/combined_LS_data.csv', index=False)
