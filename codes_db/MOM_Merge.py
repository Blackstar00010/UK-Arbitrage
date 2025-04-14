import os
import pandas as pd

directory = '../files/momentum_adj_close'
long_short = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

merged_df = pd.DataFrame()

for file in long_short:
    data = pd.read_csv(os.path.join(directory, file))

    # Keep only the 'Momentum Index' and '1' columns
    data = data[['Momentum Index', '1']]

    file_column_name = os.path.splitext(file)[0]

    # Rename the columns
    data = data.rename(columns={'Momentum Index': 'Firm Name', '1': file_column_name})

    if merged_df.empty:
        merged_df = data
    else:
        merged_df = pd.merge(merged_df, data, on='Firm Name', how='outer')

merged_df = merged_df.sort_values('Firm Name')

merged_df.to_csv('../files/mom1_data_combined_adj_close.csv', index=False)
