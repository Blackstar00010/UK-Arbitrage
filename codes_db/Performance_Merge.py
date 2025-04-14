import os
import pandas as pd

directory = '../files/position_LS/equal_weight_adj/'
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

for file in files:
    df1 = pd.read_csv('../files/mom1_data_combined_adj.csv')
    df2 = pd.read_csv(os.path.join(directory, file))

    # Shift the values in df2 by one column to the right
    df2.iloc[:, 1:] = df2.iloc[:, 1:].shift(periods=1, axis="columns")
    # Perform element-wise multiplication for matching rows
    performance = df1.iloc[1:,1:] * df2.iloc[1:,1:]

    # Drop any columns or rows that are completely filled with NaN values
    performance = performance.dropna(axis=1, how='all')
    performance = performance.dropna(how='all')

    # Output the result DataFrame
    print(file)
    print(performance)

    # Write the result to a new CSV file
    performance.to_csv(os.path.join('../files/position_LS/equal_weight_performance_adj/', 'performance_' + file), index=False)

'''

import os
import pandas as pd

# Read the data from the CSV files
# df1 = pd.read_csv('../files/mom1_data_combined.csv')

directory = '../files/position_LS/equal_weight_adj/'
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

for file in files:
    df1 = pd.read_csv('../files/mom1_data_combined_adj.csv')
    df2 = pd.read_csv(os.path.join(directory, file))

    # Find the common columns between df1 and df2
    common_columns = df1.T.columns.intersection(df2.T.columns)

    # Keep only the common columns in both dataframes
    df1 = df1.T[common_columns]
    df2 = df2.T[common_columns]

    print(df2)

    df1 = df1.T
    df2 = df2.T

    merged_df = pd.merge(df1, df2[['Firm Name']], on='Firm Name')
    df1 = merged_df

    # Save the first column (index column)
    first_column = df2.iloc[:, 0]

    # Shift all but the first column
    df2.iloc[:, 1:] = df2.iloc[:, 1:].shift(periods=1, axis="columns")

    # Concatenate the first column back
    df2 = pd.concat([first_column, df2.iloc[:, 1:]], axis=1)

    # Multiply only the numeric columns
    numeric_performance = df1.iloc[:, 1:].mul(df2.iloc[:, 1:])

    # Concatenate the index column with the result ToDo: this is the problem.
    performance = pd.concat([df1.iloc[:, 0], numeric_performance], axis=1)

    # Drop columns that are all NaN
    performance = performance.dropna(axis=1, how='all')

    # Write the result to a new CSV file
    performance.to_csv(os.path.join('../files/position_LS/equal_weight_performance_adj/', 'performance_' + file), index=False)
'''