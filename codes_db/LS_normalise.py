import os
import pandas as pd

directory = '../files/position_LS/equal_weight'
files = [f for f in os.listdir(directory) if f.endswith('.csv')]

for file in files:
    df = pd.read_csv(os.path.join(directory, file), index_col=0)

    columns = df.columns

    # Normalize each column
    for column in columns:
        df[column] = df[column] / df[column].abs().sum()

    file_name, file_extension = os.path.splitext(file)

    df.to_csv(os.path.join(directory, file_name + '_normalized' + file_extension))

