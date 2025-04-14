import pandas as pd

# Load the CSV file
df = pd.read_csv('../files/history/index_close_m.csv')

# Convert the date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set the date range
start_date = '1990-01-01'
end_date = '2022-12-01'

# Filter the data
mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
filtered_df = df.loc[mask]
filtered_df = filtered_df.T
filtered_df = filtered_df.iloc[:2]

# Save the filtered data to a new CSV file
filtered_df.to_csv('../files/monthly_return.csv', index=False)
