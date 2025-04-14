import pandas as pd

# Creates a new table only containing the rows of dates that are first business day of the month

dir = "../files/history/"

df = pd.read_csv(dir + "price_data1986.csv")

dates = df['Date']

months = pd.DataFrame([item[5:7] for item in dates])

flags = (months == months.shift(1)).dropna()

df["Month Start Flag"] = flags

df_filtered = df[df['Month Start Flag'] == False]

df_filtered = df_filtered.drop(columns='Month Start Flag')

df_filtered.to_csv(dir + "adj_close_first_day_of_month.csv", index=False)
