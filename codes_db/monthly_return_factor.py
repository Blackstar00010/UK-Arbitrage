import pandas as pd

df = pd.read_csv('../files/monthly_return.csv', header=None)

dates = pd.to_datetime(df.iloc[1])
values = pd.to_numeric(df.iloc[2])

return_factor = values / values.shift() - 1

return_factor_df = pd.DataFrame({
    'Date': dates,
    'Return Factor': return_factor
})

return_factor_df.T.to_csv('../files/month_return.csv', index=False)
