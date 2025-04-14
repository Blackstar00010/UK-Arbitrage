import pandas as pd

# For each month from 1990-01 to 2022-12, it creates a new table of 48 return factor
# Return Factor: ratio of the current month's value to the value from 1 months ago minus 1

df = pd.read_csv('../files/history/first_day_of_month.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

start_date = pd.to_datetime('1990-01-01')
end_date = pd.to_datetime('2022-12-01')
months = pd.date_range(start_date, end_date, freq='MS', tz='UTC')

for current_date in months:
    window = df.loc[:current_date].tail(49)

    return_factor = window / window.shift() - 1

    return_factor = return_factor.iloc[::-1]

    return_factor.index = range(1, 50)

    return_factor = return_factor.T

    return_factor = return_factor.dropna(how='any')

    filename = current_date.strftime('%Y-%m') + '.csv'
    return_factor.to_csv('../files/return_factor/' + filename, index_label='Return Factor')
