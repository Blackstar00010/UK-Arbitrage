import pandas as pd

# For each month from 1990-01 to 2022-12, it creates a new table of 48 rows of momentum factor
# Momentum Factor: ratio of the current month's value to the value from i months ago minus 1
# mom_1 = r_{t-1}
# mom_i = \prod_{j=t-i-1}^{t-2} (r_j+1) - 1, i \in 1,...,4

df = pd.read_csv('../files/history/adj_close_first_day_of_month.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

start_date = pd.to_datetime('1990-01-01')
end_date = pd.to_datetime('2022-12-01')
months = pd.date_range(start_date, end_date)

# jamesd: print shits = progress bar
print('[', end='')
for current_date in months:
    window = df.loc[:current_date].tail(50)

    mom = pd.DataFrame(index=range(1, 50), columns=window.columns)
    for i in range(1, 50):
        if i == 1:
            mom.loc[i] = window.iloc[-1] / window.iloc[-2] - 1
        else:
            mom.loc[i] = window.iloc[-2] / window.shift(i-1).iloc[-2] - 1

    mom.index = range(1, 50)

    mom = mom.T

    # Delete rows with all NaN values
    mom = mom.dropna(how='any')

    filename = current_date.strftime('%Y-%m') + '.csv'
    mom.to_csv('../files/momentum_adj_close/' + filename, index_label='Momentum Index')

    print('-', end='')
    if int(current_date.strftime('%m')) == 12:
        print(f'] {current_date.strftime("%Y")} done!\n[')
if int(current_date.strftime('%m')) != 12:
    print(f'] {current_date.strftime("%Y-%m")} done!\n[')
