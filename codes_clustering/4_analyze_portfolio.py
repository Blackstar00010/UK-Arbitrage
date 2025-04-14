from PCA_and_ETC import *

FTSE = False
FTSE_not = False
US = True

if US:
    col = ['06/1979-12/1999', '01/2000-12/2002(9/11)', '01/2003-12/2006', '01/2007-12/2009(GFC)',
           '01/2010-12/2019', '01/2020-12/2022(Covid-19)', 'Overall']

    period = [range(1, 249), range(249, 285), range(285, 333), range(333, 369), range(369, 489), range(488, 520),
              range(1, 520)]

if FTSE_not:
    col = ['03/1990-12/1991', '01/1992-12/1994(Black_Wednesday)', '01/1995-12/2001(Dotcom_Bubble)', '01/2002-12/2006',
           '01/2007-12/2009(GFC)',
           '01/2010-12/2013(eurozone_crisis)', '01/2014-12/2015', '01/2016-12/2019(Brexit)',
           '01/2020-12/2022(Covid-19)', '01/2023-07/2023', 'Overall']

    period = [range(1, 23), range(23, 59), range(59, 143), range(143, 203), range(203, 239), range(239, 287),
              range(287, 311), range(311, 359), range(359, 395), range(395, 402), range(1, 402)]

if FTSE:
    col = ['03/1990-12/1991', '01/1992-12/1994(Black_Wednesday)', '01/1995-12/2001(Dotcom_Bubble)', '01/2002-12/2006',
           '01/2007-12/2009(GFC)',
           '01/2010-12/2013(eurozone_crisis)', '01/2014-12/2015', '01/2016-12/2019(Brexit)',
           '01/2020-12/2022(Covid-19)', 'Overall']

    period = [range(1, 23), range(23, 59), range(59, 143), range(143, 203), range(203, 239), range(239, 287),
              range(287, 311), range(311, 359), range(359, 395), range(1, 395)]

result = pd.read_csv('../files/result/total_result_modified.csv')

if US:
    new_order = [4, 0, 1, 2, 3, 5]
    result = result.reindex(new_order).reset_index(drop=True)

if FTSE or FTSE_not:
    new_order = [7, 4, 0, 2, 6, 1, 9, 8, 5, 3, 10, 11]
    result = result.reindex(new_order).reset_index(drop=True)

print('profit_factor')
profit_factor = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        pf = row[row > 0].sum() / np.abs(row[row < 0].sum())
        profit_factor.iloc[j, i] = pf

profit_factor['metric'] = 'Profit_factor'
print(profit_factor.to_string())

print('sharpe_ratio')
sharpe_ratio = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        sf = (np.exp(row.mean() * 12) - 1) / (np.exp(row.std() * np.sqrt(12)) - 1)
        sharpe_ratio.iloc[j, i] = sf

sharpe_ratio['metric'] = 'Sharpe'
print(sharpe_ratio.to_string())

print('sortino_ratio')
sortino_ratio = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        sf = (np.exp(row.mean() * 12) - 1) / (np.exp(row[row < 0].std() * np.sqrt(12)) - 1)
        sortino_ratio.iloc[j, i] = sf

sortino_ratio['metric'] = 'Sortino'
print(sortino_ratio.to_string())

print('Maximum_drawdown(MDD)')
MDD = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        row2 = np.exp(row.astype(float)) - 1
        cumulative_returns = np.cumprod(1 + row2) - 1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1)
        max_drawdown = drawdown.min()
        MDD.iloc[j, i] = max_drawdown

MDD['metric'] = 'MDD'
print(MDD.to_string())

print('Calmar_ratio')
Calmar_ratio = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]
        row2 = np.exp(row.astype(float)) - 1
        cumulative_returns = np.cumprod(1 + row2) - 1
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / (peak + 1)
        max_drawdown = drawdown.min()
        calmar = (np.exp(row.mean() * 12) - 1) / abs(max_drawdown)
        Calmar_ratio.iloc[j, i] = calmar

Calmar_ratio['metric'] = 'Calmar'
print(Calmar_ratio.to_string())

print('annual mean return')
mean_return = pd.DataFrame(index=col, columns=result.iloc[:, 0])
for i in range(len(result.index)):
    for j in range(len(period)):
        row = result.iloc[i, period[j]]

        returns = np.exp(np.mean(row) * 12) - 1

        mean_return.iloc[j, i] = returns

mean_return['metric'] = 'return'
print(mean_return.to_string())

profit_factor = pd.concat([profit_factor, sharpe_ratio, sortino_ratio, MDD, Calmar_ratio, mean_return], axis=0)
profit_factor.to_csv('../files/result/total.csv', index=True)
