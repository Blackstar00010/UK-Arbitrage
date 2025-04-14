import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import manifold
from sklearn.decomposition import PCA

# turn off warning
warnings.filterwarnings("ignore")


def momentum_prefix_finder(df: pd.DataFrame):
    """
    :param df:
    :return:
    """
    possible_prefix = ['', 'Momentum', 'MOM', 'mom', 'Momentum_', 'MOM_', 'mom_', 'Mom_']
    for i in range(1, 10):
        for aposs in possible_prefix:
            if aposs + str(i) in df.columns:
                return aposs
    return ''


def merge_LS_Table(data, LS_merged_df, file):
    # Keep only the 'Firm Name' and 'Long Short' columns
    data = data[['Firm Name', 'Long Short']]

    # Change the column name into file name (ex: 1990-01)
    file_column_name = os.path.splitext(file)[0]
    data = data.rename(columns={'Long Short': file_column_name})

    if LS_merged_df.empty:
        LS_merged_df = data
    else:
        LS_merged_df = pd.merge(LS_merged_df, data, on='Firm Name', how='outer')

    return LS_merged_df


def product_LS_Table(LS_merged_df: pd.DataFrame, MOM_merged_df: pd.DataFrame, result_df: pd.DataFrame, subdir,
                     save=False):
    # Sort LS_Value according to Firm Name
    # LS_merged_df = LS_merged_df.sort_values('Firm Name')

    # Set Firm Name column into index
    LS_merged_df.set_index('Firm Name', inplace=True)

    # 마지막 row 버리면 한칸씩 밀어버리는 것과 동치
    LS_merged_df = LS_merged_df.drop(LS_merged_df.columns[-1], axis=1)
    LS_merged_df = LS_merged_df.fillna(0)
    LS_merged_df.sort_values('Firm Name', inplace=True)
    LS_merged_df.columns = MOM_merged_df.columns

    if save:
        LS_merged_df.to_csv(f'../files/LS_merge_{subdir}.csv', index=False)

    prod = MOM_merged_df * LS_merged_df
    prod = pd.DataFrame(prod)

    # prod index set to df1.index
    prod.set_index(MOM_merged_df.index, inplace=True)
    # cumulative return은 1990-02부터 2022-12이기 때문에 prod.columns=df1.columns
    prod.columns = MOM_merged_df.columns

    if save:
        prod.to_csv(f'../files/prod_{subdir}.csv')

    # Count the non-zero LS that is the number of total firm invested(395 by 1 matrix/index=Date)
    non_zero_count = LS_merged_df.astype(bool).sum()

    non_zero_count2 = pd.DataFrame(non_zero_count).T
    # non_zero_count2.to_csv(f'../files/traded_{subdir}.csv')

    # sum about all rows(395 by 1 matrix/index=Date)
    column_sums = prod.sum()

    # calculate mean and make into DataFrame
    column_means = column_sums.values / non_zero_count.values
    column_means = pd.DataFrame(column_means)
    column_means.index = column_sums.index

    # Concat the means DataFrame to the result DataFrame(395 by 1 matrix->1 by 395 matrix)
    result_df = pd.concat([result_df, column_means.T], ignore_index=True)

    return result_df


def save_and_plot_result(output_dir, clustering_name, result_df: pd.DataFrame, file_names, FTSE=False, apply_log=True,
                         new_Plot=False):
    # Add a new column to the result DataFrame with the file names
    result_df['Clustering Method'] = file_names

    # Separate the 'Clustering Method' column from the date columns
    clustering_method = result_df['Clustering Method']
    date_columns_df = result_df.drop('Clustering Method', axis=1)

    # Convert the date columns to datetime format and sort them
    date_columns_df.columns = pd.to_datetime(date_columns_df.columns, errors='coerce')
    date_columns_df = date_columns_df.sort_index(axis=1)

    # Concat the 'Clustering Method' column back with the sorted date columns
    result_df = pd.concat([clustering_method, date_columns_df], axis=1)
    result_df.set_index('Clustering Method', inplace=True)

    if FTSE:
        file_names.append('FTSE 100')
        file = '../files/ftse_return.csv'
        df = pd.read_csv(file)
        df = df.iloc[1:]
        df = df.iloc[0:, 2:]
        df.columns = result_df.columns[0:-7]  # columns name should be same with result_df
        result_df = pd.concat([result_df.iloc[:, 0:-7], df], axis=0)  # add monthly_return right below result_df

    result_df.index = file_names
    result_df = result_df.astype(float)  # set data type as float(df.value was str actually.)
    result_df = result_df.fillna(0)

    # Add 1 to all data values
    result_df.iloc[:, 0:] = result_df.iloc[:, 0:] + 1

    # transform into log scale
    result_df.iloc[:, 0:] = np.log(result_df.iloc[:, 0:]) if apply_log else result_df.iloc[:, 0:]
    result_df.to_csv(os.path.join('../files/result/', f'{clustering_name}_result_original.csv'), index=True)

    result_modified = pd.DataFrame(
        index=['count', 'annual return mean', 'annual return std'],
        columns=result_df.index)

    annual = True
    if annual:
        for i in range(len(result_modified.columns)):
            result_modified.iloc[0, i] = len(result_df.columns)
            result_modified.iloc[1, i] = np.exp(np.mean(result_df.iloc[i, :]) * 12) - 1
            result_modified.iloc[2, i] = np.exp(np.std(result_df.iloc[i, :]) * np.sqrt(12)) - 1

        sharpe_ratio = pd.DataFrame(index=['Sharpe ratio'], columns=result_modified.columns)
        for i in range(len(result_modified.columns)):
            sharpe_ratio.iloc[0, i] = result_modified.iloc[1, i] / result_modified.iloc[2, i]

        result_modified = pd.concat([result_modified, sharpe_ratio], axis=0)

        t_test = pd.DataFrame(index=['t-statistic'], columns=result_modified.columns)
        for i in range(len(result_modified.columns)):
            t_statistic, p_value = stats.ttest_ind_from_stats(result_modified.iloc[1, i], result_modified.iloc[2, i],
                                                              result_modified.iloc[0, i], result_modified.iloc[1, -1],
                                                              result_modified.iloc[2, -1], result_modified.iloc[0, -1])
            t_test.iloc[0, i] = t_statistic

        result_modified = pd.concat([result_modified, t_test], axis=0)

        Down_std = pd.DataFrame(index=['Downside deviation'], columns=result_modified.columns)
        for i in range(len(result_modified.columns)):
            row = result_df.iloc[i, :]
            sf = (np.exp(row[row < 0].std() * np.sqrt(12)) - 1)
            Down_std.iloc[0, i] = sf

        result_modified = pd.concat([result_modified, Down_std], axis=0)

        sortino_ratio = pd.DataFrame(index=['Sortino ratio'], columns=result_modified.columns)
        for i in range(len(result_modified.columns)):
            row = result_df.iloc[i, :]
            sf = (np.exp(row.mean() * 12) - 1) / (np.exp(row[row < 0].std() * np.sqrt(12)) - 1)
            sortino_ratio.iloc[0, i] = sf

        result_modified = pd.concat([result_modified, sortino_ratio], axis=0)

        Gross_profit = pd.DataFrame(index=['Gross profit'], columns=result_modified.columns)
        for i in range(len(result_modified.columns)):
            row = result_df.iloc[i, :]
            pf = row[row > 0].sum()
            Gross_profit.iloc[0, i] = pf

        result_modified = pd.concat([result_modified, Gross_profit], axis=0)

        Gross_loss = pd.DataFrame(index=['Gross loss'], columns=result_modified.columns)
        for i in range(len(result_modified.columns)):
            row = result_df.iloc[i, :]
            pf = row[row < 0].sum()
            Gross_loss.iloc[0, i] = pf

        result_modified = pd.concat([result_modified, Gross_loss], axis=0)

        profit_factor = pd.DataFrame(index=['Profit Factor'], columns=result_modified.columns)
        for i in range(len(result_modified.columns)):
            row = result_df.iloc[i, :]
            pf = row[row > 0].sum() / np.abs(row[row < 0].sum())
            profit_factor.iloc[0, i] = pf

        result_modified = pd.concat([result_modified, profit_factor], axis=0)

        profitable_years = pd.DataFrame(index=['Profitable years'], columns=result_modified.columns)
        for i in range(len(result_modified.columns)):
            row = result_df.iloc[i, :]
            row.index = pd.to_datetime(row.index)
            row2 = pd.DataFrame(row.index.year, index=row.index)
            row = pd.concat([row, row2], axis=1)
            sum_by_year = row.groupby(0)[result_modified.columns[i]].sum()
            pf = sum_by_year[sum_by_year > 0].count()

            profitable_years.iloc[0, i] = pf

        result_modified = pd.concat([result_modified, profitable_years], axis=0)

        unprofitable_years = pd.DataFrame(index=['Unprofitable years'], columns=result_modified.columns)
        for i in range(len(result_modified.columns)):
            row = result_df.iloc[i, :]
            row.index = pd.to_datetime(row.index)
            row2 = pd.DataFrame(row.index.year, index=row.index)
            row = pd.concat([row, row2], axis=1)
            sum_by_year = row.groupby(0)[result_modified.columns[i]].sum()
            pf = sum_by_year[sum_by_year < 0].count()

            unprofitable_years.iloc[0, i] = pf

        result_modified = pd.concat([result_modified, unprofitable_years], axis=0)

        MDD = pd.DataFrame(index=['Maximum drawdown'], columns=result_modified.columns)
        for i in range(len(result_df.index)):
            row = result_df.iloc[i, :]
            row2 = np.exp(row.astype(float)) - 1
            cumulative_returns = np.cumprod(1 + row2) - 1
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / (peak + 1)
            max_drawdown = drawdown.min()
            MDD.iloc[0, i] = max_drawdown

        result_modified = pd.concat([result_modified, MDD], axis=0)

        Calmar_ratio = pd.DataFrame(index=['Calmar ratio'], columns=result_modified.columns)
        for i in range(len(result_modified.columns)):
            row = result_df.iloc[i, :]
            row2 = np.exp(row.astype(float)) - 1
            cumulative_returns = np.cumprod(1 + row2) - 1
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / (peak + 1)
            max_drawdown = drawdown.min()
            calmar = (np.exp(row.mean() * 12) - 1) / abs(max_drawdown)
            Calmar_ratio.iloc[0, i] = calmar

        result_modified = pd.concat([result_modified, Calmar_ratio], axis=0)

    monthly = True
    if monthly:
        month = pd.DataFrame(
            index=['Mean', 'Standard deviation', 'Standard error', 't-statistic', 'Min', '25%', '50%', '75%', 'Max',
                   'Skew', 'Kurtosis'], columns=result_modified.columns)
        for i in range(len(result_modified.columns)):
            month.iloc[0, i] = np.mean(result_df.iloc[i, :])
            month.iloc[1, i] = np.std(result_df.iloc[i, :])
            month.iloc[2, i] = np.std(result_df.iloc[i, :], ddof=1) / np.sqrt(len(result_df.iloc[i, :]))
            month.iloc[4, i] = np.min(result_df.iloc[i, :])
            month.iloc[5, i] = np.percentile(result_df.iloc[i, :], 25)
            month.iloc[6, i] = np.percentile(result_df.iloc[i, :], 50)
            month.iloc[7, i] = np.percentile(result_df.iloc[i, :], 75)
            month.iloc[8, i] = np.max(result_df.iloc[i, :])
            month.iloc[9, i] = result_df.iloc[i, :].skew()
            month.iloc[10, i] = result_df.iloc[i, :].kurtosis()

            X = sm.add_constant(result_df.iloc[i, :].shift(1).dropna())
            y = result_df.iloc[i, :][1:]
            model = sm.OLS(y, X)
            newey_west = model.fit(cov_type='HAC', cov_kwds={'maxlags': 1})
            t_statistic, p_value = newey_west.tvalues
            month.iloc[3, i] = t_statistic

        result_modified = pd.concat([result_modified, month], axis=0)

    result_modified.to_csv(os.path.join(output_dir, clustering_name + '_statistcs_modified.csv'), index=True)
    result_df.to_csv(os.path.join(output_dir, clustering_name + '_result_modified.csv'), index=True)

    if new_Plot:
        result_df.iloc[:, :] = result_df.iloc[:, :].cumsum(axis=1) if apply_log else result_df.iloc[:, :].cumprod(
            axis=1)

        color_dict = {
            'K_mean': 'red',  # Standard red
            'DBSCAN': 'lightcoral',  # Darker shade of red
            'Agglomerative': 'darkred',  # Darkest shade of red

            'Bisecting_K_mean': 'blue',  # Standard blue
            'HDBSCAN': 'steelblue',  # Darker shade of blue
            'BIRCH': 'navy',  # Darkest shade blue

            'OPTICS': 'deepskyblue',  # Bright blue
            'Meanshift': 'royalblue',  # Darker shade of skyblue
            'GMM': 'midnightblue',  # Darkest shade of skyblue

            'Cointegration': 'darkgrey',  # Darker shade of grey
            'Reversal': 'lightgrey',  # Lighter shade of grey
            'FTSE 100': 'grey',  # Standard grey

            'CL_10_64': 'blue',  # Standard blue
            'CL_10_128': 'navy',  # Darker shade of blue
            'CL_20_64': 'lightsteelblue',  # Darkest shade blue
            'CL_20_128': 'royalblue',  # Bright blue
            'CL_30_64': 'slateblue',  # Darkest shade blue
            'CL_30_128': 'darkslateblue',  # Bright blue
            'CL_50_64': 'indigo',  # Darkest shade blue
            'CL_50_128': 'blueviolet',  # Bright blue
            'CL_100_64': 'cyan',  # Darkest shade blue
            'CL_100_128': 'darkcyan',  # Bright blue

        }

        plt.figure(figsize=(10, 6))
        handles = []  # List to store the line handles for the legend
        for key in color_dict:
            if key in result_df.index:
                idx = result_df.index.get_loc(key)
                line, = plt.plot(result_df.columns, result_df.iloc[idx].fillna(method='ffill'),
                                 label=key, color=color_dict[key])
                handles.append(line)

        plt.title('RETURN')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Value')

        plt.xticks(rotation=45)
        plt.legend(handles=handles)  # Use the handles list to order the legend
        plt.tight_layout()
        plt.show()

    if not new_Plot:
        # Calculate the cumulative product
        result_df.iloc[:, :] = result_df.iloc[:, :].cumsum(axis=1) if apply_log else result_df.iloc[:, :].cumprod(
            axis=1)

        plt.figure(figsize=(10, 6))
        for i in range(len(result_df)):
            plt.plot(result_df.columns[1:-7], result_df.iloc[i, 1:-7].fillna(method='ffill'),
                     label=result_df.iloc[i, 0])

        plt.title('RETURN')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Value')

        plt.xticks(rotation=45)
        plt.legend(result_df.index)  # Add a legend to distinguish different lines
        plt.tight_layout()
        plt.show()


def merge_Long_and_Short_Table(data, df2, df3, file):
    # Keep only the 'Firm Name' and 'Long Short' columns
    data2 = data[['Firm Name', 'Long']]
    data3 = data[['Firm Name', 'Short']]

    # Change the column name into file name (ex: 1990-01)
    file_column_name = os.path.splitext(file)[0]
    data2 = data2.rename(columns={'Long': file_column_name})
    data3 = data3.rename(columns={'Short': file_column_name})

    if df2.empty:
        df2 = data2
    else:
        df2 = pd.merge(df2, data2, on='Firm Name', how='outer')

    if df3.empty:
        df3 = data3
    else:
        df3 = pd.merge(df3, data3, on='Firm Name', how='outer')

    return df2, df3


def save_cluster_info(clustering_name, stat_list: list, file_names):
    # file_names.remove('FTSE 100')
    stat_df = pd.DataFrame(stat_list,
                           index=file_names,
                           columns=['Number of clusters', 'Number of stock in clusters',
                                    'Number of outliers', 'Number of stock traded']).T

    stat_df.to_csv(os.path.join('../files/result/', f'{clustering_name}_cluster_info.csv'), index=True)


def generate_PCA_Data(data: pd.DataFrame):
    """
    :param data: momentum_data
    :return: Mom1+PCA_Data
    """
    prefix = momentum_prefix_finder(data)

    # mom1 save and data Normalization
    mom1 = data.astype(float).loc[:, prefix + '1']
    data_normalized = (data - data.mean()) / data.std()
    mat = data_normalized.astype(float)
    # mom1을 제외한 mat/PCA(2-49)
    mat = mat.drop(columns=[prefix + '1'])
    mat = mat.dropna(how='all', axis=1)

    # mom49를 제외한 mat/PCA(1-48)
    # mat = mat.drop(columns=[prefix + '49'])
    # mat = mat.dropna(how='all', axis=1)

    # 1. Searching optimal n_components
    n_components = min(len(data), 20)

    pca = PCA(n_components)
    pca.fit(mat)
    total_variance = np.sum(pca.explained_variance_ratio_)

    while total_variance > 0.99:
        n_components -= 1
        pca = PCA(n_components)
        pca.fit(mat)
        total_variance = np.sum(pca.explained_variance_ratio_)

    while total_variance < 0.99:
        n_components += 1
        pca = PCA(n_components)
        pca.fit(mat)
        total_variance = np.sum(pca.explained_variance_ratio_)

    # 2. PCA
    pca_mat = PCA(n_components=n_components).fit(data).transform(data)
    cols = [f'pca_component_{i + 1}' for i in range(pca_mat.shape[1])]
    mat_pd_pca = pd.DataFrame(pca_mat, columns=cols)
    mat_pd_pca_matrix = mat_pd_pca.values

    # 3. combined mom1 and PCA data
    first_column_matrix = np.array(mom1).reshape(-1, 1)
    combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))
    df_combined = pd.DataFrame(combined_matrix)
    df_combined.columns = df_combined.columns.astype(str)
    df_combined.index = data.index

    return df_combined


def read_and_preprocess_data(input_dir, file) -> pd.DataFrame:
    """
    Only for reading YYYY-MM.csv files. Recommend using smart_read() for more general cases.
    :param input_dir: '../files/momentum_adj'
    :param file: YYYY-MM.csv
    :return: DataFrame
    """

    df = pd.read_csv(os.path.join(input_dir, file), index_col=0)
    # Replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows with NaN values
    df.dropna(inplace=True)
    return df


def t_SNE(title, data, cluster_labels):
    '''
    :param data: Mom1+PCA_Data
    :param cluster_labels: cluster_labels
    '''

    '''이웃 data와 유사성을 얼마나 중요하게 고려할지 정하는 척도.
    data set이 클수록 큰 perplexities 필요'''
    perplexities = [15, 20, 25]

    # t-SNE를 사용하여 2차원으로 차원 축소
    for i in range(3):
        perplexity = perplexities[i]

        tsne = manifold.TSNE(
            n_components=2,
            random_state=0,
            perplexity=perplexity,
            learning_rate="auto",
            n_iter=1000,
        )

        X_tsne = tsne.fit_transform(data)
        plt.figure(figsize=(8, 6))
        plt.suptitle("Perplexity=%d" % perplexity)
        sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='plasma')

        plt.title('t-SNE Visualization' + title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')

        # 클러스터 라벨을 추가하여 범례(legend) 표시
        handles, labels = sc.legend_elements()
        plt.legend(handles, labels)
        plt.show()
        print()


# PCA_Result Check
if __name__ == "__main__":
    # 파일 불러오기 및 PCA함수
    input_dir = '../files/characteristics'
    file = '1990-11.csv'
    data = read_and_preprocess_data(input_dir, file)

    prefix = momentum_prefix_finder(data)

    # mom1 save and data Normalization
    mom1 = data.astype(float).loc[:, prefix + '1']
    data_normalized = (data - data.mean()) / data.std()
    mat = data_normalized.astype(float)
    # mom1을 제외한 mat/PCA(2-49)
    # mat = np.delete(mat, 0, axis=1)

    # mom49를 제외한 mat/PCA(1-48)
    mat = mat.drop(columns=[prefix + '49'])
    mat = mat.dropna(how='all', axis=1)

    # 1. Searching optimal n_components
    n_components = min(len(data), 20)

    pca = PCA(n_components)
    pca.fit(mat)
    total_variance = np.sum(pca.explained_variance_ratio_)

    while total_variance > 0.99:
        n_components -= 1
        pca = PCA(n_components)
        pca.fit(mat)
        total_variance = np.sum(pca.explained_variance_ratio_)

    while total_variance < 0.99:
        n_components += 1
        pca = PCA(n_components)
        pca.fit(mat)
        total_variance = np.sum(pca.explained_variance_ratio_)

    # 2. PCA
    pca_mat = PCA(n_components=n_components).fit(data).transform(data)
    cols = [f'pca_component_{i + 1}' for i in range(pca_mat.shape[1])]
    mat_pd_pca = pd.DataFrame(pca_mat, columns=cols)
    mat_pd_pca_matrix = mat_pd_pca.values

    # 3. combined mom1 and PCA data
    first_column_matrix = np.array(mom1).reshape(-1, 1)
    combined_matrix = np.hstack((first_column_matrix, mat_pd_pca_matrix))
    df_combined = pd.DataFrame(combined_matrix)
    df_combined.columns = df_combined.columns.astype(str)
    df_combined.index = data.index

    # Result
    print(file)
    print("original shape:", mat.shape)
    print("transformed shape:", pca_mat.shape)
    print('variance_ratio:', pca.explained_variance_ratio_)
    print('sum of variance_ratio:', np.sum(pca.explained_variance_ratio_))
    print(mat_pd_pca)
    print(df_combined)
