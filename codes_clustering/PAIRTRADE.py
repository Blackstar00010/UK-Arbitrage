import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def MDD(rt):
    rt = pd.DataFrame(rt)
    ret = rt.cumsum()

    mdd = ret.apply(lambda x: (x.dropna().loc[((np.maximum.accumulate(x.dropna()) - x.dropna()).idxmax())] -
                               x.dropna().loc[(
                                   x.dropna().loc[
                                   :((np.maximum.accumulate(x.dropna()) - x.dropna()).idxmax())]).idxmax()]))

    mdd = np.exp(mdd)
    mdd = 1 - mdd

    return mdd


def sharpe_ratio_(rt, num):
    sharpe = (np.exp(rt.mean() * num) - 1) / (np.exp(rt.std() * np.sqrt(num)) - 1)

    return sharpe


def sortino_ratio_(rt, num):
    d_risk = rt.apply(lambda x: x.dropna()[x.dropna() < 0].std())

    sortino = (np.exp(rt.mean() * num) - 1) / (np.exp(d_risk * np.sqrt(num)) - 1)

    return sortino


def calmar_ratio_(rt_, num):
    MDD_ = MDD(rt_)
    calmar = ((np.exp(rt_.mean() * num)) - 1) / MDD_

    return calmar


def momentum(ret, max_momentum):
    if ret.shape[0] < max_momentum:
        return 'The length of data is shorter than max_momentum'
    else:

        ret = ret.iloc[-(max_momentum + 1):, :].dropna(axis=1)

        col_name = list(pd.Series(['mom'] * max_momentum) + pd.Series(list(range(1, max_momentum + 1))).astype(str))
        etf_list = list(ret.columns)
        momentem_df = pd.DataFrame(index=etf_list, columns=col_name)
        for etf in etf_list:
            momentem_df.loc[etf, 'mom1'] = ret[etf].iloc[-1]
            for j in range(2, max_momentum + 2):
                mom = 'mom' + str(j)
                momentem_df.loc[etf, mom] = ret[etf].iloc[-j:-1].sum()
        return momentem_df


def PCA_df(df):
    df = df.iloc[:, 1:]
    normalized_df = StandardScaler().fit(df)
    normalized_df = normalized_df.transform(df)

    pca = PCA()
    pca_df = pca.fit_transform(normalized_df)
    explained_var = pca.explained_variance_ratio_.cumsum()
    num_PC = len(explained_var[explained_var < 0.99]) + 1

    final_df = pca_df[:, :num_PC]
    final_df = pd.DataFrame(final_df, index=df.index)

    return final_df


def kmeans_w_pct_ol(dataframe, K, percentile):
    km = KMeans(n_clusters=K, max_iter=1000)
    km = km.fit(dataframe)

    result = pd.DataFrame(index=dataframe.index)
    result['label'] = km.labels_

    nearest_dis = pd.DataFrame(euclidean_distances(dataframe)).apply(lambda x: x[x > 0].min())
    eps = np.percentile(nearest_dis.sort_values(), percentile)
    centroids = km.cluster_centers_

    for i in range(K):
        result.loc[result['label'] == i, 'central_dis'] = cdist(dataframe.iloc[km.labels_ == i].astype(float),
                                                                centroids[i].reshape((1, centroids.shape[1])),
                                                                'euclidean')
    result['OL'] = ((result['central_dis'] - eps) >= 0).astype(float)

    return result[['label', 'OL']]


def Kmedoid_o(dataframe, K, percentile):
    km = KMedoids(n_clusters=K, max_iter=1000)
    km = km.fit(dataframe)

    result = pd.DataFrame(index=dataframe.index)
    result['label'] = km.labels_

    nearest_dis = pd.DataFrame(euclidean_distances(dataframe)).apply(lambda x: x[x > 0].min())
    eps = np.percentile(nearest_dis.sort_values(), percentile)
    centroids = km.cluster_centers_

    for i in range(K):
        result.loc[result['label'] == i, 'central_dis'] = cdist(dataframe.iloc[km.labels_ == i].astype(float),
                                                                centroids[i].reshape(1, centroids.shape[1]),
                                                                'euclidean')
    result['OL'] = ((result['central_dis'] - eps) >= 0).astype(float)

    return result[['label', 'OL']]


def AG_cluster(dataframe, percentile):
    dis = pd.DataFrame(manhattan_distances(dataframe)).apply(lambda x: x[x > 0].min()).sort_values()
    eps = np.percentile(dis, percentile)
    ag_cluster = AgglomerativeClustering(n_clusters=None, affinity='l1', linkage='average', distance_threshold=eps).fit(
        dataframe)

    result = pd.DataFrame(ag_cluster.labels_, index=dataframe.index, columns=['label'])

    return result


def portfolio_generation(momentum_df, cluster_df):
    if cluster_df.shape[1] == 2:
        non_outlier = cluster_df[cluster_df['OL'] == 0]
    else:
        non_outlier = cluster_df

    cluster_list = list(non_outlier['label'].unique())
    K = len(cluster_list)
    LONG = []
    SHORT = []
    diff_ = []
    for i in range(K):
        cur_cluster = cluster_list[i]

        temp_df_ = momentum_df.loc[list(non_outlier[non_outlier['label'] == cur_cluster].index), 'mom1']
        if len(temp_df_) == 1:
            continue
        temp_df_ = temp_df_.sort_values()

        temp_long_ = temp_df_.iloc[:int(temp_df_.shape[0] / 2)]
        temp_short_ = temp_df_.iloc[-int(temp_df_.shape[0] / 2):]

        for j in range(len(temp_long_)):
            diff_ = diff_ + [np.log(1 + (np.exp(temp_short_.iloc[-(j + 1)]) - np.exp(temp_long_.iloc[j])))]

    if len(diff_) == 0:
        return LONG, SHORT

    if len(diff_) >= 4:
        diff_.sort()
        diff_ = diff_[1:]
        diff_ = diff_[:-1]
    else:
        diff_.sort()

    diff_cut = pd.Series(diff_).std()

    for i in range(K):
        cur_cluster = cluster_list[i]

        temp_df = momentum_df.loc[list(non_outlier[non_outlier['label'] == cur_cluster].index), 'mom1']
        if len(temp_df) == 1:
            continue
        temp_df = temp_df.sort_values()

        temp_long = temp_df.iloc[:int(temp_df.shape[0] / 2)]
        temp_short = temp_df.iloc[-int(temp_df.shape[0] / 2):]

        for j in range(len(temp_long)):
            if np.log(1 + (np.exp(temp_short.iloc[-(j + 1)]) - np.exp(temp_long.iloc[j]))) > diff_cut:
                LONG = LONG + [temp_long.index[j]]
                SHORT = SHORT + [temp_short.index[-(j + 1)]]

    return LONG, SHORT


def daily_performance(price_data, reb_time, max_momentum, ub, K, percentile, cluster):
    performance_df = pd.DataFrame()

    week_price = price_data[price_data.index.hour == reb_time]

    reb_date = list(week_price.index)
    for i in range(max_momentum, len(reb_date) - 1):
        print(i)
        cur_reb = reb_date[i]
        next_reb_date = reb_date[i + 1]

        mom_ret = week_price[week_price.index <= cur_reb]
        mom_ret = np.log(mom_ret / mom_ret.shift(1)).iloc[1:]

        momentum_df = momentum(mom_ret, max_momentum)
        final_df = PCA_df(momentum_df)
        if cluster == 'km':
            cluster_df = kmeans_w_pct_ol(final_df, K, percentile)

        elif cluster == 'db':
            cluster_df = dbscan(final_df, percentile)

        elif cluster == 'kmo':
            cluster_df = Kmedoid_o(final_df, K, percentile)

        else:
            cluster_df = AG_cluster(final_df, percentile)

        long_short = portfolio_generation(momentum_df, cluster_df)
        long = long_short[0]
        short = long_short[1]

        temp_return = price_data[price_data.index <= next_reb_date]
        temp_return = temp_return[temp_return.index >= cur_reb]
        temp_return = temp_return.fillna(method='ffill', limit=4)
        temp_return = (temp_return / temp_return.shift(1))
        temp_return.iloc[0] = 1

        temp_cum = temp_return.cumprod()

        if len(long) > 0:
            for s_l in short:
                temp_short = temp_cum[s_l]
                temp_short = temp_short[temp_short > 2].dropna()
                if len(temp_short) > 0:
                    close_index = temp_short.index[0]
                    temp_return.loc[temp_return.index >= close_index, s_l] = 1

            temp_per = pd.DataFrame()
            for pset in range(len(long)):

                pper = (temp_return[long[pset]].cumprod() + (2 - temp_return[short[pset]]).cumprod()) / 2

                edidx = pper[pper > ub]

                if len(edidx) > 0:
                    edidx = edidx.index[0]
                    pper.loc[pper.index > edidx] = pper.loc[edidx]

                temp_per = pd.concat([temp_per, pper], axis=1)

            temp_per = temp_per.mean(axis=1)
            temp_per = (temp_per / temp_per.shift(1)).iloc[1:]
            temp_per = np.log(temp_per)
            temp_per = temp_per / 2
            temp_per.iloc[0] = temp_per.iloc[0] + np.log(0.9986)



        else:
            temp_per = temp_return[long].sum(axis=1)

        performance_df = pd.concat([performance_df, temp_per])

    return performance_df
