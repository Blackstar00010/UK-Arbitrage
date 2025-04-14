from PCA_and_ETC import *
from sklearn.cluster import *
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
import math
import random
import matplotlib.cm as cm

class Clustering:
    def __init__(self, data: pd.DataFrame):
        self.test = None
        self.PCA_Data = data
        self.index = data.index

        self.K_Mean = []
        self.DBSCAN = []
        self.Agglomerative = []

        self.bisecting_K_mean = []
        self.HDBSCAN = []
        self.BIRCH = []

        self.OPTIC = []
        self.Gaussian = []
        self.meanshift = []

        self.K_Mean_labels = []
        self.DBSCAN_labels = []
        self.Agglomerative_labels = []

        self.bisecting_K_mean_labels = []
        self.HDBSCAN_labels = []
        self.BIRCH_labels = []

        self.OPTIC_labels = []
        self.Gaussian_labels = []
        self.meanshift_labels = []

    def perform_kmeans(self, k_value: int, alpha: float = 0.5):
        """
        :param k_value: Number of Clusters
        :param alpha: outlier threshold
        :return: clustering result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # sample 갯수가 K보다 작은 경우 k_value = n_sample
        n_sample = self.PCA_Data.shape[0]
        if n_sample <= k_value:
            k_value = n_sample

        # Clustering
        kmeans = KMeans(init='k-means++', n_clusters=k_value, n_init=10, max_iter=500,
                        random_state=random.randint(1, 100)).fit(self.PCA_Data)
        cluster_labels = kmeans.labels_
        self.test = kmeans
        self.K_Mean_labels = cluster_labels

        # Outlier Detection
        distance_to_own_centroid = [distance.euclidean(self.PCA_Data[i], kmeans.cluster_centers_[cluster_labels[i]])
                                    for i in range(len(self.PCA_Data))]

        nbrs = NearestNeighbors(n_neighbors=3, p=2).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        nearest_neighbor_distances = distances[:, 1]

        sorted_nearest_neighbor_distances = sorted(nearest_neighbor_distances)
        epsilon = sorted_nearest_neighbor_distances[int(len(sorted_nearest_neighbor_distances) * alpha)]
        outliers = [i for i, dist in enumerate(distance_to_own_centroid) if dist < epsilon]

        clusters_indices = [[] for _ in range(k_value)]
        for i, label in enumerate(cluster_labels):
            if i in outliers:
                continue
            clusters_indices[label].append(i)

        # clusters_indices.insert(0, list(outliers))
        clusters_indices.insert(0, [])

        # number index를 firm name으로 바꾸어 2차원 리스트로 저장.
        final_cluster = [[] for _ in clusters_indices]
        for i, num in enumerate(clusters_indices):
            for j in num:
                final_cluster[i].append(self.index[j])

        final_cluster = [cluster for cluster in final_cluster if cluster]
        self.K_Mean = final_cluster

    def perform_DBSCAN(self, threshold: float):
        """
        :param threshold: outlier threshold
        :return: clustering result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # 각 데이터 포인트의 MinPts 개수의 최근접 이웃들의 거리의 평균 계산
        # 1번째는 자기자신이니까 ms+1
        ms = int(math.log(len(self.PCA_Data)))
        nbrs = NearestNeighbors(n_neighbors=ms + 1, p=1).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        eps = np.percentile(avg_distances, threshold * 100)

        # Clustering
        dbscan = DBSCAN(min_samples=ms, eps=eps, metric='manhattan').fit(self.PCA_Data)
        cluster_labels = dbscan.labels_
        self.test = dbscan
        self.DBSCAN_labels = cluster_labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.DBSCAN = clust

    def perform_HA(self, threshold: float):
        """
        :param threshold: outlier threshold
        :return: clustering result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # 이웃한 두 개 점 사이 거리의 평균 계산
        nbrs = NearestNeighbors(n_neighbors=3, p=1).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.max(distances[:, 1:], axis=1)
        outlier_distance = np.percentile(avg_distances, threshold * 100)

        # Clustering
        agglo = AgglomerativeClustering(n_clusters=None, metric='manhattan', linkage='average',
                                        distance_threshold=outlier_distance).fit(self.PCA_Data)
        cluster_labels = agglo.labels_
        self.test = agglo
        self.Agglomerative_labels = cluster_labels

        # Outlier Detection
        outlier = []
        for i, avg_distance in enumerate(avg_distances):
            if avg_distance > outlier_distance:
                outlier.append(i)

        for i, cluster_label in enumerate(cluster_labels):
            if i in outlier:
                cluster_labels[i] = -1

        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.Agglomerative = clust

    def perform_bisectingkmeans(self, k_value: int, alpha: float = 0.5):
        """
        :param k_value: Number of Cluster
        :param alpha: Outlier threshold
        :return: clustering result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # sample 갯수가 K보다 작은 경우 k_value = n_sample
        n_sample = self.PCA_Data.shape[0]
        if n_sample <= k_value:
            k_value = n_sample

        # Clustering
        kmeans = MiniBatchKMeans(init='k-means++', n_clusters=k_value, n_init=10, max_iter=500,
                                 random_state=random.randint(1, 100)).fit(self.PCA_Data)
        cluster_labels = kmeans.labels_
        self.test = kmeans
        self.bisecting_K_mean_labels = cluster_labels

        # Outlier Detection
        distance_to_own_centroid = [distance.euclidean(self.PCA_Data[i], kmeans.cluster_centers_[cluster_labels[i]])
                                    for i in range(len(self.PCA_Data))]

        nbrs = NearestNeighbors(n_neighbors=3, p=2).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        nearest_neighbor_distances = distances[:, 1]

        sorted_nearest_neighbor_distances = sorted(nearest_neighbor_distances)
        epsilon = sorted_nearest_neighbor_distances[int(len(sorted_nearest_neighbor_distances) * alpha)]
        outliers = [i for i, dist in enumerate(distance_to_own_centroid) if dist < epsilon]

        clusters_indices = [[] for _ in range(k_value)]
        for i, label in enumerate(cluster_labels):
            if i in outliers:
                continue
            clusters_indices[label].append(i)
        clusters_indices.insert(0, list(outliers))

        # number index를 firm name으로 바꾸어 2차원 리스트로 저장.
        final_cluster = [[] for _ in clusters_indices]
        for i, num in enumerate(clusters_indices):
            for j in num:
                final_cluster[i].append(self.index[j])

        final_cluster = [cluster for cluster in final_cluster if cluster]
        self.bisecting_K_mean = final_cluster

    def perform_HDBSCAN(self, threshold):
        """
        :param threshold: Outlier threshold
        :return: clustering result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # 각 데이터 포인트의 MinPts 개수의 최근접 이웃들의 거리의 평균 계산
        ms = int(math.log(len(self.PCA_Data)))
        nbrs = NearestNeighbors(n_neighbors=ms + 1, p=1).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.max(distances[:, 1:], axis=1)
        eps = np.percentile(avg_distances, threshold * 100)

        # Clustering
        Hdbscan = HDBSCAN(min_cluster_size=ms, cluster_selection_epsilon=eps,allow_single_cluster=False).fit(self.PCA_Data)
        cluster_labels = Hdbscan.labels_
        self.test = Hdbscan
        self.HDBSCAN_labels = cluster_labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.HDBSCAN = clust

    def perform_BIRCH(self, threshold):
        """
        :param threshold: Outlier threshold
        :return: clustering result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # 이웃한 두 개 점 사이 거리의 평균 계산
        nbrs = NearestNeighbors(n_neighbors=3, p=1).fit(self.PCA_Data)
        distances, indices = nbrs.kneighbors(self.PCA_Data)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        outlier_distance = np.percentile(avg_distances, threshold * 100)

        # Clustering
        birch = Birch(threshold=outlier_distance, n_clusters=None).fit(self.PCA_Data)
        cluster_labels = birch.labels_
        self.test = birch
        self.BIRCH_labels = cluster_labels

        # Outlier Detection
        outlier = []
        for i, avg_distance in enumerate(avg_distances):
            if avg_distance > outlier_distance:
                outlier.append(i)

        for i, cluster_label in enumerate(cluster_labels):
            if i in outlier:
                cluster_labels[i] = -1

        unique_labels = sorted(list(set(cluster_labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(cluster_labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.BIRCH = clust

    def perform_OPTICS(self, threshold):
        """
        :param threshold: outlier threshold
        :return: clustering result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # Clustering
        optics = OPTICS(cluster_method='xi', xi=threshold, min_cluster_size=0.1, metric='manhattan').fit(
            self.PCA_Data)
        labels = optics.labels_
        self.test = optics
        self.OPTIC_labels = labels

        unique_labels = sorted(list(set(labels)))

        clust = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(labels):
            clust[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clust.insert(0, [])

        self.OPTIC = clust

    def perform_meanshift(self, quantile):
        """
        :param quantile: outlier threshold
        :return: clustering result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # The following bandwidth can be automatically detected using
        bandwidth = estimate_bandwidth(self.PCA_Data, quantile=quantile)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False).fit(self.PCA_Data)
        cluster_labels = ms.labels_
        self.test = ms
        self.meanshift_labels = cluster_labels

        # Get the unique cluster labels
        unique_labels = sorted(list(set(self.meanshift_labels)))

        clusters = [[] for _ in unique_labels]
        for i, cluster_label in enumerate(self.meanshift_labels):
            clusters[unique_labels.index(cluster_label)].append(self.index[i])

        # outlier가 없으면 빈리스트 추가
        if -1 not in unique_labels:
            clusters.insert(0, [])

        self.meanshift = clusters

    def perform_GMM(self, n_components: float):
        """
        :param n_components: Number of clusters
        :return: clustering result
        """
        self.PCA_Data = pd.DataFrame(self.PCA_Data)
        self.PCA_Data = self.PCA_Data.values[:, 1:].astype(float)

        # 1. Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_components, init_params='k-means++', covariance_type='full').fit(
            self.PCA_Data)
        cluster_labels = gmm.predict(self.PCA_Data)
        self.test = gmm
        self.Gaussian_labels = cluster_labels

        clusters = [[] for _ in range(n_components)]

        for i, cluster_num in enumerate(cluster_labels):
            clusters[cluster_num].append(i)

        clusters = [sublist for sublist in clusters if sublist]

        # Outliers
        probabilities = gmm.score_samples(self.PCA_Data)
        threshold = np.percentile(probabilities, 70)

        outliers = []
        for i, probability in enumerate(probabilities):
            if probability < threshold:
                outliers.append(i)

        # a에 있는 값을 b에서 빼기
        for value in outliers:
            for i, row in enumerate(clusters):
                if value in row:
                    clusters[i].remove(value)

        # 1차원 리스트로 전환된 outlier를 cluster 맨앞에 저장.
        clusters.insert(0, outliers)

        for i, cluster in enumerate(clusters):
            for t, num in enumerate(cluster):
                cluster[t] = self.index[num]

        self.Gaussian = clusters

    def visualize_silhouette(self, n_cluster, label):
        fig, axs = plt.subplots(figsize=(4 * 1, 4), nrows=1, ncols=1)

        # Calculate the silhouette score
        sil_avg = silhouette_score(self.PCA_Data, label)
        sil_values = silhouette_samples(self.PCA_Data, label)

        y_lower = 10
        axs.set_title('Number of Clusters: ' + str(n_cluster) + '\n' \
                                                                'Silhouette Score: ' + str(round(sil_avg, 3)))
        axs.set_xlabel("The silhouette coefficient values")
        axs.set_ylabel("Cluster label")
        axs.set_xlim([-0.1, 1])
        axs.set_ylim([0, len(self.PCA_Data) + (n_cluster + 1) * 10])
        axs.set_yticks([])  # Clear the y-axis labels / ticks
        axs.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # Loop through clusters to create silhouette plots
        for i in range(n_cluster):
            ith_cluster_sil_values = sil_values[label == i]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_cluster)  # Adjusted the cluster color
            axs.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, \
                              facecolor=color, edgecolor=color, alpha=0.7)
            axs.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs.axvline(x=sil_avg, color="red", linestyle="--")

        plt.show()


class ResultCheck:

    def __init__(self, data: pd.DataFrame):
        self.PCA_Data = data
        self.prefix = momentum_prefix_finder(self.PCA_Data)
        self.table = []

    def ls_table(self, cluster: list, output_dir, file, save=True, raw=False):
        """
        output columns = ['Firm Name', 'Momentum_1', 'Long Short', 'Cluster Index']
        :param cluster:
        :param output_dir:
        :param file:
        :param save:
        :param raw:
        :return:
        """
        if raw:
            mom1_col_name = self.prefix + '1'
        else:
            mom1_col_name = '0'

        # '''
        clusters = []
        for i in range(len(cluster)):
            for firms in cluster[i]:
                clusters.append([firms, i])
        clusters = pd.DataFrame(clusters, columns=['Firm Name', 'Cluster Index'])
        clusters = clusters.set_index('Firm Name')
        clusters['Momentum_1'] = self.PCA_Data[mom1_col_name]
        clusters = clusters.sort_values(by=['Cluster Index', 'Momentum_1'], ascending=[True, False])
        spread_vec = (clusters.reset_index()['Momentum_1'] -
                      clusters.sort_values(by=['Cluster Index', 'Momentum_1'],
                                           ascending=[True, True]).reset_index()['Momentum_1'])
        clusters = clusters.reset_index()
        clusters['spread'] = spread_vec
        clusters['in_portfolio'] = (clusters['spread'].abs() > clusters['spread'].std()*3)
        clusters['Long Short'] = clusters['in_portfolio'] * (-clusters['spread'] / clusters['spread'].abs())
        clusters['Long Short'] = clusters['Long Short'].fillna(0)
        clusters = clusters.drop(columns=['spread', 'in_portfolio'])
        clusters.loc[clusters['Cluster Index'] == 0, 'Long Short'] = 0
        clusters['Long'] = clusters['Long Short'].apply(lambda x: 1 if x == 1 else 0)
        clusters['Short'] = clusters['Long Short'].apply(lambda x: -1 if x == -1 else 0)
        clusters.sort_values('Cluster Index', inplace=True)
        clusters = clusters[['Firm Name', 'Momentum_1', 'Long Short', 'Long','Short','Cluster Index']]

        if save:
            clusters.to_csv(os.path.join(output_dir, file), index=False)
            print(f'Exported to {output_dir}!')

        # '''

        self.table = clusters

    def reversal_table(self, data: pd.DataFrame, output_dir, file, save=True):
        """

        :param data:
        :param output_dir:
        :param file:
        :param save:
        :return:
        """
        LS_table_reversal = pd.DataFrame(columns=['Firm Name', 'Momentum_1', 'Long Short'])
        firm_lists = data.index
        firm_sorted = sorted(firm_lists, key=lambda x: data.loc[x, self.prefix + '1'])
        long_short = [0] * len(firm_sorted)
        t = int(len(firm_lists) * 0.1)
        for i in range(t):
            long_short[i] = 1
            long_short[-i - 1] = -1

        for i, firm in enumerate(firm_sorted):
            LS_table_reversal.loc[len(LS_table_reversal)] = [firm, data.loc[firm, self.prefix + '1'], long_short[i]]

        LS_table_reversal['Long'] = LS_table_reversal['Long Short'].apply(lambda x: 1 if x == 1 else 0)
        LS_table_reversal['Short'] = LS_table_reversal['Long Short'].apply(lambda x: -1 if x == -1 else 0)

        if save:
            # Save the output to a CSV file in the output directory
            LS_table_reversal.to_csv(os.path.join(output_dir, file), index=False)
            print(f'Exported to {output_dir}!')
        return LS_table_reversal

    def Plot_clusters(self, cluster: list, plttitle=None):
        firm_names = self.PCA_Data.index
        data_array = self.PCA_Data.values[:, 1:].astype(float)

        for j, firms in enumerate(cluster):
            if j == 0:
                print(f'Noise: {firms}')
                title = 'Noise'
            else:
                print(f'Cluster {j}: {firms}')
                title = f'Cluster {j}'

            # Plot the line graph for firms in the cluster
            for firm in firms:
                firm_index = list(firm_names).index(firm)
                firm_data = data_array[firm_index]

                plt.plot(range(1, len(firm_data) + 1), firm_data, label=firm)

            plt.xlabel('Characteristics')
            plt.ylabel('Data Value')
            plt.title(title) if plttitle is None else plt.title(plttitle)

            # List the firm names on the side of the graph
            if len(firms) <= 10:
                plt.legend(loc='center right')
            else:
                plt.legend(loc='center right', title=f'Total Firms: {len(firms)}', labels=firms[:10] + ['...'])

            plt.show()

    def count_outlier(self, cluster: list):
        all_cluster = [item for sublist in cluster for item in sublist]
        firm_len = len(all_cluster)
        if not cluster:
            return 0

        elif not cluster[0]:
            return 0
        else:
            percentile = len(cluster[0]) / firm_len
            return percentile

    def count_stock_of_traded(self):
        count_non_zero = (self.table['Long Short'] != 0).sum()
        proportion = count_non_zero / len(self.table)
        return proportion
