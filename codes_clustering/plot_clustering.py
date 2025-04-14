import Clustering as C
from PCA_and_ETC import *

# file to check
file = '2002-08.csv'

input_dir = '../files/characteristics'

# turn off warning
warnings.filterwarnings("ignore")

K_mean_Plot = True
if K_mean_Plot:
    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.ResultCheck(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.perform_kmeans(7)

    # Plot clustering result
    # Do_Result_Plot.Plot_clusters(Do_Clustering.K_Mean)

    # Plot t_SNE result
    t_SNE('K-mean', df_combined, Do_Clustering.K_Mean_labels)  # 0

dbscan_Plot = True
if dbscan_Plot:
    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.ResultCheck(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.perform_DBSCAN(0.8)

    # Plot clustering result
    # Do_Result_Plot.Plot_clusters(Do_Clustering.DBSCAN)

    # Plot t_SNE result
    t_SNE('DBSCAN', df_combined, Do_Clustering.DBSCAN_labels)  # -1

Agglormerative_Plot = True
if Agglormerative_Plot:
    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.ResultCheck(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.perform_HA(0.9)

    # Plot clustering result
    # Do_Result_Plot.Plot_clusters(Do_Clustering.Agglomerative)

    # Plot t_SNE result
    t_SNE('Hirarchical Agglomerative', df_combined, Do_Clustering.Agglomerative_labels)  # 0

hdbscan_Plot = True
if hdbscan_Plot:
    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.ResultCheck(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.perform_HDBSCAN(0.5)

    # Plot clustering result
    # Do_Result_Plot.Plot_clusters(Do_Clustering.HDBSCAN)

    # Plot t_SNE result
    t_SNE('HDBSCAN', df_combined, Do_Clustering.HDBSCAN_labels)  # -1

optics_Plot = True
if optics_Plot:
    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.ResultCheck(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.perform_OPTICS(0.5)

    # Plot clustering result
    # Do_Result_Plot.Plot_clusters(Do_Clustering.OPTIC)

    # Plot t_SNE result
    t_SNE('OPTICS', df_combined, Do_Clustering.OPTIC_labels)  # -1

birch_Plot = True
if birch_Plot:
    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.ResultCheck(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.perform_BIRCH(0.5)

    # Plot clustering result
    # Do_Result_Plot.Plot_clusters(Do_Clustering.BIRCH)

    # Plot t_SNE result
    t_SNE('BIRCH', df_combined, Do_Clustering.BIRCH_labels)  # 0

meanshift_Plot = True
if meanshift_Plot:
    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.ResultCheck(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.perform_meanshift(0.9)

    # Plot clustering result
    # Do_Result_Plot.Plot_clusters(Do_Clustering.meanshift)

    # # Plot t_SNE result
    t_SNE('meanshift', df_combined, Do_Clustering.meanshift_labels)  # -1

BGM_Plot = True
if BGM_Plot:
    # convert mom_data into PCA_data
    data = read_and_preprocess_data(input_dir, file)
    df_combined = generate_PCA_Data(data)

    # Call initial method
    Do_Clustering = C.Clustering(df_combined)
    Do_Result_Plot = C.ResultCheck(df_combined)

    # Do clustering and get 2D list of cluster index
    Do_Clustering.perform_GMM(7)

    # Plot clustering result
    # Do_Result_Plot.Plot_clusters(Do_Clustering.Gaussian)

    # Plot t_SNE result
    t_SNE('GMM', df_combined, Do_Clustering.Gaussian_labels)  # 0
