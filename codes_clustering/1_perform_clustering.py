import Clustering as C
from PCA_and_ETC import *

MOM_merged_df = pd.read_csv('../files/mom1_data_combined_adj_close2.csv')
MOM_merged_df.set_index('Firm Name', inplace=True)
MOM_merged_df.drop(MOM_merged_df.columns[0], axis=1, inplace=True)
output_dir = '../files/result'
# hyper parameter K(3, 5, 10, 25, 50, 75, 100, 200, 300) should be tested manually.(paper follow) Done!
K_mean_Save = False
if K_mean_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics_us'

    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

    for i in [10,20,30,50,100]:
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

        for file in files:
            print(file)
            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            raw = False
            if not raw:
                df_combined = generate_PCA_Data(data)
            else:
                df_combined = data
            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.ResultCheck(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.perform_kmeans(i, 0.5)

            Do_Result_Save.ls_table(Do_Clustering.K_Mean, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.K_Mean_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.K_Mean)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.K_Mean_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df, "_", save=False)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result(output_dir,'K_mean', result_df, file_names, FTSE=False, apply_log=True, new_Plot=False)
    save_cluster_info('K_mean', stat_lists, file_names)

# hyper parameter eps percentile np.range(0.1, 1, 0.1) should be tested manually.(paper follow) Done!
dbscan_Save = True
if dbscan_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics_us'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

    for i in np.arange(0.1, 1, 0.1):
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

        for file in files:
            print(file)
            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)

            fundamental = True
            if not fundamental:
                a = momentum_prefix_finder(data)
                columns_to_drop = [col for col in data.columns if a not in col]
                data = data.drop(columns=columns_to_drop)

            raw = False
            if not raw:
                df_combined = generate_PCA_Data(data)
            else:
                df_combined = data
            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.ResultCheck(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.perform_DBSCAN(i)

            Do_Result_Save.ls_table(Do_Clustering.DBSCAN, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.DBSCAN_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.DBSCAN)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.DBSCAN_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df, "_", save=False)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result(output_dir,'DBSCAN', result_df, file_names, FTSE=False, apply_log=True, new_Plot=False)
    save_cluster_info('DBSCAN', stat_lists, file_names)

# hyper parameter distance percentile np.range(0.1, 1, 0.1) should be tested manually.(paper follow) Done!
agglomerative_Save = False
if agglomerative_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics_us'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

    for i in np.arange(0.1, 1, 0.1):
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

        for file in files:
            print(file)
            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            raw = False
            if not raw:
                df_combined = generate_PCA_Data(data)
            else:
                df_combined = data
            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.ResultCheck(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.perform_HA(i)

            Do_Result_Save.ls_table(Do_Clustering.Agglomerative, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.Agglomerative_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Agglomerative)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.Agglomerative_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df, "_", save=False)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result(output_dir,'Agglomerative', result_df, file_names, FTSE=False, apply_log=True, new_Plot=False)
    save_cluster_info('Agglomerative', stat_lists, file_names)

# hyper parameter K(3, 5, 10, 25, 50, 75, 100, 200, 300) should be tested manually.(paper follow) Done!
bisecting_Save = False
if bisecting_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

    for i in [3, 5, 10, 25, 50, 75, 100, 200, 300]:
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

        for file in files:
            print(file)
            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            raw = False
            if not raw:
                df_combined = generate_PCA_Data(data)
            else:
                df_combined = data
            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.ResultCheck(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.perform_bisectingkmeans(i, 0.5)

            Do_Result_Save.ls_table(Do_Clustering.bisecting_K_mean, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.bisecting_K_mean_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.bisecting_K_mean)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.bisecting_K_mean_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df, "_", save=False)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result(output_dir,'Bisecting', result_df, file_names, FTSE=True, apply_log=True, new_Plot=False)
    save_cluster_info('Bisecting', stat_lists, file_names)

# hyper parameter distance percentile np.range(0.1, 1, 0.1) should be tested manually.(agglomerative) Done!
hdbscan_Save = False
if hdbscan_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

    for i in np.arange(0.1, 1, 0.1):
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

        for file in files:
            print(file)
            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)

            fundamental = True
            if not fundamental:
                a = momentum_prefix_finder(data)
                columns_to_drop = [col for col in data.columns if a not in col]
                data = data.drop(columns=columns_to_drop)

            raw = False
            if not raw:
                df_combined = generate_PCA_Data(data)
            else:
                df_combined = data
            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.ResultCheck(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.perform_HDBSCAN(i)

            Do_Result_Save.ls_table(Do_Clustering.HDBSCAN, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.HDBSCAN_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.HDBSCAN_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df, "_", save=False)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result(output_dir,'HDBSCAN', result_df, file_names, FTSE=True, apply_log=True, new_Plot=False)
    save_cluster_info('HDBSCAN', stat_lists, file_names)

# hyper parameter distance percentile np.range(0.1, 1, 0.1) should be tested manually.(K_mean/agglomerative) more..
birch_Save = False
if birch_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

    for i in np.arange(0.1, 1, 0.1):
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

        for file in files:
            print(file)
            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            raw = False
            if not raw:
                df_combined = generate_PCA_Data(data)
            else:
                df_combined = data
            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.ResultCheck(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.perform_BIRCH(i)

            Do_Result_Save.ls_table(Do_Clustering.BIRCH, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.BIRCH_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.BIRCH)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.BIRCH_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df, "_", save=False)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result(output_dir,'BIRCH', result_df, file_names, FTSE=True, apply_log=True, new_Plot=False)
    save_cluster_info('BIRCH', stat_lists, file_names)

# hyper parameter eps percentile np.arange(0.01, 0.1, 0.01) should be tested manually.(DBSCAN) Done!
optics_Save = False
if optics_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

    for i in np.arange(0.01, 0.1, 0.01):
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

        for file in files:
            print(file)
            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            raw = False
            if not raw:
                df_combined = generate_PCA_Data(data)
            else:
                df_combined = data
            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.ResultCheck(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.perform_OPTICS(i)

            Do_Result_Save.ls_table(Do_Clustering.OPTIC, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.OPTIC_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.OPTIC)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.OPTIC_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df, "_", save=False)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result(output_dir,'OPTICS', result_df, file_names, FTSE=True, apply_log=True, new_Plot=False)
    save_cluster_info('OPTICS', stat_lists, file_names)

# hyper parameter bandwidth percentile np.range(0.1, 1, 0.1) should be tested manually.(arbitrarily) Done!
meanshift_Save = False
if meanshift_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

    for i in np.arange(0.1, 1, 0.1):
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

        for file in files:
            print(file)
            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            raw = False
            if not raw:
                df_combined = generate_PCA_Data(data)
            else:
                df_combined = data
            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.ResultCheck(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.perform_meanshift(i)

            Do_Result_Save.ls_table(Do_Clustering.meanshift, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.meanshift_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.meanshift)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.meanshift_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df, "_", save=False)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result(output_dir,'meanshift', result_df, file_names, FTSE=True, apply_log=True, new_Plot=False)
    save_cluster_info('meanshift', stat_lists, file_names)

# hyper parameter n components [3,5,10,20,30,40,50,60,70] should be tested manually.(arbitrarily) more..
GMM_Save = False
if GMM_Save:
    file_names = []
    result_df = pd.DataFrame()
    stat_lists = []

    input_dir = '../files/characteristics'
    files = sorted(filename for filename in os.listdir(input_dir))
    cl = 0
    outliers_count = 0
    figure = 0

    for i in [2, 3, 5, 10, 20, 30, 40, 50, 60]:
        file_names.append(f'{i}')
        LS_merged_df = pd.DataFrame()

        for file in files:
            print(file)
            # convert mom_data into PCA_data
            data = read_and_preprocess_data(input_dir, file)
            raw = False
            if not raw:
                df_combined = generate_PCA_Data(data)
            else:
                df_combined = data
            # Call initial method
            Do_Clustering = C.Clustering(df_combined)
            Do_Result_Save = C.ResultCheck(df_combined)

            # Do clustering and get 2D list of cluster index
            Do_Clustering.perform_GMM(i)

            Do_Result_Save.ls_table(Do_Clustering.Gaussian, output_dir, file, save=False, raw=raw)

            LS_merged_df = merge_LS_Table(Do_Result_Save.table, LS_merged_df, file)

            cl += len((set(Do_Clustering.Gaussian_labels)))
            outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Gaussian)
            figure += Do_Result_Save.count_stock_of_traded()

            print("Number of clusters is:", len(set(Do_Clustering.Gaussian_labels)))

        result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df, "_", save=False)
        cl = int(cl / len(files))
        outliers_count = outliers_count / len(files)
        figure = figure / len(files)
        stat_list = [cl, 1 - outliers_count, outliers_count, figure]
        stat_lists.append(stat_list)
        print(f'avg of clusters: {cl}')
        print(f'total outliers: {outliers_count}')
        print(f'number of stock traded: {figure}')

    save_and_plot_result(output_dir,'GMM', result_df, file_names, FTSE=True, apply_log=True, new_Plot=False)
    save_cluster_info('GMM', stat_lists, file_names)
