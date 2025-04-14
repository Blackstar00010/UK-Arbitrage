import Clustering as C
from PCA_and_ETC import *

Reversal_Save = False
if Reversal_Save:
    input_dir = '../files/characteristics_us'
    output_dir = '../files/clustering_result/Reversal'
    files = sorted(filename for filename in os.listdir(input_dir))

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        Do_Result_Save = C.ResultCheck(data)

        # Save LS_Table CSV File
        Do_Result_Save.reversal_table(data, output_dir, file)

K_mean_Save = True
if K_mean_Save:
    input_dir = '../files/characteristics_us'
    output_dir = '../files/clustering_result/K_mean'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    top_df = pd.DataFrame(columns=['month', 1, 2, 3, 'number of clusters'])

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_kmeans(100)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.K_Mean)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.K_Mean, output_dir, file, save=True, raw=False)

        if True:
            # 각 sublist의 원소 개수를 저장할 리스트 생성
            sublist_lengths = [len(sublist) for sublist in Do_Clustering.K_Mean]

            sublist_lengths=sublist_lengths[1:]

            # sublist_lengths를 기반으로 top3 원소 개수를 찾음
            top3_lengths = sorted(set(sublist_lengths), reverse=True)[:2]

            if len(top3_lengths) == 1:
                top3_lengths.append(0)

            if len(top3_lengths) == 0:
                top3_lengths.append(0)
                top3_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    1: sum(top3_lengths),
                                    2: [top3_lengths[0]],
                                    3: [top3_lengths[1]],
                                    'number of clusters': [len(sublist_lengths)]})

            # 이 새로운 행을 기존 DataFrame에 추가합니다.
            top_df = pd.concat([top_df, new_row], ignore_index=True)

    top_df.to_csv(os.path.join('../files/result2/', 'K_Mean.csv'), index=False)

dbscan_Save = False
if dbscan_Save:
    input_dir = '../files/characteristics_us'
    output_dir = '../files/clustering_result/DBSCAN'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    top_df = pd.DataFrame(columns=['month', 1, 2, 3, 'number of clusters'])

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_DBSCAN(0.9)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.DBSCAN)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.DBSCAN, output_dir, file, save=True, raw=False)

        if True:
            # 각 sublist의 원소 개수를 저장할 리스트 생성
            sublist_lengths = [len(sublist) for sublist in Do_Clustering.DBSCAN]

            sublist_lengths=sublist_lengths[1:]

            # sublist_lengths를 기반으로 top3 원소 개수를 찾음
            top3_lengths = sorted(set(sublist_lengths), reverse=True)[:2]

            if len(top3_lengths) == 1:
                top3_lengths.append(0)

            if len(top3_lengths) == 0:
                top3_lengths.append(0)
                top3_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    1: sum(top3_lengths),
                                    2: [top3_lengths[0]],
                                    3: [top3_lengths[1]],
                                    'number of clusters': [len(sublist_lengths)]})

            # 이 새로운 행을 기존 DataFrame에 추가합니다.
            top_df = pd.concat([top_df, new_row], ignore_index=True)

    top_df.to_csv(os.path.join('../files/result2/', 'DBSCAN.csv'), index=False)

Agglormerative_Save = False
if Agglormerative_Save:
    input_dir = '../files/characteristics_us'
    output_dir = '../files/clustering_result/Agglomerative'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    top_df = pd.DataFrame(columns=['month', 1, 2, 3, 'number of clusters'])

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_HA(0.8)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Agglomerative)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.Agglomerative, output_dir, file, save=True, raw=False)


        if True:
            # 각 sublist의 원소 개수를 저장할 리스트 생성
            sublist_lengths = [len(sublist) for sublist in Do_Clustering.Agglomerative]

            sublist_lengths=sublist_lengths[1:]

            # sublist_lengths를 기반으로 top3 원소 개수를 찾음
            top3_lengths = sorted(set(sublist_lengths), reverse=True)[:2]

            if len(top3_lengths) == 1:
                top3_lengths.append(0)

            if len(top3_lengths) == 0:
                top3_lengths.append(0)
                top3_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    1: sum(sublist_lengths),
                                    2: [top3_lengths[0]],
                                    3: [top3_lengths[1]],
                                    'number of clusters': [len(sublist_lengths)]})

            # 이 새로운 행을 기존 DataFrame에 추가합니다.
            top_df = pd.concat([top_df, new_row], ignore_index=True)

    top_df.to_csv(os.path.join('../files/result2/', 'Agglo.csv'), index=False)

Bisecting_Save = False
if Bisecting_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Bisecting_K_mean'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    top_df = pd.DataFrame(columns=['month', 1, 2, 3, 'number of clusters'])

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_bisectingkmeans(25)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.bisecting_K_mean)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.bisecting_K_mean, output_dir, file, save=True, raw=False)

        if True:
            # 각 sublist의 원소 개수를 저장할 리스트 생성
            sublist_lengths = [len(sublist) for sublist in Do_Clustering.bisecting_K_mean]

            sublist_lengths=sublist_lengths[1:]

            # sublist_lengths를 기반으로 top3 원소 개수를 찾음
            top3_lengths = sorted(set(sublist_lengths), reverse=True)[:2]

            if len(top3_lengths) == 1:
                top3_lengths.append(0)

            if len(top3_lengths) == 0:
                top3_lengths.append(0)
                top3_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    1: sum(top3_lengths),
                                    2: [top3_lengths[0]],
                                    3: [top3_lengths[1]],
                                    'number of clusters': [len(sublist_lengths)]})

            # 이 새로운 행을 기존 DataFrame에 추가합니다.
            top_df = pd.concat([top_df, new_row], ignore_index=True)

    top_df.to_csv(os.path.join('../files/result2/', 'Bisecting.csv'), index=False)

hdbscan_Save = False
if hdbscan_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/HDBSCAN'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    top_df = pd.DataFrame(columns=['month', 1, 2, 3, 'number of clusters'])

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_HDBSCAN(0.8)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.HDBSCAN)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.HDBSCAN, output_dir, file, save=True, raw=False)

        if True:
            # 각 sublist의 원소 개수를 저장할 리스트 생성
            sublist_lengths = [len(sublist) for sublist in Do_Clustering.HDBSCAN]

            sublist_lengths=sublist_lengths[1:]

            # sublist_lengths를 기반으로 top3 원소 개수를 찾음
            top3_lengths = sorted(set(sublist_lengths), reverse=True)[:2]

            if len(top3_lengths) == 1:
                top3_lengths.append(0)

            if len(top3_lengths) == 0:
                top3_lengths.append(0)
                top3_lengths.append(0)


            new_row = pd.DataFrame({'month': [file[:-4]],
                                    1: sum(top3_lengths),
                                    2: [top3_lengths[0]],
                                    3: [top3_lengths[1]],
                                    'number of clusters': [len(sublist_lengths)]})

            # 이 새로운 행을 기존 DataFrame에 추가합니다.
            top_df = pd.concat([top_df, new_row], ignore_index=True)

    top_df.to_csv(os.path.join('../files/result2/', 'HDBSCAN.csv'), index=False)

birch_Save = False
if birch_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/BIRCH'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    top_df = pd.DataFrame(columns=['month', 1, 2, 3, 'number of clusters'])

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_BIRCH(0.9)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.BIRCH)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.BIRCH, output_dir, file, save=True, raw=False)

        if True:
            # 각 sublist의 원소 개수를 저장할 리스트 생성
            sublist_lengths = [len(sublist) for sublist in Do_Clustering.BIRCH]

            sublist_lengths=sublist_lengths[1:]

            # sublist_lengths를 기반으로 top3 원소 개수를 찾음
            top3_lengths = sorted(set(sublist_lengths), reverse=True)[:2]

            if len(top3_lengths) == 1:
                top3_lengths.append(0)

            if len(top3_lengths) == 0:
                top3_lengths.append(0)
                top3_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    1: sum(top3_lengths),
                                    2: [top3_lengths[0]],
                                    3: [top3_lengths[1]],
                                    'number of clusters': [len(sublist_lengths)]})

            # 이 새로운 행을 기존 DataFrame에 추가합니다.
            top_df = pd.concat([top_df, new_row], ignore_index=True)

    top_df.to_csv(os.path.join('../files/result2/', 'BIRCH.csv'), index=False)

optics_Save = False
if optics_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/OPTICS'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    top_df = pd.DataFrame(columns=['month', 1, 2, 3, 'number of clusters'])

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_OPTICS(0.04)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.OPTIC)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.OPTIC, output_dir, file, save=True, raw=False)

        if True:
            # 각 sublist의 원소 개수를 저장할 리스트 생성
            sublist_lengths = [len(sublist) for sublist in Do_Clustering.OPTIC]

            sublist_lengths=sublist_lengths[1:]

            # sublist_lengths를 기반으로 top3 원소 개수를 찾음
            top3_lengths = sorted(set(sublist_lengths), reverse=True)[:2]

            if len(top3_lengths) == 1:
                top3_lengths.append(0)

            if len(top3_lengths) == 0:
                top3_lengths.append(0)
                top3_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    1: sum(top3_lengths),
                                    2: [top3_lengths[0]],
                                    3: [top3_lengths[1]],
                                    'number of clusters': [len(sublist_lengths)]})

            # 이 새로운 행을 기존 DataFrame에 추가합니다.
            top_df = pd.concat([top_df, new_row], ignore_index=True)

    top_df.to_csv(os.path.join('../files/result2/', 'OPTICS.csv'), index=False)

meanshift_Save = False
if meanshift_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/Meanshift'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    top_df = pd.DataFrame(columns=['month', 1, 2, 3, 'number of clusters'])

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_meanshift(0.8)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.meanshift)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.meanshift, output_dir, file, save=True, raw=False)

        if True:
            # 각 sublist의 원소 개수를 저장할 리스트 생성
            sublist_lengths = [len(sublist) for sublist in Do_Clustering.meanshift]

            sublist_lengths=sublist_lengths[1:]

            # sublist_lengths를 기반으로 top3 원소 개수를 찾음
            top3_lengths = sorted(set(sublist_lengths), reverse=True)[:2]

            if len(top3_lengths) == 1:
                top3_lengths.append(0)

            if len(top3_lengths) == 0:
                top3_lengths.append(0)
                top3_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    1: sum(top3_lengths),
                                    2: [top3_lengths[0]],
                                    3: [top3_lengths[1]],
                                    'number of clusters': [len(sublist_lengths)]})

            # 이 새로운 행을 기존 DataFrame에 추가합니다.
            top_df = pd.concat([top_df, new_row], ignore_index=True)

    top_df.to_csv(os.path.join('../files/result2/', 'meanshift.csv'), index=False)

GMM_Save = False
if GMM_Save:
    input_dir = '../files/characteristics'
    output_dir = '../files/clustering_result/GMM'
    files = sorted(filename for filename in os.listdir(input_dir))
    outliers_count = 0
    top_df = pd.DataFrame(columns=['month', 1, 2, 3, 'number of clusters'])

    for file in files:
        print(file)

        # convert mom_data into PCA_data
        data = read_and_preprocess_data(input_dir, file)
        df_combined = generate_PCA_Data(data)

        # Call initial method
        Do_Clustering = C.Clustering(df_combined)
        Do_Result_Save = C.ResultCheck(df_combined)

        # Do clustering and get 2D list of cluster index
        Do_Clustering.perform_GMM(60)
        outliers_count += Do_Result_Save.count_outlier(Do_Clustering.Gaussian)

        # Save LS_Table CSV File
        Do_Result_Save.ls_table(Do_Clustering.Gaussian, output_dir, file, save=True, raw=False)

        if True:
            # 각 sublist의 원소 개수를 저장할 리스트 생성
            sublist_lengths = [len(sublist) for sublist in Do_Clustering.Gaussian]

            sublist_lengths=sublist_lengths[1:]

            # sublist_lengths를 기반으로 top3 원소 개수를 찾음
            top3_lengths = sorted(set(sublist_lengths), reverse=True)[:2]

            if len(top3_lengths) == 1:
                top3_lengths.append(0)

            if len(top3_lengths) == 0:
                top3_lengths.append(0)
                top3_lengths.append(0)

            new_row = pd.DataFrame({'month': [file[:-4]],
                                    1: sum(top3_lengths),
                                    2: [top3_lengths[0]],
                                    3: [top3_lengths[1]],
                                    'number of clusters': [len(sublist_lengths)]})

            # 이 새로운 행을 기존 DataFrame에 추가합니다.
            top_df = pd.concat([top_df, new_row], ignore_index=True)

    top_df.to_csv(os.path.join('../files/result2/', 'GMM.csv'), index=False)
