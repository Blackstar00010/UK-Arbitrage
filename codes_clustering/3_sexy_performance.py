from PCA_and_ETC import *
from scipy.stats.mstats import winsorize
# turn off warning
warnings.filterwarnings("ignore")

MOM_merged_df = pd.read_csv('../files/mom1_data_combined_adj_close2.csv')
MOM_merged_df.set_index('Firm Name', inplace=True)
MOM_merged_df.drop(MOM_merged_df.columns[0], axis=1, inplace=True)
count3 = 0
count4 = 0
count5 = 0
count6 = 0

count3 += MOM_merged_df[MOM_merged_df >= 0].count().sum()
count4 += MOM_merged_df[MOM_merged_df < 0].count().sum()
count5 += MOM_merged_df[MOM_merged_df == 0].count().sum()
count6 += MOM_merged_df.isna().sum().sum()

count_greater_than_0_5 = (MOM_merged_df >= 0.5).any(axis=1).sum()
count_less_than_0_5 = (MOM_merged_df <= -0.5).any(axis=1).sum()
count_both_0_5 = ((MOM_merged_df <= -0.5) | (MOM_merged_df >= 0.5)).any(axis=1).sum()

print("\n-0.5보다 작고 0.5보다 큰 숫자가 있는 행의 개수:", count_both_0_5)
print("0.5보다 큰 숫자가 있는 행의 개수:", count_greater_than_0_5)
print("-0.5보다 작은 숫자가 있는 행의 개수:", count_less_than_0_5)
print('0보다 큰 숫자가 있는 칸 갯수:', count3)
print('0보다 작은 숫자가 있는 칸 갯수:', count4)
print('mom1=0인 칸 갯수:', count5)
t = 9749 * 391 - count6
print('NaN이 아닌 칸 갯수:', t)
print(MOM_merged_df.shape)
print("\nWinsorized Data:")
print("Min:", np.min(MOM_merged_df))
print("Max:", np.max(MOM_merged_df))
print("Mean:", np.mean(MOM_merged_df))

base_directory = '../files/clustering_result/'
output_dir='../files/result'
output_dir2='../files/Long'
output_dir3='../files/Short'

# Get all subdirectories in the base directory
subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

file_names = []
result_df = pd.DataFrame()
result_df2 = pd.DataFrame()
result_df3 = pd.DataFrame()

# Save subdir name in file_names at the beginning.
for subdir in subdirectories:
    # DBSCAN / Gaussian_Mixture_Model / HDBSCAN / Hierarchical_Agglomerative / ...
    file_names.append(subdir)

file_names2 = [name for name in file_names if name not in ['Cointegration']]

for subdir in subdirectories:
    print(subdir)
    directory = os.path.join(base_directory, subdir)

    LS_merged_df = pd.DataFrame()

    files = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))
    for file in files:
        data = pd.read_csv(os.path.join(directory, file))
        LS_merged_df = merge_LS_Table(data, LS_merged_df, file)
        LS_merged_df = LS_merged_df[~LS_merged_df.iloc[:,0].duplicated(keep='first')]

    result_df = product_LS_Table(LS_merged_df, MOM_merged_df, result_df, subdir, save=False)

save_and_plot_result(output_dir,'total', result_df, file_names, FTSE=False, apply_log=True, new_Plot=True)

# for subdir in subdirectories:
#     print(subdir)
#     directory = os.path.join(base_directory, subdir)
#
#     if subdir == 'Cointegration':
#         continue
#
#     # elif subdir == 'Reversal':
#     #     continue
#
#     LS_merged_df2 = pd.DataFrame()
#     LS_merged_df3 = pd.DataFrame()
#
#     files = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))
#     for file in files:
#         data = pd.read_csv(os.path.join(directory, file))
#         LS_merged_df2, LS_merged_df3 = merge_Long_and_Short_Table(data, LS_merged_df2, LS_merged_df3, file)
#
#     result_df2 = product_LS_Table(LS_merged_df2, MOM_merged_df, result_df2, subdir, save=False)
#     result_df3 = product_LS_Table(LS_merged_df3, MOM_merged_df, result_df3, subdir, save=False)
#
# save_and_plot_result(output_dir2,'total', result_df2, file_names2, FTSE=False, apply_log=True, new_Plot=True)
# save_and_plot_result(output_dir3,'total', result_df3, file_names2, FTSE=False, apply_log=True, new_Plot=True)
