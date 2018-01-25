import os
import numpy as np
import pandas as pd
from subprocess import check_output
#print(check_output(["ls", "./sub"]).decode("utf8"))

# Global parameter
# Mean Median PushOut+Median MinMax+Mean MinMax+Median  'MinMax+BestBase' 'MinMax+Mean'
stacking_mode = 'Mean'
base_path = '/data/zrb/Iceberg-Classifier/output/10-crop-ResNet50+InceptionV3-2018-01-23-23-48/'
filelist = [
    'ResNet50+InceptionV3-2018-01-16-20-34/',
    'ResNet50+InceptionV3-2018-01-16-11-05/',
    'ResNet50+InceptionV3-2018-01-22-15-46/'
]


for m in filelist:
    #data load
    sub_path = base_path+m
    all_files = os.listdir(sub_path)
    print(all_files)

    '''
    #build dir
    if not os.path.exists('./stacking_sub'):
        print("Mkdir stacking_sub")
        os.makedirs('./stacking_sub')
    '''

    # Read and concatenate submissions
    outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
    concat_sub = pd.concat(outs, axis=1)
    cols = list(map(lambda x: "is_iceberg_" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)
    #sub_base = pd.read_csv(base_path)

    # get the data fields ready for stacking
    concat_sub['is_iceberg_max'] = concat_sub.iloc[:, 1:len(concat_sub.columns)].max(axis=1)
    concat_sub['is_iceberg_min'] = concat_sub.iloc[:, 1:len(concat_sub.columns)].min(axis=1)
    concat_sub['is_iceberg_mean'] = concat_sub.iloc[:, 1:len(concat_sub.columns)].mean(axis=1)
    concat_sub['is_iceberg_median'] = concat_sub.iloc[:, 1:len(concat_sub.columns)].median(axis=1)
    #concat_sub['is_iceberg_base'] = sub_base['is_iceberg']

    #Minimum threshold
    cutoff_lo = 0.8
    cutoff_hi = 0.2

    print("Stacking mode:" + stacking_mode)
    if stacking_mode == 'Mean':
        concat_sub['is_iceberg'] = concat_sub['is_iceberg_mean']
        concat_sub[['id', 'is_iceberg']].to_csv(base_path+m + '/stack_mean.csv',index=False, float_format='%.6f')
    elif stacking_mode == 'Median':
        concat_sub['is_iceberg'] = concat_sub['is_iceberg_median']
        concat_sub[['id', 'is_iceberg']].to_csv('./stacking_sub/stack_median.csv',index=False, float_format='%.6f')
    elif stacking_mode == 'PushOut+Median':
        concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:, 1:len(concat_sub.columns)] > cutoff_lo, axis=1), 1,
                                            np.where(np.all(concat_sub.iloc[:, 1:len(concat_sub.columns)] < cutoff_hi, axis=1),
                                                     0, concat_sub['is_iceberg_median']))
        concat_sub[['id', 'is_iceberg']].to_csv('./stacking_sub/stack_pushout_median.csv',index=False, float_format='%.6f')
    elif stacking_mode == 'MinMax+Mean':
        concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:, 1:len(concat_sub.columns)] > cutoff_lo, axis=1),
                                            concat_sub['is_iceberg_max'],
                                            np.where(np.all(concat_sub.iloc[:, 1:len(concat_sub.columns)] < cutoff_hi, axis=1),
                                                     concat_sub['is_iceberg_min'],
                                                     concat_sub['is_iceberg_mean']))
        concat_sub[['id', 'is_iceberg']].to_csv('./stacking_sub/stack_minmax_mean.csv',index=False, float_format='%.6f')
    elif stacking_mode == 'MinMax+Median':
        concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:, 1:len(concat_sub.columns)] > cutoff_lo, axis=1),
                                            concat_sub['is_iceberg_max'],
                                            np.where(np.all(concat_sub.iloc[:, 1:len(concat_sub.columns)] < cutoff_hi, axis=1),
                                                     concat_sub['is_iceberg_min'],
                                                     concat_sub['is_iceberg_median']))
        concat_sub[['id', 'is_iceberg']].to_csv('./stacking_sub/stack_minmax_median.csv',index=False, float_format='%.6f')
    elif stacking_mode == 'MinMax+BestBase':
        concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:, 1:len(concat_sub.columns)] > cutoff_lo, axis=1),
                                            concat_sub['is_iceberg_max'],
                                            np.where(np.all(concat_sub.iloc[:, 1:len(concat_sub.columns)] < cutoff_hi, axis=1),
                                                     concat_sub['is_iceberg_min'],
                                                     concat_sub['is_iceberg_base']))
        concat_sub[['id', 'is_iceberg']].to_csv('./stacking_sub/stack_minmax_bestbase.csv',index=False, float_format='%.6f')
    else:
        print("Stacking mode error, default MinMax+BestBase")
        concat_sub['is_iceberg'] = np.where(np.all(concat_sub.iloc[:, 1:len(concat_sub.columns)] > cutoff_lo, axis=1),
                                            concat_sub['is_iceberg_max'],
                                            np.where(
                                                np.all(concat_sub.iloc[:, 1:len(concat_sub.columns)] < cutoff_hi, axis=1),
                                                concat_sub['is_iceberg_min'],
                                                concat_sub['is_iceberg_base']))
        concat_sub[['id', 'is_iceberg']].to_csv('./stacking_sub/stack_minmax_bestbase.csv', index=False, float_format='%.6f')
