import pandas as pd
from scipy import interpolate
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt


def preprocess():
    # Load data
    raw_acc = pd.read_csv('dataset/20210917_1_이루민_2/E4AccData.csv', usecols=['Time', 'Acc_X', 'Acc_Y', 'Acc_Z'])
    raw_gsr = pd.read_csv('dataset/20210917_1_이루민_2/E4GsrData.csv', usecols=['Time', 'GSR'])
    raw_tmp = pd.read_csv('dataset/20210917_1_이루민_2/E4TmpData.csv', usecols=['Time', 'Tmp'])
    raw_sk = pd.read_csv('dataset/20210917_1_이루민_2/SkeletonData.csv', skiprows=[0], header=None)
    raw_pf = pd.read_csv('dataset/20210917_1_이루민_2/PerformanceData.csv', usecols=['Time', 'Performance'])
    raw_lb = pd.read_csv('dataset/20210917_1_이루민_2/LabelingData.csv', usecols=['Time', 'Labeling'])

    """
    1. 각 데이터를 0.03125초 간격으로 재구성 (Skeleton data는 부정확한 값 제거 후 재구성)
    2. 각 데이터 정규분포로 변환
    3. 데이터프레임 통합 후 csv 파일로 저장
    """

    # Labeling data
    raw_lb2 = raw_lb['Labeling']
    raw_lb2 = raw_lb2.round()

    under0 = raw_lb2 < 0
    raw_lb.loc[under0] = np.nan
    raw_lb = raw_lb.dropna()

    # Time setting
    lb_time = raw_lb.iloc[:, 0]

    useless_pf = raw_pf[raw_pf['Performance'] == -1]
    rpf_time = useless_pf['Time'].max()

    starttime = max(raw_acc['Time'][0], raw_gsr['Time'][0], raw_tmp['Time'][0], rpf_time)
    endtime = min(raw_acc['Time'][len(raw_acc) - 1], raw_gsr['Time'][len(raw_gsr) - 1],
                  raw_tmp['Time'][len(raw_tmp) - 1], raw_pf['Time'][len(raw_pf) - 1])

    start = lb_time > starttime
    end = lb_time < endtime
    time = lb_time.loc[start & end]

    # Final Labeling data
    final_lb = pd.DataFrame({'Time': time, 'Labeling': raw_lb2})

    # Acc data
    acc_time = raw_acc['Time']
    raw_acc_x = raw_acc['Acc_X']
    raw_acc_y = raw_acc['Acc_Y']
    raw_acc_z = raw_acc['Acc_Z']

    acc_xf = interpolate.interp1d(acc_time, raw_acc_x)
    acc_yf = interpolate.interp1d(acc_time, raw_acc_y)
    acc_zf = interpolate.interp1d(acc_time, raw_acc_z)

    acc_x = acc_xf(time)
    acc_y = acc_yf(time)
    acc_z = acc_zf(time)

    acc_x = ss.zscore(acc_x)
    acc_y = ss.zscore(acc_y)
    acc_z = ss.zscore(acc_z)

    final_acc = pd.DataFrame({'Time': time, 'Acc_X': acc_x, 'Acc_Y': acc_y, 'Acc_Z': acc_z})

    # GSR data
    gsr_time = raw_gsr['Time']
    raw_gsr2 = raw_gsr['GSR']
    gsr_f = interpolate.interp1d(gsr_time, raw_gsr2)
    gsr = gsr_f(time)
    gsr = ss.zscore(gsr)
    final_gsr = pd.DataFrame({'Time': time, 'GSR': gsr})

    # Tmp data
    tmp_time = raw_tmp['Time']
    raw_tmp2 = raw_tmp['Tmp']
    tmp_f = interpolate.interp1d(tmp_time, raw_tmp2)
    tmp = tmp_f(time)
    tmp = ss.zscore(tmp)
    final_tmp = pd.DataFrame({'Time': time, 'Tmp': tmp})

    # Skeleton data

    sk_time = raw_sk.iloc[:, 0]

    final_sk_before = pd.DataFrame({'Time': sk_time})
    final_sk_after = pd.DataFrame({'Time': time})

    for i in range(0, 19):
        df = pd.DataFrame()
        sk_x = df['X_{}'.format(i + 1)] = raw_sk.iloc[:, 1 + 4 * i]
        sk_y = df['Y_{}'.format(i + 1)] = raw_sk.iloc[:, 2 + 4 * i]
        sk_z = df['Z_{}'.format(i + 1)] = raw_sk.iloc[:, 3 + 4 * i]
        sk_confidence = df['Confidence'] = raw_sk.iloc[:, 4 + 4 * i]

        under50 = df['Confidence'] < 0.5
        df.loc[under50] = np.nan

        cdf = pd.DataFrame()
        cdf['X_{}'.format(i + 1)] = df['X_{}'.format(i + 1)].diff().fillna(np.nan)
        cdf['Y_{}'.format(i + 1)] = df['Y_{}'.format(i + 1)].diff().fillna(np.nan)
        cdf['Z_{}'.format(i + 1)] = df['Z_{}'.format(i + 1)].diff().fillna(np.nan)
        cdf['Confidence'] = df['Confidence']

        sk_x = ss.zscore(df['X_{}'.format(i + 1)], nan_policy='omit')
        sk_y = ss.zscore(df['Y_{}'.format(i + 1)], nan_policy='omit')
        sk_z = ss.zscore(df['Z_{}'.format(i + 1)], nan_policy='omit')

        final_sk_before['X_{}'.format(i + 1)] = sk_x
        final_sk_before['Y_{}'.format(i + 1)] = sk_y
        final_sk_before['Z_{}'.format(i + 1)] = sk_z

    #print('final_sk_before', final_sk_before)

    for i in range(0, 19):
        df = pd.DataFrame()

        sk_x2 = final_sk_before.iloc[:, 1 + 3 * i]
        sk_y2 = final_sk_before.iloc[:, 2 + 3 * i]
        sk_z2 = final_sk_before.iloc[:, 3 + 3 * i]

        sk_xf = interpolate.interp1d(sk_time, sk_x2, fill_value="extrapolate")
        sk_yf = interpolate.interp1d(sk_time, sk_y2, fill_value="extrapolate")
        sk_zf = interpolate.interp1d(sk_time, sk_z2, fill_value="extrapolate")

        sk_x3 = sk_xf(time)
        sk_y3 = sk_yf(time)
        sk_z3 = sk_zf(time)

        final_sk_after['X_{}'.format(i + 1)] = sk_x3
        final_sk_after['Y_{}'.format(i + 1)] = sk_y3
        final_sk_after['Z_{}'.format(i + 1)] = sk_z3

    #print('final_sk_after', final_sk_after)


    # Performance data
    pf_time = raw_pf['Time']
    raw_pf2 = raw_pf['Performance']

    pf_f = interpolate.interp1d(pf_time, raw_pf2)
    pf = pf_f(time)
    pf[pf[:] <= 1] = 1
    final_pf = pd.DataFrame({'Time': time, 'Performance': pf})

    # Data 모두 하나의 csv 파일로 통합

    AccGsr = pd.merge(final_acc, final_gsr, on='Time')
    AccGsrTmp = pd.merge(AccGsr, final_tmp, on='Time')
    AccGsrTmpSkeleton = pd.merge(AccGsrTmp, final_sk_after, on='Time')
    AccGsrTmpPerformance = pd.merge(AccGsrTmp, final_pf, on='Time')
    AccGsrTmpSkePer = pd.merge(AccGsrTmpSkeleton, final_pf, on='Time')

    merge1 = pd.merge(AccGsrTmp, final_lb, on='Time') # Acc + Gsr + Tmp + Labeling Data
    merge2 = pd.merge(AccGsrTmpSkeleton, final_lb, on='Time') # Acc + Gsr + Tmp + Skeleton + Labeling Data
    merge3 = pd.merge(AccGsrTmpPerformance, final_lb, on='Time') # Acc + Gsr + Tmp + Performance + Labeling Data
    merge4 = pd.merge(AccGsrTmpSkePer, final_lb, on='Time') # Acc + Gsr + Tmp + Skeleton + Performance + Labeling Data

    '''
    #merge1 = pd.merge(final_acc, final_gsr, on='Time')
    #merge2 = pd.merge(final_tmp, merge1, on='Time')
    merge3 = pd.merge(merge2, final_lb, on='Time')
    merge4 = pd.merge(final_pf, final_lb, on='Time')
    merge5 = pd.merge(merge2, merge4, on='Time')
    '''

    merge1.to_csv('dataset/20210917_1_이루민_2/이루민_1_ProcessedData_1.csv', index=False)
    merge2.to_csv('dataset/20210917_1_이루민_2/이루민_1_ProcessedData_2.csv', index=False)
    merge3.to_csv('dataset/20210917_1_이루민_2/이루민_1_ProcessedData_3.csv', index=False)
    merge4.to_csv('dataset/20210917_1_이루민_2/이루민_1_ProcessedData_4.csv', index=False)
