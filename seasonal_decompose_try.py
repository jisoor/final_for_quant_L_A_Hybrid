import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from pmdarima import auto_arima
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping
from talib import abstract
import json
#  pip install -r requirements.txt
import talib
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



class_name = 'Nasdaq'
val_df = pd.DataFrame(columns=['class_name', 'Lstm_loss', 'Arima_order'])  # 컬럼으로 이루어진 데이터프레임 만들기
val_df.set_index('class_name', inplace=True)  # 인덱스 설정
val_df.loc['Nasdaq'] = np.nan  # 인덱스 행 추가 (np.nan)으로

# 인덱스 datetime
data = pd.read_csv('NASDAQ Composite_2007-01-03-2022-01-25.csv', header=0).tail(1500)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

print(data)
print(type(data.index))

# Initialize moving averages from Ta-Lib, store functions in dictionary
talib_moving_averages = ['SMA', 'EMA', 'WMA', 'DEMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'T3', 'TEMA', 'TRIMA']
functions = {}
for ma in talib_moving_averages:
    functions[ma] = abstract.Function(ma)

print('SMA', functions['SMA'])

data = data.rename(columns={'Adj Close': 'close'})  # 이름을 'close' 'High', 'Low', 'close' 'change' 인덱스는 걍 순서
data = data.rename(columns={'High': 'high'})
data = data.rename(columns={'Low': 'low'})
data.drop(['Change'], axis=1, inplace=True)

print('데이터 길이', len(data))  # 1500

# 이동평균 기간 4~99까지 에서 최적의 첨도 K구하기
kurtosis_results = {'period': []}
for i in range(4, 100):
    kurtosis_results['period'].append(i)
    for ma in talib_moving_averages:  # ['SMA', 'EMA', 'WMA', 'DEMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'T3', 'TEMA', 'TRIMA'] 총 10개
        # 1500개에서 252개 뺴고 1248개만 train?. 마지막 60.. 60일 이동평균?
        ma_output = functions[ma](data[:-252], i).tail(
            60)  # input_arrays parameter missing required data keys: high, low
        k = kurtosis(ma_output, fisher=False)  # 60개 데이터들에 대한 첨도값을 구해줌(솟아오른 정도)
        if ma not in kurtosis_results.keys():  # key값이 없다면
            kurtosis_results[ma] = []  # { key : [k 첨도 값_SMA, k 첨도 값_EMA, k 첨도 값_WMA ...... , k 첨도 값_TRIMA  }
        kurtosis_results[ma].append(k)

kurtosis_results = pd.DataFrame(kurtosis_results)  # ( 11, 96) DF
# kurtosis_results.to_csv('./datasets_3/kurtosis_results_without_change.csv', index=True)

optimized_period = pd.DataFrame(index=['period'])  # 이것인가..<= 데이터프레임 만들어줘야함

for ma in talib_moving_averages:
    difference = np.abs(kurtosis_results[ma] - 3)  # 각각 값에 대해 뺴져서 리스트로?  / 최적의 첨도값 3과의 차이의 절대값
    print(type(difference))  # <class 'pandas.core.series.Series'>
    df = pd.DataFrame({'difference': difference, 'period': kurtosis_results['period']})
    df = df.sort_values(by=['difference'], ascending=True).reset_index(drop=True)
    print(df.head())
    if df.at[0, 'difference'] < 3 * 0.05:  # at은 loc랑 비슷  / df.at[0, difference]는 첨도가 3의 가장 가까운 값 => period
        optimized_period[ma] = [df.at[0, 'period']]  # 컬럼값 넣어주려면 값이 하나라도 리스트여야 함.
        print('최적의 이동평균은? ', df.at[0, 'period'])
    else:
        print(ma + ' is not viable, best K greater or less than 3 +/-5%')
print('\nOptimized periods 이평 길이에대한 df:', optimized_period)  # df 인덱스가 기본이고, 열은 'Midprice' 값은 하나의 최적 period가 들어가있음
print(type(optimized_period))

simulation = {}
for ma in optimized_period.columns:
    # 저변동성 / 고 변동성 시계열로 각각 나누기

    print(type(data[['close','high', 'low']]))
    low_vol = functions[ma](data[['close','high', 'low']], int(optimized_period.loc['period'][ma]))  # int로 만들어줘야./ 총 1248곘지만 앞에 이평 길이-1 만큼 Nan값
    low_vol.dropna()
    low_vol.index = pd.to_datetime(low_vol.index)
    low_vol_monthly = low_vol.resample('M').sum()

    print(type(low_vol)) # series
    print(low_vol)
    print(type(low_vol.index))

    print('======== low vol ========= 을 분해햅자')
    result_additive = seasonal_decompose(low_vol_monthly, model='additive')
    fig = result_additive.plot()

    acf = plot_acf(data['close'])
    acf0 = plot_acf(low_vol)  # <= 시리즈값만 인식해서 열을 따로 불러와줌  # 보면, 이동평균 해준 low_vol data는 v완벽히 정상성이 되었음
    plt.show()


    # Generate ARIMA and LSTM predictions
    print('\nWorking on ' + ma + ' predictions')