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
import yfinance as yf
import datetime
from tensorflow.keras.models import load_model



import pickle
# h = np.array(high)
# l = np.array(low)
# c = np.array(close)
# output_atr = np.array(talib.ATR(h,l,c,14))

# 내일값 예측위해 필요한 데이터수
# 가장 최신데이터의 (period-1) + (lstm_len-1)

def mean_absolute_percentage_error(actual, prediction):
    actual = pd.Series(actual)
    prediction = pd.Series(prediction)
    return 100 * np.mean(np.abs((actual - prediction))/actual)


if __name__ == '__main__':
    # 티커명 참조해서 csv 다운로드 받기
    futures = [('BZ=F', 'BRENT_OIL'), ('CC=F', 'COCOA'), ('KC=F', 'Coffee'), ('HG=F', 'COPPER'), ('ZC=F', 'CORN'),('CT=F', 'COTTON'),
               ('CL=F', 'CRUDE_OIL'), ('YM=F', 'DOW'), ('GF=F', 'FEEDER_CATTLE'), ('GC=F', 'GOLD'), ('HE=F', 'LEAN_HOGS'), ('LE=F', 'LIVE_CATTLE'),
               ('LBS=F', 'LUMBER'), ('NQ=F', 'NASDAQ'), ('NG=F', 'NATURAL_GAS'), ('ZO=F', 'OAT'), ('PA=F', 'PALLADIUM'),('PL=F', 'PLATINUM'), ('ZR=F', 'ROUGH_RICE'),
               ('RTY=F', 'RUSSEL2000'), ('SI=F', 'SILVER'), ('ZS=F', 'SOYBEAN'), ('ZM=F', 'SOYBEAN_MEAL'), ('ZL=F', 'SOYBEAN_OIL'),
               ('ES=F', 'SPX'), ('SB=F', 'SUGAR'), ('ZT=F', 'US2YT'), ('ZF=F', 'US5YT'), ('ZN=F', 'US10YT'),('ZB=F', 'US30YT'), ('KE=F', 'WHEAT')]
    class_name = 'brent_oil'


    # 기존의 ma와 이동평균 가져오기
    ma = 'TRIMA'
    optimized_period = 64
    lstm_len = 4
    Today = datetime.date.today()


    # 2022-01-26 부터 필요할 듯함( 기존에 트레인 시킨 마지막 데이타 이후로부터)
    # 2022-01-26 ~ 02-09
    data = yf.download('BZ=F', start='2021-06-27', end=Today) #적당히 6~8개월 어치 데이터 가져오기
    data = data.reset_index(drop=True)
    print(len(data)) # 1/26 ~ 2/9(어제)까지 대략 12개의 종가를 예측, backtesting 시켜보자 .(2/10일은 공휴일이었나봄)  # 159개

    # 우리가 예측을 위해 필요한 데이터는 (64-1) + (4-1) = 최신 101개이여야 함
    data_len = ((optimized_period-1) + (lstm_len-1))*-1
    data = data[data_len:]

    # Initialize moving averages from Ta-Lib, store functions in dictionary
    talib_moving_averages = ['SMA', 'EMA', 'WMA', 'DEMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'T3', 'TEMA', 'TRIMA']
    functions = {}
    for ma in talib_moving_averages:
        functions[ma] = abstract.Function(ma)

    # CSV should have columns: ['date', 'open', 'high', 'low', 'close', 'volume']
    data = data.rename(columns={'Adj Close':'close'})  # 이름을 'close' 'High', 'Low', 'close' 'change' 인덱스는 걍 순서
    data = data.rename(columns={'High': 'high'})
    data = data.rename(columns={'Low': 'low'})
    data = data.rename(columns={'Date': 'date'})
    data.drop(['Open','Close', 'Volume'], axis=1, inplace=True)  # drop 열 추가
    print(data)


    # 저변동성 / 고 변동성 시계열로 각각 나누기
    simulation = {}
    low_vol = functions[ma](data, optimized_period) # int로 만들어줘야./ 총 1248곘지만 앞에 이평 길이-1 만큼 Nan값
    print(low_vol)  # 3개 뺴고 다 난값
    high_vol = data['close'] - low_vol
    print(high_vol)  #  3개 뺴고 다 난값

    # Generate ARIMA and LSTM predictions

    # 모델 불러오기
    with open('./models/{}_Arima_model.pickle'.format(class_name), 'rb') as f:
        model = pickle.load(f)
    # order 튜플로 불러오기.
    order = (2, 1, 2)

    ############## 어떻게 될지 모르겠음 . 시도해보기 #############################
    # data = 2022-01-26 ~ 2022-02-10(가장 최신 종가데이터)를 넣어서 내일 값 예측시키기.
    predict_arima_price = low_vol[-1:] # 마지막 1개 가져와서 예측.
    model = pm.ARIMA(order=order)
    model.fit(predict_arima_price)  # 우리는 2007년 ~ 2022-01-25까지 다 train시킨 모델을 가져온 것이기 떄문에 01-26부터 데이터만 추가시키면 되지않낭?
    arima_prediction = model.predict()[0]  # ㅁ내일 값 예측
    print('arima 내일 예측 값:', arima_prediction) # 내일 예측 값: 79.01496184233464


    # LSTM 예측
    predict_lstm_price = high_vol[-4:]
    dataset = np.reshape(data.values, (1, lstm_len, 1))  # ( 1, lstm_len, 1) 아닌강>?
    model = load_model('./models/{}_Lstm_model.h5'.format(class_name))
    with open('./minmaxscaler/{}_minmaxscaler.pickle'.format(class_name), 'rb') as f:
        minmaxscaler = pickle.load(f)
    dataset_scaled = minmaxscaler.transform(dataset)
    lstm_prediction = model.predict(dataset_scaled)  # dataset_scaled(1, 30, 1)안해줘도??
    prediction = minmaxscaler.inverse_transform(lstm_prediction)
    print('lstm 내일 예측 값 :', lstm_prediction)

    final_prediction = arima_prediction + lstm_prediction
    print('최종 내일 예측 값은?? ====> ', final_prediction)




