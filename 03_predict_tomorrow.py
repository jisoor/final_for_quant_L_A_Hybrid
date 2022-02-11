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


def get_arima(data, data_len): #
    # prepare train and test data
    # 모델 불러오기
    with open('model_pickle', 'rb') as f:
        model = pickle.load(f)
    # order 튜플로 불러오기.
    order = (3, 1, 3)
    data = data.reset_index(drop=True)

    ############## 어떻게 될지 모르겠음 . 시도해보기 #############################
    # data = 2022-01-26 ~ 2022-02-10(가장 최신 종가데이터)를 넣어서 내일 값 예측시키기.
    model = pm.ARIMA(order=order)
    model.fit(data)   # 우리는 2007년 ~ 2022-01-25까지 다 train시킨 모델을 가져온 것이기 떄문에 01-26부터 데이터만 추가시키면 되지않낭?
    prediction = model.predict()[0] # ㅁ내일 값 예측
    print('내일 예측 값:',  prediction)
    print(model.info())


    # 벡테스팅용 예측 ( 01/26, 01/27 .... 02/10거 다 예측해준 결과)
    # prediction = []
    # train = data.at[0, 'close'] # 맨첨 데이터는 한개짜리.
    # for i in range(len(data)):  # 2022-01-26 ~ 2022-02-10(가장 최신 종가데이터).
    #     model = pm.ARIMA(order=order)
    #     model.fit(train)  # 우리는 2007년 ~ 2022-01-25까지 다 train시킨 모델을 가져온 것이기 떄문에 01-26부터 데이터만 추가시키면 되지않낭?
    #     print('working on', i + 1, 'of', len(data), '-- ' + str(int(100 * (i + 1) /len(data))) + '% complete')
    #     prediction.append(model.predict()[0])  # 252개 데이터 예측.
    #     train.append(data.iloc[i+1]['close'])  # train list는 1450개가 됌. arima는 이전값이 필요하기 때문에 예측할 때도 이전값을 계속 업데이트 시킨다.
    # print('예측 값들:', prediction)

    # Generate error data
    # mse = mean_squared_error(data, prediction)
    # rmse = mse ** 0.5
    # mape = mean_absolute_percentage_error(pd.Series(data), pd.Series(prediction))
    # return prediction, mse, rmse, mape


def get_lstm(data, train_len, test_len, lstm_len=4):
    # prepare train and test data
    data = data.tail[-4:].reset_index(drop=True)
    dataset = np.reshape(data.values, (len(data), 1))  # ( 1450, 1)
    model = load_model('./models/{}_Lstm_model.h5'.format(class_name))
    with open('./minmaxscaler/{}_minmaxscaler.pickle'.format(class_name), 'rb') as f:
        minmaxscaler = pickle.load(f)
    dataset_scaled = minmaxscaler.transform(dataset)
    tmr_predict = model.predict(dataset_scaled)   # dataset_scaled(1, 30, 1)안해줘도??
    tmr_predict = minmaxscaler.inverse_transform(tmr_predict)
    print(tmr_predict)

    return tmr_predict
    # mse = mean_squared_error(data.tail(len(tmr_predict)).values, tmr_predict)
    # rmse = mse ** 0.5
    # mape = mean_absolute_percentage_error(data.tail(len(tmr_predict)).reset_index(drop=True), pd.Series(tmr_predict))
    # return tmr_predict, mse, rmse, mape


if __name__ == '__main__':
    # 티커명 참조해서 csv 다운로드 받기
    futures = [('BZ=F', 'BRENT_OIL'), ('CC=F', 'COCOA'), ('KC=F', 'Coffee'), ('HG=F', 'COPPER'), ('ZC=F', 'CORN'),('CT=F', 'COTTON'),
               ('CL=F', 'CRUDE_OIL'), ('YM=F', 'DOW'), ('GF=F', 'FEEDER_CATTLE'), ('GC=F', 'GOLD'), ('HE=F', 'LEAN_HOGS'), ('LE=F', 'LIVE_CATTLE'),
               ('LBS=F', 'LUMBER'), ('NQ=F', 'NASDAQ'), ('NG=F', 'NATURAL_GAS'), ('ZO=F', 'OAT'), ('PA=F', 'PALLADIUM'),('PL=F', 'PLATINUM'), ('ZR=F', 'ROUGH_RICE'),
               ('RTY=F', 'RUSSEL2000'), ('SI=F', 'SILVER'), ('ZS=F', 'SOYBEAN'), ('ZM=F', 'SOYBEAN_MEAL'), ('ZL=F', 'SOYBEAN_OIL'),
               ('ES=F', 'SPX'), ('SB=F', 'SUGAR'), ('ZT=F', 'US2YT'), ('ZF=F', 'US5YT'), ('ZN=F', 'US10YT'),('ZB=F', 'US30YT'), ('KE=F', 'WHEAT')]
    class_name = 'brent_oil'
    Today = datetime.date.today()
    days_98 = datetime.timedelta(days=98)
    lstm_len_day = datetime.timedelta(days=4-1)

    # 2022-01-26 부터 필요할 듯함( 기존에 트레인 시킨 마지막 데이타 이후로부터)
    data = yf.download('BZ=F', start='2022-01-26', end=Today) # start_day + today => 전체 데이터 수가 98 + 3 = 101이여야 함/ days_98 + lstm_len_day
    len(data) # 1/26 ~ 2/11(오늘)까지 대략 12개의 종가를 예측, backtesting 시켜보자 .
    exit()
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

    data.drop(['Change'], axis=1, inplace=True)  # drop 열 추가


    # 기존의 ma와 이동평균 가져오기
    ma = 'SMA'
    optimized_period = 50

    simulation = {}
    # 저변동성 / 고 변동성 시계열로 각각 나누기
    low_vol = functions[ma](data, optimized_period) # int로 만들어줘야./ 총 1248곘지만 앞에 이평 길이-1 만큼 Nan값
    print(low_vol)
    high_vol = data['close'] - low_vol
    print(high_vol)
    # Generate ARIMA and LSTM predictions
    print('\nWorking on ' + ma + ' predictions')

    low_vol_prediction, low_vol_mse, low_vol_rmse, low_vol_mape = get_arima(low_vol, 1500-252-int(optimized_period.iloc[0][0]), 252) # 이평으로 스무스해진 데이터(평균일정)=> # 1400, 252 이케 해도 될듯
    print('ARIMA error, skipping to next MA type')
    # exit()
    high_vol_prediction, high_vol_mse, high_vol_rmse, high_vol_mape = get_lstm(high_vol, 1500-252-int(optimized_period.iloc[0][0]), 252)  # 원본 종가 - 이평 의 데이터(분산된 느낌??)

    final_prediction = pd.Series(low_vol_prediction) + pd.Series(high_vol_prediction) # series, 합산 => 최종예측 데이터
    mse = mean_squared_error(final_prediction.values, data['close'].tail(252).values)  # test데이터에서 예측한 값 252와 실제 마지막 값 252 의 mse 구해보기
    print('mse의 타입', type(mse))     # float
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(data['close'].tail(252).reset_index(drop=True), final_prediction)


    # Generate prediction accuracy
    actual = data['close'].tail(252).values
    result_1 = []
    result_2 = []
    for i in range(1, len(final_prediction)): # 테스트 데이터(252개)로 정확도 측정하기.
        # Compare prediction to previous close price
        if final_prediction[i] > actual[i-1] and actual[i] > actual[i-1]:
            result_1.append(1) # 숫자 1을 추가하라.(정답)
        elif final_prediction[i] < actual[i-1] and actual[i] < actual[i-1]:
            result_1.append(1) # 숫자 1을 추가하라.(정답)
        else:
            result_1.append(0) # 숫자 0을 추가하라.(오답)

        # Compare prediction to previous prediction
        if final_prediction[i] > final_prediction[i-1] and actual[i] > actual[i-1]:
            result_2.append(1)
        elif final_prediction[i] < final_prediction[i-1] and actual[i] < actual[i-1]:
            result_2.append(1)
        else:
            result_2.append(0)
    print(result_1)  # [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
    print(result_2)  # [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1]
    accuracy_1 = np.mean(result_1)
    accuracy_2 = np.mean(result_2)

    print('===================== 시뮬레이션 결과 ==================')
    simulation[ma] = {'low_vol': {'prediction': low_vol_prediction, 'mse': low_vol_mse,
                                  'rmse': low_vol_rmse, 'mape': low_vol_mape},
                      'high_vol': {'prediction': high_vol_prediction, 'mse': high_vol_mse,
                                   'rmse': high_vol_rmse},
                      'final': {'prediction': final_prediction.values.tolist(), 'mse': mse,
                                'rmse': rmse, 'mape': mape},
                      'accuracy': {'prediction vs close': accuracy_1, 'prediction vs prediction': accuracy_2}}

    # save simulation data here as checkpoint
    with open('./datasets_3/simulation_data.json', 'w') as fp:
        json.dump(simulation, fp)

    for ma in simulation.keys():
        print('\n' + ma)
        print('Prediction vs Close:\t\t' + str(round(100*simulation[ma]['accuracy']['prediction vs close'], 2))
              + '% Accuracy')
        print('Prediction vs Prediction:\t' + str(round(100*simulation[ma]['accuracy']['prediction vs prediction'], 2))
              + '% Accuracy')
        print('MSE:\t', simulation[ma]['final']['mse'],
              '\nRMSE:\t', simulation[ma]['final']['rmse'],
              '\nMAPE:\t', simulation[ma]['final']['mape'])
        # MIDPRICE
        # Prediction vs Close: 49.0 % Accuracy
        # Prediction vs Prediction: 52.59 % Accuracy
        # MSE: 98759.1100664388
        # RMSE: 314.2596220745497
        # MAPE: 1.6777265314384462



    # 피클 담글 변수
    # class_name ( 자산 네임 )
    # loss_value ( lstm 모델의 val_loss ), order
    # 예측 plot도 만들어야 함.
    # 모델. 민멕스 스켈러 저장