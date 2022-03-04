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

def get_arima(data, train_len, test_len): #  len(data) , 252
    # prepare train and test data
    data = data.tail(test_len + train_len).reset_index(drop=True) # Nan값 만 빠지고
    train = data.head(train_len).values.tolist()
    test = data.tail(test_len).values.tolist()

    # auto_arima로 모델 초기화
    model = auto_arima(train, max_p=3, max_q=3, seasonal=False, trace=True, # 이평선으로 계절성을 제거해주었으므로 seasonal=False
                       error_action='ignore', suppress_warnings=True)

    # 최적의 모델 파라미터 찾기.
    model.fit(train)  # 1198개 train데이터 입히기.
    order = model.get_params()['order']   # (3, 2, 2)
    print('ARIMA order:', order, '\n')    # ARIMA order: (3, 2, 2)


    # test 데이터로 예측 하기
    prediction = []
    for i in range(test_len):  # 252번 만큼.
        model = pm.ARIMA(order=order)
        model = model.fit(train)   # 맨 첨 데이터 1198개
        print('working on', i+1, 'of', test_len, '-- ' + str(int(100 * (i + 1) / test_len)) + '% complete')
        prediction.append(model.predict()[0]) # 252개 데이터 예측.
        train.append(test[i]) # train list는 1450개가 됌. arima는 이전값이 필요하기 때문에 예측할 때도 이전값을 계속 업데이트 시킨다.
        if i >=  test_len - 1:
            with open('./{}/{}_Arima_model.pickle'.format(class_name, class_name), 'wb') as f:
                pickle.dump(model, f)                            # arima 모델 저장( 2007년 ~ 2022-01-25 까지 저장)

    print('예측 값들:',  prediction)
    # Generate error data
    mse = mean_squared_error(test, prediction)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(pd.Series(test), pd.Series(prediction))
    return prediction, mse, rmse, mape


def get_lstm(data, train_len, test_len, lstm_len=4):
    # prepare train and test data
    data = data.tail(test_len + train_len).reset_index(drop=True)
    dataset = np.reshape(data.values, (len(data), 1))  # ( 3591, 1)
    minmaxscaler = MinMaxScaler(feature_range=(0, 1))   # 이렇게도 가능하군..
    dataset_scaled = minmaxscaler.fit_transform(dataset)
    # minmaxscaler 저장

    with open('./{}/{}_minmaxscaler.pickle'.format(class_name, class_name), 'wb') as f:
        pickle.dump(minmaxscaler, f)

    x_train = []
    y_train = []
    x_test = []
    for i in range(lstm_len, train_len):  # 4, 1198
        x_train.append(dataset_scaled[i - lstm_len:i, 0])  # 0~3, 1~4, 2~5, .... , 996~999   (4, 1)로 저장됨
        y_train.append(dataset_scaled[i, 0])               #  4,   5,   6 , .... ,   1000
    for i in range(train_len, len(dataset_scaled)):    # 1198 ~ 1450 (252개)
        x_test.append(dataset_scaled[i - lstm_len:i, 0])  # 1194 ~ 1197, 1195 ~ 1198 , ....... , 1446 ~ 1449

    x_train = np.array(x_train)
    print('======== x_train 셰입 ========== ' , x_train.shape) # ( 996, 4)
    y_train = np.array(y_train)
    print(y_train.shape)        # ( 996, )
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(x_train.shape)        #  => (996, 4, 1)
    x_test = np.array(x_test)
    print(x_test.shape)         # => (252, 4)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    print(x_test.shape)             # (252, 4, 1)


    # Set up & fit LSTM RNN
    # 모델 조정해보자 ( (1, 소프트맥스로 해보기 2.  tanh로 해보기
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='tanh')) # (units=lstm_len)  activation='tanh'
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid')) # 'softmax' 아닌가 ??.
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)

    fit_hist = model.fit(x_train, y_train, epochs=500, batch_size=2, verbose=2, callbacks=[early_stopping])
    print(list(fit_hist.history)) # ['loss']
    plt.plot(fit_hist.history['loss'][:], label='loss')
    plt.show()
    plt.pause(1)
    plt.close()
    loss_value = fit_hist.history['loss'][-1] # loss
    print(loss_value)
    model.save('./{}/{}_Lstm_model.h5'.format(class_name, class_name))  # lstm  모델 저장


    # Generate predictions
    prediction = model.predict(x_test)
    prediction = minmaxscaler.inverse_transform(prediction).tolist()

    output = []
    for i in range(len(prediction)):
        output.extend(prediction[i]) # extend와 append의 차이점 extend는 내용물을 넣어준다(리스트면, 리스트 안의 내용만 꺼내서) / 걍  for문 안쓰고 prediction = output.extend(prediction) 하면...
    prediction = output
    print( '==============prediction : =========' , prediction)  # 리스트로 252개?
    print(len(prediction))
    # Generate error data
    mse = mean_squared_error(data.tail(len(prediction)).values, prediction)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(data.tail(len(prediction)).reset_index(drop=True), pd.Series(prediction))
    return prediction, mse, rmse, mape

if __name__ == '__main__':
    # 티커명 참조해서 csv 다운로드 받기
    futures = [('BZ=F', 'BRENT_OIL'), ('CC=F', 'COCOA'), ('KC=F', 'Coffee'), ('HG=F', 'COPPER'), ('ZC=F', 'CORN'),('CT=F', 'COTTON'),
               ('CL=F', 'CRUDE_OIL'), ('YM=F', 'DOW'), ('GF=F', 'FEEDER_CATTLE'), ('GC=F', 'GOLD'), ('HE=F', 'LEAN_HOGS'), ('LE=F', 'LIVE_CATTLE'),
               ('LBS=F', 'LUMBER'), ('NQ=F', 'NASDAQ'), ('NG=F', 'NATURAL_GAS'), ('ZO=F', 'OAT'), ('PA=F', 'PALLADIUM'),('PL=F', 'PLATINUM'), ('ZR=F', 'ROUGH_RICE'),
               ('RTY=F', 'RUSSEL2000'), ('SI=F', 'SILVER'), ('ZS=F', 'SOYBEAN'), ('ZM=F', 'SOYBEAN_MEAL'), ('ZL=F', 'SOYBEAN_OIL'),
               ('ES=F', 'SPX'), ('SB=F', 'SUGAR'), ('ZT=F', 'US2YT'), ('ZF=F', 'US5YT'), ('ZN=F', 'US10YT'),('ZB=F', 'US30YT'), ('KE=F', 'WHEAT')]

    # 기존의 ma와 이동평균 가져오기
    class_name = 'brent_oil'
    ma = 'TRIMA'
    optimized_period = 64
    order = (2, 1, 2)
    lstm_len = 4
    Today = datetime.date.today()
    ticker = 'BZ=F'



    # 2022-01-26 부터 필요할 듯함( 기존에 트레인 시킨 마지막 데이타 이후로부터)
    # 2022-01-26 ~ 03-02 약 26개 정도 예측해야함 -> 총필요한 데이터: 26 + (optimized_period-1)
    data = yf.download(ticker, start='2021-10-27', end=Today) #적당히 6~8개월 어치 데이터 가져오기
    print(data)
    print(len(data)) # 89
    print(data['2022-01-26':'2022-03-02'])
    print(len(data['2022-01-26':'2022-03-02'])) # 25개
    # 총 필요한 데이터 : 2022-01-26 ~ 03-02 약 25개 정도 예측해야함 -> 25 + (optimized_period-1) => 25 + (64 -1) = 88
    data = data.reset_index(drop=True)
    data = data[-88:]
    print(len(data))

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
    print('길이', len(data))
    exit()

    # 저변동성 / 고 변동성 시계열로 각각 나누기
    low_vol = functions[ma](data, optimized_period) # int로 만들어줘야./ 총 1248곘지만 앞에 이평 길이-1 만큼 Nan값
    print(low_vol)  # 3개 뺴고 다 난값
    high_vol = data['close'] - low_vol
    print(high_vol)  #  3개 뺴고 다 난값

    # Generate ARIMA and LSTM predictions

    # 모델 불러오기
    with open('./{}/{}_Arima_model.pickle'.format(class_name , class_name), 'rb') as f:
        model = pickle.load(f)
    predict_arima_price = low_vol[-1:] # 마지막 1개 가져와서 예측.
    model = pm.ARIMA(order=order)
    model.fit(predict_arima_price)  # 우리는 2007년 ~ 2022-01-25까지 다 train시킨 모델을 가져온 것이기 떄문에 01-26부터 데이터만 추가시키면 되지않낭?
    arima_prediction = model.predict()[0]  # ㅁ내일 값 예측
    print('arima 내일 예측 값:', arima_prediction) # 내일 예측 값: 79.01496184233464


    ############## 어떻게 될지 모르겠음 . 시도해보기 #############################
    # data = 2022-01-26 ~ 2022-02-10(가장 최신 종가데이터)를 넣어서 내일 값 예측시키



    # test 데이터로 예측 하기
    prediction = []
    for i in range(len(data['2022-01-26':'2022-03-02'])):  # 25번 만큼.
        model = pm.ARIMA(order=order)
        model = model.fit(data)  # 맨 첨 데이터 1198개
        # print('working on', i + 1, 'of', test_len, '-- ' + str(int(100 * (i + 1) / test_len)) + '% complete')
        # prediction.append(model.predict()[0])  # 252개 데이터 예측.
        # train.append(test[i])  # train list는 1450개가 됌. arima는 이전값이 필요하기 때문에 예측할 때도 이전값을 계속 업데이트 시킨다.
        # if i >= test_len - 1:
        #     with open('./{}/{}_Arima_model.pickle'.format(class_name, class_name), 'wb') as f:
        #         pickle.dump(model, f)  # arima 모델 저장( 2007년 ~ 2022-01-25 까지 저장)

    print('예측 값들:', prediction)
    # Generate error data
    mse = mean_squared_error(test, prediction)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(pd.Series(test), pd.Series(prediction))
    predict_arima_price = low_vol[-1:] # 마지막 1개 가져와서 예측.
    model = pm.ARIMA(order=order)
    model.fit(predict_arima_price)  # 우리는 2007년 ~ 2022-01-25까지 다 train시킨 모델을 가져온 것이기 떄문에 01-26부터 데이터만 추가시키면 되지않낭?
    arima_prediction = model.predict()[0]  # ㅁ내일 값 예측
    print('arima 내일 예측 값:', arima_prediction) # 내일 예측 값: 79.01496184233464

############## 이 밑만 해결하면 댐 ###########################
    # LSTM 예측
    predict_lstm_price = high_vol[-4:]
    print(high_vol[-6:])
    dataset = np.reshape(predict_lstm_price.values, ( lstm_len, 1))  # ( 1, lstm_len, 1) 아닌강>?
    print('dataset', dataset)
    model = load_model('./{}/{}_Lstm_model.h5'.format(class_name, class_name))
    with open('./{}/{}_minmaxscaler.pickle'.format(class_name, class_name), 'rb') as f:
        minmaxscaler = pickle.load(f)
    dataset_scaled = minmaxscaler.transform(dataset)

    lstm_prediction = model.predict(np.reshape(dataset_scaled, (1,4,1)))  # dataset_scaled(1, 4, 1)안해줘도??
    prediction = minmaxscaler.inverse_transform(lstm_prediction)
    print('lstm 내일 예측 값 :', prediction)

    final_prediction = arima_prediction + prediction
    print('최종 내일 예측 값은?? ====> ', final_prediction)

# pip install numpy == 1.19.2


