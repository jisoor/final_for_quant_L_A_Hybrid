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
import pickle
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMAResults
# h = np.array(high)
# l = np.array(low)
# c = np.array(close)
# output_atr = np.array(talib.ATR(h,l,c,14))

# 최적의 ma 가지고 lstm 모델, arima모델 돌려서 모델 파일 및 minmaxscaler 저장하는 파일.
# 찾은 ma 의 arima model order / lstm의 loss값 도 df로 저장
# ma와 period 가져오기 ( dataframe으로 잘라서 가져오기)


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
    val_df['Arima_order'] = [order]

    # test 데이터로 예측 하기
    prediction = []
    for i in range(test_len):  # 252번 만큼.
        model = pm.ARIMA(order=order)
        model = model.fit(train)   # 맨 첨 데이터 1198개
        print('working on', i+1, 'of', test_len, '-- ' + str(int(100 * (i + 1) / test_len)) + '% complete')
        prediction.append(model.predict()[0]) # 252개 데이터 예측.
        train.append(test[i]) # train list는 1450개가 됌. arima는 이전값이 필요하기 때문에 예측할 때도 이전값을 계속 업데이트 시킨다.
        if i >=  test_len - 1:
            with open('./models/{}_Arima_model.pickle'.format(class_name), 'wb') as f:
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

    with open('./minmaxscaler/{}_minmaxscaler.pickle'.format(class_name), 'wb') as f:
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
    model.save('./models/{}_Lstm_model.h5'.format(class_name))  # lstm  모델 저장

    # loss_value 값 저장
    val_df['Lstm_loss'] = [loss_value]

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
    # CSV should have columns: ['date', 'open', 'high', 'low', 'close', 'volume']
    class_name = 'brent_oil'
    val_df = pd.DataFrame(columns=['class_name', 'Lstm_loss', 'Arima_order']) # 컬럼으로 이루어진 데이터프레임 만들기
    val_df.set_index('class_name', inplace=True)   # 인덱스 설정
    val_df.loc[class_name] = np.nan   # 인덱스 행 추가 (np.nan)으로

    data = pd.read_csv('futures_BRENT_OIL.csv', index_col=0, header=0).reset_index(drop=True) # 왜 1500개만 했지?? 그게 나으려나..
    print(data)
    print( '데이터 길이', len(data))  # 3591
    # Initialize moving averages from Ta-Lib, store functions in dictionary
    talib_moving_averages = ['SMA', 'EMA', 'WMA', 'DEMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'T3', 'TEMA', 'TRIMA']
    functions = {}
    for ma in talib_moving_averages:
        functions[ma] = abstract.Function(ma)

    # print('SMA', functions['SMA'])
    data = data.rename(columns={'Adj_Close':'close'})  # 이름을 'close' 'High', 'Low', 'close' 'change' 인덱스는 걍 순서
    data = data.rename(columns={'High': 'high'})
    data = data.rename(columns={'Low': 'low'})
    data.drop(['Change'], axis=1, inplace=True)

    ############### 최적의 ma 선택하기 초기화( 직접 손으로 써도 되고, 데이터프레임으로 잘라서 가져와도 됌)##########################
    optimized_period = 64
    ma = 'TRIMA'

    simulation = {}
    # 저변동성 / 고 변동성 시계열로 각각 나누기
    # period 가 50 이니깐
    low_vol = functions[ma](data ,optimized_period)  # int로 만들어줘야./ 총 1248곘지만 앞에 이평 길이-1 만큼 Nan값
    print(low_vol)
    high_vol = data['close'] - low_vol
    print(high_vol)
    # Generate ARIMA and LSTM predictions
    print('\nWorking on ' + ma + ' predictions')

    # train_test_set 나누기
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(data, test_size=0.2, shuffle=False)
    print(len(train_set), len(test_set))
    print(test_set.tail())

    # 길이 설정
    train_len = len(train_set)
    test_len = len(test_set)
    train = len(train_len)-(optimized_period-1)
    # 이평으로 스무스해진 데이터(평균일정)
    low_vol_prediction, low_vol_mse, low_vol_rmse, low_vol_mape = get_arima(low_vol, train , test_len)
    # (원본 종가 - 이평) 의 데이터(분산된 느낌??)
    high_vol_prediction, high_vol_mse, high_vol_rmse, high_vol_mape = get_lstm(high_vol, train , test_len)


    final_prediction = pd.Series(low_vol_prediction) + pd.Series(high_vol_prediction)  # series, 합산 => 최종예측 데이터
    mse = mean_squared_error(final_prediction.values,
                             data['close'].tail(252).values)  # test데이터에서 예측한 값 252와 실제 마지막 값 252 의 mse 구해보기
    print('mse의 타입', type(mse))  # float
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(data['close'].tail(252).reset_index(drop=True), final_prediction)

    # Generate prediction accuracy
    actual = data['close'].tail(252).values
    result_1 = []
    result_2 = []
    for i in range(1, len(final_prediction)):  # 테스트 데이터(252개)로 정확도 측정하기.
        # Compare prediction to previous close price
        if final_prediction[i] > actual[i - 1] and actual[i] > actual[i - 1]:
            result_1.append(1)  # 숫자 1을 추가하라.(정답)
        elif final_prediction[i] < actual[i - 1] and actual[i] < actual[i - 1]:
            result_1.append(1)  # 숫자 1을 추가하라.(정답)
        else:
            result_1.append(0)  # 숫자 0을 추가하라.(오답)

        # Compare prediction to previous prediction
        if final_prediction[i] > final_prediction[i - 1] and actual[i] > actual[i - 1]:
            result_2.append(1)
        elif final_prediction[i] < final_prediction[i - 1] and actual[i] < actual[i - 1]:
            result_2.append(1)
        else:
            result_2.append(0)
    print(
        result_1)  # [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
    print(
        result_2)  # [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1]
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
    with open('./brent_oil/simulation_data.json', 'w') as fp:
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

    val_df.to_csv('./brent_oil/{}_lstm_loss_arima_order'.format(class_name), index=True)


    # 피클 담글 변수
    # class_name ( 자산 네임 )
    # loss_value ( lstm 모델의 val_loss ), order
    # 예측 plot도 만들어야 함.
    # 모델. 민멕스 스켈러 저장