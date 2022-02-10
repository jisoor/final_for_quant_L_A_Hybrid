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
# h = np.array(high)
# l = np.array(low)
# c = np.array(close)
# output_atr = np.array(talib.ATR(h,l,c,14))


def mean_absolute_percentage_error(actual, prediction):
    actual = pd.Series(actual)
    prediction = pd.Series(prediction)
    return 100 * np.mean(np.abs((actual - prediction))/actual)


def get_arima(data, train_len, test_len):
    # prepare train and test data
    data = data.tail(test_len + train_len).reset_index(drop=True)
    train = data.head(train_len).values.tolist()
    test = data.tail(test_len).values.tolist()

    # auto_arima로 모델 초기화
    model = auto_arima(train, max_p=3, max_q=3, seasonal=False, trace=True, # 이평선으로 계절성을 제거해주었으므로 seasonal=False
                       error_action='ignore', suppress_warnings=True)
    # model_0 = pm.auto_arima(y_train  # 데이터
    #                         , d=1  # 차분 차수, ndiffs 결과!
    #                         , start_p=0, max_p=3, start_q=0, max_q=3
    #                         , m=1  # 음 중요한게...주기성이 있으면 따로 추가를 해줘야하는데
    #                         , seasonal=True  # 계절성 ARIMA가 아니라면 필수!
    #                         , stepwise=True, trace=True)

    # 최적의 모델 파라미터 찾기.
    model.fit(train)
    order = model.get_params()['order']   # (3, 2, 2)
    print('ARIMA order:', order, '\n')    # ARIMA order: (3, 2, 2)
    val_df['Arima_order'] = [order]

    # test 데이터로 예측 하기
    prediction = []
    for i in range(len(test)):  # 252번 만큼.
        model = pm.ARIMA(order=order) # (3,1,1)로 해봐야쥐
        model.fit(train)
        print('working on', i+1, 'of', test_len, '-- ' + str(int(100 * (i + 1) / test_len)) + '% complete')
        prediction.append(model.predict()[0])
        train.append(test[i])

    # Generate error data
    mse = mean_squared_error(test, prediction)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(pd.Series(test), pd.Series(prediction))
    return prediction, mse, rmse, mape


def get_lstm(data, train_len, test_len, lstm_len=4):
    # prepare train and test data
    data = data.tail(test_len + train_len).reset_index(drop=True)
    dataset = np.reshape(data.values, (len(data), 1))  # ( 1500, 1)
    minmaxscaler = MinMaxScaler(feature_range=(0, 1))   # 이렇게도 가능하군..
    dataset_scaled = minmaxscaler.fit_transform(dataset)
    x_train = []
    y_train = []
    x_test = []

    for i in range(lstm_len, train_len):  # 4, 1000
        x_train.append(dataset_scaled[i - lstm_len:i, 0])  # 0~3, 1~4, 2~5, .... , 996~999   (4, 1)로 저장됨
        y_train.append(dataset_scaled[i, 0])               #  4,   5,   6 , .... ,   1000
    for i in range(train_len, len(dataset_scaled)):    # 1000 ~ 1500
        x_test.append(dataset_scaled[i - lstm_len:i, 0])  # ( 500, 4)

    x_train = np.array(x_train)         # (1, 4, 1) ??
    print('======== x_train 셰입 ========== ' , x_train.shape) # ( 996, 4)
    y_train = np.array(y_train)   # ( 996, 1) ??
    print(y_train.shape)        # ( 996, )
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    print(x_train.shape)        # (996, 1, 1 ) ??  => (996, 4, 1)
    x_test = np.array(x_test)
    print(x_test.shape)         # (1, 500, 4)??   => (252, 4)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    print(x_test.shape)             # (252, 4, 1)


    # Set up & fit LSTM RNN
    # 모델 조정해보자 ( (1, 소프트맥스로 해보기 2.  tanh로 해보기
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=2, input_shape=(x_train.shape[1], 1), activation='tanh')) # (units=lstm_len)  activation='tanh'
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='softmax')) # 'softmax' 아닌가 ??.
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stopping = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)

    fit_hist = model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2, callbacks=[early_stopping])
    print(list(fit_hist.history)) # ['loss']
    plt.plot(fit_hist.history['loss'][:], label='loss')
    plt.show()
    plt.pause(1)
    plt.close()
    loss_value = fit_hist.history['loss'][-1] # loss
    print(loss_value)

    # loss_value 값 저장
    val_df['Lstm_loss'] = [loss_value]

    # Generate predictions
    prediction = model.predict(x_test)
    prediction = minmaxscaler.inverse_transform(prediction).tolist()

    output = []
    for i in range(len(prediction)):
        output.extend(prediction[i])
    prediction = output
    print( '==============prediction : =========' , prediction)  # 리스트로 252개?
    print(len(prediction))
    # Generate error data
    mse = mean_squared_error(data.tail(len(prediction)).values, prediction)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(data.tail(len(prediction)).reset_index(drop=True), pd.Series(prediction))
    return prediction, mse, rmse, mape


if __name__ == '__main__':
    # Load historical data
    # CSV should have columns: ['date', 'open', 'high', 'low', 'close', 'volume']
    class_name = 'Nasdaq'
    val_df = pd.DataFrame(columns=['class_name', 'Lstm_loss', 'Arima_order']) # 컬럼으로 이루어진 데이터프레임 만들기
    val_df.set_index('class_name', inplace=True)   # 인덱스 설정
    val_df.loc['Nasdaq'] = np.nan   # 인덱스 행 추가 (np.nan)으로

    data = pd.read_csv('NASDAQ Composite_2007-01-03-2022-01-25.csv', index_col=0, header=0).tail(1500).reset_index(drop=True) # 왜 1500개만 했지?? 그게 나으려나..
    print(data)
    # Initialize moving averages from Ta-Lib, store functions in dictionary
    talib_moving_averages = ['SMA', 'EMA', 'WMA', 'DEMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'T3', 'TEMA', 'TRIMA']
    functions = {}
    for ma in talib_moving_averages:
        functions[ma] = abstract.Function(ma)

    print('SMA', functions['SMA'])

    data = data.rename(columns={'Adj Close':'close'})  # 이름을 'close' 'High', 'Low', 'close' 'change' 인덱스는 걍 순서
    data = data.rename(columns={'High': 'high'})
    data = data.rename(columns={'Low': 'low'})
    data.drop(['Change'], axis=1, inplace=True)

    print( '데이터 길이', len(data))  # 1500

    # 이동평균 기간 4~99까지 에서 최적의 첨도 K구하기
    kurtosis_results = {'period': []}
    for i in range(4, 100):
        kurtosis_results['period'].append(i)
        for ma in talib_moving_averages:  # ['SMA', 'EMA', 'WMA', 'DEMA', 'KAMA', 'MIDPOINT', 'MIDPRICE', 'T3', 'TEMA', 'TRIMA'] 총 10개
            #1500개에서 252개 뺴고 1248개만 train?. 마지막 60.. 60일 이동평균?
            ma_output = functions[ma](data[:-252], i).tail(60)  # input_arrays parameter missing required data keys: high, low
            # print('이평 결과',ma, ':', ma_output[:5], ma_output[-5:])  # 1188 ~ 1247( i =4일 때)
            # print('이건뭔데', functions[ma](data[:-252], i))  # (1248, 1) i=4 일떄 앞에 3개씩은 Nan값이고 그럼. 총 1248개의 이평값(이동평균값)이 나온다.(0~3 => 3, 1~4=>4 .... )
            # Determine kurtosis "K" value
            k = kurtosis(ma_output, fisher=False) # 60개 데이터들에 대한 첨도값을 구해줌(솟아오른 정도)
            # print('k는??', k )
            # add to dictionary
            if ma not in kurtosis_results.keys():  # key값이 없다면
                kurtosis_results[ma] = []          # { key : [k 첨도 값_SMA, k 첨도 값_EMA, k 첨도 값_WMA ...... , k 첨도 값_TRIMA  }
            kurtosis_results[ma].append(k)

    kurtosis_results = pd.DataFrame(kurtosis_results)   # ( 11, 96) DF
    kurtosis_results.to_csv('./datasets_5/kurtosis_results_without_change.csv', index=True)



    optimized_period = pd.DataFrame(index=['period'])   # 이것인가..<= 데이터프레임 만들어줘야함
    # Determine period with K closest to 3 +/-5%   K 첨도 값이 3 +- 5% 사이면 변동성이 low volatiliy => ARIMA로 예측측    optimized_period = {}
    # 최적의 함술 구하는 건지 최적의 period를 구하는 건지(이동평균 기준)
    for ma in talib_moving_averages:
        difference = np.abs(kurtosis_results[ma] - 3)   # 각각 값에 대해 뺴져서 리스트로?  / 최적의 첨도값 3과의 차이의 절대값
        print(type(difference))     # <class 'pandas.core.series.Series'>
        df = pd.DataFrame({'difference': difference, 'period': kurtosis_results['period']})
        df = df.sort_values(by=['difference'], ascending=True).reset_index(drop=True)
        print(df.head())
        if df.at[0, 'difference'] < 3 * 0.05: # at은 loc랑 비슷  / df.at[0, difference]는 첨도가 3의 가장 가까운 값 => period
            optimized_period[ma] = [df.at[0, 'period']]   # 컬럼값 넣어주려면 값이 하나라도 리스트여야 함.
            print('최적의 이동평균은? ', df.at[0, 'period'])
        else:
            print(ma + ' is not viable, best K greater or less than 3 +/-5%')
    print('\nOptimized periods 이평 길이에대한 df:', optimized_period)  # df 인덱스가 기본이고, 열은 'Midprice' 값은 하나의 최적 period가 들어가있음
    print(type(optimized_period))


    simulation = {}
    for ma in optimized_period.columns:
        # 저변동성 / 고 변동성 시계열로 각각 나누기
        low_vol = functions[ma](data, int(optimized_period.loc['period'][ma])) # int로 만들어줘야./ 총 1248곘지만 앞에 이평 길이-1 만큼 Nan값
        print(low_vol)
        high_vol = data['close'] - low_vol
        print(high_vol)
        # Generate ARIMA and LSTM predictions
        print('\nWorking on ' + ma + ' predictions')
        try:
            low_vol_prediction, low_vol_mse, low_vol_rmse, low_vol_mape = get_arima(low_vol, 1000, 252) # 이평으로 스무스해진 데이터(평균일정)=> # 1400, 252 이케 해도 될듯
        except:
            print('ARIMA error, skipping to next MA type')
            continue

        high_vol_prediction, high_vol_mse, high_vol_rmse, high_vol_mape = get_lstm(high_vol, 1000, 252)  # 원본 종가 - 이평 의 데이터(분산된 느낌??)

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
        with open('./datasets_5/simulation_data.json', 'w') as fp:
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

    val_df.to_csv('./datasets_5/{}_lstm_loss_arima_order'.format(class_name), index=True)


    # 피클 담글 변수
    # class_name ( 자산 네임 )
    # loss_value ( lstm 모델의 val_loss ), order
    # 예측 plot도 만들어야 함.
    # 모델. 민멕스 스켈러 저장