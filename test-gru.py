import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import time


# 아마존의 2013년 부터 2018년까지 일일 주가를 학습 데이터로
# 2019년 데이터를 테스트 데이터로 사용 

AMZN = yf.download('AMZN', 
                  start = '2013-01-01',
                  end = '2019-12-31',
                  progress = False)

# 수정종가(Adj close), 시가(Open), 최고가(High), 최저가(Low), 종가(Close). 거래량(Volume)
all_data = AMZN[['Adj Close', 'Open', 'High','Low',"Close","Volume"]].round(2)
all_data.head(10)

print("There are "+str(all_data[:'2018'].shape[0])+" observations in the training data")
print("There are "+str(all_data['2019':].shape[0])+" observations in the test data")
all_data['Adj Close'].plot()


def ts_train_test_normalize(all_data, time_steps, for_periods):
    """
    입력: 
        data: 날짜와 가격 데이터가 있는 데이터프레임
    출력: 
        X_train, y_train: 2013/1/1-2018/12/31 데이터 
        X_test : 2019년 이후 데이터 
        sc :     훈련 데이터에 맞게 인스턴스화된 MinMaxScaler 객체
    """
    # 훈련 및 테스트 세트 생성
    ts_train = all_data[:'2018'].iloc[:,0:1].values  # 2018년까지의 데이터
    ts_test = all_data['2019':].iloc[:,0:1].values   # 2019년 이후 데이터
    ts_train_len = len(ts_train)  # 훈련 데이터의 길이
    ts_test_len = len(ts_test)    # 테스트 데이터의 길이
    
    # 데이터 스케일링
    from sklearn.preprocessing import MinMaxScaler 
    sc = MinMaxScaler(feature_range=(0,1))        # MinMaxScaler 인스턴스 생성
    ts_train_scaled = sc.fit_transform(ts_train)  # 훈련 데이터 스케일링
    
    # s 샘플과 t 타임 스텝의 훈련 데이터 생성
    X_train = [] 
    y_train = [] 
    for i in range(time_steps, ts_train_len-1):
        X_train.append(ts_train_scaled[i-time_steps:i, 0])   # 입력 데이터
        y_train.append(ts_train_scaled[i:i+for_periods, 0])  # 출력 데이터
    X_train, y_train = np.array(X_train), np.array(y_train)  # 배열로 변환
    
    # 효율적인 모델링을 위한 X_train 재구성
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1 ))
    
    # 입력 데이터 준비
    inputs = pd.concat((all_data["Adj Close"][:'2018'], all_data["Adj Close"]['2019':]), axis=0).values  # Adj Close 열 결합
    inputs = inputs[len(inputs)-len(ts_test)-time_steps:]  # 필요한 부분만 잘라냄
    inputs = inputs.reshape(-1,1)  # 재구성
    inputs = sc.transform(inputs)  # 스케일링
    
    # X_test 준비
    X_test = [] 
    for i in range(time_steps, ts_test_len + time_steps - for_periods):
        X_test.append(inputs[i-time_steps:i,0])  # 입력 데이터
    X_test = np.array(X_test)  # 배열로 변환
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # 재구성
    
    return X_train, y_train , X_test, sc  # 반환


X_train, y_train, X_test, sc = ts_train_test_normalize(all_data, 5,2)
X_train.shape[0], X_train.shape[1]


# Convert the 3D shape of X_train to a data frame so we can see: 
X_train_see = pd.DataFrame(np.reshape(X_train, (X_train.shape[0], X_train.shape[1])))
y_train_see = pd.DataFrame(y_train)
pd.concat([X_train_see, y_train_see], axis = 1)


# Convert the 3D shape of X_test to a data frame so we can see: 
X_test_see = pd.DataFrame(np.reshape(X_test, (X_test.shape[0], X_test.shape[1])))
pd.DataFrame(X_test_see)


print("There are " + str(X_train.shape[0]) + " samples in the training data")
print("There are " + str(X_test.shape[0]) + " samples in the test data")


def actual_pred_plot(preds):
    """
    실제 값과 예측 값을 그래프로 그립니다.
    """
    actual_pred = pd.DataFrame(columns = ['Adj. Close', 'prediction'])  # DataFrame 생성
    actual_pred['Adj. Close'] = all_data.loc['2019':,'Adj Close'][0:len(preds)]  # 실제 값
    actual_pred['prediction'] = preds[:,0]  # 예측 값
    
    from keras.metrics import MeanSquaredError  # MeanSquaredError 메트릭 임포트
    m = MeanSquaredError()  # MeanSquaredError 인스턴스 생성
    m.update_state(np.array(actual_pred['Adj. Close']), np.array(actual_pred['prediction']))  # 상태 업데이트
    
    return (m.result().numpy(), actual_pred.plot())  # Mean Squared Error와 그래프 반환


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score

def confirm_result(y_test, y_pred):
    MAE = mean_absolute_error(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MSLE = mean_squared_log_error(y_test, y_pred)
    RMSLE = np.sqrt(mean_squared_log_error(y_test, y_pred))
    R2 = r2_score(y_test, y_pred)
    
    pd.options.display.float_format = '{:.5f}'.format
    Result = pd.DataFrame(data=[MAE,RMSE, RMSLE, R2],
                         index = ['MAE','RMSE', 'RMSLE', 'R2'],
                         columns=['Results'])
    return Result


def GRU_model(X_train, y_train, X_test, sc):
    # create a model 
    from keras.models import Sequential 
    from keras.layers import Dense, SimpleRNN, GRU
    from keras.optimizers.legacy import SGD 
    
    # The GRU architecture 
    my_GRU_model = Sequential()
    my_GRU_model.add(GRU(units = 50, 
                         return_sequences = True, 
                         input_shape = (X_train.shape[1],1), 
                         activation = 'tanh'))
    my_GRU_model.add(GRU(units = 50, 
                         activation = 'tanh'))
    my_GRU_model.add(Dense(units = 2))
    
    # Compiling the RNN 
    my_GRU_model.compile(optimizer = SGD(lr = 0.01, decay = 1e-7, 
                                         momentum = 0.9, nesterov = False), 
                         loss = 'mean_squared_error')
    
    # Fitting to the trainig set 
    my_GRU_model.fit(X_train, y_train, epochs = 50, batch_size = 150, verbose = 0)
    
    GRU_prediction = my_GRU_model.predict(X_test)
    GRU_prediction = sc.inverse_transform(GRU_prediction)
    
    return my_GRU_model, GRU_prediction 


# 학습 시작 전 시간 저장
start_time = time.time()


# 그래프
my_GRU_model, GRU_prediction = GRU_model(X_train, y_train, X_test, sc)
GRU_prediction[1:10]
# actual_pred_plot(GRU_prediction)


# 학습 후 시간 저장
end_time = time.time()


# 걸린 시간 계산
elapsed_time = end_time - start_time


y_pred_gru = pd.DataFrame(GRU_prediction[:, 0])
y_test_gru=all_data.loc['2019':,'Adj Close'][0:len(GRU_prediction)]
y_test_gru.reset_index(drop=True, inplace=True)


print(confirm_result(y_test_gru, y_pred_gru))
print(f"GRU 모델 학습에 걸린 시간은 {elapsed_time:.2f}초 입니다.")