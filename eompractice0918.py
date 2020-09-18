'''
종가 예측 예제에 원핫인코딩 전처리해서 돌려봄
파이썬 3.6, 텐서플로우 1.0
This script shows how to predict stock prices using a basic RNN
'''
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
#data-02-stock_daily.csv 주식데이터 가지고 RNN알고리즘 적용하여 종가 예측하는 예제

tf.set_random_seed(777)  # reproducibility

#스케일링의 종류: Standard Scaler, MinMaxScaler MaxAbsScaler(최대절대값과 0이 각각 1, 0이  되도록), RobustScaler(중앙값과 IQR 사용)
# MinMaxScaler  최대/최소 값이 각각 1,0이 되도록 스케일링
def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7
data_dim = 9  #컬럼 갯수 넣어줌
hidden_dim = 5  #원래 10이었던 것을 바꿔보자
#hidden_dim=5로 바꿀 경우 RMSE 0.0461, loss:73.174...별로 영향도 없어

output_dim = 1
learning_rate = 0.01
#학습 70%할 경우 고정 시 아래 학습률 변경할 때 RMSE, loss값
# 0.001로 할 경우, RMSE: 0.0456039, loss:71.4000396
# 0.05로 할 경우, RMSE 0.0466, loss:74.674
# 0.005로 할 경우, RMSE 0.0467, loss 74.5014,
iterations = 500

# Open, High, Low, Volume, Close
xy = np.loadtxt('train_test_set20918.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]  # Close as label

# build a dataset
dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

# train/test split
train_size = int(len(dataY) * 0.7)
# 0.7일때 RMSE: 0.0456039, loss:71.4000396
# 0.8일떄 RMSE: 0.0470961, loss:78.13455
# 0.9일때 RMSE: 0.0498497, loss: 85.27803
# loss는 훈련 손실값
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])

# build a LSTM network
# 블록형태라서 은닉층 dense를 늘려가는 것은 여기서는 안됨
# 93행에서 hidden_dim의 숫자 정도는 바꿔볼 수 있겠다.
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
#RMSE 평균 제곱근 오차 , MSE 평균 제곱 오차에 루트 씌운 것, 최소화 하는 게 좋다. 98행 관련
#특이값이 많은 경우 MAE쓰는게 좋다.
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
#MAPE로 식을 바꿔야 하나? 공모전 케이스 평가 지표
#MAPE는 MSE보다 특이치에 강건하다 (변동폭이 크지 않다)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Revenue")
    plt.show()
