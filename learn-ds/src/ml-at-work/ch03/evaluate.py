from math import sqrt

# 평균제곱근오차
sum = 0

for predict, actual in zip(predicts, actuals):
    sum += (predict - actual) ** 2

sqrt(sum / len(predicts))


# 사이킷런에는 mean_squared_error 함수로 따로 구현되어 있음
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_actual, y_predicted))