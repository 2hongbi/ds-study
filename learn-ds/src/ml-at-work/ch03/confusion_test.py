from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

# 데이터를 훈련 데이터와 테스트 데이터로 분할하기
data_train, data_test, label_train, label_test = train_test_split(data, label)

# 분류기로 예측하기
classfier = svm.SVC(kernel='linear')
label_pred = classfier.fit(data_train, label_train).predict(data_test)

# 혼동행렬 계산
cm = confusion_matrix(label_test, label_pred)
print(cm)