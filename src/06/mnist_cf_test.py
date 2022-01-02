from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# MNIST 데이터셋 가져오기 / 이미 학습된 모델을 불러와 사용하므로 테스트셋만 사용
_, (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0 # 데이터 정규화

# 모델 불러오기
model = load_model('mnist_model.h5')
model.summary()
model.evaluate(x_test, y_test, verbose=2)

# 테스트셋에서 20번째 이미지 출력 / 실제 예측된 값과 비교하기 위해 이미지로 출력
plt.imshow(x_test[20], cmap='gray')
plt.show()

# 테스트셋의 20번쨰 이미지 클래스 분류
picks = [20]
# 오류 발생 / tf 2.6 이후로 predict_classes가 없기 때문에 발생하는 오류
# predict = model.predict_classes(x_test[picks])  # 입력 데이터에 대해 클래스를 예측(분류)한 값을 반환
y_prob = model.predict(x_test[picks])
predict = y_prob.argmax(axis=-1)
print('손글씨 이미지 예측값 : ', predict)