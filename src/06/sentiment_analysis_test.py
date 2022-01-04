# ex 6-3 (p.163) 문장 감정 분류 CNN 모델
# 필요한 모듈 임포트
import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate

# 데이터 읽어오기
train_file = './../../data/chatbot_data.csv'
data = pd.read_csv(train_file, delimiter=',')
features = data['Q'].tolist()   # Q : 질문
labels = data['label'].tolist()     # label : 감정

# 단어 인덱스 시퀀스 벡터
corpus = [preprocessing.text.text_to_word_sequence(text) for text in features]  # text_to_word_sequence : 단어 시퀀스 생ㅎ
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
word_index = tokenizer.word_index

MAX_SEQ_LEN = 15    # 단어 시퀀스 벡터 크기
# pad_sequences() : 시퀀스의 패딩 처리를 손쉽게 함
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

# 학습용, 검증용, 테스트용 데이터셋 생성
# 학습셋: 검증셋 : 테스트셋 = 7:2:1 (or 6:2:2)
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, labels))
ds = ds.shuffle(len(features))

train_size = int(len(padded_seqs) * 0.7)
val_size = int(len(padded_seqs) * 0.2)
test_size = int(len(padded_seqs) * 0.1)

train_ds = ds.take(train_size).batch(20)
val_ds = ds.skip(train_size).take(val_size).batch(20)
test_ds = ds.skip(train_size + val_size).take(test_size).batch(20)

# 하이퍼파라미터 설정
dropout_prob = 0.5
EMB_SIZE = 128
EPOCH = 5
VOCAB_SIZE = len(word_index) + 1    # 전체 단어 수

# CNN 모델 정의
input_layer = Input(shape=(MAX_SEQ_LEN,))   # 입력 계층은 케라스의 Input()으로 생성, shape: 입력 노드에 들어올 데이터의 형상 지정
embedding_layer = Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_SEQ_LEN)(input_layer)
# 50% 확률로 dropout 생성. 학습 과정에서 발생할지도 모르는 오버피팅(과적합)에 대비
dropout_emb = Dropout(rate=dropout_prob)(embedding_layer)

# 크기가 3, 4, 5인 합성곱 필터를 128개씩 사용한 합성곱 계층을 3개 생성.이는 3, 4, 5-gram 언어 모델의 개념과 비슷
conv1 = Conv1D(
    filters=128,
    kernel_size=3,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool1 = GlobalMaxPool1D()(conv1)    # GlobalMaxPool1D : 최대 풀링 연산 수행

conv2 = Conv1D(
    filters=128,
    kernel_size=4,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool2 = GlobalMaxPool1D()(conv2)

conv3 = Conv1D(
    filters=128,
    kernel_size=5,
    padding='valid',
    activation=tf.nn.relu)(dropout_emb)
pool3 = GlobalMaxPool1D()(conv3)

# 3, 4, 5-gram 이후 합치기 > 완전 연결 계층에 전달. 각각 병렬로 처리된 합성곱 계층의 특징맵 결과를 하나로 묶어줌
concat = concatenate([pool1, pool2, pool3])

hidden = Dense(128, activation=tf.nn.relu)(concat)
dropout_hidden = Dropout(rate=dropout_prob)(hidden)
logits = Dense(3, name='logits')(dropout_hidden)    # 점수 score
predictions = Dense(3, activation=tf.nn.softmax)(logits)

# 모델 생성
model = Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',   # 클래스 분류 모델을 학습할 때 주로 손실값을 계산
              metrics=['accuracy']) # 케라스 모델 평가 시 정확도 확인 위해 accuracy 사용

# 모델 학습 / verbose 인자가 1인 경우, 모델 학습 시 진행 과정을 상세하게 보여줌(0 : 생략)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCH, verbose=1)

# 모델 평가(테스트 데이터셋 이용)
loss, accuracy = model.evaluate(test_ds, verbose=1)
print('Accuracy :  %f' % (accuracy * 100))
print('loss : %f' % (loss))

# 모델 저장
model.save('cnn_model.h5')
