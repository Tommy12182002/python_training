# 全結合verをkerasで実装

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils

# mnistデータの読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# データをfloat型にして正規化する
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test  = X_test.reshape(10000, 784).astype('float') / 255

# ラベルをone_hot形式に変換
y_train = np_utils.to_categorical(y_train, 10)
y_test  = np_utils.to_categorical(y_test, 10)

# モデルの構造を定義
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# Function APIの書き方だとこう
# inputs = Input(shape=(784,))
#
# nw = Dense(512, activation='relu')(inputs)
# nw = Dropout(.2)(nw)
# nw = Dense(512, activation='relu')(nw)
# nw = Dropout(.2)(nw)
# predictions = Dense(10, activation='softmax')(nw)


# モデルの構築
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['accuracy'])

# データで訓練
hist = model.fit(X_train, y_train)

# テストデータで評価する
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('loss=', loss)
print('accuracy=', accuracy)
