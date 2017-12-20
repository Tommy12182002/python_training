from keras.models               import Sequential
from keras.datasets             import mnist
from keras.layers.convolutional import Conv2D
from keras.layers.pooling       import MaxPooling2D
from keras.layers.core          import Dense, Dropout, Activation, Flatten
from keras.optimizers           import Adam
from keras.utils                import np_utils

from keras import backend as K

# ------------------------------
# データ準備
# ------------------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# データをfloat型にして正規化する
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float') / 255.0

img_rows = 28
img_cols = 28

if K.image_data_format() == 'channels_first':
    X_train     = X_train.reshape(-1, 1, img_rows, img_cols)
    X_test      = X_test.reshape(-1, 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train     = X_train.reshape(-1, img_rows, img_cols, 1)
    X_test      = X_test.reshape(-1, img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# ラベルはone-hot encodingを施す
y_train = y_train.astype('int32')
y_test  = y_test.astype('int32')
y_train = np_utils.to_categorical(y_train, 10)
y_test  = np_utils.to_categorical(y_test, 10)

# ------------------------------
# モデルの定義
# ------------------------------
# 場合分けでインプットデータのテンソルの形を変える
# mnistデータは28×28ピクセルで60000個のデータで、28×28ピクセルで1チャネルのデータに変える
# reshapeの-1は、28×28ピクセルで60000個のデータを28×28ピクセルで1チャネルによしなに変えてくれる
model = Sequential()

# 畳み込み層1
# フィルタは5×5ピクセルで28個→出力データは32チャネル
# 入力データは28×28ピクセルの1チャンネル
# input_shapeを指定するのは1層目だけ
model.add(Conv2D(32, (5, 5), input_shape=input_shape))
model.add(Activation('relu'))
# プーリング層1
model.add(MaxPooling2D(pool_size=(2, 2)))

# 畳み込み層2
# フィルタは5×5ピクセルで64個→出力データは64チャネル
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
# プーリング層2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattenして全結合する
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

# 過学習予防のためドロップアウトして最後に出力(出力データ数はone-hotなので10個)
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# ------------------------------
# 学習の開始
# ------------------------------
epochs = 20
batch_size = 100

adam = Adam(lr=1e-4)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print('loss=', loss)
print('accuracy=', accuracy)
