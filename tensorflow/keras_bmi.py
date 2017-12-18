# デバッグ用のtensorflow
import keras.backend as K
from tensorflow.python import tfdbg

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
import pandas as pd, numpy as np

# デバッグ用設定
def set_debug():
    sess = K.get_session()
    sess = tfdbg.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)

# csv読み込みと正規化
csv = pd.read_csv('bmi.csv')
csv['weight'] /= 100
csv['height'] /= 200

# 重み
X = csv[['weight', 'height']].as_matrix()
bclass = {'thin': [1,0,0], 'normal': [0,1,0], 'fat': [0,0,1]}

# ラベル
y = np.empty((20000, 3))
for i, v in enumerate(csv['label']):
    y[i] = bclass[v]

# 訓練データとテストデータに分ける
X_train, y_train = X[1:15001], y[1:15001]
X_test, y_test   = X[15001:20000], y[15001:20000]

# 1層目
model = Sequential()
model.add(Dense(512, input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))

# 2層目
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))

# 3層目
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

# 訓練
hist = model.fit(
    X_train, y_train,
    batch_size=100,
    nb_epoch=20,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
    verbose=1)

# 検証
loss, accuracy = model.evaluate(X_test, y_test)
print('loss=', loss)
print('accuracy=', accuracy)
