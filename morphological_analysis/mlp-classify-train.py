from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import json

# カテゴリ数・バッチ数・エポック数をそれぞれ定義
nb_classes = 9
batch_size = 64
np_epoch   = 20

# 学習用データとテストデータの用意
data = json.load(open('./newstext/data.json'))
X    = data['X'] # テキスト
Y    = data['Y'] # カテゴリ

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
Y_train                          = np_utils.to_categorical(Y_train, nb_classes)
print(len(X_train), len(Y_train))

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(len(X[0]),)))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# データで訓練
model.fit(X_train, Y_train)

# 分類問題を解くためのモデル生成(繰り返し利用できるようにファイルとして保存)
model.fit(X_train, Y_train)
model.model.save('model.h5')

print('fit done.')
