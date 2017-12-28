from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics
import json

def build_model():
    global max_words
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(max_words,)))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# カテゴリ数・バッチ数・エポック数をそれぞれ定義
nb_classes = 9
batch_size = 64
np_epoch   = 20

# 学習用データとテストデータの用意
data = json.load(open('./newstext/data.json'))

X = data['X'] # テキスト
Y = data['Y'] # カテゴリ

max_words = len(X[0])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
Y_train = np_utils.to_categorical(Y_train, nb_classes)
print(len(X_train), len(Y_train))

# 分類問題をscikitを使って解く(kerasのmodelでラップ)
model = KerasClassifier(build_fn=build_model, nb_epoch=np_epoch, batch_size=batch_size)
model.fit(X_train, Y_train)

# 予測と精度測定
y         = model.predict(X_test)
ac_score  = metrics.accuracy_score(Y_test, y)
cl_report = metrics.classification_report(Y_test, y)

print('正解率 =', ac_score)
print('レポート =\n', cl_report)
