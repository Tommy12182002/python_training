from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.models import load_model
import json

# テストデータの用意
data = json.load(open('./newstext/data.json'))
X    = data['X'] # テキスト
Y    = data['Y'] # カテゴリ

nb_classes           = 9
_, X_test, _, Y_test = train_test_split(X, Y)
Y_test               = np_utils.to_categorical(Y_test, nb_classes)
print(len(X_test), len(Y_test))

# 学習済みのモデルを使って予測と精度測定
model = load_model('model.h5')
loss, accuracy = model.evaluate(X_test, Y_test, verbose=1)

print('loss=', loss)
print('accuracy=', accuracy)
