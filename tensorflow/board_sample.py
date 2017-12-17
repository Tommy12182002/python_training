import numpy as np
import pandas as pd
import tensorflow as tf

# ファイル読み込み
csv = pd.read_csv('bmi.csv')

# 適当に正規化
csv['height'] = csv['height'] / 200
csv['weight'] = csv['weight'] / 100

#  ラベルを三次元のクラスで表す
bclass = {'thin': [1,0,0], 'normal': [0,1,0], 'fat': [0,0,1]}
# 三次元の配列が入っている配列にする
csv['label_array'] = csv['label'].apply(lambda x : np.array(bclass[x]))

# テストデータの準備(重さ・高さの配列と、三次元配列の配列を抽出)
test_csv = csv[15000:20000]
test_pat = test_csv[['weight', 'height']]
test_ans = list(test_csv['label_array'])

# 身長・体重、答えのラベルを格納する変数
x = tf.placeholder(tf.float32, [None, 2], name="x")
y_ = tf.placeholder(tf.float32, [None, 3], name="y_")

# 重みとバイアスの宣言
with tf.name_scope('interface') as scope:
    W = tf.Variable(tf.zeros([2, 3]), name="W")
    b = tf.Variable(tf.zeros([3]), name="b")

    # 予測結果を定義(matmulはnumpyのdotと同じ) ソフトマックスで出力した結果
    with tf.name_scope('softmax') as scope:
        y = tf.nn.softmax(tf.matmul(x, W) + b)

# 訓練（損失関数を定義して、勾配降下法で損失関数を最小にする）
#　答えのラベルと予測結果を使用する
with tf.name_scope('loss') as scope:
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(cross_entropy)

# 精度を求める（答えと予測結果が正しいか）
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

# セッションの開始
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 学習
for step in range(3500):
    # 100件ずつミニバッチで学習させる
    i = (step * 100) % 14000
    rows = csv[1 + i : 1 + i + 100]

    # 学習データとその答えの用意
    x_pat = rows[['weight', 'height']]
    y_ans = list(rows['label_array'])
    fd = {x: x_pat, y_: y_ans}

    sess.run(train, feed_dict=fd)

    # 500件ごとに損失関数と精度を表示する
    if step % 500 == 0:
        # 学習データを使って誤差を求める
        cre = sess.run(cross_entropy, feed_dict=fd)
        # テストデータを使って精度を求める
        acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
        print('step=', step, 'cre=', cre, 'acc=', acc)

acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
print('最終的な正解率=', acc)

tw = tf.summary.FileWriter('log_dir', graph=sess.graph)
