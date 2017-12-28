import os, glob, json

# 分かちファイルから深層学習に読ませるための情報に変換

root_dir      = './newstext'
# 単語IDの辞書ファイル
dic_file      = root_dir + '/word-dic.json'
data_file     = root_dir + '/data.json'
data_file_min = root_dir + '/data-min.json'

# 単語ID保持用辞書(_MAXは連番の最大値を示す)
word_dic = { '_MAX': 0 }

# 単語IDをdictionaryに登録する
def register_dic():
    files = glob.glob(root_dir + '/*/*.wakati', recursive=True)
    for news_file in files:
        file_to_ids(news_file)

# ファイル内の単語から単語IDのdictionaryを生成する
def file_to_ids(news_file):
    with open(news_file, 'r') as f:
        text_to_ids(f.read())

def text_to_ids(text):
    text   = text.strip()
    words  = text.split(' ')
    result = []

    for word in words:
        word = word.strip()
        if not word: continue
        if not word in word_dic:
            wid = word_dic[word] = word_dic['_MAX']
            word_dic['_MAX'] += 1
            print(wid, word)
        else:
            wid = word_dic[word]
        result.append(wid)

    return result

# ジャンルごとにファイルを読み、カテゴリごとの各単語の出現数(学習データ)、カテゴリのID(学習データに対する正解ラベル)を求める
def count_freq(limit = None):
    X              = []
    Y              = []
    category_names = []

    for category in os.listdir(root_dir):
        category_dir = '{0}/{1}'.format(root_dir,category)
        if not os.path.isdir(category_dir): continue
        category_idx = len(category_names)
        category_names.append(category)
        files = glob.glob(category_dir + '/*.wakati')

        # カウンターは1スタート
        for index, path in enumerate(files, 1):
            print(path)
            count = count_file_freq(path)
            X.append(count)
            Y.append(category_idx)

            if limit is not None and index >= limit: break

    return X, Y

def count_file_freq(file):
    # 全単語のカウントを0で初期化
    count = [0 for n in range(word_dic['_MAX'])]
    with open(file, 'r') as f:
        text = f.read().strip()
        ids  = text_to_ids(text)
        for wid in ids:
            count[wid] += 1

    return count

# 単語IDの辞書生成
if os.path.exists(dic_file):
    word_dic = json.load(open(dic_file))
else:
    # なければ生成
    register_dic()
    word_dic = json.dump(word_dic, open(dic_file, 'w'))

# ファイルごとの単語出現頻度のベクトル生成
# テスト用の小規模データと学習用に全単語データ両方を生成
X, Y = count_freq(20)
json.dump({ 'X': X, 'Y': Y }, open(data_file_min, 'w'))
X, Y = count_freq()
json.dump({ 'X': X, 'Y': Y }, open(data_file, 'w'))

print('done')
