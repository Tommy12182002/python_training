from janome.tokenizer import Tokenizer
import os, re, json, random

# 辞書ファイルの作成（単語ごとに分けて前後の関係を保存）
def make_dic(words):
    # 文章の先頭を@を使用して表現する
    tmp = ['@']
    dic = {}

    for i in words:
        word = i.surface
        # 改行や空文字の場合はスキップ
        if word == '' or word == '\r\n' or word == '\n': continue
        tmp.append(word)
        if len(tmp) < 3: continue
        if len(tmp) > 3: tmp = tmp[1:]
        set_to_dic(dic, tmp)
        if word == '。': tmp = ['@']
    return dic

def set_to_dic(dic, words):
    w1, w2, w3 = words
    if not w1 in dic: dic[w1]                 = {}
    if not w2 in dic[w1]: dic[w1][w2]         = {}
    if not w3 in dic[w1][w2]: dic[w1][w2][w3] = 0
    dic[w1][w2][w3] += 1

# 作文する
def make_sentence(dic):
    ret = []
    if not '@' in dic: return 'invalid dic'
    top = dic['@']
    w1  = choice_word(top)
    w2  = choice_word(top[w1])
    ret.append(w1)
    ret.append(w2)
    while True:
        w3 = choice_word(dic[w1][w2])
        ret.append(w3)
        if w3 == '。': break
        w1, w2 = w2, w3
    return ''.join(ret)

def choice_word(sel):
    keys = sel.keys()
    return random.choice(list(keys))

# 解析するテキストファイルを読み込む
sjis_file = 'kokoro.txt.sjis'
dic_file  = 'markov-kokoro.json'

if not os.path.exists(dic_file):
    # 辞書ファイル(単語単位のngram情報)を作成する
    sjis = open(sjis_file, 'rb').read()
    text = sjis.decode('shift_jis')
    # 不要な部分を削除する
    text = re.split(r'\-{5,}',text)[2] # ヘッダを削除
    text = re.split(r'底本：', text)[0] # フッタを削除
    text = text.strip()
    text = text.replace('｜', '') # ルビの開始記号を削除
    text = re.sub(r'《.+?》', '', text) # ルビを削除
    text = re.sub(r'［＃.+?］', '', text) # 入力注を削除

    # 形態素解析して辞書ファイルを作成
    t     = Tokenizer()
    words = t.tokenize(text)
    dic   = make_dic(words)
    json.dump(dic, open(dic_file, 'w', encoding='utf-8'))
else:
    dic = json.load(open(dic_file, 'r'))

# 作文する
for i in range(3):
    s = make_sentence(dic)
    print(s)
    print('---')
