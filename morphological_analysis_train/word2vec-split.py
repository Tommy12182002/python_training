import os, re
from janome.tokenizer import Tokenizer

# 形態素解析
def tokenize(text):
    t = Tokenizer()
    # テキストの先頭にあるヘッダとフッタを削除 --- (※2)
    text = re.split(r'\-{5,}',text)[2]
    text = re.split(r'底本：', text)[0]
    text = text.strip()
    # ルビを削除
    text = text.replace('｜', '')
    text = re.sub(r'《.+?》', '', text)
    # 脚注を削除
    text = re.sub(r'［＃.+?］', '', text)

    lines   = text.split('\r\n')
    results = []
    for line in lines:
        res = []
        tokens = t.tokenize(line)
        for tok in tokens:
            if tok.base_form == "*": # 単語の基本系を採用
                w = tok.surface
            else:
                w = tok.base_form
            ps = tok.part_of_speech # 品詞情報
            hinsi = ps.split(',')[0]
            if hinsi in ['名詞', '形容詞', '動詞', '記号']:
                res.append(w)
        results.append(" ".join(res))
    return results

# 辞書データの作成
persons = ['夏目漱石', '太宰治', '芥川龍之介']
sakuhin_count = {}
for person in persons:
    person_dir            = './text/' + person
    sakuhin_count[person] = 0
    results               = []

    for sakuhin in os.listdir(person_dir):
        print(person, sakuhin)
        sakuhin_count[person] += 1
        sakuhin_file = person_dir + '/' + sakuhin

        try:
            bindata = open(sakuhin_file, 'rb').read()
            text    = bindata.decode('shift_jis')
            lines   = tokenize(text)
            results += lines
        except Exception as e:
            print('[error]', sakuhin_file, e)
            continue

    # ファイルへ保存
    fname = './text/' + person + '.wakati'
    with open(fname, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print(person)

print('作品数：', sakuhin_count)
