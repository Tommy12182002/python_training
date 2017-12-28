from janome.tokenizer import Tokenizer
import os, glob

ja_tokenizer = Tokenizer()

def ja_tokenize(text):
    res = []
    lines = text.split('\n')
    # 最初の2行はヘッダ
    lines = lines[2:]

    for line in lines:
        malist = ja_tokenizer.tokenize(line)
        for tok in malist:
            ps = tok.part_of_speech.split(',')[0]
            if not ps in ['名詞', '形容詞', '動詞']: continue
            w = tok.base_form
            if w == '*' or w == '': w = tok.surface
            if w == ''  or w == '\n': continue
            res.append(w)
        res.append('\n')
    return res

root_dir = './newstext'
for path in glob.glob(root_dir+'/*/*.txt', recursive=True):
    # ライセンスファイルはスキップ
    if path.find('LICENSE') > 0: continue
    print(path)

    path_wakati = path + '.wakati'
    if os.path.exists(path_wakati): continue

    # ニューステキストを読み取って分かちファイルにする
    text  = open(path, 'r').read()
    words = ja_tokenize(text)
    wt    = ' '.join(words)
    open(path_wakati, 'w', encoding='utf-8').write(wt)
