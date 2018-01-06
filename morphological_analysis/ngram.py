def ngram(str, num):
    res  = []
    slen = len(str) - num + 1
    for i in range(slen):
        extract_str = str[i:i+num]
        res.append(extract_str)
    return res

# 2つの文章の類似率を調べる
def diff_ngram(str1, str2, num):
    str1_ngram   = ngram(str1, num)
    str2_ngram   = ngram(str2, num)
    match_result = []
    count        = 0
    for str1_unit in str1_ngram:
        for str2_unit in str2_ngram:
            if str1_unit == str2_unit:
                count += 1
                match_result.append(str1_unit)
    return count / len(str1_ngram), match_result

a = '今日、渋谷で美味しいトンカツを食べた。'
b = '渋谷で食べた今日のトンカツは美味しかった。'

result2, word_list2 = diff_ngram(a, b, 2)
result3, word_list3 = diff_ngram(a, b, 3)

print('2-gram:', result2, word_list2)
print('3-gram:', result3, word_list3)
