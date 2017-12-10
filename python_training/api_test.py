import urllib.request
import sys

# データ取得
#url = 'http://api.aoikujira.com/time/get.php'
#print(sys.argv[1])
url = 'http://api.aoikujira.com/zip/xml/get.php'
values = {
    'fmt': 'xml',
    'zn': '1500042'
}
# hashをquerystringの形に変換
params = urllib.parse.urlencode(values)
url = url + '?' + params
res = urllib.request.urlopen(url)
data = res.read()

# 取得されるものはバイナリなのでstringに変換する
text = data.decode('utf-8')
print(text) 

