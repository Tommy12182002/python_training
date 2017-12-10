import urllib.request
url = 'https://uta.pw/shodou/img/28/214.png'
urllib.request.urlretrieve(url, 'test.png')
print('done')
