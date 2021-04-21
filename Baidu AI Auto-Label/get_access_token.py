import requests

# client_id 为官网获取的, client_secret 为官网获取的
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=8TLdFM2arcHyDgT2GWVYPYiW&client_secret=0V7xmPOjwP1judqTaM8uaYBEmfvFCrFp'
response = requests.get(host)
if response:
    print(response.json())
