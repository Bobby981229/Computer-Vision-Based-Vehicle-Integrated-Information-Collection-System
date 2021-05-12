"""
Recognize the make, model, color, year of the vehicle by BaiDu API
"""
import json
import http.client
import hashlib
from urllib import parse
import random
import base64
import json
import requests


def translation(text):
    """
    Translate text via Baidu API
    :param text: text need to translate
    :return:  text after translating
    """
    appid = '20210412000774999'  # App ID in Baidu API
    secretKey = 'GVf5TNLMrjlZ1sCushsZ'  # Secret Key in Baidu API

    httpClient = None
    myurl = '/api/trans/vip/translate'
    fromLang = 'zh'  # Original language - Chinese
    toLang = 'en'  # Translation Language - English
    salt = random.randint(32768, 65536)

    sign = appid + text + str(salt) + secretKey
    m1 = hashlib.md5()
    m1.update(sign.encode("utf-8"))
    sign = m1.hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + parse.quote(
        text) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        response = httpClient.getresponse()
        # Transcoding
        html = response.read().decode('utf-8')
        html = json.loads(html)
        dst = html["trans_result"][0]["dst"]
        # print(dst)
        return dst
    except Exception as e:
        print(e)
    finally:
        if httpClient:
            httpClient.close()


def car_model(path):
    """
    Vehicle model recognition via Baidu API
    :param path: Vehicle image path
    :return: Recognition results
    """
    # # client_id -> ak， client_secret -> sk
    # host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=KWyIaUr2hg3jaaMBKGLGOnRP&client_secret=OopxV3wv036xXyb2GXeXsL6aCffO2hCL'
    # response = requests.get(host)
    # if response:
    #     print(response.json())
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/car"
    # Binary way to open image files
    f = open(path, 'rb')
    img = base64.b64encode(f.read())

    params = {"image": img, "top_num": 5}
    access_token = '24.56492885436cef84eb0071e21e8b3be3.2592000.1620830776.282335-23977855'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)

    if response:
        # print(response.json())
        all_result = json.loads(response.text)
        brand_name = all_result['result'][0]['name']
        model_year = all_result['result'][0]['year']
        car_colour = all_result['color_result']
        # Recognition: make, colour, model, year
        result_trans = translation(brand_name + car_colour + model_year)
        print(result_trans)

        return result_trans


