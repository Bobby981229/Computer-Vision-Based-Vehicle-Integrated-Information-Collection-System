import requests
import cv2
import numpy as np
import json
import base64
import os


def get_token(client_id, client_secret):
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + client_id + '&client_secret=' + client_secret
    res = requests.get(host, headers=headers).text
    res = json.loads(res)
    return res['access_token']


def get_license_plate(access_token, i):
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/license_plate?access_token=" + access_token
    # img=cv2.imread(img_path)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    with open(img_path, 'rb') as f:
        image_binary = f.read()
    image_encode = base64.b64encode(image_binary)  # Image requires base64 encryption
    # print(image_encode)
    postdata = {'image': image_encode,
                'multi_detect': 'false'}
    respond = requests.post(url, data=postdata, headers=headers)  # Send a POST request to get license plate information
    respond.encoding = 'utf-8'
    words_result = json.loads(respond.text)
    if 'words_result' in words_result.keys():
        number = words_result['words_result']['number']
        print(number)
        # cv2.imwrite(save_path+number+'.png',img)
        cv2.imencode('.png', img)[1].tofile(save_dir + number + '.png')
    else:
        print('%s Unrecognized ' % name)
        cv2.imencode('.png', img)[1].tofile(save_dir + name)


if __name__ == '__main__':
    client_id = '8TLdFM2arcHyDgT2GWVYPYiW'  # API Key URL https://ai.baidu.com/forum/topic/show/943028
    client_secret = '0V7xmPOjwP1judqTaM8uaYBEmfvFCrFp'  # Secret Key URL https://ai.baidu.com/forum/topic/show/943028
    headers = {
        'user-agent': "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36"}
    access_token = get_token(client_id, client_secret)
    img_dir = "../Unmarked Pictures/"  # image path
    img_name = os.listdir(img_dir)
    save_dir = '../Marked Pictures/'  # save path
    for name in img_name:
        img_path = img_dir + name
        get_license_plate(access_token, name)