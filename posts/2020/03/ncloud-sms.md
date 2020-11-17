---
title: "[Python] Send ncloud sms message"
tags:
  - python
published: true
date: 2020-03-17 06:43:12
description: 네이버 클라우드 플랫폼의 서비스 중 하나인
  https://www.ncloud.com/product/applicationService/sens 로 SMS를 발송하는 예제.
  ncloud서비스를 다 써본건 아니지만, `make_signature`는 전 서비스에 다 똑같이 쓸 수 있을 것 같은 기분이다.
  ```python import time import req...
category: python
slug: /2020/03/ncloud-sms/
template: post
---
네이버 클라우드 플랫폼의 서비스 중 하나인 https://www.ncloud.com/product/applicationService/sens 로 SMS를 발송하는 예제. ncloud서비스를 다 써본건 아니지만, `make_signature`는 전 서비스에 다 똑같이 쓸 수 있을 것 같은 기분이다.

```python
import time
import requests
import hashlib
import hmac
import base64

def send_sms(phone_number, subject, message):
  def make_signature(access_key, secret_key, method, uri, timestmap):
    secret_key = bytes(secret_key, 'UTF-8')

    message = method + " " + uri + "\n" + timestamp + "\n" + access_key
    message = bytes(message, 'UTF-8')
    signingKey = base64.b64encode(hmac.new(secret_key, message, digestmod=hashlib.sha256).digest())
    return signingKey

  #  URL
  url = 'https://sens.apigw.ntruss.com/sms/v2/services/ncp:sms:kr:99999999999:sample/messages'
  # access key
  access_key = 'access_key'
  # secret key
  secret_key = 'secret_key'
  # uri
  uri = '/sms/v2/services/ncp:sms:kr:99999999999:sample/messages'
  timestamp = str(int(time.time() * 1000))

  body = {
    "type":"LMS",
    "contentType":"COMM",
    "countryCode":"82",
    "from":"01012345678",
    "content": message,
    "messages":[
        {
            "to": phone_number,
            "subject": subject,
            "content": message
        }
    ]
  }

  key = make_signature(access_key, secret_key, 'POST', uri, timestamp)
  headers = {
    'Content-Type': 'application/json; charset=utf-8',
    'x-ncp-apigw-timestamp': timestamp,
    'x-ncp-iam-access-key': access_key,
    'x-ncp-apigw-signature-v2': key
  }


  res = requests.post(url, json=body, headers=headers)
  print(res.json())
  return res.json()
```
