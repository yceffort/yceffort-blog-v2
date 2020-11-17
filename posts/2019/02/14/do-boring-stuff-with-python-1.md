---
title: 업무 자동화 (1) - 구글 스프레드 시트 API 활용하기
date: 2019-02-14 07:56:22
published: true
tags:
  - python
description: 구글 스프레드 시트를 파이썬에서 조작해보자. 내가 할일은 1. 스프레드시트를 읽고 2. 스프레드시트에 쓰는 두가지
  작업이다. ```python import pickle import os.path from googleapiclient.discovery
  import build from google_auth_oauthlib.flow import Installe...
category: python
slug: /2019/02/14/do-boring-stuff-with-python-1/
template: post
---
구글 스프레드 시트를 파이썬에서 조작해보자. 내가 할일은 1. 스프레드시트를 읽고 2. 스프레드시트에 쓰는 두가지 작업이다.

```python
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

class GoogleSheetInit():

    def __init__(self):
        self.creds = None

    def initialize(self):
        creds = None

        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file('../credentials.json', SCOPES)
                creds = flow.run_local_server()
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        self.creds = creds
```

구글 docs에 접근하는 방법은 두가지인데, 한가지는 oauth2기반 인증과, 다른 한가지는 api_key방식 인증이다. api_key 인증 방식은 요청시에 parameter로 api key를 보내는 방식인데, 안타깝게도 보안상의 문제로 인해 전체 공개된 문서에만 접근할 수 있다.

따라서 제한적으로 공개되어 있는 문서에 접근하기 위해서는 oauth2 방식을 활용해야 한다.

```python
google_sheet = GoogleSheetInit()
google_sheet.initialize()
```

실행하게 되면 브라우저에서 구글 계정 인증을 받게 된다. 계정인증을 거친 뒤에는 인증 정보가 `token.pickle`에 남아서 이 후부터는 별도의 인증없이 접근할 수 있다. 그리고 해당 인증정보를 파이썬 코드에서 사용할 수 있도록 creds가 반환된다.

자세한 api 스펙은 [여기](https://developers.google.com/sheets/api/guides/values)에서 참조하면 된다.

### 스프레드시트 읽기

```python
service = build('sheets', 'v4', credentials=self.creds)
sheet = service.spreadsheets()
result = sheet.values().get(spreadsheetId=self.sheet_id, range=self.sheet_range).execute()
values = result.get('values', )
```

sheet_id는 해당 스프레드 시트의 id인데, url에 나와있다. 그리고 sheet_range는 `시트이름!A1:Z1` 이런식으로 접근하면 된다.

```python
enumerate(values)
```

으로 접근할 수 있다.

### 스프레드시트 쓰기

```python
request = sheet.values().update(spreadsheetId=self.sheet_id, range=range, valueInputOption='RAW', body={ "values": [[value]]})
response = request.execute()
```

body영역은 실제 스프레드시트에 쓰려고 하는 영역의 크기만큼 설정하면 된다. 위의 예제에서는 단순히 셀 1개에만 쓰는 케이스다.
