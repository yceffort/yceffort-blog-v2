---
title: Pytorch (2-3) - 뉴스 카테고리 분류하기
date: 2019-01-28 09:32:01
published: true
tags:
  - pytorch
description: 뉴스 말뭉치를 다운로드 받아서 분석해보자. 말뭉치는
  [여기](http://www.kristalinfo.com/download/hkib-20000-40075.tar.gz)에서 받을 수 있다.
  과거 뉴스 데이터를 다운로드해서, 어떤 카테코리인지 분류하는 학습을 진행해보자. 먼저 구글드라이브에 해당 파일을 업로드해서 진행했다. 물론
  아래와 같은 코드로 cola...
category: pytorch
slug: /2019/01/28/pytorch-2-multi-perceptron(3)/
template: post
---

뉴스 말뭉치를 다운로드 받아서 분석해보자. 말뭉치는 [여기](http://www.kristalinfo.com/download/hkib-20000-40075.tar.gz)에서 받을 수 있다. 과거 뉴스 데이터를 다운로드해서, 어떤 카테코리인지 분류하는 학습을 진행해보자.

먼저 구글드라이브에 해당 파일을 업로드해서 진행했다. 물론 아래와 같은 코드로 colab docker에 업로드 할 수 있지만

```python
from google.colab import files
upload = files.upload()
```

속도가 너무 느리다. ㅠ.ㅠ 그래서 그냥 구글 드라이브에 올려서 진행했다.

```python
from google.colab import drive
drive.mount('/content/gdrive')
```

이렇게하면, 구글드라이브의 내용을 `/contnet/gdrive`에 마운트 할 수 있다. 마운트 된 gdrive는 파일 시스템에 접근하는 것처럼 손쉽게 접근할 수 있다. 오오 구글신 오오...

그리고 한글 형태소를 분석해야 하므로, 한글 형태소 분석을 지원하는 라이브러리를 깔았다.

```
!pip3 install konlpy
```

한글 형태소 라이브러리에 관한 글은 여기저기에 많으니 따로 설명하지 않겠다.

```python
import os
import re

from sklearn import datasets, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from konlpy.tag import Hannanum
from konlpy.tag import Kkma

import pandas as pd
import numpy as np
```

```python
target_dir = 'HKIB-20000'
cat_dirs = ['healths', 'economy', 'science', 'education', 'culture', 'society', 'industry', 'leisure', 'politics']
cat_prefixes = ['건강', '경제', '과학', '교육', '문화', '사회', '산업', '여가', '정치']

files = os.listdir(data_path+'/'+target_dir)
files
```

```
['hkib20000-cat03-file3.categories',
 'hkib20000-cat07-all.categories',
 'hkib20000-cat07-file3.categories',
 'hkib20000-cat03-file1.categories',
 'hkib20000-cat07-file4.categories',
 'hkib20000-cat07-file2.categories',
 'hkib20000-cat03-file4.categories',
 'hkib20000-cat03-file5.categories',
 'hkib20000-cat03-all.categories',
 'hkib20000-cat07-file5.categories',
 'hkib20000-cat03-file2.categories',
 'hkib20000-cat07-file1.categories',
 'HKIB-20000_003.txt',
 'HKIB-20000_002.txt',
 'HKIB-20000_001.txt',
 'HKIB-20000_005.txt',
 'HKIB-20000_004.txt']
```

```python
# 데이터 정리
files = os.listdir(data_path+'/'+target_dir)

# 5분할된 텍스트 파일을 각각 처리
for file in files:
    # 데이터가 담긴 파일만 처리
    if not file.endswith('.txt'):
        continue

    # 각 텍스트 파일을 처리
    with open(data_path + 'HKIB-20000/' + file) as currfile:
        doc_cnt = 0
        docs = []
        curr_doc = None

        # 기사 단위로 분할하여 리스트를 생성
        for curr_line in currfile:
            if curr_line.startswith('@DOCUMENT'):
                if curr_doc is not None:
                    docs.append(curr_doc)
                curr_doc = curr_line
                doc_cnt = doc_cnt + 1
                continue
            curr_doc = curr_doc + curr_line

        # 각 기사를 대주제 별로 분류하여 기사별 파일로 정리
        for doc in docs:
            doc_lines = doc.split('\n')
            doc_no = doc_lines[1][9:]

            # 주제 추출
            doc_cat03 = ''
            for line in doc_lines[:10]:
                if line.startswith("#CAT'03:"):
                    doc_cat03 = line[10:]
                    break

            # 추출한 주제 별로 디렉토리 정리
            for cat_prefix in cat_prefixes:
                if doc_cat03.startswith(cat_prefix):
                    dir_index = cat_prefixes.index(cat_prefix)
                    break

            # 문서 정보를 제거하고 기사 본문만 남기기
            filtered_lines = []
            for line in doc_lines:
                if not (line.startswith('#') or line.startswith('@')):
                    filtered_lines.append(line)

            # 주제별 디렉토리에 기사를 파일로 쓰기
            filename = 'hkib-' + doc_no + '.txt'
            filepath = data_path + 'HKIB-20000/' + cat_dirs[dir_index]

            if not os.path.exists(filepath):
                os.makedirs(filepath)
            f = open(filepath + '/' + filename, 'w')
            f.write('\n'.join(filtered_lines))
            f.close()
```

이렇게 해서, 각각의 주제별로 나눈 다음에 txt파일을 생성하였다. 고오급 스킬이 필요한 것은 아니고, 단순 노가다의 문제다.

```python
dirs = cat_dirs

x_ls = []
y_ls = []

tmp1 = []
tmp2 = ''

tokenizer = Kkma()

for i, d in enumerate(dirs):
  files = os.listdir(data_path+'HKIB-20000/'+d)

  for file in files:
    f = open(data_path+'HKIB-20000/'+d+'/'+file, 'r', encoding='UTF-8')
    raw = f.read()

    reg_raw = re.sub(r'[-\'@#:/◆▲0-9a-zA-Z<>!-"*\(\)]', '', raw)
    reg_raw = re.sub(r'[ ]+', ' ', reg_raw)
    reg_raw = reg_raw.replace('\n', ' ')

    tokens = tokenizer.nouns(reg_raw)

    for token in tokens:
      tmp1.append(token)

    tmp2 = ' '.join(tmp1)
    x_ls.append(tmp2)
    tmp1 = []

    y_ls.append(i)

    f.close()
```

내용에서 특수문자를 제거하고, `x_ls`에 설명변수, `y_ls`에 목적변수를 각각 넣었다. 처음에는 딱 두개카테코리만 0, 1 로 분석해서 진행했다. 여유가 되다면 (시간이 남고 컴퓨터도 빠르다면...) 모든 파일을 분석해보는 것도 재밌을 수 있다.

```python
x_array = np.array(x_ls)
y_array = np.array(y_ls)

cntvec = CountVectorizer()
x_cntvecs = cntvec.fit_transform(x_array)
x_cntarray = x_cntvecs.toarray()

pd.DataFrame(x_cntarray)
```

CounterVectyorizer를 통해서 단어 별로 쪼갰다. 그 결과 1001행 \* 33572열 크기의 표가 생성되었다. 프린트를 해보면

```python
for k, v in sorted(cntvec.vocabulary_.items(), key=lambda x: x[1]):
  print(k, v)
```

```
가가 0
가가치 1
가검물 2
가게 3
가격 4
가격등 5
가격명 6
가격문란 7
가격변동 8
가격산정 9
가격상승요인 10
....
```

이런식으로 추출해 놓은 단어가 나온다.

이제 본격적으로 분석해보자.

```python
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(x_tfidf_array, y_array, test_size=0.2)
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()

test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).float()

train = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train, batch_size=100, shuffle=True)
```

이제 신경망을 구성해보자.

입력층에는 33572개의 단어가 있었고, 중간노드수는 256개, 128객 그리고 출력층은 2개 (두 개의 카테고리만 분석)로 구성해두었다.

```python
# 신경망 구성
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(33572, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc5(x)
        return F.log_softmax(x)

# 인스턴스 생성
model = Net()
```

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.05)

for epoch in range(20):
  total_loss = 0

  for train_x, train_y in train_loader:
    train_x, train_y = Variable(train_x), Variable(train_y)
    optimizer.zero_grad()
    output = model(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()
    total_loss += loss.data.item()

  print(epoch + 1, total_loss)
```

```python
test_x, test_y = Variable(test_X), Variable(test_Y)
result = torch.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())
accuracy
```

```
0.9203980099502488
```

92%의 정확성으로 구별해 내었다.
