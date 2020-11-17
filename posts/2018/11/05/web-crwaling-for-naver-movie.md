---
title: '초보를 위한 웹크롤링: 네이버 영화 댓글 크롤링하기'
date: 2018-11-06 04:44:01
published: true
tags:
  - python
description:
  "e 파이썬과 파이썬 라이브러리 (beatifulSoup)를 활용하여 네이버 영화 댓글 크롤링 해보기 ## 1.
  크롤링하려는 웹페이지의 구조를 살펴보기  인크레더블 평점 댓글 페이지를 먼저 살펴보겠습니다.
  [여기](https://movie.naver.com/movie/bi/mi/point.nhn?code=136990&onlyActualPoin\
  tYn=Y#po..."
category: python
slug: /2018/11/05/web-crwaling-for-naver-movie/
template: post
---

e
파이썬과 파이썬 라이브러리 (beatifulSoup)를 활용하여 네이버 영화 댓글 크롤링 해보기

## 1. 크롤링하려는 웹페이지의 구조를 살펴보기

인크레더블 평점 댓글 페이지를 먼저 살펴보겠습니다. [여기](https://movie.naver.com/movie/bi/mi/point.nhn?code=136990&onlyActualPointYn=Y#pointAfterTab)

![naver-crawling-1](../images/naver-crawling-1.png)

우리가 할일은 저 페이징 (1, 2, 3, 4...)을 전부다 조회하면서, 댓글을 가져오는 일입니다. 한번 페이지를 눌러보면서 웹페이지 주소의 변화가 일어나는지 살펴봅니다.

별다른 변화가 일어나지 않습니다. 웹페이지 주소에 변화는 없지만 어디선가 다른 방식으로 데이터를 가져오고 있다는 것을 의미합니다.

어디에서 데이터를 가져오는지 살펴보기 위해, 페이지 아무곳에서 오른쪽단추를 누르고 검사를 누릅니다. 그리고 네트워크 탭을 누릅니다. 그다음, 페이지를 새로고침하고, 다시한번 페이지 버튼 (1, 2, 3)을 눌러봅니다.

간단하게 네트워크 탭을 설명해보면, 웹페이지를 로딩하면서 발생하는 모든 네트워크 이벤트 (이미지, 정보 등)가 기록되는 곳입니다.

![naver-crawling-2](../images/naver-crawling-2.png)

위 이미지에서 주목해야 할 부분이 있습니다. 바로 페이지 번호에 따라서 변화가 있는 요소 인데요. 이미지 두번째 줄의 [주소](https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=136990&type=after&onlyActualPointYn=Y&isActualPointWriteExecute=false&isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page=4) URL을 보면, 마지막에 `page=4` 같은 형식으로 요청을 보내고 있다는 것을 알수 있습니다. 이링크를 들어가볼까요? 오른쪽 단추로 누르고, open in a new tab을 클릭합니다.

![naver-crawling-3](../images/naver-crawling-3.png)

댓글 영역을 html로 불러오고 있었네요. 우리는 이제 저 page=1에 적절한 페이지 번호만 입력하면, 모든 댓글을 기계적으로 가져올 수 있다는 것을 깨닫게 되었습니다.

## 2. requests, beautifulSoup 라이브러리 활용하기

댓글 구조가 어떻게 되어 있는지는 확인했습니다. 이제 우리가 할일은 파이썬으로 정보를 가져오는 일만 남았습니다. 이를 위해서 우리는 두가지 라이브러리를 사용하고자 합니다.

```python
import requests
from bs4 import BeautifulSoup
```

`requests`는 파이썬에서 http 요청을 보낼 때 쓸 라이브러리고, `BeautifulSoup`는 html에서 정보를 간단하게 빼오기 위해 사용할 라이브러리 입니다.

```python
test_url = "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=136990&type=after&page=1"
resp = requests.get(test_url)
html = BeautifulSoup(resp.content, 'html.parser')
html
```

```html
<!DOCTYPE html>

<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta content="IE=edge" http-equiv="X-UA-Compatible"/>
<title>네이버 영화</title>
<link href="https://ssl.pstatic.net/static/m/movie/icons/naver_movie_favicon.ico" rel="shortcut icon" type="image/x-icon"/>
<link href="/css/common.css?20181031144347" rel="stylesheet" type="text/css">
<link href="/css/movie_tablet.css?20181031144347" rel="stylesheet" type="text/css"/>
<link href="/css/movie_end.css?20181031144347" rel="stylesheet" type="text/css"/>
<script src="/js/deploy/movie.all.js?20181031144347" type="text/javascript"></script>
</link></head>
<body>
<!-- content -->
<input id="movieCode" name="movieCode" type="hidden" value="160487"/>
<input id="onlyActualPointYn" name="onlyActualPointYn" type="hidden" value="N"/>
<input id="order" name="order" type="hidden" value="sympathyScore"/>
<input id="page" name="page" type="hidden" value="622"/>
<input id="point" name="point" type="hidden" value="0"/>
<div class="ifr_area basic_ifr">
<div class="input_netizen ">
<form id="pointWriteArea">
<fieldset>
<legend><span class="blind">네티즌 평점 입력란</span></legend>
<!-- 모바일 기기로 접근 시 : 클래스 t_layer_view 추가 -->
<ul class="t_layer_view">
<li class="drag_star">
<div class="lft">
<div class="star_score">
<strong class="blind">평점선택</strong>
<!-- [D] st_off 영역에 마우스 오버시 : 클래스 st_over 추가
...
```

길어서 생략했지만, 암튼 가져왔습니다. 이제 저기에서 어디가 댓글 영역인지 찾아야 합니다.

![naver-crawling-4](../images/naver-crawling-4.png)

바로 이 영역이요. 여기는 이제 우리가 눈에 불을 켜고 찾아야할 곳입니다. 찾아보니, 한가지 사실을 알 수 있었습니다. `<div class="score_result>` 라는 곳 아래에, ul, 그리고 그 아래에 li가 우리가 찾던 댓글 영역이었습니다. 그리고 그 li가 10개 반복되어 있네요.

`<div class="score_result>` 이 구조에 대해 간단하게 설명드리면, `div`는 태그명입니다. `class="score_result"`는 `class`라는 속성으로 `score_result`값을 가지고 있는 거지요.

```python
score_result = html.find('div', {'class': 'score_result'})
lis = score_result.findAll('li')
lis[0]
```

결과

```html
<li>
  <div class="star_score">
    <span class="st_off"><span class="st_on" style="width:100.0%"></span></span
    ><em>10</em>
  </div>
  <div class="score_reple">
    <p>처음에 만두 같은게 나와서 상영관 잘못 들어온줄</p>
    <dl>
      <dt>
        <em>
          <a
            href="#"
            onclick="javascript:showPointListByNid(14315731, 'after');parent.clickcr(this, 'ara.uid', '', '', event); return false;"
            target="_top"
          >
            <span>크리스탈(a8d5****)</span>
          </a>
        </em>
        <em>2018.07.18 09:07</em>
      </dt>
      <dd>
        <a
          class="go_report2"
          href="#"
          onclick="parent.clickcr(this, 'ara.report', '', '', event); common.report('false','a8d5****', 'mwNBVSIMqNRoWDHCxeN1YFkIC/JtxnOSpwYTkFE2Qos=', '처음에 만두 같은게 나와서 상영관 잘못 들어온줄 ', '14315731', 'point_after', false);return false;"
          ><em>신고</em></a
        >
      </dd>
    </dl>
  </div>
  <div class="btn_area">
    <a
      class="_sympathyButton"
      href="#"
      onclick="parent.clickcr(this, 'ara.sym', '', '', event);"
      ><span class="blind">공감</span></a
    ><strong><span class="sympathy_14315731 count">2406</span></strong>
    <a
      class="_notSympathyButton"
      href="#"
      onclick="parent.clickcr(this, 'ara.opp', '', '', event);"
      ><span class="blind">비공감</span></a
    ><strong><span class="notSympathy_14315731 count v2">141</span></strong>
  </div>
</li>
```

사용법은 간단합니다. 굳이 설명이 필요 없을 정도로요. 이제 찾고 싶은 요소를 하나씩 검색해봅시다.

#### 댓글 내용

```python
review_text = lis[0].find('p').getText()
review_text
```

```python
'처음에 만두 같은게 나와서 상영관 잘못 들어온줄 '
```

#### 평점

```python
score = lis[0].find('em').getText()
score
```

```python
'1'
```

#### 댓글 좋아요, 싫어요

```python
like = lis[0].find('div', {'class': 'btn_area'}).findAll('span')[1].getText()
dislike = lis[0].find('div', {'class': 'btn_area'}).findAll('span')[3].getText()
like, dislike
```

```
('7', '12')
```

#### 닉네임

```python
nickname = lis[0].findAll('a')[0].find('span').getText()
nickname
```

```python
'크리스탈(a8d5****)'
```

#### 작성일

```python
from datetime import datetime
created_at = datetime.strptime(li.find('dt').findAll('em')[-1].getText(), "%Y.%m.%d %H:%M")
created_at
```

```python
datetime.datetime(2018, 7, 18, 9, 7)
```

필요한 정보를 다 가져온 것 같습니다. 이제 이것을 method로 만들어봅시다.

```python
def get_data(url):
    resp = requests.get(url)
    html = BeautifulSoup(resp.content, 'html.parser')
    score_result = html.find('div', {'class': 'score_result'})
    lis = score_result.findAll('li')
    for li in lis:
        nickname = li.findAll('a')[0].find('span').getText()
        created_at = datetime.strptime(li.find('dt').findAll('em')[-1].getText(), "%Y.%m.%d %H:%M")

        review_text = li.find('p').getText()
        score = li.find('em').getText()
        btn_likes = li.find('div', {'class': 'btn_area'}).findAll('span')
        like = btn_likes[1].getText()
        dislike = btn_likes[3].getText()

        watch_movie = li.find('span', {'class':'ico_viewer'})

        # 간단하게 프린트만 했습니다.
        print(nickname, review_text, score, like, dislike, created_at, watch_movie and True or False)
```

데이터를 가져오는 방법은, 보시면 아시겠지만 특별한 노하우가 없습니다. 이요소 저요소 찾아보면서, 다양한 방법을 시도하는 것이 좋습니다. 제가 쓴 코드가 결코 좋은 코드라고 말할 수는 없습니다만, 편하신 방법으로 찾으면 될 것 같습니다.

## 3. 전체 댓글 수 가져오기

한 가지 놓친게 있다면, 전체 댓글수를 가져오는 것입니다. 전체 댓글 수를 가져오고, 그 숫자 만큼 나누기 10을 하여 페이지 조회를 해야할 것입니다.

```python
result = html.find('div', {'class':'score_total'}).find('strong').findChildren('em')[1].getText()
int(result.replace(',', ''))
```

```python
9926
```

## 4. 완성

```python
test_url = 'https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=136990&type=after'
resp = requests.get(test_url)
html = BeautifulSoup(resp.content, 'html.parser')
result = html.find('div', {'class':'score_total'}).find('strong').findChildren('em')[1].getText()
total_count = int(result.replace(',', ''))

for i in range(1, int(total_count / 10) + 1):
    url = test_url + '&page=' + str(i)
    print('url: "' + url + '" is parsing....')
    get_data(url)
```

```bash
일산빵셔틀(rnra****) 평론가 임수연씨의 마블보다 재밌다는 평을보고 코웃음치고 보러갔는데 마블빠인 내가봐도 마블보다 재밌었다 히어로물은 언제까지나 마블의 독주일꺼라는 착각을 씻어내준 가족액션 히어로물 진짜 너무재밌다   10 488 32 2018-07-18 23:49:00 False
url: "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=136990&type=after&page=1&page=4" is parsing....
크리스탈(a8d5****) 처음에 만두 같은게 나와서 상영관 잘못 들어온줄   10 2406 141 2018-07-18 09:07:00 False
gksq**** 14년동안 기다린 보람이있다.인크레더블 3도 해주세요 ㅠㅠ 제발 ㅠㅠ   10 1539 66 2018-07-18 10:39:00 False
yski**** 잭잭이랑 에드나 케미 미쳤닼ㅋㅋㅋ   10 1309 36 2018-07-18 10:07:00 False
배센도(bbtj****) 어릴 때 1편을 보고 성인이 된 올해 2편을 봤다. 또 보고 싶다. 3편도 나오면 좋겠다.   10 1013 34 2018-07-18 12:28:00 False
space(tmd5****) 속편도 이리 완벽할 수 있구나..!   10 898 40 2018-07-18 09:27:00 False
황진이의두번째팬티(sion****) 중심히어로와 빌런이 여성이라는 점, 그 둘의 대화 내용, 아내에게 열등감을 느끼던 남편이 육아를 도맡고 아내의 바깥일을 내조하며 그녀를 진짜 히어로로 인정하는 과정이 인상 깊었다. 꿈이 많은 내게 선물같은 영화였다. 또 보러 가겠다.   10 1230 382 2018-07-18 20:40:00 False
불(catc****) 이런 게 최고의 애니매이션이 아니면 뭐란 말인가?? 미취학아동 때 1을 보고 대학생이 되어서 2를 보는 기분이란...ㅠㅠㅠ헬렌의 활약과 잭잭의 귀여움, 그리고 개인적으로는 에드나의 매력까지 올해 본 영화 중 최고!!   10 758 33 2018-07-18 12:39:00 False
쿠앤크(zhfl****) 잭잭 납치하러 갈 파티원 구합니다(1/10000)   8 639 105 2018-07-18 09:21:00 False
아머두어라두(dkqj****) 관람객내 어릴적 베스트 영화의 속편이 너무 잘만들어져서 울컥했습니다.   10 499 24 2018-07-18 18:05:00 True
일산빵셔틀(rnra****) 평론가 임수연씨의 마블보다 재밌다는 평을보고 코웃음치고 보러갔는데 마블빠인 내가봐도 마블보다 재밌었다 히어로물은 언제까지나 마블의 독주일꺼라는 착각을 씻어내준 가족액션 히어로물 진짜 너무재밌다   10 488 32 2018-07-18 23:49:00 False
url: "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=136990&type=after&page=1&page=5" is parsing....
크리스탈(a8d5****) 처음에 만두 같은게 나와서 상영관 잘못 들어온줄   10 2406 141 2018-07-18 09:07:00 False
gksq**** 14년동안 기다린 보람이있다.인크레더블 3도 해주세요 ㅠㅠ 제발 ㅠㅠ   10 1539 66 2018-07-18 10:39:00 False
yski**** 잭잭이랑 에드나 케미 미쳤닼ㅋㅋㅋ   10 1309 36 2018-07-18 10:07:00 False
배센도(bbtj****) 어릴 때 1편을 보고 성인이 된 올해 2편을 봤다. 또 보고 싶다. 3편도 나오면 좋겠다.   10 1013 34 2018-07-18 12:28:00 False
space(tmd5****) 속편도 이리 완벽할 수 있구나..!   10 898 40 2018-07-18 09:27:00 False
황진이의두번째팬티(sion****) 중심히어로와 빌런이 여성이라는 점, 그 둘의 대화 내용, 아내에게 열등감을 느끼던 남편이 육아를 도맡고 아내의 바깥일을 내조하며 그녀를 진짜 히어로로 인정하는 과정이 인상 깊었다. 꿈이 많은 내게 선물같은 영화였다. 또 보러 가겠다.   10 1230 382 2018-07-18 20:40:00 False
불(catc****) 이런 게 최고의 애니매이션이 아니면 뭐란 말인가?? 미취학아동 때 1을 보고 대학생이 되어서 2를 보는 기분이란...ㅠㅠㅠ헬렌의 활약과 잭잭의 귀여움, 그리고 개인적으로는 에드나의 매력까지 올해 본 영화 중 최고!!   10 758 33 2018-07-18 12:39:00 False
쿠앤크(zhfl****) 잭잭 납치하러 갈 파티원 구합니다(1/10000)   8 639 105 2018-07-18 09:21:00 False
아머두어라두(dkqj****) 관람객내 어릴적 베스트 영화의 속편이 너무 잘만들어져서 울컥했습니다.   10 499 24 2018-07-18 18:05:00 True
일산빵셔틀(rnra****) 평론가 임수연씨의 마블보다 재밌다는 평을보고 코웃음치고 보러갔는데 마블빠인 내가봐도 마블보다 재밌었다 히어로물은 언제까지나 마블의 독주일꺼라는 착각을 씻어내준 가족액션 히어로물 진짜 너무재밌다   10 488 32 2018-07-18 23:49:00 False
url: "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=136990&type=after&page=1&page=6" is parsing....
크리스탈(a8d5****) 처음에 만두 같은게 나와서 상영관 잘못 들어온줄   10 2406 141 2018-07-18 09:07:00 False
gksq**** 14년동안 기다린 보람이있다.인크레더블 3도 해주세요 ㅠㅠ 제발 ㅠㅠ   10 1539 66 2018-07-18 10:39:00 False
yski**** 잭잭이랑 에드나 케미 미쳤닼ㅋㅋㅋ   10 1309 36 2018-07-18 10:07:00 False
배센도(bbtj****) 어릴 때 1편을 보고 성인이 된 올해 2편을 봤다. 또 보고 싶다. 3편도 나오면 좋겠다.   10 1013 34 2018-07-18 12:28:00 False
space(tmd5****) 속편도 이리 완벽할 수 있구나..!   10 898 40 2018-07-18 09:27:00 False
황진이의두번째팬티(sion****) 중심히어로와 빌런이 여성이라는 점, 그 둘의 대화 내용, 아내에게 열등감을 느끼던 남편이 육아를 도맡고 아내의 바깥일을 내조하며 그녀를 진짜 히어로로 인정하는 과정이 인상 깊었다. 꿈이 많은 내게 선물같은 영화였다. 또 보러 가겠다.   10 1230 382 2018-07-18 20:40:00 False
불(catc****) 이런 게 최고의 애니매이션이 아니면 뭐란 말인가?? 미취학아동 때 1을 보고 대학생이 되어서 2를 보는 기분이란...ㅠㅠㅠ헬렌의 활약과 잭잭의 귀여움, 그리고 개인적으로는 에드나의 매력까지 올해 본 영화 중 최고!!   10 758 33 2018-07-18 12:39:00 False
쿠앤크(zhfl****) 잭잭 납치하러 갈 파티원 구합니다(1/10000)   8 639 105 2018-07-18 09:21:00 False
아머두어라두(dkqj****) 관람객내 어릴적 베스트 영화의 속편이 너무 잘만들어져서 울컥했습니다.   10 499 24 2018-07-18 18:05:00 True
일산빵셔틀(rnra****) 평론가 임수연씨의 마블보다 재밌다는 평을보고 코웃음치고 보러갔는데 마블빠인 내가봐도 마블보다 재밌었다 히어로물은 언제까지나 마블의 독주일꺼라는 착각을 씻어내준 가족액션 히어로물 진짜 너무재밌다   10 488 32 2018-07-18 23:49:00 False
url: "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?code=136990&type=after&page=1&page=7" is parsing....
크리스탈(a8d5****) 처음에 만두 같은게 나와서 상영관 잘못 들어온줄   10 2406 141 2018-07-18 09:07:00 False
gksq**** 14년동안 기다린 보람이있다.인크레더블 3도 해주세요 ㅠㅠ 제발 ㅠㅠ   10 1539 66 2018-07-18 10:39:00 False
yski**** 잭잭이랑 에드나 케미 미쳤닼ㅋㅋㅋ   10 1309 36 2018-07-18 10:07:00 False
배센도(bbtj****) 어릴 때 1편을 보고 성인이 된 올해 2편을 봤다. 또 보고 싶다. 3편도 나오면 좋겠다.   10 1013 34 2018-07-18 12:28:00 False
```

단순히 제 예제에서는 print만 했지만, 이를 데이터 베이스로 저장한다던지, csv로 저장한다던지 하는 방법이 있습니다.

csv작성을 위해서는 [여기](https://realpython.com/python-csv/)를, mysql같은 db에 저장하기 위해서는, 적절한 파이썬 라이브러리 (pymysql 등)를 찾으시면 되겠네요.
