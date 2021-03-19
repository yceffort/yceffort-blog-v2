---
title: '애플 단축어와 GCP로 내 건강정보 업로드하기'
tags:
  - apple
  - gcp
published: true
date: 2021-03-17 23:19:59
description: '간만에 했본 간단하고 재밌는 일'
---

어디 공부할 만 한 좋은 포스팅이 없나 찾던 중, 애플의 단축어 명령으로 내 폰에 있는 건강정보를 업로드 할 수 있다는 포스팅을 보았다. https://blog.maximeheckel.com/posts/build-personal-health-api-shortcuts-serverless 그래서 이걸 나도 해보면 어떨까 싶어서, 최근에 빠져있는 걷기+달리기에 관련되니 정보를 업로드 해보는 작업을 해보았다.

## 1. 단축어 설정

일단 아이폰에 있는 단축어 앱을 사용해서 내 건강정보를 업로드 할 수 있는 환경을 만들어야 한다.

![shortcut1](./images/shortcut1.jpeg)
![shortcut2](./images/shortcut2.jpeg)
![shortcut3](./images/shortcut3.jpeg)
![shortcut4](./images/shortcut4.jpeg)

> 추가: 자동화와 연결해서 자기전에 자동으로 업로드 하도록 했다. 그러나 아이폰이 잠겨있을 경우에는 업로드가 되지 않으므로, 아이폰 잠금을 해제해 둬야 한다. 🤪


당연히, 헤더 정보에 secret 키 등으로 접근을 하지 못하게 막아둬야 한다. (누가 여기에 업로드 할 일이 있을지는 모르겠지만,,)

## 2. Google Cloud Function 작업

```javascript
exports.health = functions.https.onRequest(async (req, res) => {
  // {'health': {'run': '28.88118937860876', 'timestamps': '2021-03-17T08:45:00+09:00', 'unit': 'km'}}
  const {
    health: { run, timestamps },
  } = req.body

  const { secret } = req.headers

  if (secret !== APPLE_HEALTH_SECRET) {
    return res.send('permission denied! 🤬')
  }

  const healthRef = db.collection('apple_health')

  const date = new Date(timestamps)
  const timeZoneFromDB = +9.0
  const tzDifference = timeZoneFromDB * 60 + date.getTimezoneOffset()
  const offsetDate = new Date(date.getTime() + tzDifference * 60 * 1000)

  const key = `${offsetDate.getFullYear()}${(offsetDate.getMonth() + 1)
    .toString()
    .padStart(2, 0)}${offsetDate.getDate()}`

  const data = (await healthRef.doc('daily').get()).data()

  healthRef.doc('daily').set({
    ...data,
    [key]: run,
  })

  // 응답을 단순 text로 쏘면 아이폰에서 알림을 내보낼 수 있다.
  res.send('성공')
})
```

## 3. 결과

![result](./images/shortcut-result.gif)

![result](./images/shortcut-result.png)

지금은 단축어 앱을 클릭해야만 업로드 되는 구조지만, 이 작업을 그대로 단축어 > 자동화에 옮기면 자동으로 처리할 수 있다. 애플 워치에서 운동이 끝난 뒤라든가, 수면 준비 시작 시간이 되면 업로드 한다든가,,

처음에 단축어 앱이 생겼을 때, 그냥 단순히 QR 체크인 꺼내는 용도로만 썼었는데 이렇게 강력한 기능이 있는지 몰랐다. 이렇게 반성하면서 🤪 이걸로 해볼만한 더 재밌는일이 있을지 고민해봐야겠다. 심박수나 달리기 정보를 strava처럼 그래프로 보여주는 것들을 하면 재밌을 것 같다.