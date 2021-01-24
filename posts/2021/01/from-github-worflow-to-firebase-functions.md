---
title: 'Github action cron이 제시간에 실행되지 않는 문제'
tags:
  - javascript
  - firebase
  - github
published: true
date: 2021-01-24 21:16:38
description: 'Github에 실망했습니다 ㅠㅠ'
---

## github workflow로 cron 사용

github action이 생긴 이후로 workflow를 활용해서 cron job을 처리하고 있었다.

```yaml
name: cron

on:
  schedule:
    - cron: '0 5 * * 1-5'

jobs:
  cron:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - uses: actions/setup-node@v1
        with:
          node-version: '12'
          check-latest: true

      - name: CI
        run: |
          npm ci
      - name: Run Cron
        run: |
          npm run job
```

그러나 언제선가 부터 작업 시작 시간이 굉장히 안맞기 시작했다. 공짜로 사용하는 것이다 보니, 그러려니 하고 넘어갔었는데 예상 시간 보다 40~50분 넘어가서 작업이 실행되었다. 심지어 09시 (utc 기준 00시)에 실행되는 작업은 2시간 뒤에 실행되곤 했다.

![workflow-cron](./images/workflow-cron.png)

> 00시에 걸어둔 작업이었는데 실제로는 02시 30분에 실행되었다. github도 이런 문제가 있을수 있다는 걸 아는건지(?) 실제 실행시간이 표시되고 있지는 않았다.

그러나 이런 문제는 나만 겪는 것은 아닌듯 했다.

- https://stackoverflow.com/questions/65132563/why-is-github-actions-workflow-scheduled-with-cron-not-triggering-at-the-right-t
- https://github.community/t/github-actions-on-schedule-executed-in-delay/152972

github이 뭔가 조치를 취해주기를 기다렸지만(?) 포스팅 작성 시간 현재 (1월 24일) 까지 문제는 계속되었다. 그리고 별개로 로그를 살펴보니 이러한 증상은 2021년 들어서 더 심해진 것 같았다.

그래서 firebase의 cloud function으로 갈아탔다. 단순히 http요청으로 함수를 실행하게 도와주는 것 뿐만 아니라 cron으로 정해진 작업을 실행할 수도 있었다.

```javascript
exports.cronJob = functions.pubsub
  .schedule('0 14 * * 1-5')
  .timeZone('Asia/Seoul')
  .onRun((_) => {
    job()
  })
```

github과 다르게 시간대도 설정할 수 있어서 좋았다 (github에서는 몇 번 이점을 까먹고 새벽에 스케쥴이 돌아가기도 했다.) 한가지 배운 것은, `firebase init`으로 function을 초기화 할 경우 기본 디렉토리가 `./functions`로 설정되는데, 이를 다음과 같이 설정하면 root로 바꿀 수 있다.

**firebase.json**

```json
{
  "functions": {
    "source": ".",
    "runtime": "nodejs12"
  }
}
```

![functions](./images/functions-cron.png)

당연히 작업도 원하는 시간대에 정확하게 실행되었다.

언제부터인가 AWS를 안쓰고, GCP와 firebase를 위주로 계속 쓰고 있는데 나쁘지 않은 것 같다. 적어도 인터페이스는 GCP와 firebase가 더 나은 것 같다. 내가 구글빠라서 그런걸 수도 있고.