---
title: '2020년 사이드 프로젝트 회고'
tags:
  - diary
  - TIL
  - retrospective
published: true
date: 2020-12-15 23:24:36
description: '이거 좀 재밌네여'
---

2020년에도 많은 사이드 프로젝트를 진행했다. 대다수의 사이드 프로젝트는 웹 애플리케이션으로 제작했고, 서버와 프론트 데이터베이스를 구축했어야 했고, 이것들을 어떻게 구축했는지, 그리고 어떤 것을 느꼈는지 간단하게 요약하고자 한다.

## 개발환경

- nextjs
- react
- typescript (급할 땐 javascript도 함)
- firestore
- cloud functions
- github
- vercel
- styled component

### nextjs

nextjs는 Server Side React 환경을 구축하는데 있어서 필수적인 라이브러리로 자리잡은 것 같다. 비단 SSR 때문 만이 아니더라도, 라우팅을 하는데 있어서도 편하게 적용할 수 있었다. `/pages` 폴더 하단에 디렉토리 구조로 만 설정해두면, 알아서 라우팅을 구성하기 때문에 굉장히 좋았다.

다만 이 방법의 한계는 당연하게도 라우팅이 복잡해 질수록 디렉토리 관리가 어려워 진다는 것이다. 가령 path param으로 `/id`를 가져오기 위해서 `[id].js`를 만들어 두면, 이 파일을 찾는 것도 피곤하고 관리하기도 어려웠다.

![next-path](./images/next-path.png)

> `[id].js` 라는 파일이 많아지면,, 이제 감당할 수 없는 미래...

그래서 `api`에서는 귀찮아지니까 그냥 다 query param으로 가져오는 방식으로 변경해버렸다.

혹은 불가피 하게 라우팅이 복잡해지면, `koa`를 별도 서버를 둬서 페이지 분기를 여기에서 처리하도록 만들었다. 이는 이전 회사에서 즐겨 쓰던 방식으로, 굳이 복잡하게 디렉토리 구조를 만들지 않아도 되서 유용했다. (감사합니다.)

하지만 대부분의 사이드 프로젝트가 라우팅이 복잡했던 것은 아니므로, 이 방법은 많이 쓰지 않았다.

### react

jquery 부터 웹 개발을 해오면서, 이제는 react가 어느정도 front 개발의 표준으로 자리잡은 것 같다. 물론 이러한 react에 대한 아쉬움 내지는 성토의 글도 있었지만, 많은 수의 프로젝트들이 react로 쓰여지면서 커뮤니티가 커지고, 그에 따른 많은 편의성을 얻은 것도 사실이다. 나 또한 모든 사이드 프로젝트에서 웹 애플리케이션이 필요할 때 마다 리액트를 선택했고, 이는 탁월한 선택이었다. 이에 관해 흥미로운 글이 있었는데 한번 읽어보면 좋을 것 같다.

https://jake.nyc/words/no-one-ever-got-fired-for-choosing-react/

누구도, 리액트를 고른다고 해서 해고되는 것은 아니다.

> I have heard from many developers who have told me that they accepted the argument that vanilla JavaScript or microlibraries would let them write leaner, meaner, faster apps. After a year or two, however, what they found themselves with was a slower, bigger, less documented and unmaintained in-house framework with no community. As apps grow, you tend to need the abstractions that a framework offers. Either you or the community write the code.

### typescript

사실 타입스크립트가 협업을 하는 경우에 한해서 좋다고 생각해서, 초창기 사이드 프로젝트를 진행할 때에는 자바스크립트로 진행을 했었다. 물론 어느 정도 타입과 이런저런 제약에 벗어나서 정말 편리했고, 일정수준 프로젝트가 커지지 않는다면 생산성도 향상됐다.

문제는 내 스스로가 만든 데이터 구조가 복잡해지면서 내 머리로 조차 기억하지 못하고 급기야 버그를 만들기 시작했다는 것이었다. (오오 33살,,,) 그렇게 타입과 갖가지 버그로 인해 혼란을 겪고 나니, 이럴거면 초창기에 타입을 잘 선언해두고 프로젝트를 꾸몄다면 이런 혼선을 줄일 수 있었을 거라는 아쉬움이 남았다.

그래서 앞으로는 그냥 혼자 하든, 둘이 하든 지간에 타입스크립트로 진행하고자 마음먹었다. 잘 선언해둔 타입으로 프로젝트를 꾸미게 되면, 이후에 데이터 구조를 가져다 쓸 때도, 리팩토링을 할 때도 많은 도움을 얻을 것 같다.

### firestore

부끄럽고 미련하게도 초창기에는 SQL을 node와 연결해서 썼었다. 그러나 스키마 구성, 트랜잭션 관리, 테이블 관리 등을 겪고 나니 배보다 배꼽이 더 큰 느낌이었고, 빠르게 가져다 쓸 수 있는 NoSQL을 찾던 와중에 Firebase의 firestore 을 쓰기 시작했다. SQL 데이터 베이스를 구성하는데 걸린 시간을 제로에 가깝게 줄일 수 있었고, 데이터 관리 또한 JSON 형태로 아주 편하게 가져다 쓸 수 있었다.

다만 join 이나 or, in 과 같은 식의 복합쿼리는 안되기 때문에 어느 정도의 불편함을 감수해야 했다.

https://firebase.google.com/docs/firestore/query-data/queries?hl=ko

> Cloud Firestore는 다음 유형의 쿼리를 지원하지 않습니다.

> - 이전 섹션에서 설명한 것과 같이 여러 필드에 범위 필터가 있는 쿼리
> - 논리적 OR 쿼리: 이 경우 각 OR 조건에 해당하는 별도의 쿼리를 만들고 앱에서 쿼리 결과를 병합해야 합니다.
> - != 절을 사용하는 쿼리: 이 경우 쿼리를 초과 및 미만 쿼리로 분할해야 합니다. 예를 들어 where("age", "!=", "30") 쿼리 절은 지원되지 않지만 where("age", "<", "30") 절이 있는 쿼리 하나와 where("age", ">", 30) 절이 있는 쿼리 하나를 결합하면 동일한 결과 집합을 얻을 수 있습니다.

예컨데 장바구니와 같은 데이터 베이스를 꾸밀일이 있었는데, 이를 조회하기 위해서 불가피하게 2n 회로 데이터 조회를 요청할 수 밖에 없었다.

![firestore](./images/firestore-quota.png)

그러다 보니 개발 단계에서 마저도 읽기 횟수가 눈에 띄게 튀기 시작했다. 아직 눈에 띄게 document가 있지 않은 상태라 무료 수준에서 커버를 치고 있지만, document가 늘어 나면 이제 모든 쿼리를 쓰는데 신중에 신중을 기하게 될 것이다. (한번 읽기 마다 `0.06$`)

https://firebase.google.com/docs/firestore/pricing

없는 살림에 GCP를 이것저것 쓰고 있기 때문에 굉장히 덜덜덜 하고 있는데,, 부디 나에게 예산 알림이 오는날이 없었으면 한다. 그리고 그날이 온다면, mock을 잘만들어서, 쿼리를 잘만들어서 대처해보는 걸로...

### Cloud functions

cloud function은 slack, survey monkey, jandi 등의 웹 훅 용으로 유용하게 썼다. 함수 하나만 클라우드에 올려서 서버 없이 쓸 수 있기 때문에 굉장히 유용했다. firestore에 비해서 가격도 비교적 저렴해서 (그리고 애초에 많이 호출되는 함수도 아니었지만) 안심하고 여러 함수를 서버없이 잘 사용했다. 예전에 이거 하자고 인스턴스 따고 보안규칙 따고 생쇼를 했던 과거에 비하면, 너무나도 편리하다.

![cloud functions](./images/cloud-functions.png)

[최근에는 블로그 썸네일 생성기로도 사용하기도 했다.](https://yceffort.kr/2020/12/generate-serverless-thumbnail) 감사합니다. 다음엔 꼭 픽셀 폰 싸서 보답하겠습니다.

### vercel

heroku의 시대가 가고, 이제는 vercel 의 시대가 온 것 같다. 웹 애플리케이션을 프로덕션에 올리는 용도로는 망설임없이 vercel을 썼다. heroku에 비해 기능은 rich 하지 않지만 필요한 기능만 들어있고, (heroku는 너무 이것저것 많은 기분이다) 배포도 용이하며, 무엇보다 github 에 integration 할 수 있다는 점이 매력적이다.

https://github.com/yceffort/yceffort-blog-v2/pull/222#issuecomment-740369078

또한 가격도 heroku 보다 저렴했다. 웹 애플리케이션 당 가격이 아니고 한 member 당 가격을 책정하기 때문에 20달러에 프로 계정으로 나 혼자 잘 먹고 잘 쓰고 있다. (프로 계정은 최대 10명의 멤버, 50 도메인, 1000gb의 bandwidth 제한이 있다.)

![vercel1](./images/vercel1.png)

![vercel2](./images/vercel2.png)

### github

마이크로소프트에 인수된 뒤로 나날이 발전하고 있는 github 도 빼 놓을 수 없다. github action 덕분에 CI, CD 등도 편리하게 수행할 수 있었고, 각종 bot을 통해서 다양한 작업을 할 수 있었다. 또한 cron job 도 편리하게 이용할 수 없었다. 기존에는 클라우드에 인스턴스 띄워서 크론 설정하고, https://healthchecks.io/ 로 healthcheck 까지 확인했다면, 이제는 github action 하나면 충분하다.

또한 유용하게 썼던 것 중 하나는 github code spaces 다. 개발 환경이 불안정했던 외부에서도 인터넷만 연결되어 있다면, codespaces로 편리하게 개발을 할 수가 있었다.

pro 계정을 활용하면서 private repository도 자유롭게 꾸민 것이 개인적으로 많은 도움도 되었다. 내 작업물 대다수가 private 으로 되어 있어서 많은 코드를 세상과 공유(?) 하지 못했지만, 그래도 덕분에 각종 예민한 private key를 편하게 관리할 수 있었다.

올해에는 dark 모드 까지 지원되었는데, 앞으로도 github에서 더욱 rich한 기능을 지원해 줬으면 좋겠다.

### styled-component 그리고 css

css-in-js 에 대해서는 이제는 하나의 흐름이라고 개인적으로 생각했다.

- Global namespace
- Dependencies
- Dead Code Elimination
- Minification
- Sharing Constants
- Non-deterministic Resolution

하지만 지나가다가 본 글이 있는데, 보면서 이런저런 생각을 하게 됐다.

- https://blueshw.github.io/2020/09/27/css-in-js-vs-css-modules/
- https://blueshw.github.io/2020/09/14/why-css-in-css/

각자의 장단이 있는 것 같고, 글쓴이의 의도에도 적극 공감한다.

그러나 이와 별개로 문제는 내가 css에 대한 이해와 디자인 감각이 현저히 떨어진다는 것이다.🤪 내년에는 정녕 css를 공부해야 하는 것일까. 이쁘게는 못하더라도 (이미 디자인에 감각이 없다는 것을 블로그가 증명하고 있다), 기본적인 css에 대한 이해도 아직은 조금 부족하고, 더 공부해야 겠다는 생각이 든다.

## 마치며

일하는 회사에서 이것저것 새로운 최신의 기술, 내가 배웠던 다른 코드들을 모두 적용해 볼 수 있으면 정말 좋다. 돈도 받고, 공부도 하고, 개인적으로 성장도 할 수 있고 회사 연말 평가에 반영할 수도 있다.

그러나 아쉽게도 회사 업무에서 펼칠 수 있는 상상의 나래는 한정적이다. 기존에 쓰고 있는 기술은 한정적이고 뒤쳐져 있으며, 설득해야 하는 사람은 많고 새로운 것에 대한 반론 또한 존재한다. 이미 잘되고 있는 레거시가 있는데 굳이 왜? 그렇다고 내가 모든 책임을 다 안고 시도하기엔, 일단 나부터가 쫄린다. 로컬에선 잘됐는데? 알파에선 잘됐는데? 라는 변명으로 막을 것인가? 또한 개발자만 있는게 아니다. 기획자 분들도, 테스터 분들도, 그리고 높으신 양반들도 있다. 개발자 입장에서는 아무런 변화가 없는 코드라고 자신할 수 있지만, 그들의 눈에서는 테스트해야할 골칫거리가 하나 더 생기는 것 뿐일지도 모른다.

![sorry](./images/sorry.png)

> 미안합니다,,,

이런 저런 이유로 미루어, 사이드프로젝트는 개발자로서의 성장, 그리고 킬링타임을 위해 필수인 것 같다. 비록 아무도 안보는 블로그지만, 그게 오히려 좋다. CSR을 SSR 바꾸고, 새로운 기술을 적용해보고 공부해 볼 수도 있고 트렌드에 뒤쳐지지 않을 수도 있다. 버그로 프로젝트가 망가져도 트래픽은 잃을 지언정 아무도 뭐라고 하지 않으며, 부담없이 새로운일을 해볼 수 있다. 개발자가 된 이후로 몇년만에 처음으로, 각종 컨퍼런스의 타이틀을 보고 '다 아는 기술이구만' 이라며 고개를 끄덕여 봤다.

새로운 기술, 코드가 있는데 회사에서 시도할 수가 없다? 회사에서 잘 만들어 놓은 사내 서비스 보다 AWS, GCP를 써보고 싶다? 체크카드를 꺼내서 payment를 등록하고 지금 당장 나만의 프로젝트를 만들어서 공부해보자. 기술은 많고, 내가 써본 것은 손에 꼽는다. 이대로 회사일만 하기엔, 밖에는 재밌는게 너무 많다.
