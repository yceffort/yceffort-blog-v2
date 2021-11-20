---
title: '자바스크립트 코드가 가져야할 책임감 (2)'
tags:
  - javascript
  - web
published: true
date: 2021-11-20 20:12:01
description: ''
---

## Introduction

오늘날 웹 환경은 우리가 생각하는 것 보다 더 빠르게 성장할 것을 요구하고 있다. 이러한 압박으로 부터 자유롭기 위해, 우리는 가능한 생산적인 수단을 사용해야 한다. 이 말을 다르게 풀어보자면, 웹 애플리케이션을 구축할 때 오버헤드를 만들거나, 성능과 접근성을 저해할 수 있는 패턴을 반복적으로 사용할 가능성도 거친다.

웹 개발은 오늘날 많은 뉴비 개발자들이 뛰어들고 있는 가장 '쉽게 보이는' 개발이지만, 사실 그렇게 쉬운 영역은 아니다. (물론 진입장벽을 치기 위해서 하는 말은 아니다) 다들 쉽게 웹 개발을 시작하지만, 다들 하다보면 첫 개발부터 무언가 완벽하지 않다는 것을 깨닫게 된다. 물론 처음부터 완벽할 필요는 없다. 우리는 그 이후에 개선을 해나갈 수 있으며, 여기서 하고자 하는 말은 그 '개선'에 관한 것이다. 완벽은 아직 멀었다.

## 최적화 목록 점검하기

### 트리쉐이킹

먼저 자신이 속한 웹 개발 환경이 트리쉐이킹을 효과적으로 하고 있는지 확인해봐야 한다. 트리쉐이킹에 관한 글은 많다. 트리쉐이킹에 한번더 요약하자면, 사용되지 않는 코드를 프로덕션 번들에서 제거하는 과정이다.

- https://yceffort.kr/2021/08/javascript-tree-shaking
- https://yceffort.kr/2020/07/how-commonjs-is-making-your-bundles-larger
- https://ui.toast.com/weekly-pick/ko_20180716
- https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking

트리쉐이킹은 webpack, rollup, parcel과 같은 도구를 사용하면 즉시 해결할 수 있다. (번들러가 아닌 태스트러너인 grunt나 gulp는 도움이 되지 않는다.) 트리쉐이킹을 효과적으로 하기 위해서는, 아래 지침을 따라야 한다.

1. 애플리케이션 로직과 프로젝트에 설치하는 패키지를 모두 ES6로 작성하거나 활용해야 한다. CommonJS를 트리쉐이킹하는 것은 현실적으로 불가능하다.
2. 번들러가 빌드시에 ES6 모듈을 다른 모듈 형식으로 변환해서는 안된다. babel에서 이러한 상황이 발생하는 경우, es6코드가 commonjs로 변환하지 않도록 [@babel/preset-env 설정](https://babeljs.io/docs/en/babel-preset-env)을 반드시 [modules: false](https://babeljs.io/docs/en/babel-preset-env#modules) 로 해야 한다.

트리쉐이킹의 효과는 애플리케이션 개발 환경마다 조금씩 차이가 있을 수 있다. 또한 import하는 module이 [side effect](https://en.wikipedia.org/wiki/Side_effect_(computer_science))를 도입하느냐에 따라 달라지기도 하는데, 이는 사용하지 않는 `exports`를 제거하는 번들러에 영향을 미칠 수 있다.


