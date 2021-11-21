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

### 코드 스플릿

아마도 어떤 형식으로든 코드스플릿을 쓰고 있을 가능성이 높지만, 어떻게 동작되고 있는지 다시 한번 살펴볼 필요가 있다. 코드 스플릿 방법에 관계 없이, 코드 스플릿에 대해서 다음과 같은 질문에 대해 답할 수 있어야 한다.

https://developers.google.com/web/fundamentals/performance/optimizing-javascript/code-splitting/

1. 코드 [entry point](https://webpack.js.org/concepts/entry-points/) 간에 중복 코드를 제거 하고 있는지?
2. [dymaic import()]()를 사용하여 레이지 로딩을 하고 있는지?

중복 코드를 줄이는 것은 성능에 매우 필수적이므로 꼭 해야 한다. 레이지 로딩은 첫 페이지의 초기 자바스크립트 용량을 줄임으로써 성능을 향상 시킨다. 또한 [Bundle Buddy](https://github.com/samccone/bundle-buddy)와 같은 도구를 사용하면 문제가 있는지 확인할 수도 있다.

레이지 로딩을 어디서 부터 손봐야할지 살펴보는 것은 다소 어려울 수도 있다. 기존 프로젝트에서 레이지 로딩을 적용할 곳을 찾을 때에는, 먼저 클릭이나 키보드 이벤트 등 코드 베이스 전반에 걸쳐 사용자 인터랙션 포인트가 발생하는 곳을 찾는 것이 좋다. 사용자 상호작용으로 실행 되는 코드들은 `dynamic import`를 사용하기에 적합한 후보다.

데이터 사용량이 큰 문제가 아니라면 [rel=prefetch](https://www.w3.org/TR/resource-hints/#prefetch) 리소스 힌트를 사용하여 낮은 우선순위로 스크립트를 로드할 수도 있다. 설령 [이 리소스 힌트를 지원하지 않는 브라우저](https://caniuse.com/link-rel-prefetch)라 하더라도 어차피 마크업 상에서 무시되기 때문에 크게 신경쓰지 않아도 된다.

### 써드 파티 코드 분석

웹 애플리케이션을 구성할 때, 가능한 사이트의 의존적인 리소스는 자체적으로 호스팅하는 것이 좋다. 어떤 이유로든지 제3자로부터 리소스를 가져와야 할 경우, [번들러 구성에서 이를 externals로 표시](https://webpack.js.org/configuration/externals/)하는 것이 좋다.그렇지 않으면 웹 사이트 방문자가 로컬에 있는 코드와 똑같은 코드를 써드파티로 부터 다운로드 할 수도 있다.

예를 한가지 들어보자. 만약 사이트에서 public CDN에서 lodash를 불러온다고 가정해보자. 그리고 내 프로젝트 개발을 하기 위해 로컬에서 lodash를 설치했다. 그러나 lodash를 external로 표시하지 않을 경우, 프로덕션 번들링에 lodash가 또 들어가버리게 될 것이다.

써드파티 의존성을 자체적으로 호스팅해야할지 확인이 없다면 [dns-prefetch](https://css-tricks.com/prefetching-preloading-prebrowsing/#dns-prefetching), [preconnect](https://css-tricks.com/prefetching-preloading-prebrowsing/#preconnect)m [preload](https://www.smashingmagazine.com/2016/02/preload-what-is-it-good-for/)을 도입해보는 것을 검토해보자. 이렇게하면 [사이트가 인터랙션이 가능해지는 시간](https://developers.google.com/web/tools/lighthouse/audits/time-to-interactive)을 낮출수도 있고, 만약 사이트의 콘텐츠를 렌더링하는게 중요하다면 [Speed Index](https://developers.google.com/web/tools/lighthouse/audits/time-to-interactive)에도 좋은 영향을 미칠 수 있다.

### 오버헤드를 줄이기 위한 또다른 방법

자바스크립트 생태계는 마치 엄청나게 큰 시장과도 같고, 개발자로서 우리는 오픈소스가 제공하는 다양한 코드에 때로는 경외심을 느끼기도 한다. 프레임워크와 라이브러리를 활용해 애플리케이셔늘 확장하는데 들어가는 시간과 노력을 줄이고, 모든 작업을 신속하게 마무리할 수도 있따.

개인적으로는 프로젝트에서 프레임워크와 라이브러리의 사용을 최소화 하는 것을 선호하지만, 솔직히 이를 사용하는 것은 아주 큰 유혹으로 느껴지기도 한다.