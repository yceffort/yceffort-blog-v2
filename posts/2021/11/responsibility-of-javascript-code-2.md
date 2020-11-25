---
title: '자바스크립트 코드가 가져야할 책임감 (2)'
tags:
  - javascript
  - web
published: true
date: 2021-11-20 20:12:01
description: '알지만 왠지 선뜻 내키지 않는 최적화, 이유가 무엇일까 🤔'
---

## Introduction

오늘날 웹 환경은 우리가 생각하는 것 보다 더 빠르게 성장할 것을 요구하고 있다. 이러한 압박으로 부터 자유롭기 위해, 우리는 가능한 생산적인 수단을 사용해야 한다. 이 말을 다르게 풀어보자면, 웹 애플리케이션을 구축할 때 오버헤드를 만들거나, 성능과 접근성을 저해할 수 있는 패턴을 반복적으로 사용할 가능성도 거친다.

웹 개발은 오늘날 많은 뉴비 개발자들이 뛰어들고 있는 가장 '쉬워 보이는' 개발이지만, 사실 그렇게 쉬운 영역은 아니다. (물론 진입장벽을 치기 위해서 하는 말은 아니다) 다들 쉽게 웹 개발을 시작하지만, 다들 하다보면 첫 개발부터 무언가 완벽하지 않다는 것을 깨닫게 된다. 물론 처음부터 완벽할 필요는 없다. 우리는 그 이후에 개선을 해나갈 수 있으며, 여기서 하고자 하는 말은 그 '개선'에 관한 것이다. 완벽은 아직 멀었다.

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

트리쉐이킹의 효과는 애플리케이션 개발 환경마다 조금씩 차이가 있을 수 있다. 또한 import하는 module이 [side effect](<https://en.wikipedia.org/wiki/Side_effect_(computer_science)>)를 도입하느냐에 따라 달라지기도 하는데, 이는 사용하지 않는 `exports`를 제거하는 번들러에 영향을 미칠 수 있다.

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

써드파티 의존성을 자체적으로 호스팅해야할지 확인이 없다면 [dns-prefetch](https://css-tricks.com/prefetching-preloading-prebrowsing/#dns-prefetching), [preconnect](https://css-tricks.com/prefetching-preloading-prebrowsing/#preconnect) [preload](https://www.smashingmagazine.com/2016/02/preload-what-is-it-good-for/)을 도입해보는 것을 검토해보자. 이렇게하면 [사이트가 인터랙션이 가능해지는 시간](https://developers.google.com/web/tools/lighthouse/audits/time-to-interactive)을 낮출수도 있고, 만약 사이트의 콘텐츠를 렌더링하는게 중요하다면 [Speed Index](https://developers.google.com/web/tools/lighthouse/audits/time-to-interactive)에도 좋은 영향을 미칠 수 있다.

### 오버헤드를 줄이기 위한 또다른 방법

자바스크립트 생태계는 마치 엄청나게 큰 시장과도 같고, 개발자로서 우리는 오픈소스가 제공하는 다양한 코드에 때로는 경외심을 느끼기도 한다. 프레임워크와 라이브러리를 활용해 애플리케이셔늘 확장하는데 들어가는 시간과 노력을 줄이고, 모든 작업을 신속하게 마무리할 수도 있다.

개인적으로는 프로젝트에서 프레임워크와 라이브러리의 사용을 최소화 하는 것을 선호하지만, 솔직히 이를 사용하는 것은 아주 큰 유혹으로 느껴지기도 한다. 하지만 우리는 패키지를 설치함에 있어 항상 비판적인 자세를 유지할 필요가 있다.

리액트는 아주 정말로 유명하지만 서도, [Preact](https://preactjs.com/)는 리액트보다 더 작고, 대부분의 API를 공유하고 있으며, 리액트 애드온 등으로 호환성도 유지할 수 있다. [Luxon](https://moment.github.io/luxon/#/)과 [date-fns](https://date-fns.org/)는 [moment.js](https://momentjs.com/)의 효과적인 대안이다.

- https://yceffort.kr/2020/12/why-moment-has-been-deprecated

lodash와 같은 라이브러리는 정말로 유용한 많은 메소드를 제공하지만, 사실 이는 ES6문법을 활용하면 쉽게 대체할 수 있다.

- https://github.com/you-dont-need/You-Dont-Need-Lodash-Underscore#_chunk

선호하는 도구가 무엇이든 간에, 우리가 생각해봐야 할 것은 동일하다. 더 작은 대안이 있는가? 혹은 자체적으로 구현이 가능한가?

## 브라우저별로 다른 스크립트 제공하기

요즘 대부분의 애플리케이션은 [ES6를 지원하지 않는 브라우저](https://caniuse.com/es6)에서 사용할 수 있는 코드로 변환하기 위해 babel을 사용하고 있을 가능성이 높다. 반대로 생각해보자. es6를 지원하는 브라우저가 더 많은데, 여전히 es6를 지원하지 않는 브라우저를 위해서 트랜스파일링이 된 번들링을 계속해서 제공해야 할까? 두개의 다른 빌드를 제공하면 되지 않을까?

1. 이전 브라우저에서 작동하는데 필요한 모든 도구, 폴리필을 포함하고 있다. 대부분의 애플리케이션이 현재 이런 상태일 것이다.
2. 모던 브라우저를 타겟으로 한 또다른 번들링을 만들어, 폴리필, 트랜스파일링 등을 모두 제거한다. 이 번들은 대부분의 애플리케이션이 제공하고 있지 않다.

이를 달성하기 위해서는 어떻게 해야할까?

[가장 단순한 패턴](https://v8.dev/features/modules#browser)은 바로 이것이다.

```html
<!-- Modern browsers load this file: -->
/js/app.mjs
<!-- Legacy browsers load this file: -->
/js/app.js
```

그러나 이 패턴을 사용하면, IE11, Edge 15 ~ 18 에서는 두 번들링을 모두 다운로드 한다는 문제가 있다.

https://gist.github.com/jakub-g/5fc11af85a061ca29cc84892f1059fec

```javascript
var scriptEl = document.createElement('script')

if ('noModule' in scriptEl) {
  // 모던 스크립트
  scriptEl.src = '/js/app.mjs'
  scriptEl.type = 'module'
} else {
  // 레거시 스크립트
  scriptEl.src = '/js/app.js'
  scriptEl.defer = true // 순서가 중요하다면 defer를 false로
}

// Inject!
document.body.appendChild(scriptEl)
```

https://caniuse.com/mdn-html_elements_script_nomodule

## 가능한 트랜스파일은 적게!

> transpile less!

https://twitter.com/_developit/status/1110229993999777793

바벨을 그만 쓰자는 이야기는 아니다. 바벨은 절대로 없어서는 안된다. 하지만, 바벨은 내가 모르는 사이에 더 많은 것들을 하므로 이를 자세히 알아보는 것이 좋다. 이러한 작은 습관은 바벨이 만드는 코드에 긍정적인 영향을 미칠 수 있다.

```javascript
function logger(message, level = 'log') {
  console[level](message)
}
```

여기서 주의해야할 것은 기본값이 `log`인 함수다. 이 함수를 트랜스파일링 하면 어떻게 될까?

```javascript
'use strict'

function logger(message) {
  var level =
    arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 'log'
  console[level](message)
}
```

> https://babeljs.io/repl#?browsers=%3E%200.25%25%2C%20not%20dead&build=&builtIns=false&corejs=3.6&spec=false&loose=false&code_lz=GYVwdgxgLglg9mABAGzgczQUwE4AoC2mAzkQIZYA0KmAbpsogLyIBEqaLAlIgN4BQiRBARE4yTAG1xdZAF0CxMlk4BuPgF8gA&debug=false&forceAllTransforms=false&shippedProposals=false&circleciRepo=&evaluate=false&fileSize=false&timeTravel=false&sourceType=module&lineWrap=true&presets=env%2Creact%2Cstage-2&prettier=false&targets=&version=7.16.4&externalPlugins=&assumptions=%7B%7D

분명 편리한 기본값 지정을 위해서 저렇게 코드를 썼건만, 몇바이트였던 코드가 바벨을 거치면서 프로덕션 코드에서는 훨씬 더 커졌다.

```javascript
function logger(...args) {
  const [level, message] = args

  console[level](message)
}
```

```javascript
'use strict'

function logger() {
  for (
    var _len = arguments.length, args = new Array(_len), _key = 0;
    _key < _len;
    _key++
  ) {
    args[_key] = arguments[_key]
  }

  var level = args[0],
    message = args[1]
  console[level](message)
}
```

> https://babeljs.io/repl#?browsers=%3E%200.25%25%2C%20not%20dead&build=&builtIns=false&corejs=3.6&spec=false&loose=false&code_lz=GYVwdgxgLglg9mABAGzgczQUwE4AoB0hAhtmgM4CUiA3gFCKIQJlSIDaymAbpsgDSIAtpjJkiWALqIAvIhLkA3LXqNmcTh268JuYaPGYKSgL5A&debug=false&forceAllTransforms=false&shippedProposals=false&circleciRepo=&evaluate=false&fileSize=false&timeTravel=false&sourceType=module&lineWrap=true&presets=env%2Creact%2Cstage-2&prettier=false&targets=&version=7.16.4&externalPlugins=&assumptions=%7B%7D

`...args`는 분명 편리하지만, `babel`은 함수의 인수가 몇개가 올지 추론할 수 없기 때문에 위와 같이 트랜스파일링 해버렸다.

위와 같은 상황을 방지하기 위해서는, 아래와 같이 `||`을 사용하는 것이 좋다.

```javascript
function logger(message, level) {
  console[level || 'log'](message)
}
```

결과가 같다.

```javascript
'use strict'

function logger(message, level) {
  console[level || 'log'](message)
}
```

물론 이처럼 주의해야 할 것이 기본 파라미터만은 아니다. 화살표 함수나 전개 연산자들도 트랜스파일 하면 꽤나 복잡해진다.

이러한 기능을 모두 사용하지 않으려면, 다음과 같은 방법으로 영향도를 줄일수도 있다.

1. 라이브러리 작성자라면, [@babel/plugin-transform-runtime](https://babeljs.io/docs/en/babel-plugin-transform-runtime)와 함께 [@babel/runtime](https://babeljs.io/docs/en/babel-runtime)을 사용하여 바벨이 코드에 추가하는 helper함수를 제거할 수 있다.
2. 앱의 폴리필을 위해서는, [@babel/preset-env `useBuiltIns:"usage"`](https://babeljs.io/docs/en/babel-preset-env#usebuiltins)을 [@babel/polyfill](https://babeljs.io/docs/en/babel-polyfill)과 함께 선택적으로 사용할 수 있다.

개인적인 의견이지만, 모던 브라우저용으로 생성된 번들링을 트랜스파일링 하지 않는 것이 최선의 방법이라고 생각한다. 물론 이는 `jsx`나 널리 사용되지 않는 기능들 같이 무조건 어떤 브라우저든 상관없이 변환해야 하는 경우에는 불가능할 수도 있다. 그렇다면 이 앞선 도구가 꼭 필요한 것인지 되물어볼 필요가 있다. 만약 바벨이 코드 툴체인의 일부가 무조건 되어야 한다면, 바벨이 하고 있는 것들을 잘 살펴보고 개선할 필요가 있다.

## 성능향상은 레이스가 아니다

가능한 빨리 무언가를 얻으려고 하는 것은 때로는 사용자 경험의 고통으로 이어질 수도 있다. 물론 웹 개발 커뮤니티가 경쟁이라는 미명아래 더 빨리 반복하는 것에 집착하고 있기 때문에 조금은 속도를 늦출 필요가 있다고 생각한다. 그렇게 함으로써, 경쟁사만큼 빠른 이터레이션은 거치지 못할 지언정, 애플리케이션 경험은 더욱 향상될 수 있다.

코드 베이스에 성능 향상을 적용 하기전에, 이 모든 것이 하루 밤 사이에 되지 않는 것도 또한 알아둬야 한다. 웹 개발도 하나의 직업이다. 진정으로 영향력 있는 개발을 하기 위해서는 끊임 없이 고민하고 오랜시간 동안 헌신 할 때 비로소 이뤄진다. 꾸준히 향상에 집중해보자. 측정과 테스트를 반복하다보면 사이트의 사용자 경험이 향상되어 시간이 지남에 따라 조금씩 빨라질 것이다.
