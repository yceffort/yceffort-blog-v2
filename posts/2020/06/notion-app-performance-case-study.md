---
title: Notion 성능 최적화
tags:
  - javascript
published: true
date: 2020-06-29 07:42:01
description: '[Case Study: Analyzing Notion app
  performance](https://3perf.com/blog/notion/)를 제멋대로 요약한 글입니다. 왠만하면 저 글을 참고하세요.
  ```toc tight: true, from-heading: 2 to-heading: 3 ```  ## 자바스크립트의 비용  보통 `로딩
  속도`를 이야기하면...'
category: javascript
slug: /2020/06/notion-app-performance-case-study/
template: post
---

[Case Study: Analyzing Notion app performance](https://3perf.com/blog/notion/)를 제멋대로 요약한 글입니다. 왠만하면 저 글을 참고하세요.

## Table of Contents

## 자바스크립트의 비용

보통 `로딩 속도`를 이야기하면, 네트워크 성능을 떠올리는 경우가 많다. 네트워킹이라는 관점에서는 노션은 꽤 괜찮았다. [HTTP/2](https://developers.google.com/web/fundamentals/performance/http2?hl=ko)를 사용하고, 파일을 gzip으로 압축했으며, [CDN 프록시](https://cdn.hosting.kr/cdn%EC%9D%B4%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80%EC%9A%94/)를 위해 클라우드페어를 잘 쓰고 있었다. 그러나 `로딩 속도`를 차지 하는 다른 한켠에는 `처리 성능`이 포함되어 있다. gzip을 압축해제하고, 이미지는 디코드 되야 하며, 자바스크립트는 실행되어야 한다. 이런 것들이 처리 성능에 포함되어 있다.

더 좋은 품질의 네트워크를 사용하면 향상되면 네트워크 성능과는 다르게, 처리 성능은 그렇지 않다. 오로지 사용자의 CPU가 더 좋아야 한다. 그리고 스마트폰의 사용자의 CPU라고 한다면 - 특히 안드로이드 폰의 경우 구리다.

![스마트폰 별 노션 앱 로딩 속도](https://3perf.com/static/c7e7dd1756462191f79441053ce9d5a7/28bdc/cost-of-js.png)

감사합니다 아이폰 센세

노션의 경우, 처리 성능이 차지 하는 부분은 더 크다. 앱에서 아용하는 리소스를 캐싱하여 네트워크의 비용을 줄이는 것은 쉽다. 그러나 처리 성능은 앱을 시작할 때 마다 지불해야 한다. 즉, 어떤 스마트폰 사용자는 매번 앱을 실행할 때 마다 10초이상 스플래쉬 스크린 (애플리케이션이 실행되기 전에 보여지는 화면)을 봐야 한다.

노션의 테스트폰 중 하나인 넥서스5의 경우 `vendor`와 `app`을 실행하는데 4.9초가 걸렸다. 이 시간은 즉 페이지와 앱이 상호작용 하지 못하고 비어있게 된다.

![0.4 + 4.5초가 되어야 비로소 의미있는 First Paint가 실행된다.](https://3perf.com/static/7dc48c81c2034a8df43c79164c023198/4e22f/waterfall-nexus-js.png)

브라우저 Dev Tool을 사용하여 무슨 일이 일어 나고 있는지 확인해보자.

![](https://3perf.com/static/1751d6c59e1e18a0a9de2334fd2d0b63/28bdc/js-trace.png)

먼저 0.4초 동안 `vendor`번들이 컴파일 된다. 그리고 `app`번들이 컴파일되며, 그리고 두 번들이 실행되기 시작하고 - 이작업에만 3.3초가 소요된다. 어떻게 이 시간을 줄일 수 있을까?

## 자바스크립트 실행을 지연시키기.

먼저 번들 실행 과정을 살펴보자.

![](https://3perf.com/static/c59467702c58246f5c3cf04b4cd54843/28bdc/js-trace-execution.png)

- 함수는 모두 `bkwR`과 같은 네글자로 되어 있다. 웹팩이 번들을 만들때, 각 모듈을 함수로 감싼다. 그리고 이 감싼 것들에 ID를 부여한다. 이 ID들이 바로 함수명이 된다. (이 것은 [optimization.moduleIdes:'hashed'](https://v4.webpack.js.org/configuration/optimization/#optimizationmoduleids)나 [HashedModuleIdsPlugins](https://webpack.js.org/plugins/hashed-module-ids-plugin/)를 사용하면 발생한다.)

before

```javascript
import formatDate from './formatDate.js`
//....
```

after

```javascript
 fOpr: function(module, __webpack_exports__, __webpack_require__) {
  "use strict";
   __webpack_require__.r(__webpack_exports__);
   var _formatDate__WEBPACK_IMPORTED_MODULE_0__ =
     __webpack_require__("xN6P");
   // ...
  },
```

- 그리고 저기서 자주 보이는 `s`함수는 사실 `__webpack_require__`다. 이는 웹팩의 내부 함수로 모듈을 요구할 때 사용된다. 다시 말해 코드에서 `import`를 사용하면, 웹팩이 `__Webpack_require__()`로 바꾼다.

번들 초기화는 굉장히 많은 시간을 할애하는데, 그 이유는 모든 모듈을 실행하기 때문이다. 각 모듈은 실행하는데 몇 밀리초가 걸릴뿐이지만, 노션의 경우 이러한 모듈이 1100개가 넘게 있다. 이것을 해결하는 유일한 방법은 초기화에 더 적은 모듈을 실행하는 것이다.

### 코드 스플리팅

첫 화면을 띄우는 시간을 줄이는 가장 좋은 방법은 당장 필요하지 않은 기능들을 나누는 코드 스플릿 방식이다. 웹팩에서는, [import()](https://webpack.js.org/guides/code-splitting/)를 사용한다.

```html
// Before
<button onClick="{openModal}" />

// After <Button onClick={() => import('./Modal').then(m => m.openModal())} />
```

코드 스플릿은 여러분이 할 수 있는 가장 최선의 성능최적화다. 이는 많은 성능상 이점을 가져다 준다. 코드 스플릿팅을 하게되면, 로딩 시간을 60% 감소시킬 수 있다. 노션의 경우 40~45% 를 절감하는 효과를 가져왔다.

[코드 스플릿팅을 하는 몇가지 일반적인 방식이 있다.](https://medium.com/js-dojo/3-code-splitting-patterns-for-vuejs-and-webpack-b8fff1ea0ba4)

- 페이지 별로 번들을 나누기
- below-the-fold (신문을 접었을 때 볼 수 없는 영역. 웹페이지에서는 스크롤하지 않으면 볼 수 없는 부분을 의미한다.) 의 코드를 나누기
- 조건에 따라 노출되는 컨텐츠를 나누기 (당장 사용자에게 노출되지 않은 다이나믹 UI)

노션의 경우 페이지가 없으며 (페이지 그 자체가 하나의 글이므로) 페이지 또한 사용자에 따라 굉장히 유동적이기 때문에 below-the-fold방식도 처리하기 어렵다. 여기에서 노션이 사용할 수 있는 유일한 방법은 조건에 따라 노출되는 컨텐츠를 나누는 방식이다. 그래서 노션은 다음과 같은 부분을 적용해 보기로 했다.

- Settings, import, trash와 같이 사용자가 자주 사용하지 않는 UI
- 사이드바, share, page options와 같이 자주 사용하지만 앱 시작하는데 바로 보여줄 필요가 없는 UI. 이 영역 들은 앱이 시작된 이후에 준비해도 된다.
- 페이지 로딩을 가로막는 무거운 요소들. 몇몇 글 조각들은 꽤 무겁다. 일례로 코드 블록의 경우 `Prism.js`를 활용하여 68 종류의 언어를 지원하는데, 이는 압축되어있지만 최소 120KB가 나간다.

### ModuleConcatenationPlugin이 제대로 작동하는지 확인하기

웹팩에서 [module concatenation](https://webpack.js.org/plugins/module-concatenation-plugin/) 이라는 기능이 있는데, 이는 작은 ES 모듈을 하나로 합치는 역할을 한다. 이는 모듈 처리과정에서의 오버헤드를 줄여주며, 불필요한 코드를 삭제해준다. 이 모듈이 제대로 작동하는지 확인하기 위해서는

- 바벨이 ES 모듈을 Commonjs로 컴파일 하지 않는지 확인한다. [@babel/preset-env](https://babeljs.io/docs/en/babel-preset-env)는 ES모듈을 CommonJS로 트랜스파일 하지 않는다.
- [optimization.concatenateModules](https://webpack.js.org/configuration/optimization/#optimizationconcatenatemodules)옵션이 명시적으로 꺼져있진 않은지 확인한다.
- 웹팩 프로덕션 빌드를 [--display-optimization-bailout](https://webpack.js.org/plugins/module-concatenation-plugin/#debugging-optimization-bailouts)옵션과 함께 실행해서, module concatenation이 안되는 경우가 있는지 확인한다.

> 모든 imports가 `__webpack_require__` 함수로 변경된다는 것을 기억하는가? 같은 함수가 초기화 단계에서 1100번 넘게 호출되면 어떻게 될까? 이 함수는 엄청난 시간을 잡아먹게 된다 (...)
> ![](https://3perf.com/static/befda82857003648779614db0961125d/713f0/hot-path.png)
>
> [그러나 이부분은 딱히 최적화 될것 같지 않다.](https://github.com/webpack/webpack/issues/2219)

### Babel `plugin-transform-modules-common-js`의 `lazy`옵션을 활용하기

> 해당 옵션은 module concatenation이 꺼져있을때만 가능하다. 즉 위의 항목과는 호환되지 않는다.

[@babel/plugin-transform-modules-commonjs](https://babeljs.io/docs/en/babel-plugin-transform-modules-commonjs#lazy)는 바벨의 공식 플러그인으로, ES imports구문을 Commonjs의 `require()`로 바꿔 준다.

```javascript
// Before
import formatDate from './formatDate.js'
export function getToday() {
  return formatDate(new Date())
}

// After
const formatDate = require('./formatDate.js')
exports.getToday = function getToday() {
  return formatDate(new Date())
}
```

그리고 `lazy`옵션이 활성화 되면, 아래과 같이 바뀌게 된다.

```javascript
// After, with `lazy: (path) => true`, simplified
exports.getToday = function getToday() {
  return require('./formatDate.js')(new Date())
}
```

고맙게도, `getToday`가 호출되지 않는다면 `./formatDate.js`도 import 되지 않는다. 그러나 여기엔 몇가지 하자가 있는데

- 현재 코드베이스를 `lazy`로 변경하는 것은 까다로울 수 있다. 몇 모듈들은 다른 모듈의 부수효과에 의지하고 있을 수 도 있는데, 이는 딜레이를 유발한다. 그리고 [플러그인 문서](https://babeljs.io/docs/en/babel-plugin-transform-modules-commonjs#lazy)에 나와있듯이, `lazy`옵션은 순환 참조를 깨버린다.
- 웹팩 5버전 이하에서 [웹팩의 트리쉐이킹](https://webpack.js.org/guides/tree-shaking/)을 지원하지 못한다.
- 위에서 언급했던 것처럼 module concatenation을 꺼버린다. 이는 즉 모듈 처리과정에서의 오버헤드가 높아질 수 있다는 것이다.

위 세가지 단점은 이 옵션을 사용하는데 있어 머뭇거리게 만드는 요소다. 그러나 적절하게만 사용된다면, 비용을 줄이는데 도움을 줄 수 있다.

> 몇개의 모듈이 이렇게 지연 실행 될 수 있을까? Chrome Dev Tools에서 이에 대한 해답을 찾을 수 있다.
> 자바스크립트가 무거운 페이지를 연다음, Ctrl+Shift+P (Windows) ⌘⇧P (macOS), 을 누르고 “start coverage” 를 치고 엔터를 누르자.
> 페이지가 새로고침되면서, 최초 렌더링시에 얼마나 많은 코드가 실행되었는지 보여준다.
> 노션의 경우 39%가 `vendor`, 61%가 `app` 번들에서 페이지 렌더링 이후에 사용되지 않는다.
> ![](https://3perf.com/static/7aed4a03203dae92b7f153801229ff8d/28bdc/coverage.png)
>
> 오직 빨간 부분만 페이지 렌더링에 사용되었다.

## 사용하지 않는 JS 코드 삭제하기

![](https://3perf.com/static/1751d6c59e1e18a0a9de2334fd2d0b63/28bdc/js-trace.png)

`compile script` 과정에서 1.6초가 소요되고 있다. 이 과정에서 무슨일이 일어나고 있는걸까?

V8엔진은 다른 자바스크립트 엔진처럼, 자바스크립트를 [just-in-time compilation](https://blog.sessionstack.com/how-javascript-works-inside-the-v8-engine-5-tips-on-how-to-write-optimized-code-ac089e62b12e)로 실행한다. 이 말인 즉슨, 모든 코드들은 실행하기 전에 머신에서 컴파일 되야 한다는 것을 의미한다. 따라서 코드가 많으면 많을 수록 컴파일하는데 더 많은 시간을 할애한다. [2018년 기준 평균적으로 보통 총 실행 시간의 10~30%를 자바스크립트을 컴파일하고 파싱하는데 사용하는 것으로 알려졌다.](https://blog.sessionstack.com/how-javascript-works-inside-the-v8-engine-5-tips-on-how-to-write-optimized-code-ac089e62b12e) 따라서 이 과정을 줄이는 유일한 방법은 자바스크립트 코드의 양을 줄이는 것이다. (...)

### 코드 스플리팅

또 나왔다. 코드 스플리팅은 최초 번들 초기화 시간을 줄여줄 뿐만 아니라, 컴파일에 소요되는 시간도 줄여준다. 코드가 적을 수록, 컴파일도 빠르다.

### 사용하지 않는 vendor 코드 삭제

앞서 봤던 것처럼, 40% 정도의 코드는 로딩 후에 렌더링에 관여하지 않았다.

몇 코드들은 유저가 무언가를 액션을 취했을 때 필요해질 수 있다. 그러나 이런 코드가 얼마나 될까? 노션은 소스팹을 퍼블리쉬 하지 않는다. 그말인즉 [source-map-explorer](https://www.npmjs.com/package/source-map-explorer)를 활용해 본들 내부를 살펴보고 가장 큰 모듈을 볼수도 없다는 것을 의미한다. 그러나 우리는 github에서 압축되지 않는 외부 라이브러리의 코드를 통해 압축된 코드들에 대해 대충 추측할 수 있다. 이 과정에서 검거된(?) 라이브러리는 아래와 같다.

1. moment with all locales → 227 KB
2. react-dom → 111 KB
3. libphonenumber-js/metadata.min.json → 81 KB
4. lodash → 71 KB
5. amplitude-js → 55 KB
6. diff-match-patch → 54 KB
7. tinymce → 48 KB
8. chroma-js → 35 KB
9. moment-timezone → 32 KB
10. fingerprintjs2 → 29 KB

여기에서 최적화 하기 쉬운 모듈은 `moment` `lodash` `libpnoenumber-js`다. 날짜를 다루는 자바스크립트 라이브러리 `moment`는 모든 localization 데이터를 포함하면 번들링되도 160kb가 넘는다. 노션은 어차피 영어만 지원하므로, 이 `localization`은 별로 필요치 않다. 따라서

1. [moment-locales-wepback-plugin](https://www.npmjs.com/package/moment-locales-webpack-plugin)을 활용하여 사용하지 않는 `moment` locale을 지운다.
   2, `moment`를 [date-fns](https://date-fns.org/)로 바꾸는 것을 고려해본다. `moment`와 다르게, `date-fns`를 사용하면, 오로지 필요한 메소드만 import할 수 있다.

데이터 조작 유틸리티인 [lodash](https://github.com/lodash/lodash)의 경우 300개가 넘는 함수들을 제공하고 있다. 이는 좀 과도하다. 보통 많아봐야 5~30개 정도의 메소드만 사용할 뿐이다. 이를 해결할 좋은 방법은 [babel-plugin-lodash](https://github.com/lodash/babel-plugin-lodash)를 사용하는 것이다. [lodash-webpack-plugin](https://www.npmjs.com/package/lodash-webpack-plugin)도 마찬가지로 사용하지 않는 loadash 메소드를 날려준다.

[libphonenumber-js](https://github.com/catamphetamine/libphonenumber-js)는 전화번호를 파싱하고 포맷팅 해주는 라이브러리이지만, 전화번호 메타데이터를 포함하게 되면 81kb가 된다. 이것도 사용하지 않으면 삭제하는 것이 좋다.

### 폴리필 제거하기

`vendor`번들에서 의존하고 있는 다른 주요 디펜던시 중 하나는 [core-js](https://github.com/zloirock/core-js)라이브러리다.

![core-js](https://3perf.com/static/c3a621e6655ea9ce12a3f0ca9259d83c/28bdc/core-js.png)

여기엔 두가지 문제가 존재한다.

1. 불필요하다. 노션의 경우 크롬 81버전에서 테스트하는데, 해당 버전은 대부분의 모던 자바스크립트 기능을 지원한다. 그러나 이 번들에는 여전히 `Symbol` `Object.assign`등의 폴리필이 포함되어 있다.
2. 노션 앱에 불필요하다. 데스크톱 또는 모바일 앱에서도 마찬가지로 자바스크립트 엔진 또한 1번처럼 모던하다.

그러면 대신 무엇을 해야할까? 오래된 브라우저를 위한 폴리필을 지원하되, 몇가지 안쓰는 폴리필은 삭제하는 것이다. 해당 방법은 [이글](https://3perf.com/blog/polyfills/)을 참조하면 좋다.

이 폴리필은 여러차례 번들링 된다. `vendor` 번들은 `core-js` 카피 라이트를 3번이나 포함하고 있다. 매 카피라이트는 동일하지만, 다른 모듈에 의존되며 다른 의존성을 갖는다.

![](https://3perf.com/static/71b244456b9258e98be3547f06de1d59/e01d3/core-js-copyright.png)

이 말은 즉 `core-js`그 자체가 3번이나 번들링 된다는 것이다. 도대체 왜?

카피라이트 모듈은 아래와 같은 모습을 띄고 있다.

```javascript
var core = require('./_core')
var global = require('./_global')
var SHARED = '__core-js_shared__'
var store = global[SHARED] || (global[SHARED] = {})

;(module.exports = function (key, value) {
  return store[key] || (store[key] = value !== undefined ? value : {})
})('versions', []).push({
  version: core.version,
  mode: require('./_library') ? 'pure' : 'global',
  copyright: '© 2019 Denis Pushkarev (zloirock.ru)',
})
```

- `var core = require('./_core'); core.version`는 라이브러리의 버전이고
- `require('./_library') ? 'pure' : 'global'`는 [라이브러리 모드](https://github.com/zloirock/core-js/tree/v2#basic)다.

압축된 코드에서 이는

- `var r=n(<MODULE_ID>);r.version`고
- `n(<MODULE_ID>)?"pure":"global"`다.

이 모듈 ID를 번들에서 추적하다보면, 아래와 같은 것을 마주하게 된다.

![](https://3perf.com/static/884fccdc2f4bfb8cfe805a7985a2df09/e01d3/core-js-copyright-2.png)

이 말인 즉슨 `core-js`에 세가지 다른 버전이 있는데

- `2.6.9`는 글로벌 모드에
- `2.6.11`는 글로벌 모드에
- `2.6.11`는 pure 모드에

있다는 것이다.

[사실 이는 흔한 문제다.](https://twitter.com/iamakulov/status/1225069880988270592) 내 앱에서는 특정버전 `core-js`에 의존하고 있지만, 어딘가 내 다른 디펜던시에서 다른 `core-js`버전을 의존하고 있는 것이다.

이를 해결하는 방법은 `yarn why core-js`를 실행해서 왜 두가지 버전이 존재하고 있는지 확인하는것이다. 그리고 디펜던시를 조정해서 `core-js`버전을 맞추거나, 웹팩의 [resolve.alias](https://webpack.js.org/configuration/resolve/#resolvealias)를 이용해서 중복을 해결하면 된다.

## 로딩 워터폴 최적화하기

![how notion is loading](https://3perf.com/static/875c8bbd7a5a05190b50c9dffabecdfb/ffcbe/waterfall-explained-full.png)

https://webpagetest.org/result/200418_KE_d8c556d0fa8e60a79cd2370f224b3ad7/1/details/#waterfall_view_step1

여기서 몇가지 주목할 것이 있다.

- API요청은 번들이 온전히 다운로드 될때까지 일어나지 않는다
- 의미있는 페인팅(Contentful Paint, 실제 컨텐츠가 보이는 순간)은 주요 api요청이 끝나기 전까지 일어나지 않는다. (특히 35번 요청이 오래걸린다)
- API요청이 Intercom, Segment, Amplitude등 여러 써드 파티 라이브러리의 짬뽕으로 되어 있다.

### 써드 파티 라이브러리 지연시키기

써드파티 라이브러리의 경우 광고, 분석과 같은 일을 위해 종종 사용된다. 이는 비즈니스단의 문제로 - 유용하기는 하지만 문제의 시발점이기도 하다.

노션의 경우 위 3가지 써드파티 라이브러리가 자바스크립트 실행 성능을 저해하고, 메인 쓰레드의 실행을 방해하여 앱이 여전히 초기화 중인 것처럼 보이게 한다. 만약 이 3가지 써드파티 라이브러리를 날리면, 적어도 1초 정도의 시간은 벌 수 있다.

물론 이런 코드들을 날려버리면 참 좋겠지만, 지연 시키는 것도 한가지 방법이다.

```javascript
// Before
async function installThirdParties() {
  if (state.isIntercomEnabled) intercom.installIntercom()

  if (state.isSegmentEnabled) segment.installSegment()

  if (state.isAmplitudeEnabled) amplitude.installAmplitude()
}

// After
async function installThirdParties() {
  setTimeout(() => {
    if (state.isIntercomEnabled) intercom.installIntercom()

    if (state.isSegmentEnabled) segment.installSegment()

    if (state.isAmplitudeEnabled) amplitude.installAmplitude()
  }, 15 * 1000)
}
```

이렇게 바꾼다면, 앱이 완전히 실행되기 전까지는 로드 되지 않을 것이다.

> setTimeout vs requestIdleCallback vs events
> setTimeout은 최선의 접근 법은 아니지만, 쓸만하다.
> 가장 좋은 방법은 `페이지가 완전히 로드되었다`라는 이벤트를 참고 하는 것이다.
> [requestIdleCallback](https://developer.mozilla.org/en-US/docs/Web/API/Window/requestIdleCallback)는 이러한 문제를 해결하는데 최적화된 도구인 것 같지만 서도 그렇지 않다. 크로미움에서 테스트 했을때, 너무 빨리 트리거 되었다.

## API 데이터를 미리 로딩하기

그 외에 노션 api의 경우, 렌더링 이전에 무려 9개의 요청을 보내고 있었다.

![](https://3perf.com/static/3612a04461fce98c8355e6188d41bfd8/335b6/waterfall-api.png)

각 요청은 최소 70ms에서 최대 500ms가 소요되었으며, 이 요청은 각 순차적으로 이루어졌다. 즉 한 가지 요청이 끝나야 다음 것이 시작되었다. 즉 api 요청에 대한 응답이 느려진다면 지연이 더 발생한다는 것을 의미한다. 이러한 지연시간을 없애는 좋은 방법은 무엇이 있을까?

가장 좋은 방법은 서버사이드에서 데이터를 데이터를 가져오고 이를 HTML안에 때려 넣는 것이다.

```javascript
app.get('*', (req, res) => {
  /* ... */

  // Send the bundles so the browser can start loading them
  res.write(`
    <div id="notion-app"></div>
    <script src="/vendors-2b1c131a5683b1af62d9.js" defer></script>
    <script src="/app-c87b8b1572429828e701.js" defer></script>
  `)

  // Send the initial state when it’s ready
  const stateJson = await getStateAsJsonObject()
  res.write(`
    <script>
      window.__INITIAL_STATE__ = JSON.parse(${stateString})
    </script>
  `)
})
```

> 이 방법을 위해서는 아래 사항을 유념해두자
> [최적의 성능을 위해](https://joreteg.com/blog/improving-redux-state-transfer-performance) 데이터를 json으로 인코딩하자
> XSS 공격을 피하기 위해 데이터를 [jsesc](https://github.com/mathiasbynens/jsesc)로 이스케이프 처리해두자. (`json: true, isScriptContext: true`)

이 접근 방법으로 인해, 앱은 API요청을 기다릴 필요가 없다. 앱의 초기 상태값(`state`)를 window에서 구해올 수 있으며, 렌더링도 즉시 이루어질 것이다.

또 다른 방법 중 하나는 데이터를 요청하는 인라인 스크립트를 작성하는 것이다.

```html
<div id="notion-app"></div>
<script>
  fetchAnalytics()
  fetchExperiments()
  fetchPageChunk()

  function fetchAnalytics() {
    window._analyticsSettings = fetch('/api/v3/getUserAnalyticsSettings', {
      method: 'POST',
      body: '{"platform": "web"}',
    }).then((response) => response.json())
  }

  async function fetchExperiments() {
    /* ... */
  }

  async function fetchPageChunk() {
    /* ... */
  }
</script>
<script src="/vendors-2b1c131a5683b1af62d9.js"></script>
<script src="/app-c87b8b1572429828e701.js"></script>
```

데이터가 로드 된다면 앱은 거의 즉시 필요한 데이터를 얻을 수 있다. 중요한 것은, 스크립틀가 가능한 빨리 요청을 날려야 한다는 것이다. 이는 번들이 로딩 중이고 메인스레드가 유휴상태일 때 응답이 도착하여 처리될 가능성을 높여 준다.

## 그 밖에

### 응답에 `Cache-Control`을 사용하기

응답 헤더에 `Cache-Control`이 세팅되어 있지 않다는 것은, 캐싱이 꺼져 있다는 뜻은 아니지만 - [각 브라우저 별로 응답을 다른 방식으로 캐시한다는 것을 의미한다.](https://paulcalvano.com/index.php/2018/03/14/http-heuristic-caching-missing-cache-control-and-expires-headers-explained/) 이는 클라이언트 사이드에서 원치 않는 버그를 야기 할 수 있다.

![](https://pbs.twimg.com/media/EXuNrXLWkAAdGrw?format=jpg&name=large)

이를 피하기 위해서는 번들 asset 과 api 응답 요청의 `Cache-Control` 헤더에 적당한 값을 넣어주면 좋다.

> For API responses (like /api/user): prevent caching
> → Cache-Control: max-age=0, no-store
> For hashed assets (like /static/bundle-ab3f67.js): cache for as long as possible
> → Cache-Control: max-age=31556952, immutable

### 스켈레톤 활용하기

보통 앱이 뭔가 로딩 중인 것을 보여주고 싶을 때 스피너나 로딩 바를 쓰곤 한다. [그러나 때로는 스피너가 체감상 성능을 더 악화시키는 효과를 가질 때가 있다.](https://www.lukew.com/ff/entry.asp?1797) 유저는 스피너를 보고 앱이 더 느리다고 느낄 수 있다. 이러한 느낌을 피하기 위해서는, 스켈레톤 UI를 사용하면 좋다.

![skeleton](https://3perf.com/1fb74cf3a28740ab90f3d61ea37e016c/notion-skeleton.svg)

## 요약

이러한 작업을 통해서 얼마나 최적화를 할 수 있을까?

- `vendor`번들에서 30%를 차지하는 사용하지 않는 의존성과 폴리필을 제거했다고 가정해보자. 추가로 코드 스플릿 방식으로 메인 번들에서 20%를 덜어냈다고 해보자. 컴파일 과 실행과정에서 얼마나 줄었다고 단언하기는 어렵지만, 대략 10~50%정도의 효과를 기대할 수 있다. 노션의 넥서스5에서는 25% 정도의 성능을 체감할 수 있었다.
- API를 미리 로딩해서 10% 정도의 성능 효과를 볼 수 있었다.
- 써드 파티라이브러리를 지연 시킴으로서 1초 정도를 더 줄였다.

대충 계산해서, 이러한 것들을 활용해서 기존의 12.6초에서 약 3.9초 정도를 절감할 수 있었다.

알고보면, 거의 모든 앱에서 번들러 구성을 조정하고, 몇가지 정밀한 코드 변경만으로도 이뤄낼 수 있는 최적화들이 존재했다. [3perf.com](https://3perf.com/#services)에서 가장 쉬운 방법을 찾아보자.
