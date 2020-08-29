---
title: 왜 CommonJS는 번들사이즈를 크게 하는가?
tags:
  - javascript
  - web
published: true
date: 2020-07-05 09:23:12
description: "[How CommonJS is making your bundles
  larger](https://web.dev/commonjs-larger-bundles/) 를 번역 & 요약한 글입니다. ```toc
  tight: true, from-heading: 2 to-heading: 3 ```  **요약: 웹 애플리케이션을 확실하게 최적화해서
  번들링하기 위해서는, C..."
category: javascript
slug: /2020/07/how-commonjs-is-making-your-bundles-larger/
template: post
---
[How CommonJS is making your bundles larger](https://web.dev/commonjs-larger-bundles/) 를 번역 & 요약한 글입니다.

```toc
tight: true,
from-heading: 2
to-heading: 3
```

**요약: 웹 애플리케이션을 확실하게 최적화해서 번들링하기 위해서는, Common js 모듈을 사용하는 것을 피하고 ECMAS script module synatx를 사용하라**

## CommonJS란 무엇인가?

CommonJS는 2009년에 만들어진 표준으로, 자바스크립트 모듈을 만들기 위한 일종의 규칙이다. 이 방법은 원래 브라우저를 위해 개발된 것은 아니고, 서버사이드 애플리케이션을 위해 만들어졌다.

CommonJS 형식으로 모듈을 정의하면, 이를 export 할 수 있고, 다른 모듈에서 import 할 수 있다. 예를 들어, `add` `subtract` `multiply` `divide` `max`라고 하는 다섯가지 함수가 있다고 해보자.

```javascript
// utils.js
const { maxBy } = require('lodash-es');
const fns = {
  add: (a, b) => a + b,
  subtract: (a, b) => a - b,
  multiply: (a, b) => a * b,
  divide: (a, b) => a / b,
  max: arr => maxBy(arr)
};

Object.keys(fns).forEach(fnName => module.exports[fnName] = fns[fnName]);
```

이제 이것들을 다른 모듈에서 import 하여 사용할 수 있다.

```javascript
// index.js
const { add } = require('./utils');
console.log(add(1, 2));
```

2010년 초반에는 브라우저에 제대로 정착된 모듈 시스템이 부족했으므로, CommonJS는 이내 서버사이드 뿐만 아니라 클라이언트 사이드 라이브러리에도 유명한 표준으로 자리 잡았다.

## CommonJS가 최종 모듈 사이즈에 어떻게 영향을 미치는가?

서버사이드 자바스크립트 애플리케이션의 사이즈는 브라우저만큼 치명적이지는 않으므로, 애초에 딱히 CommonJS를 만들 때는 딱히 프로덕션 번들 사이즈를 줄이는 것에 대한 고려가 되지 않았었다. [https://v8.dev/blog/cost-of-javascript-2019](https://v8.dev/blog/cost-of-javascript-2019) 의 결과에 따르면, 자바스크립트의 번들 사이즈는 브라우저 애플리케이션을 느리게 하는 주범으로 밝혀졌다.

자바스크립트를 번들링하고 최소화하는 `webpack`과 `terser`의 경우, 서로 다른 방식으로 앱 크기를 줄이는 최적화를 진행한다. 빌드 시 애플리케이션을 분석하는 과정에서, 이들은 코드에서 최대한 사용하지 않는 코드를 삭제하려고 한다.

예를 들어, 위의 코드에서의 경우에는 - `add`함수만 사용하고 있으므로, `utils.js`에는 오로지 `add`만 사용하고 있으므로 `add`외에는 모든 것이 지워질 것이라 기대해볼 수 있다.

아래와 같은 webpack 설정으로 빌드를 진행해보자.

```javascript
const path = require('path');
module.exports = {
  entry: 'index.js',
  output: {
    filename: 'out.js',
    path: path.resolve(__dirname, 'dist'),
  },
  mode: 'production',
};
```

이 설정에서는 `index.js`를 엔트리 포인트로 진행하고, 프로덕션 빌드 최적화를 진행했다. `webpack` 을 실행한 뒤에는 [최종 결과물을 확인](https://github.com/mgechev/commonjs-example/blob/master/commonjs/dist/out.js)해볼 수 있는데, 아래와 같다.

```shell
$ cd dist && ls -lah
625K Apr 13 13:04 out.js
```

번들 사이즈가 625kb라는 것에 주목하라. `utils.js` 함수를 살펴보면, `lodash`로 부터 생성된 온갖 모듈들이 추가되어 있음을 볼 수 있다. `index.js`ㅇ에서는 그 어떠한 `loadsh`패키지를 사용하지 않았지만, 프로덕션 에셋에는 엄청난 부분을 차지하고 있음을 볼 수 있다.

같은 코드를 [ECMAScript modules](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/import)을 사용해보자.

```javascript
export const add = (a, b) => a + b;
export const subtract = (a, b) => a - b;
export const multiply = (a, b) => a * b;
export const divide = (a, b) => a / b;

import { maxBy } from 'lodash-es';

export const max = arr => maxBy(arr);
```

```javascript
import { add } from './utils';

console.log(add(1, 2));
```

그 결과물을 보면, 빌드한 결과 [단 40바이트](https://github.com/mgechev/commonjs-example/blob/master/esm/dist/out.js) 만으로 완성되었음을 알 수 있다.

```javascript
(()=>{"use strict";console.log(1+2)})();
```

최종 번들 결과물에는, `utils.js`에 선언된 코드 뿐만 아니라, `lodash`도 찾아 볼 수 없다. 더욱이, `terser`는 심지어 이 `add`함수를 인라인으로 처리해버렸음을 알 수 있다.

왜 CommonJS의 빌드 결과물이 16000배나 더 컸을까? 물론 이는  단순한 토이 프로젝트 였으므로 실제 웹 애플리케이션 사이즈와 비교했을 때 이정도 차이는 없겠지만, 여전히 CommonJS는 프로덕션 빌드에서 많은 부분을 차지하고 있음을 알 수 있다.

**CommonJS 모듈은 일반적으로 최적화를 진행하기가 어렵다. 그 이유는 ES Module 대비 더 다이나믹한 방식을 취하고 있기 때문이다. bundler와 minifier 가 성공적으로 애플리케이션을 최적화 할 수 있게 하려면, CommonJS 모듈을 사용하는 것 보다 ECMAScript module syntax를 전체 애플리케이션에 사용하는 것이 좋다.**

아무리 `index.js`를 ECMAScript 모듈 방식으로 처리했어도, 다른 모듈 사용을 CommonJS방식으로 한다면, 번들 사이즈는 고통 받을 것이다.

## 왜 CommonJS는 애플리케이션 사이즈를 더 크게 하는가?

이 질문에 답을 하기 위해서는, `webpack`의 `ModuleConcatenationPlugin`이 어떻게 동작하는지 살펴볼 필요가 있다. 그리고, 정적 분석에 대해 살펴보아야 한다. (static analyzability) 이 플러그인은 모든 모듈의 범위를 하나의 클로저로 연결하고, 코드가 브라우저에서 더 빠르게 실행할 수 있도록 도와준다.

> In the past, one of webpack’s trade-offs when bundling was that each module in your bundle would be wrapped in individual function closures. These wrapper functions made it slower for your JavaScript to execute in the browser. In comparison, tools like Closure Compiler and RollupJS ‘hoist’ or concatenate the scope of all your modules into one closure and allow for your code to have a faster execution time in the browser.

[ModuleConcatenationPlugin](https://webpack.js.org/plugins/module-concatenation-plugin/)

> 과거 웹팩에서는 함수를 각각의 클로저에 번들링 해두었지만, 이제는 모든 모듈을 하나의 클로저에 묶어두어 브라우저에서 더욱 빠르게 실행 될 수 있도록 한다.

```javascript
// utils.js
export const add = (a, b) => a + b;
export const subtract = (a, b) => a - b;
```

```javascript
// index.js
import { add } from './utils';
const subtract = (a, b) => a - b;

console.log(add(1, 2));
```

ECMA module을 사용한 위의 예제 `index.js`를 살펴보자. 여기에선 `substract` 함수를 정의했다. 그리고 이를 `webpack`으로 빌드하는 대신, minimization옵션을 꺼볼 것이다.

```javascript
const path = require('path');

module.exports = {
  entry: 'index.js',
  output: {
    filename: 'out.js',
    path: path.resolve(__dirname, 'dist'),
  },
  optimization: {
    minimize: false
  },
  mode: 'production',
};
Let us look at th
```

그 결과물을 보자

```javascript
/******/ (() => { // webpackBootstrap
/******/ 	"use strict";

// CONCATENATED MODULE: ./utils.js**
const add = (a, b) => a + b;
const subtract = (a, b) => a - b;

// CONCATENATED MODULE: ./index.js**
const index_subtract = (a, b) => a - b;**
console.log(add(1, 2));**

/******/ })();
```

모든 함수가 같은 네임스페이스 안에 정의되어 있음을 알 수 있다. 그리고 충돌을 막기 위해서, `index.js`의 `substract` 함수를 `index_substract`로 변경했음을 알 수 있다.

만약 위 코드에서 minifier 처리가 진행되었다면

- 사용하지 않는 `substract` `index_substract` 삭제
- 필요없는 모든 주석과 공백삭제
- `console.log`호출안에 있는 `add`함수를 인라인으로 처리

사용하지 않는 import 를 정리하는 것을 트리쉐이킹이라고 한다. 트리쉐이킹은 웹팩이 `utils.js`에서 import 하는 것과 어떤 것을 exports 하는 지를 빌드 타임에  정적으로 이해했기 때문에 (빌드 타임에) 가능했다.

이러한 기능은 ES Module이 CommonJS와 비교했을 때 더 정적으로 분석할 수 있었기 때문에 가능하다.

같은 예제를 CommonJS로 처리해보자.

```javascript
// utils.js
const { maxBy } = require('lodash-es');

const fns = {
  add: (a, b) => a + b,
  subtract: (a, b) => a - b,
  multiply: (a, b) => a * b,
  divide: (a, b) => a / b,
  max: arr => maxBy(arr)
};

Object.keys(fns).forEach(fnName => module.exports[fnName] = fns[fnName]);
```

빌드 시 파일크기가 너무 커지는 관계로, 아래 코드만 살펴보도록 하자.

```javascript
...
(() => {

"use strict";
/* harmony import */ var _utils__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(288);
const subtract = (a, b) => a - b;
console.log((0,_utils__WEBPACK_IMPORTED_MODULE_0__/* .add */ .IH)(1, 2));

})();
```

최종 번들에 `webpack`이라고 되어 있는, 번들 모듈에서 코드를 import/export 하는 일을 담당하는 코드가 삽입되어 있음을 볼 수 있다. 이번 빌드에서는, `utils.js`와 `index.js`안에 있는 심볼들을 모두 같은 네임스페이스 안에 두는 대신에, 코드 실행히에 다이나믹하게 `add`함수를 `__webpack_require__`로 불러오고 있음을 알 수 있다.

이 코드는 CommonJS가 export 명을 임의로 표현하기 때문에 필요하다. 예를 들어, 아래 코드는 완전히 유효한 구조다.

```javascript
module.exports[localStorage.getItem(Math.random())] = () => { … };
```

번들러가 빌드타임에 내보낸 심볼 명이 무엇인지 알수 있는 방법이 없다. 이는 오로지 사용자 브라우저 컨텍스트에서, 런타임시에만 사용할 수 있는 정보를 요구하기 때문이다.

> 고정되어 있는 심볼명을 사용하고 있지 않고, 이를 알아낼 수 있는 방법은 오로지 런타임 (브라우저를 실행하는 순간) 이라는 이야기 입니다.

이 때문에, minifier는 `index.js`에서 정확히 어떤 디펜던시를 가지고 있는지 이해하기 어렵기 때문에, 트리쉐이킹을 할 수 없다. 이러한 패턴을 다른 써드 파티 라이브러리 모듈에서도 찾아볼 수 있다. 만약 `node_modules`에서 CommonJs 모듈을 import 한다면, 빌드 툴 체인이 빌드를 최적화 하기가 어려워진다.

## CommonJS와 트리쉐이킹

CommonJS 모듈이 다이나믹 definition을 하기 때문에 이를 분석하는 것은  매우 어렵다. 그에 반에 ESModule은 항상 string module을 활용하여 import 하기 때문에 매우 명확하다.

만약 현재 사용하고 있는 라이브러리가 (`lodash` 같은 경우) CommonJS의 컨벤션을 따르는 경우, 웹팩의 써드 파티 라이브러리인 [Webpack Common Shake](https://github.com/indutny/webpack-common-shake)를 활용하여 사용하지 않는 export를 제거 할 수도 있다. 이 라이브러리가 트리 쉐이킹을 지원하지만, CommonJS에서 사용 가능한 모든 디펜던시를 커버하는 것은 아니다. 이 말인 즉슨, ES Modules 만큼은 보장되지 않는 다는 것이다. 추가로, `webpack`에서 빌드를 하는데 있어서 추가적인 비용이 지출된다.

## 결론

번들러가 애플리케이션 최적화를 진행하게 할 수 있도록, CommonJS 모듈을 사용하는 것을 피하고, 전체 애플리케이션에서 ECMAScript module syntax를 사용할 수 있도록 하자.

몇가지 팁을 더 추가한다.

- `Rollup.js`의 [node-resolve](https://github.com/rollup/plugins/tree/master/packages/node-resolve)플러그인을 사용하고 `modulesOnly` 플래그에오직 ECMAScript 모듈에만 의존하고 싶다고 명시하라.
- [is-esm](https://github.com/mgechev/is-esm)을 사용해서 사용하고 있는 npm 패키지가 ECMASCript 모듈인지 확인하자
- 앵귤러를 사용하고 있다면, 기본적으로 트리쉐이크가 불가능한 모듈에 대해서 경고를 띄운다.