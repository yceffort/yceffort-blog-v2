---
title: CommonJS와 ES Modules은 왜 함께 할 수 없는가?
tags:
  - javascript
published: true
date: 2020-08-11 08:48:47
description: "[이
  글](https://redfin.engineering/node-modules-at-war-why-commonjs-and-es-modules\
  -cant-get-along-9617135eeca1)을 번역 요약한 글입니다. ## CommonJS와 ES Modules은 왜 함께 할 수
  없는가?  [노드14](https://nodejs.org/en/blog/r..."
category: javascript
slug: /2020/08/commonjs-esmodules/
template: post
---

[이 글](https://redfin.engineering/node-modules-at-war-why-commonjs-and-es-modules-cant-get-along-9617135eeca1)을 번역 요약한 글입니다.

## CommonJS와 ES Modules은 왜 함께 할 수 없는가?

[노드14](https://nodejs.org/en/blog/release/v14.0.0/) 에서는 옛날 스타일의 CommonJS와 (이하 CJS) 새로운 스타일의 ESM Scripts (이하 MJS) 두개가 공존하고 있다. CJS의 경우 `require()`와 `module.exports`를 사용하며, ESM은 `import`와 `export`를 사용한다.

> 정확히는 ECMAScript Modules - Experimental Warning Removal 이다.

ESM과 CJS는 태생부터 완전히 다르다. 일단 겉으로 보기엔, ESM은 CJS와 비슷한 면이 있지만, 이를 구현한 것은 완전히 다르다. ESM에서 CJ를 서로 호출 할 수는 있지만, 꽤나 귀찮은 일이다.

1. ESM에서는 `require()`를 사용할 수는 없다. 오로지 `import`만 가능하다.
2. CJS도 마찬가지로 `import`를 사용할 수는 없다.
3. ESM에서 CJS를 `import`하여 사용할 수 있다. 그러나 오로지 default import만 가능하다. `import _ from 'lodash'` 그러나 CJS가 named export를 사용하고 있다면 named import `import { shuffle } from 'lodash`와 같은 것은 불가능하다.
4. ESM을 CJS에서 `require()`로 가져올 수는 있다. 그러나 이는 별로 권장되지 않는다. 그 이유는 이를 사용하기 위해서는 더 많은 boilerplate가 필요하고, 최악의 경우 Webpack이나 Rollup 같은 번들러도 필요 하다. 그 이유는, ESM가 `require()`에서 어떻게 동작해야 하는지 모르기 때문이다.
5. CJS는 기본값으로 지정되어 있다. 따라서 ESM 모드를 사용하기 위해서는 opt-in해야 한다. `.js`를 `.mjs`로 바꾸거나, `package.json`에 `"type": "module"` 옵션을 넣는 방법이 있다. (기존에 CJS를 쓰던 것은 `.cjs`로 바꾸면 된다.)

이러한 규칙은 고통스럽다. 이는 다양한 유저들, 특히 노드 뉴비들에게는 이해하기 어려운 과정이다.

이러한 규칙들은 고통스럽지만(?) 그 규칙 나름대로 앞으로 살펴볼 이유가 있어, 미래에도 이러한 규칙을 어기기에는 매우 어려워 질 것이다. 이 아티클에서는 자바스크립트 라이브러리 작성자들을 위한 다음 세가지 유용한 정보를 제공할 것이다.

- 라이브러리를 CJS 버전으로 제공하기
- CJS 라이브러리에 ESM 래퍼를 씌우기
- package.json에 exports map을 추가하기

## CJS와 ESM은 무엇인가?

Nodejs 초창기에는, Node 모듈은 CommonJS 모듈로 작성되었다. 따라서 `require()`로 이들을 사용했다. 다른 개발자들이 이를 사용하게 하기 위해서, `exports`를 정의하거나 named exports라 불리우는 `module.exports.foo = 'bar`를 사용하거나, 기본 값으로 `module.exports = 'baz`를 사용하기도 했다.

다음은 named exports 의 예시다.

```javascript
// @filename: util.cjs
module.exports.sum = (x, y) => x + y
// @filename: main.cjs
const { sum } = require('./util.cjs')
console.log(sum(2, 4))
```

다음은 default exports의 예시로, 따로 이름을 정해두지 않으면 default로 설정된다.

```javascript
// @filename: util.cjs
module.exports = (x, y) => x + y
// @filename: main.cjs
const whateverWeWant = require('./util.cjs')
console.log(whateverWeWant(2, 4))
```

ESM 스크립트에서는, `import`와 `export`가 언어의 일부로 추가되었다. CJS와 비슷하게, named exports와 default exports를 지원하는 두가지 문법이 존재한다.

다음은 named exports 의 예시다.

```javascript
// @filename: util.mjs
export const sum = (x, y) => x + y
// @filename: main.mjs
import { sum } from './util.mjs'
console.log(sum(2, 4))
```

다음은 default export의 예시다. CJS와 마찬가지로, 별도로 이름을 지정해두지 않아도 된다.

```javascript
// @filename: util.mjs
export default (x, y) => x + y
// @filename: main.mjs
import whateverWeWant from './util.mjs'
console.log(whateverWeWant(2, 4))
```

## ESM과 CJS는 완전히 다르다.

CommonJS에서는 `require()`는 동기로 이루어진다. 따라서 promise나 콜백 호출을 리턴하지 않는다. `require()`는 디스크로 부터 읽어서 (네트워크 일수도 있다) 그 즉시 스크립트를 실행한다. 따라서 스스로 I/O나 부수효과 (side effect)를 실행하고 `module.exports`에 설정되어 있는 값을 리턴한다.

반면에 ESM은 모듈 로더를 비동기 환경에서 실행한다. 먼저 가져온 스크립트를 바로 실행하지 않고, `import`와 `export`구문을 찾아서 스크립트를 파싱한다. 파싱 단계에서, 실제로 ESM 로더는 종속성이 있는 코드를 실행하지 않고도도, named imports에 있는 오타를 감지하여 에러를 발생시킬 수 있다.

그 다음 ESM 모듈 로더는 가져온 스크립트를 비동기로 다운로드 하여 파싱한다음, import된 스크립트를 가져오고, 더 이상 import 할 것이 없어질 때까지 import를 찾은 다음 dependencies의 모듈 그래프를 만들어 낸다. 그리고, 스크립트는 실행될 준비를 마치게 되며, 그 스크립트에 의존하고 있는 스크립트들도 실행할 준비를 마치게 되고, 마침내 실행된다.

ESM 모듈 내의 모든 자식 스크립트들은 병렬로 다운로드 되지만, 실행은 순차적으로 진행된다.

## CJS는 기본 값이다. 왜냐면 ESM은 바뀔게 넘 많아서

ESM은 자바스크립트의 많은 부분에 변경이 필요하다. ESM은 일단 기본 값으로 `use strict`가 설정되어 있어야하고, `this`는 global object를 참조하지 않고, 스코프는 다르게 작동 되는 등등 변화가 많다.

이것이 브라우저에서 조차 `<script>`가 ESM을 기본으로 지정하지 않는 이유다. ESM을 사용하기 위해서는 `type="module"`을 추가해 주어야 한다.

기본 값을 CJS에서 ESM으로 바꾸는 것은 호환성을 해치는 문제가 된다. (node의 대안으로 주목받고 있는 deno는 ESM을 기본값으로 사용하지만, 결과적으로 모든 생태계를 처음부터 다시 설계해야 했다.)

## 톱레벨에 존재하는 await 때문에 CJS는 ESM을 `require()`할 수 없다.

CJS가 ESM을 `require()` 하지 못하는 가장 단순한 이유는, ESM는 top level에서 `await`을 할 수 있지만, CJS는 그렇지 못하기 때문이다. 여기서 말하는 [top-level await](https://v8.dev/features/top-level-await)은 `async function` 밖에서 `await`을 사용하게 해주는 것이다.

해당 V8 블로그 포스트 글을 인용하자면

> [이 gist](https://gist.github.com/Rich-Harris/0b6f317657f5167663b493c722647221)에서 top-level await에 대한 우려와 함께, 자바스크립트가 미래에 해당 기능을 구현하지 말하야 하는 이유에 대해 논의 한 적이 있다. 여기서 우려하는 사안은
>
> - top-level await은 코드 실행을 블로킹할 수 있다.
> - top-level await은 리소스를 가져오는 것을 블로킹할 수 있다.
> - commonjs에서 이를 명확히 구현할 수 없다.
>   그리고 이 3가지 문제점에 대해서, stage 3 제안에서 다음과 같이 언급한다.
> - siblings 코드가 실행 가능하므로, 결정적인 블로킹 포인트가 없다.
> - top-level await은 모듈 그래프의 실행 단계에서 이루어 진다. 이 지점에서는, 모든 리소스들이 이미 fetch 되고 링크 되어 있다. 따라서 리소스 fetch를 블로킹할 염려는 없다.
> - top-level await은 오로지 [ESM]에서만 논의 될 문제다. CommonJS 모듈에서는 이를 지원할 계획이 없다.

[ESM을 `require()` 하는 방법에 대한 논의가 이어지고는있지만,](https://github.com/nodejs/modules/issues/454) 빠른 시일 내에 이것이 실현 되기는 어려워 보인다.

## CJS는 ESM에서 `import`할 수는 있지만, 썩 훌륭해보이지는 않는다.

```javascript
;(async () => {
  const { foo } = await import('./foo.mjs')
})()
```

## ESM은 cjs의 named exports를 import 할 수 없다.

이것은 가능하지만

```javascript
import _ from './lodash.cjs'
```

이것은 불가능하다.

```javascript
import { shuffle } from './lodash.cjs'
```

CJS는 named exports를 실행단계에서 연산하지만, ESM은 named exports를 파싱 단계에서 연산하기 때문이다.

다행히 이를 우회할 수 있는 방법은 있다.

```javascript
import _ from './lodash.cjs'
const { shuffle } = _
```

> 하지만 이방법은 tree shaking이 되지 않으므로 번들링 시 사이즈가 커지게 된다.

그러나 이 방법이 순서까지도 보장해주는 것은 아니다.

```javascript
import liquor from 'liquor'
import beer from 'beer'
```

만약 `liquor` `beer`모두 cjs로 되어 있다면 그 순서가 반드시 `liquor`, `beer`가 되는 것은 아니다. `beer`가 `liquor`가 반드시 실행되어야 하는 상황이라면 더욱 문제가 커질 수 있다.

## CJS와 ESM을 모두 지원하는 방법

### CJS 버전으로 라이브러리를 제공해라

이는 CJS 유저들에게도 친숙하고, 오래된 노드버전도 지원가능하다. 타입스크립트로 작성할 경우, JS > CJS로 트랜스파일하면 된다.

## CJS 라이브러리에 ESM 래퍼를 제공해라

```javascript
import cjsModule from '../index.js'
export const foo = cjsModule.foo
```

ESM 래퍼를 `esm` 디렉토리에 두고, `package.json` 에 `{"type": "module"}` 을 추가하자. `.mjs`로 이름을 변경하는 것도 방법이지만, 일부 툴에서 제대로 작동하지 않으므로 별도의 디렉토리에 넣는 것을 선호한다.

트랜스파일링이 중복으로 되는 것을 피해야 한다. 만약 typescript에서 트랜스파일링한다면, 이를 CJS와 ESM 두개 모두로 트랜스파일링 할 수 있지만, 이는 사용자가 실수로 `import`하거나 `require`하는 일이 발생하게 된다.

### `package.json`에 `exports` 를 추가하라

```json
"exports": {
    "require": "./index.js",
    "import": "./esm/wrapper.js"
}
```

한가지 명심해야 할 것은, `exports`를 추가하는 것은 시멘틱 버저닝의 브레이킹 체인지 (메이저 버전 업)을 가져온다는 것이다. 그리고 항상 온전한 파일명 `index.js`가 들어가야 한다. `index` 나 `./build`가 들어가서는 안된다.

https://nodejs.org/api/esm.html#esm_package_entry_points
