---
title: '2부) esmodule은 무엇인가?'
tags:
  - nodejs
published: true
date: 2023-06-06 11:02:49
description: '누구나 알지만 모르는 esmodule'
---

## Table of Contents

## 서론

ECMAScript module (이하 esmodule)은 패키지로 작성된 자바스크립트 코드를 재사용하기 위한 방법으로써, tc39에 기재되어 있는 공식적인 표준이다. 모듈은 각각 `import`와 `export`구문으로 가져오고 사용할 수 있다. 예시 코드를 살펴보자.

```javascript
// addTwo.mjs
function addTwo(num) {
  return num + 2
}

export { addTwo }
```

위 파일은 `addTwo.mjs`라는 파일명으로 작성된 예제이며, `addTwo`라는 함수를 내보내고 있다. 그리고 이 파일은 다음과 같이 불러오는 것이 가능하다.

```javascript
// app.mjs
import { addTwo } from './addTwo.mjs'

// Prints: 6
console.log(addTwo(4))
```

nodejs는 표준 문서에 나와있는 방식대로 esmodule 방식을 완전히 지원하고 있으며, 원래 nodejs의 표준 방식인 commonjs 방식 역시 동일하게 지원하고 있다.

## esmodule 활성화 하기

그러나 앞서 1부에서도 언급한 것처럼 원래 Commonjs방식이 nodejs의 표준이기 떄문에, esmodule 을 활성화 하기 위해서는 nodejs에 esmodule을 사용해야 한다는 것을 알려야 한다. 방법은 크게 세가지다.

- `.mjs`로 파일을 작성하기 (module javascript) -`package.json`에 `type`필드 작성하기 WIP
- `--input-type` 플래그 선언하기

위 세가지 방식 중 하나라도 되어 있지 않다면, nodejs는 commonjs 방식으로 판단하고 commonjs 모듈 로더를 사용한다.

## `import`와 지시자

많은 자바스크립트 개발자들이 알고 있지만, `import`는 보통 `from` 키워드와 함께 널리 사용된다.

```javascript
import { foo } from 'bar'
```

이렇게 `import`를 사용하기 위해서는 모듈을 제공하는 쪽에서 `export` 형태로 외부로 공개해야 하는 모듈을 선언해야 한다. 그리고 이 `from` 뒤에 오는 것은 `specifier` (지시자) 라고 하는데, 크게 3가지 타입의 지시자를 작성할 수 있다.

- 상대경로 지시자: `./config.mjs` `./config.mjs`와 같은 형태이며, 이 경우 `import`를 하고 있는 파일을 기준으로 상대 경로를 추적하여 찾게 된다. 이 상대경로를 사용할 경우 반드시 파일 확장자가 필요하다.
- 일반적인 지시자: `next` `react` 또는 `lodash/sum`과 같은 지시자를 말한다. `next`나 `react` 와 같은 경우에는 일반적으로 알고 있는 것 처럼 패키지 명칭을 가리키게 된다. `lodash/sum` 과 같이 패키지 명을 기준으로 특정 모듈을 가져올 수도 있는데, 이 경우에는 해당 패키지, 즉 `lodash`에서 `exports`를 통해서 `sum`을 내보내는 경우에만 가능하다. 만약 `exports`로 내보내지 않는 경우에는, 상대경로 지시자와 마찬가지로 확장자을 명시해줘야 한다.
- 절대경로 지시자: `file:///opt/nodejs/config.js`와 같이 전체 경로를 명시할 수도 있다.

일반적인 지시자가 노드의 모듈을 찾아가는 방식은 뒤이어 설명할 nodejs의 module resolution 알고리즘을 따른다. 그 외의 경우에는 표준 URL resolution 문법을 따라서 찾아가게 된다.

한가지 명심해야 할 것은 이 URL 기반 지시자 방식 (상대경로, 절대 경로 등) **`import`는 반드시 확장자를 명시해야 한다는 것이다.** 타입스크립트에서 esmodule 방식을 사용하였지만 딱히 확장자를 명시해본적이 없을 텐데, 이는 타입스크립트가 확장자 처리를 다 해주기 때문이다.

### URL

앞서 설명했듯이 esmodule은 URL을 기반으로 모듈을 불러오고 캐싱한다. 따라서 일부 특수문자의 경우에는 `%`를 사용한 인코딩을 사용해야 한다.

현재 nodejs는 `file:` `node:` `data:` 방식의 URL 스킴을 지원하고 있다. 다만 deno와 같이 `https://example.com/app.js' http https 모듈을 불러오는 것은 지원하지 않고 있다.

#### `file:`

웹 문서에서 흔히 같은 파일이지만 새롭게 불러오는 것을 강제하기 위해 쿼리 방식을 사용하는 것을 종종 목격할 수 있는데, `nodejs`도 이와 같은 방법이 가능하다.

```javascript
import './foo.mjs?query=1' // loads ./foo.mjs with query of "?query=1"
import './foo.mjs?query=2' // loads ./foo.mjs with query of "?query=2"
```

위 두 모듈은 완전히 별개의 모듈로 취급되어 두번 불러오게 된다.

> The volume root may be referenced via /, //, or file:///. Given the differences between URL and path resolution (such as percent encoding details), it is recommended to use url.pathToFileURL when importing a path.

#### `data:`

`data: URL` 방식은 다음과 같은 MIME 타입을 지원한다.

- esmodule 을 위한 `text/javascript`
- json을 위한 `application/json`
- wasm을 위한 `application/wasm`

```javascript
import 'data:text/javascript,console.log("hello!");'
```

위 파일을 실행하면 `console.log("hello")`를 실행한 것과 동일한 결과가 반환된다. 즉 `data:text/javascript`로 esmodule을 선언하고, 그 뒤에 해당 코드를 실행하게 된다.

```javascript
import _ from 'data:application/json,{"foo":"bar"}' assert { type: 'json' }

console.log(_) // {"foo":"bar"}
```

위 파일은 `application/json` MIME 타입을 선언하여 json을 파싱한 모습이다. 다만 `assert`라는 구문을 추가로 선언한 것을 볼 수 잇는데, 이는 `16.14.0`과 `17.1.0`에 추가된 실험적 기능으로, 모듈 지시자에 어떤 내용인지 추가적인 정보를 주는 문법 구문이다.

```javascript
const { default: json } = await import('data:application/json,{"foo":"bar"}', {
  assert: { type: 'json' },
})

console.log(json) // {"foo":"bar"}
```

이 방법은 현재 `json` 타입에 대해서만 지원하고 있다.

#### `node:`

`node:`는 nodejs 내장 모듈을 불러오고 싶을 때 사용한다. 흔히 `node:fs`와 같은 방법 대신 `fs`를 사용하여 nodejs 내장 모듈을 많이 불러왔을 텐데, 이는 앞서 소개한 node module resolution algorithm이 작동한 덕분이다. nodejs 모듈을 이러한 알고리즘을 거치지 않고 빠르게 불러오고 싶다면 `node:`를 사용하면 된다.

```javascript
import fs from 'node:fs/promises'
```

## Commonjs와의 호환성

commonjs와 esmodule은 태생부터 목적과 방향성이 다르기 때문에 공존하기 어려운 것이 사실이다. 하지만 nodejs는 다음과 같이 어느 정도의 호환성을 제공해준다.

### `import`

`import`를 사용하면 esmodule과 commonjs 모듈을 불러올 수 있다. 다만 `import`는 esmodule 에서만 지원하지만, 동적 `import()`구문은 commonjs 내부에서도 esmodule을 불러오기 위해서 지원된다.

```js
// test.mjs
export const foo = 'bar'

// main.js
// commonjs 내부에서도 `await import`를 사용할 수 있다.
;(async () => {
  const test = await import('./test.mjs')

  console.log(test.foo) // bar
})()
```

조금더 정확히 이야기하자면, commonjs에서 `import`는 `Promise` 객체를 반환한다.

```javascript
// main.js

const test = import('./test.mjs')
console.log(test) // Promise { <pending> }
```

`import`로 commonjs 모듈을 불러오는 경우, `module.exports` 객체가 `default export`로 간주된다.

```javascript
// test.js
module.exports = {
  foo: 'bar',
}

// main.mjs
import test from './test.js'

console.log(test) // {foo: 'bar'}
```

### `require`

commonjs 모듈 방식인 `require`는 `require`로 불러오는 파일을 항상 `commonjs`라고 간주한다. `require`로 esmodule로 작성된 파일을 불러오면 에러가 발생한다.

```javascript
// test.mjs
export const foo = 'bar'

// main.js
const test = require('./test.mjs')

console.log(test)
// Error [ERR_REQUIRE_ESM]: require() of ES Module /.../test.mjs not supported.
```

이러한 방식이 불가능한 이유는, `commonjs`는 항상 동적으로 (synchronous)하게 모듈을 불러오는 반면, esmodule은 비동기로 실행을 가져가기 때문이다. esmodule로 작성된 파일을 commonjs에서 불러오고 싶다면 `await import`을 사용해야 하는 이유가 이 때문이다.

### commonjs 네임스페이스

라이브러리를 처음 만들어 본 사람들이 제일 혼란 스러워 하는 것이 바로 `default`가 동작하는 방식이다. 결론부터 말하자면, `module.exports`가 esmodule 방식의 default로 동작한다. 다음과 같은 방식의 코드를 이해해 두면 편하다.

```javascript
// test.js
module.exports = {
  foo: 'bar',
}

// main.mjs
import { default as test1 } from './test.js'

import test from './test.js'

console.log(test1 === test) // true
```

이는 nodejs가 제하는 문법적 설탕으로, commonjs 모듈에서 가져오는 `default` sms
