---
title: '2부) esmodule은 무엇인가?'
tags:
  - nodejs
published: false
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

- `.mjs`로 파일을 작성하기 (module javascript)
  - 반대로 `.cjs`는 commonjs로 인식된다.
- `package.json`에 `type`필드에 `module`이라고 넣어주기
  - 이 필드에 가능한 값은 `module` 밖에 없다.
- `--input-type=module` 플래그 선언하기
  - 이 플래그는 `commonjs` `esmodule` 두가지를 인식하며, 기본값은 `commonjs`다.

위 방식중 하나라도 되어 있다면, nodejs는 esmodule로 인식하고 esmodule 방식으로 동작한다. 반ㄴ대의 경우에는 당연하게도 commonjs로 인식된다.

## `import`와 지시자

많은 자바스크립트 개발자들이 알고 있는 것 처럼, `import`는 보통 `from` 키워드와 함께 널리 사용된다.

```javascript
import { foo } from 'bar'
```

@@ -58,15 +62,15 @@ import { foo } from 'bar'

- 절대경로 지시자: `file:///opt/nodejs/config.js`와 같이 전체 경로를 명시할 수도 있다.

일반적인 지시자가 노드의 모듈을 찾아가는 방식은 뒤이어 설명할 nodejs의 module resolution 알고리즘을 따른다. 그 외의 경우에는 표준 URL resolution 문법을 따라서 찾아가게 된다.

한가지 명심해야 할 것은 이 URL 기반 지시자 방식 (상대경로, 절대 경로 등) **`import`는 반드시 확장자를 명시해야 한다는 것이다.** 타입스크립트에서 esmodule 방식을 사용하였지만 딱히 확장자를 명시해본적이 없을 텐데, 이는 타입스크립트가 확장자 처리를 다 해주기 때문이다. 일반적인 `import`에서는 확장자가 반드시 필요하다.

### URL

앞서 설명했듯이 esmodule은 URL을 기반으로 모듈을 불러오고 캐싱한다. 따라서 일부 특수문자의 경우에는 `%`를 사용한 인코딩을 사용해야 한다.

현재 nodejs는 `file:` `node:` `data:` 방식의 URL 스킴을 지원하고 있다. 다만 deno와 같이 `https://example.com/app.js' http https 모듈을 불러오는 것은 지원하지 않고 있다. 그리고 일반적인`URL`과 마찬가지로, 쿼리와 fragment가 다르면 다른 모듈로 인식한다.

#### `file:`

웹 문서에서 흔히 같은 파일이지만 새롭게 불러오는 것을 강제하기 위해 쿼리 방식을 사용하는 것을 종종 목격할 수 있는데, `nodejs`도 이와 같은 방법이 가능하다.
@@ -182,21 +186,57 @@ console.log(test)
이러한 방식이 불가능한 이유는, `commonjs`는 항상 동적으로 (synchronous)하게 모듈을 불러오는 반면, esmodule은 비동기로 실행을 가져가기 때문이다. esmodule로 작성된 파일을 commonjs에서 불러오고 싶다면 `await import`을 사용해야 하는 이유가 이 때문이다.

### commonjs 네임스페이스

commonjs 모듈은 `module.exports`로 구성되어 있으며, 이를 활용하면 어떠한 타입의 값이든 내보낼 수 있다. 그리고 esmodule에서 commonjs 을 불러올때는 다음과 같은 방법을 사용하면 된다.

```javascript
// 아래 두 방식은 모두 동일하다.
import { default as cjs } from 'cjs'
import cjsSugar from 'cjs'

console.log(cjs)
console.log(cjs === cjsSugar)
```

위 예제에서 볼 수 있듯, commonjs 모듈은 esmodule 환경에서 `default`가 바로 `module.exports`를 가리키게 된다. 이는 `*`을 사용할 때도 마찬가지다.

```javascript
import * as test from './test.js'

console.log(test)
console.log(test === await import from './test.js')
```

자바스크립트 생태계 내부에서, 기존의 코드와 호환성을 높이기 위해서, nodejs는 정적 분석 프로세스를 사용하여, `esmodule`내부에서 불러오는 모든 commonjs 모듈의 이름을 확인하여 분석한다. commonjs는 앞서 1부에서 알아보았듯, 동적으로 export 할 수 있는 특징 떄문에 실제로 무엇이 export 되는지는 실행단계에서만 알 수 있다는 단점이 있었다. 그러나 esmodule은 정적 분석을 활용하여 실제 실행전에 export 되어있는 것이 있는지 확인하는 절차를 거치는데, 이를 commonjs에서도 적용했다고 보면 된다.

```javascript
// test.cjs
module.exports = {
  hello: 'world',
}

// index.mjs
// 실제 없는 모듈을 import 함
import { hi } from './test.cjs'

console.log(hi)
```

위 코드를 실행하면, 실행되기전에 nodejs가 정적 분석으로 모듈이 없다는 것을 알아채고 에러를 뱉는다.

```text
SyntaxError: Named export 'hi' not found. The requested module './test.cjs' is a CommonJS module, which may not support all module.exports as named exports.
CommonJS modules can always be imported via the default export, for example using:
```

## esmodule 과 commonjs 의 차이점

- `require`, `exports` `module.exports`가 없다.
- `__filename` `__dirname` 이 없다. (module wrapper 가 없으므로)
- [Addons](https://nodejs.org/api/addons.html)가 불가능하다.
- `require.resolve`가 없다.
  - 이것을 대신하기 위해서는, `import.meta.resolve`를 사용하면 된다.
  - `new URL('./local', import.meta.url)`
- `NODE_PATH`가 없다.
- `require.extensions` 이 없다.
- `require.cache`가 없다.
