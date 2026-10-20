---
title: 패키지 가이드 2탄) 패키지 제작을 위한 타입 제공 가이드
marp: true
paginate: true
theme: default
tags:
  - typescript
  - npm
date: 2025-03-25
description: '생각보다 잘 모르고 있던 타입스크립트'
published: true
art:
  undraw: markdown-file
---

# 패키지 가이드 2탄) 패키지 제작을 위한 타입 제공 가이드

> with attw cli

<!-- _class: invert deck-cover -->

@yceffort

---

## are the types wrong (aka attw)

- 타입스크립트가 사실상의 표준이 된 지금, 제대로 된 타입을 제공하지 못하면 사용자들로 부터 외면 받을 수 밖에 없음
- `@types/`와 같은 [DefinitelyTyped](https://github.com/DefinitelyTyped/DefinitelyTyped) 패키지를 통해서 제공하는 방법도 있지만, 패키지 크기 나 내용이 너무나 거대해져서 다루기 어렵고, 사용자 역시 원하는 타입이 필요하다면 별도의 설치가 필요하다는 명확한 단점이 존재
- 앞서 1장에서 살펴보았던 것 처럼, 단순히 타입만 제공해서 되는 것이 아니라, 사용자의 모듈 시스템에 맞는 타입을 적재 적소에 배치하는 것이 중요
- `package.json`의 패키지 내용에 대한 타입을 제공하기 위한 가이드, [attw](https://arethetypeswrong.github.io/)

> attw 가 제공하는 지침에 대해서 간단하게 살펴보겠습니다

---

<style scoped>section { font-size: 20px; }</style>

## 💀 Resolution failed

- `package.json`에 선언한 파일 중, 타입스크립트가 이해할 수 없는 확장자가 존재하거나 이해할 수 없는 구조로 배치되어 있는 경우 해당 오류 발생
- `.css` 와 같이 타입스크립트가 해석할 수 없는 파일이지만, 번들러가 적절히 처리할 수 있어 패키지 입장에서 내보내야 하는 파일인 경우에는 오탐으로 인식될 수 있음
- `moduleResolution: node` (`node10`)을 지원하고자 하는 경우, 해당 알고리즘으로는 `package.json`의 `exports` 을 읽을 수 없어 에러가 발생할 수 있음. `moduleResolution: node`을 지원해야 하는 경우, 서브 패스 방식에 대해서 검토해보거나 `typeVersions`을 사용해 볼 수도 있음

```jsonc
{
  "name": "pkg",
  "main": "./dist/index.js",
  "exports": {
    ".": "./dist/index.js",
    // moduleResolution: node 으로는 여기에 닿을 수 없음.
    "./subpath": "./dist/subpath.js",
  },
}
```

---

## ❌ No types

- 자바스크립트 파일을 찾았지만, 이에 해당하는 타입을 찾는 경우 해당 오류 발생
- 타입스크립트는 node_modules 에서 제공하는 자바스크립트 파일을 분석하지 않고, 자바스크립트 파일을 찾으면 해당하는 타입 파일만 찾아서 분석하기에 발생하는 문제

```jsonc
{
  "name": "pkg",
  "main": "./index.js",
  "types": "./index.d.ts",
  "exports": {
    // 이렇게 되어 있다면, 타입 파일인 ./index.m.dts 를 찾으려고 시도할 것임
    // 실제로 파일이 존재한다면 상관없지만, 파일이 존재하지 않는다면 에러 발생
    // 파일이 존재하더라도, nodejs 모듈 알고리즘과 동일하게 타입 파일을 선언해준다면 굳이 fs 로 파일을 찾지 않아도 됨
    "import": "./index.mjs",
    "require": "./index.js",
  },
}
```

---

## 🎭 Masquerading as CJS

<style scoped>section { font-size: 20px; }</style>

- CommonJS 타입 파일을 찾았지만, 해당하는 자바스크립트 파일은 ESModule로 작성되어 있는 경우 해당 오류 발생
- `moduleResolution: node16`이라면
  - `.d.mts` 파일은 `.mjs` 파일을 찾으려고 시도 (ESModule 로 간주)
  - `.d.cts` 파일은 `.cjs` 파일을 찾으려고 시도 (CommonJS 로 간주)
  - `.d.ts` 파일은 `package.json` `type`필드의 값을 보고 결정
- 타입스크립트는 타입스크립트가 만든 선언파일 (.d.ts)과 Node.js 가 실제로 불러온 js 파일 (.js)가 동일한 모듈 형식을 공유한다는 전제로 동작

```json
{
  "name": "pkg",
  "exports": {
    ".": {
      // 타입스크립트가 .d.ts 를 보고, type 필드도 없기 때문에 CommonJS 로 간주
      // 하지만 Node.js 상에서 import 로 모듈을 불러오는 경우, `import` 필드의 확장자를 봄으로 ESModule 로 간주
      // 불일치 발생
      "types": "./index.d.ts",
      "import": "./index.mjs",
      "require": "./index.js"
    }
  }
}
```

---

<style scoped>section { font-size: 14px; }</style>

- `.d.ts`가 다음과 같이 선언되어 있다고 가정

  ```ts
  export declare const a: string
  export declare function b(): number
  ```

- 타입스크립트는 이를 `CommonJS`로 간주하고, 자바스크립트 파일이 다음과 같이 있다고 상상해보자

  ```js
  Object.defineProperty(exports, '__esModule', {value: true})
  exports.a = '...'
  exports.b = function b() {
    /* ... */
  }
  ```

- 그렇다면 Nodejs 에서는 다음과 같이 default import 로 불러올 수 있다고 판단할 것

  ```js
  import mod from 'pkg'
  mod.a
  mod.b()
  ```

- 하지만 실제로는 ESModule 이었고, 원래 파일은 아마 저렇게 작성되어 있을 것

  ```js
  export const a = '...'
  export function b() {
    /* ... */
  }
  ```

- 이 시점에서, 위처럼 코드를 불러올 수 없으므로 에러!

---

<style scoped>section { font-size: 14px; }</style>

- `.d.ts`가 다음과 같이 선언되어 있다고 가정

  ```ts
  declare function hello(): string
  export default hello
  ```

- 타입스크립트는 이를 `CommonJS`로 간주하고, 자바스크립트 파일이 다음과 같이 있다고 추측

  ```js
  Object.defineProperty(exports, '__esModule', {value: true})
  exports.default = function hello() {
    /* ... */
  }
  ```

- Node.js 에서 `CommonJS`의 `default` `import` 는 `module.exports` 객체를 통채로 가져오는 것과 동일합니다. 따라서 위 구조에서 제대로 접근하기 위해서는, 실제로는 아래와 같이 접근해야 함.

  ```ts
  import mod from 'pkg'
  console.log(mod)
  mod.default()
  ```

- 그러나 사실은 ESModule 코드였습니다. 실제 모듈은 아래와 같이 생겼고, 실행 시점에는 에러가 발생.

  ```js
  export default function hello() {
    /* ... */
  }
  // ..
  import mod from 'pkg'
  mod()
  mod.default // undefined
  ```

---

## 👺 Masquerading as ESM

<style scoped>section { font-size: 20px; }</style>

- ESModule 타입 파일을 찾았지만, 해당하는 자바스크립트 파일은 CommonJS로 작성되어 있는 경우 해당 오류 발생
- 에러 발생 과정은 위와 동일 (하지만 반대로 동작)
- 타입스크립트는 CommonJS 파일 환경에서는 ESModule 은 `require`로 불러올 수 없다는 규칙을 적용함 (Node.js 와 동일) 그러나 타입이 잘못되어 있다면, 실제로는 불러올 수 있는 파일을 못불러온다고 거짓 에러 발생

```json
{
  "name": "pkg",
  "type": "module",
  "exports": {
    ".": {
      //  type module 이므로 해당 파일은 ESModule 로 간주.
      // 그러나 require 로 해당 모듈을 불러오면 index.cjs 를 가져오는데 반해, 이에 맞는 타입파일은 존재하지 않음
      "types": "./index.d.ts",
      "import": "./index.js",
      "require": "./index.cjs"
    }
  }
}
```

---

## ⚠️ Entrypoint is ESM-only

- 앞선 경고 👺 Masquerading as ESM 와는 다르게, 타입 파일과 실제 구현한 파일 모두 ESModule 임에도 불구하고, CommonJS 방식으로 불러오려고 해서 발생하는 문제
- 즉 모듈 자체가 ESModule 로만 작성되어 있어, CommonJS 환경에서는 쓸수 없다는 뜻
- 잘못되었다기 보다는 사용자에게 경고를 해줘야 한다는 의미
- <https://gist.github.com/sindresorhus/a39789f98801d908bbc7ff3ecc99d99c>

---

## 🐛 Used fallback condition

<style scoped>section { font-size: 18px; }</style>

- Node.js 알고리즘에 따르면, `exports` 객체 내에서 `import` 하고 싶은 키를 찾은 경우 실제 파일이 있건 없건 더 검색하는 동작을 해서는 안되는데 (= 그냥 거기서 끝내야 하는데), [타입스크립트는 이 부분을 잘못 수행 하고 있어 발생하는 문제](https://github.com/microsoft/TypeScript/issues/50762)

```json
{
  "exports": {
    ".": {
      "types": {
        "foo": "./didnt-match.d.ts"
      },
      "import": {
        "types": "./doesnt-exist.d.ts",
        "default": "./exists.mjs"
      }
    }
  }
}
```

1. 해당 패키지의 엔드리에 대한 타입파일을 찾을려고 시도
2. `types`를 찾음. 그러나 `.`가 없고 `foo` 만 있으므로 실패
3. `import`를 찾음. `types`를 찾음. 여기가 원래 찾으려던 위치가 맞음.
   - `Node.js`는 이 파일이 있건 말건 그냥 여기서 끝내야함
   - 타입스크립트는 해당하는 파일이 없으면 fallback 동작을 함 > 다음 default 를 통해서라도 찾으려고 함 (`./exists.m.dts`)
   - 그러나 이는 잘못된 동작임

---

## ⁉️ 퀴즈

<style scoped>section { font-size: 18px; }</style>

- 아래 상황에서는 무슨 버그가 존재할 수 있을까요?

```json
{
  "name": "pkg",
  "main": "./index.js",
  "types": "./index.d.ts",
  "exports": {
    "import": "./index.mjs",
    "default": "./index.js"
  }
}
```

- `import` 조건에 일치한다면, `.mjs` 규칙에 따라 `.m.dts`를 찾으려고 시도합
- `import` 조건에 걸렸으므로 파일 존재 유무와 관계 없이 `index.m.dts`를 토대로 시도하고, 없다면 그냥 에러를 발생시켜야 함.
- 그러나 타입스크립트 버그로, `default`까지 찾으려고 시도하게 됨
- `index.js`가 있고, 이에 해당하는 `index.d.ts`까지 찾을 수 있어 해당 파일을 사용
- 그러나 실제로는 `index.d.ts`는 CommonJS 타입 파일이므로, 결과적으로 에러가 발생

---

<style scoped>section { font-size: 16px; }</style>

## 🤨 CJS default export

- ESModule 로 작성된 모듈을 CommonJS 로 옮겨온다면, `default`를 다음과 같이 해석해서 가져온다.

```js
Object.defineProperty(exports, "__esModule", { value: true });
// default 키워드 유지를 위함
exports.default = /* ... */;
```

```js
import mod from 'pkg'
console.log(mod)
```

- 번들러는 `__esModule`가 있다면, 위 구문에 대해서 `[Function: f]`를 가져오지만 (미리 암묵적으로 약속된 동작), Node.js 는 그런거 없이 `{ default: [Function: f] }`를 가져온다.
- 패키지를 번들러에서 사용하는지, `Node.js`에서 직접 사용하는지에 따라서 다른 결과를 초래
- 위와 같은 상황은 `ESModule`로 작성된 패키지를 특별한 조치 없이 `CommonJS` 로 트랜스파일 하는 경우 발생할 수 있음
  - 가능한 `default`대신 `named`로 내보내는 것을 권장
  - 꼭 필요하다면, 다음과 같은 내보내기를 할 수 있도록 번들

    ```js
    Object.defineProperty(exports, '__esModule', {value: true})
    function f() {
      /* ... */
    }
    module.exports = f
    module.exports.default = f
    ```

---

## ❗️ Incorrect default export

<style scoped>section { font-size: 20px; }</style>

- `CommonJS`의 default export `module.exports = something` 는 내보내기 자체가 `something`이라는 뜻임
- 만약 그에 해당하는 타입을 `export default something` 는 `default` 에 `something`을 할당하는 것이라는 뜻임
- 올바르게 타입을 선언하고 싶다면, 아래와 같은 구조로 작성해야함.

  | JavaScript syntax                                | Type declaration syntax |
  | ------------------------------------------------ | ----------------------- |
  | `module.exports = x`                             | `export = x`            |
  | `exports.default = x; exports.__esModule = true` | `export default x`      |
  | `export default x`                               | `export default x`      |

---

## ❓ Missing `export =`

- 앞서 소개한 내용처럼, default export 호환을 위해서 `module.exports` 와 `module.exports.default`는 동일한 값으로 설정
- 하지만 타입파일에는 `export default`만 선언한다면, `module.exports`가 있다는 사실을 제대로 알려주지 못해서 불필요한 workaround 가 발생할 수 있음
  - `mod` 로 바로 접근해도 되는데, `mod.default`로 접근하는 불편함 발생
- `export = something` 을 사용해서 CommonJS 를 올바르게 반영해야 하고, 추가로 병합해야할 내용들이 있으면 이것 또한 점검해서 확인 필요 (뒤이어 예제에서 설명)

---

## 예제로 확인해보기 (1)

<style scoped>section { font-size: 18px; }</style>

```js
// my-class.js
class MyClass {
  constructor() {
    console.log("I'm MyClass!")
  }

  doSomething() {
    console.log('Doing something...')
  }
}

// CommonJS 주 내보내기
module.exports = MyClass

// 번들러/호환성 위한 패턴: default를 자기 자신으로
MyClass.default = MyClass
```

```ts
// my-class.d.ts
declare class MyClass {
  // `static default: typeof MyClass;`
  // -> 런타임에서 MyClass.default = MyClass; 구조를 반영
  static default: typeof MyClass

  doSomething(): void
}

// `export = MyClass;` -> "이 모듈이 CJS로 MyClass를 내보낸다"는 뜻
export = MyClass
```

---

## 예제로 확인해보기 (2)

<style scoped>section { font-size: 12px; }</style>

```js
// my-lib.js
class Widget {
  constructor(name) {
    this.name = name
  }

  render() {
    console.log(`Rendering ${this.name}`)
  }
}

// 런타임에서도 추가 정보를 담고 싶다면, 예: Widget.WidgetProps 객체
Widget.WidgetProps = {
  color: 'string',
  size: 'number',
}

Widget.default = Widget
module.exports = Widget
```

```ts
// my-lib.d.ts
declare class Widget {
  static default: typeof Widget // 런타임에서 Widget.default = Widget;

  constructor(name: string)
  render(): void
}

// "네임스페이스 병합"을 통해, Widget에 붙는 추가 타입을 함께 선언
declare namespace Widget {
  // 인터페이스 선언 예시
  export interface WidgetProps {
    color: string
    size: number
  }

  // (만약 런타임에도 Widget.WidgetProps = {...}가 있다면  타입 시스템에서 이를 객체 타입으로 선언하는 것도 가능)
  // const WidgetProps: {
  //   color: string;
  //   size: number;
  // };
}

// CJS export
export = Widget
```

---

## `🚭 Unexpected module syntax`

- CommonJS 로 인식되는 파일 내부에 `import` 구문이 발생하거나, ESModule 로 인식되는 구문에 `require`가 존재할 때 발생하는 문제
- 타입필드와 확장자로 모듈 시스템을 구별하지만, 정작 코드 내부가 잘못되어 있는 경우 발생함
- 이러한 문제는 '프론트엔드 번들러' 에서만 동작하는 것을 목표로하는 패키지에서 자주 나타남
  - 프론트엔드 번들러는, Nodejs 의 모듈 알고리즘을 엄격하게 지키지 않고 번들링하기 때문에, 이러한 문제가 발생할 수 있음.
    - 예: (`type: module`이 없는데) js 파일이고, 내부에서 `import`를 사용하는 경우
  - 또는 node 구형 버전을 타겟으로 번들되어 있는 경우

---

## `🥴 Internal resolution error`

- 타입 선언 파일 내부에서 발견한 `import` 구문이 정상적으로 해석되지 않은 경우 발생하는 에러.
- 실제 자바스크립트 파일은 존재하지만, 타입파일이 존재하지 않는 경우에 많이 발생
- 타입스크립트는 '타입파일과 실제 자바스크립트 파일 구조가 일치한다' 라는 전제하에 타입 체크를 수행하기 때문에, 이러한 문제가 발생할 수 있음
- 이런 문제는 `.d.ts`파일과`.js` 파일이 서로 다른 빌드 과정으로 생성되서 불일치가 발생하거나, 수동으로 `.d.ts` 파일을 작성하는 과정에서 종종 발생함
  - `@naverpay/pite` : vite 로 js를 빌드하고 tsup 으로 d.ts를 빌드하기 때문에 자칫하면 이런 문제가 발생할 수 있음
  - `react`: 리액트는 `flow`기반으로 작성되어 있어, `@types/react`를 직접 작성하는데 이 경우에도 발생할 수 있음.

---

## `🕵️‍♂️ Named exports`

- `CommonJS`로 작성된 모듈을 `ESModule`환경에서 named exports 를 사용할 수 있다고 가정하지만, 실제로 파일 실행 시점에는 해당 내보내기 가 존재하지 않는 경우 발생. 이러한 문제가 발생하는 이유는 다음과 같이 크게 두가지
  - 자바스크립트에 없는 `export`를 타입에 선언했을 때에 발생하는 문제
  - 또는 [cjs-module-lexer](https://www.npmjs.com/package/cjs-module-lexer)가 분석할 수 없는 형태로 작성된 `CommonJS` 파일을 `ESModule`에서 `import`하려고 할 때 발생하는 문제
    - `cjs-module-lexer`: `ESModule`이 도입되면서 `CommonJS`를 `ESModule`에서 어떻게 읽어올 것인지에 대한 고민이 있었는데, 이를 해결하기 위해 만들어진 syntax lexer 라이브러리로, `CommonJS` 모듈 내에서 무엇을 내보내는지 분석할 수 있게 도와주는 도구

---

## `cjs-module-lexer`가 분석할수 없는 코드가 뭔데?

<style scoped>section { font-size: 20px; }</style>

```js
// module.cjs
const a = 'I am a'
const b = 'I am b'

module.exports = {
  a,
  b,
}
// index.mjs
import {a} from './module.cjs'
console.log('Value of a:', a) // ok
```

> 최선의 노력으로 `module.exports`에 할당된 객체를 파싱하지만, 실제 파서가 아니므로 모든 경우를 처리할수 없으며 처리할 수 없는 코드가 발생해버리면 그냥 포기 (bailout) 해버립니다.

```js
// module.cjs
module.exports = {
  a: 'I am A!',
  b: 'I am B!',
}
// index.mjs
import {a} from './module.cjs'
// Named export 'a' not found..??
console.log('Value of a:', a)
```

---

<style scoped>section { font-size: 18px; }</style>

```js
// module.cjs
module.exports.a = 'I am a'
module.exports.b = 'I am b'
// index.mjs
import {a} from './module.cjs'
console.log('Value of a:', a) // ok
```

- `module.exports.a`는 `exports` 라는 객체에 `a`를 붙인다는 것을 정적으로 분석 가능하므로 파악 가능
- 근데 문자열은 왜 안됐을까? 🤔
- 문자열을 만나면 (`"`) `cjs-module-lexer`는 이후 분석을 포기해버림
  - 빠르게 동작하는 것을 목표로 만들어져서, AST 와 같이 추상구문트리로 분석하는 것이 아니라, 단순히 토큰단위로 분석하는 정도의 도구 이기 때문
  - `"`를 만났다? > 뒤에 `+`와 같은 연산이 있을수도 있다? > 그럼 그냥 분석열 여기서 부터 포기해버리자

```js
// module.cjs
const a = 'I am a'
const b = 'I am b'

module.exports = {
  a,
  c: 'hello',
  b,
}
// index.mjs
import {a, b} from './module.cjs'
// Named export 'b' not found
console.log('Value of a:', a, b)
```

---

## 요약

<style scoped>section { font-size: 22px; }</style>

- `package.json`의 경로마다 `exports`내부에서 타입을 한방에 바로 찾을 수 있도록 경로 설정 및 타입 파일 선언을 잘 해줘야 한다.
- `export default`와 `module.exports = {}` 의 차이를 극복하고자 다양한 트릭들이 사용되고 있다. 그러나 이러한 트릭들은 이해하기도 까다롭고 다루기도 번거로우므로 그냥 가급적 default exports 를 자제하고 named exports 를 사용해서 내보내는 것이 더 깔끔하고 정신건강에 이롭다.
- Node.js 버전의 최소버전을 ESModule 을 정상적으로 지원하는 16 이상으로 올리는 것이 권장되며, 타입스크립트의 `moduleResolution` 도 `node`가 아닌 `node16` `nodenext`, `bundler` 등으로 사용하는 것이 좋다. 이렇게 해야 타입스크립트와 Node.js 사이에 모듈을 해석하는 방식을 통일할 수 있고, `ESModule`을 읽는 방식을 더욱 엄격하게 수행할 수 있다.
- 타입스크립트와 자바스크립트 빌드 과정을 통합하고 확장자 및 환경에 맞는 모듈 타입을 제공하는 것이 중요하다.
- [@naverpay/pite](https://github.com/NaverPayDev/pite) 많관부!
