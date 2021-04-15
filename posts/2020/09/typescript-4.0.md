---
title: 'Typescript 4.0 릴리즈 노트 '
tags:
  - typescript
published: true
date: 2020-09-21 13:33:17
description: '한 발 늦었지만 타입스크립트 4.0에서 추가된 기능을 알아보자'
category: typescript
template: post
---

## Table of Contents

출처는 [여기](https://devblogs.microsoft.com/typescript/announcing-typescript-4-0/)

## 가변 튜플

튜플은 원소의 수와, 각 원소의 타입이 정확히 지정된 배열의 타입을 지정하고 싶을 때 사용하는 방법이다.

```typescript
const hello: [string, number] = ['yc', 33]
```

먼저 tuple 타입 신택스에 있는 spread 연산자 `...`가 제네릭하게 될 수 있다. (외부에서 타입을 지정할 수 있다.) 블로그에서 나온 예시로는, `tail`함수를 예로 들고 있다.

```typescript
// 이 ... 이 3.x 버전 이하에서는 에러가 났지만, 4.0 부터는 쓸 수 있게 되었다.
function tail<T extends any[]>(arr: readonly [any, ...T]) {
  const [_ignored, ...rest] = arr
  return rest
}

const myTuple = [1, 2, 3, 4] as const
const myArray = ['hello', 'world']

// type [2, 3, 4]
const r1 = tail(myTuple)

// type [2, 3, 4, ...string[]]
const r2 = tail([...myTuple, ...myArray] as const)
```

[예제코드](https://www.typescriptlang.org/play?ts=4.0.2#code/GYVwdgxgLglg9mABFAhjANgHgCqIKYAeUeYAJgM6IpgCeA2gLoB8AFCgE7sBci7eKpBOhqI61GgBpEAOlnYGASkQBvAFCINiCAnJRRAfRgBzMHD6kps6X10NEAXiqcA3Os18oIdkhtRXAX1VVbTBdRABbGmwQAAd0PAdRAEYpACYpAGYpABY7FEoQ3VdCvUiAQU4UEUc6ACIACzx0dDhaqVqAdzN0UlqGV1UAekHkGhiEunTELMRc4J09diTE1AwWSOi4vAUB4dHx0SmZ7MtZXXYYMCNGBnnQxdSVtHQWOisN2PjT6XLKmjyCgsdkA)

또한 tuple 내부에서 전개 연산자를 어디서든 쓸 수 있다.

```typescript
type Strings = [string, string]
type Numbers = [number, number]

// [string, string, number, number, boolean]
type StrStrNumNumBool = [...Strings, ...Numbers, boolean]
```

아마도 함수 합성을 하는데 있어서 중요한 역할을 할 것으로 보인다. 블로그에 예제에서 보여준 것처럼, 자바스크립트의 `bind`함수에 타입체크를 할 때도 많이 이용될 것으로 보인다.

[다양한 예제들](https://github.com/microsoft/TypeScript/pull/39094)

```typescript
// Variadic tuple elements

type Foo<T extends unknown[]> = [string, ...T, number]

type T1 = Foo<[boolean]> // [string, boolean, number]
type T2 = Foo<[number, number]> // [string, number, number, number]
type T3 = Foo<[]> // [string, number]

// Strongly typed tuple concatenation
function concat<T extends unknown[], U extends unknown[]>(
  t: [...T],
  u: [...U],
): [...T, ...U] {
  return [...t, ...u]
}

const ns = [0, 1, 2, 3] // number[]

const t1 = concat([1, 2], ['hello']) // [number, number, string]
const t2 = concat([true], t1) // [boolean, number, number, string]
const t3 = concat([true], ns) // [boolean, ...number[]]

// Inferring parts of tuple types
declare function foo<T extends string[]>(...args: [...T, () => void]): T

foo(() => {}) // []
foo('hello', 'world', () => {}) // ["hello", "world"]
foo('hello', 42, () => {}) // Error, number not assignable to string

// Inferring to a composite tuple type
function curry<T extends unknown[], U extends unknown[], R>(
  f: (...args: [...T, ...U]) => R,
  ...a: T
) {
  return (...b: U) => f(...a, ...b)
}

const fn1 = (a: number, b: string, c: boolean, d: string[]) => 0

const c0 = curry(fn1) // (a: number, b: string, c: boolean, d: string[]) => number
const c1 = curry(fn1, 1) // (b: string, c: boolean, d: string[]) => number
const c2 = curry(fn1, 1, 'abc') // (c: boolean, d: string[]) => number
const c3 = curry(fn1, 1, 'abc', true) // (d: string[]) => number
const c4 = curry(fn1, 1, 'abc', true, ['x', 'y']) // () => number
```

## 튜플 요소에 이름 지정하기

튜플에 이름을 지정할 수 있다.

```typescript
type Range = [start: number, end: number]
type Foo = [first: number, second?: string, ...rest: any[]]
```

대신 하나라도 이름을 지정하면, 그 뒤에도 주르륵 이름을 지정해야 한다.

```typescript
type Bar = [first: string, number]
// error! Tuple members must all have names or all not have names.
```

## 생성자에서 클래스 프로퍼티 추론

`noImplicitAny` 속성이 켜져 있다면, 아래와 같은 추론이 가능해진다.

```typescript
class Square {
  // Previously: implicit any!
  // Now: inferred to `number`!
  area
  sideLength

  constructor(sideLength: number) {
    this.sideLength = sideLength
    this.area = sideLength ** 2
  }
}
```

그리고 위의 정보만으로는 추론할 수 없는 경우에는 `undefined`에러가 난다.

```typescript
class Square {
  sideLength

  constructor(sideLength: number) {
    if (Math.random()) {
      this.sideLength = sideLength
    }
  }

  get area() {
    return this.sideLength ** 2
    //
    // error! Object is possibly 'undefined'.
  }
}
```

사실 저렇게 하는 것보다, 명확하게 타입을 선언하고 초기화 해주는 것이 낫다.

## 간략 할당 연산자

자바스크립트에는 아래와 같이 간략하게 할당하는 연산자가 존재한다.

```javascript
// Addition
// a = a + b
a += b

// Subtraction
// a = a - b
a -= b

// Multiplication
// a = a * b
a *= b

// Division
// a = a / b
a /= b

// Exponentiation
// a = a ** b
a **= b

// Left Bit Shift
// a = a << b
a <<= b
```

새로운 ECMAScript의 표준에 맞춰, 타입스크립트에도 `&&=` `||=` `??=` 도 지원하게 된다.

```typescript
a = a && b
a = a || b
a = a ?? b
```

```typescript
const obj = {
  get prop() {
    console.log('getter has run')

    // Replace me!
    return Math.random() < 0.5
  },
  set prop(_val: boolean) {
    console.log('setter has run')
  },
}

function foo() {
  console.log('right side evaluated')
  return true
}

console.log('This one always runs the setter')
obj.prop = obj.prop || foo()

console.log('This one *sometimes* runs the setter')
obj.prop ||= foo()
```

## catch binding에 unknown사용

이전 까지는 에러 객체 타입이 `any`여서 아무렇게나 사용할 수 있었다. 그러나 이제 `unknown`으로 간주되면서, 보다 안전하게 사용될 수 있다.

```typescript
try {
  // ...
} catch (e) {
  // error!
  // Property 'toUpperCase' does not exist on type 'unknown'.
  console.log(e.toUpperCase())

  if (typeof e === 'string') {
    // works!
    // We've narrowed 'e' down to the type 'string'.
    console.log(e.toUpperCase())
  }
}
```

그러나 이 기능은 개발자들에게 익숙해기 전까지 `--strict`모드를 켜놔야만 적용된다. 조만간 이를 위한 lint룰도 추가될 예정이다.

## Custom JSX Factory

JSX에서 [fragment](https://reactjs.org/docs/fragments.html)란 자식 엘리먼트 여러개를 반환하는 JSX 엘리먼트 타입을 의미한다. 과거 타입스크립트에 프래그먼트를 추가할 때는 별로 이에 대한 고민이 이뤄지지 않았지만, 현재는 많은 라이브러리들이 JSX를 사용하며 이에 관련된 API의 지원도 늘려가고 있다.

타입스크립트 4.0부터 `jsxFragmentFactory` 옵션을 사용해서 프래그먼트 팩토리를 커스터마이징 할 수 있다.

이렇게 하면 대신 React의 `Fragment`대신 `Fragment`를, `createElement`대신 `h`를 사용해야 한다.

`tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "esnext",
    "module": "commonjs",
    "jsx": "react",
    "jsxFactory": "h",
    "jsxFragmentFactory": "Fragment"
  }
}
```

파일마다 JSX 팩토리를 다르게 사용하려면 `/** @jsxFrag */` 주석을 함께 사용해야 한다. 아래 예를 보자.

```typescript
// Note: these pragma comments need to be written
// with a JSDoc-style multiline syntax to take effect.
/** @jsx h */
/** @jsxFrag Fragment */

import { h, Fragment } from 'preact'

let stuff = (
  <>
    <div>Hello</div>
  </>
)
```

위 코드는 아래 처럼 빌드 된다.

```javascript
// Note: these pragma comments need to be written
// with a JSDoc-style multiline syntax to take effect.
/** @jsx h */
/** @jsxFrag Fragment */
import { h, Fragment } from 'preact'
let stuff = h(Fragment, null, h('div', null, 'Hello'))
```

## `--noEmitOnError` 빌드 시에 속도 향상

이전 버전 까지는 `----incremental`를 `--noEmitOnError`와 함께 사용하면 빌드가 매우 느렸다. 이는 `--noEmitOnError` 플래그를 사용하면, 이전 컴파일 결과에 대한 정보가 `.tsbuildinfo`파일에 캐싱되지 않았기 때문이다. 그리고 이를 해결해서 속도를 향상 시켰다.

## `--incremental`과 `--noEmit`

위와 동일

## VSCode 에디터 지원 개선

VSCode에서 원하는 타입스크립트 버전을 선택할 수 있도록 한다.

## 옵셔널 체이닝 지원

- 옵셔널 체이닝 지원
- null 병합연산자 지원

## `/** @deprecated */` 지원

JSDoc 에서 `/** @deprecated */`를 사용할 수 있도록 지원한다.

## 에디터 시작시 파셜 시맨틱 모드 지원

에디터 실행시간이 길다는 유저들의 불만이 많았다. 이를 해결 하기 위해 언어 지원 서비스 전체가 로딩되기 전에 바로 현재 파일에서 활용할 수 있는 부분 시맨틱 모드를 추가했다. 이는 에디터를 실행했을 때 현재 파일 만이라도 언어지원 서비스를 지원하는 것이다.

좌측 (구버전)과 우버전 (신버전) 을 비교하면 확연한 속도 차이를 느낄 수 있다.

https://devblogs.microsoft.com/typescript/wp-content/uploads/sites/11/2020/08/partialModeFast.mp4

## 더 똑똑해진 자동 임포트

타입스크립트로 작성된 라이브러리라 할지라도, 프로젝트에 한번도 로드 되지 않은 임포트 문이 자동으로 불러와지지 않는 문제가 있었다. 이는 자동 로드 기능이 해당 프로젝트에 한번이라도 로드된 패키지들을 대상으로만 동작하기 때문이다.

를 보완하기 위해, `package.json`의 `dependencies`와 `peerDependencies`를 처리하는 로직을 따로 추가했으며, 이 정보는 auto import 를 위해서만 사용된다.

## Breaking Changes

### `lib.d.ts` 수정

`lib.d.ts`가 수정되었다. 특히, DOM 타입 관련된 부분의 수정이 있었다. 그중 가장 유의깊게 봐야할 것은 [document.origin](https://developer.mozilla.org/en-US/docs/Web/API/Document/origin)대신에 [self.origin][https://developer.mozilla.org/en-us/docs/web/api/windoworworkerglobalscope/origin]를 사용하는 것이다.

### 프로퍼티, 게터, 세터 오버라이딩 금지

이전 버전까지는 `useDefineForClassFields`가 있어야만 오류로 처리했지만, 이제부터는 이 옵션이 없어도 에러가 발생한다. 이는 상속관계에서만 발생한다.

```typescript
class Base {
  get foo() {
    return 100
  }
  set foo() {
    // ...
  }
}

class Derived extends Base {
  foo = 10
  //  ~~~
  // error!
  // 'foo' is defined as an accessor in class 'Base',
  // but is overridden here in 'Derived' as an instance property.
}
```

```typescript
class Base {
  prop = 10
}

class Derived extends Base {
  get prop() {
    //  ~~~~
    // error!
    // 'prop' is defined as a property in class 'Base', but is overridden here in 'Derived' as an accessor.
    return 100
  }
}
```

### 옵셔널 항목만 delete 가능

`strictNullChecks`가 켜져 있는 상태에서 delete를 사용하면, 그 대상이 반드시 `any` `unknown` `never`이거나 옵셔널 항목이어야 한다. 그렇지 않으면 에러가 발생한다.

```typescript
interface Thing {
  prop: string
}

function f(x: Thing) {
  delete x.prop
  //     ~~~~~~
  // error! The operand of a 'delete' operator must be optional.
}
```

### Node Factory 가 deprecated 되었다

AST(추상 구문 트리) 노드 생성을 위해, 팩토리 함수를 제공했지만, 이제는 새로운 API 형태로 노드 팩토리를 제공한다. 자세한 내용은 [여기](https://github.com/microsoft/TypeScript/pull/35282)를 참고
