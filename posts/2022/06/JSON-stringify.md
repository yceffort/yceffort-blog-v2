---
title: 'JSON.stringify 만들어보기'
tags:
  - javascript
  - typescript
published: true
date: 2022-06-17 12:19:04
description: 'V8로는 아니더라도 내부 동작 직접 구현해보기'
---

## Table of Contents

## JSON이 지원하는 타입

JSON 무려 [공식 홈페이지](https://www.json.org/json-en.html)가 존재하는데, 여기서 어떤 데이터 타입을 지원하는지 나와있다. JSON은 우리가 매일 쓰고 또 그다지 어렵지 않기 때문에 그렇게 복잡하게 생각해본적이 없는데, 공식 문서의 그래프를 보면 살짝 어지러워진다. 이래저래 읽는게 귀찮고 복잡하므로, 타입스크립트로 간단하게 요약해보자면 다음과 같다.

```typescript
type JSONType =
  | null
  | boolean
  | number
  | string
  | JSONType[]
  | { [key: string]: JSONType }
```

JSON은 언어에 종속적이지 않기 때문에, 자바스크립트에만 있는 고유의 타입, `undefined` `Symbol` `BigInt` 등과 `Function` `Class` `Map` 등도 지원하지 않는다.

## 현기증 나는 `JSON.stringify`

`JSON.stringify`를 계속 쓰다보면, 이 함수의 동작은 참 일관적이지 않다는 것을 깨닫게 된다.

```typescript
JSON.stringify(1) // '1'
JSON.stringify(null) // 'null'
JSON.stringify('foo') // '"foo"'
JSON.stringify({ foo: 'bar' }) // '{"foo":"bar"}'
JSON.stringify(['foo', 'bar']) // '["foo","bar"]'
```

### JSON이 지원하지 않는 타입은 undefined

여기까지는 우리가 모두 이해하는 수준이다. 그러나 앞서 언급했던, `JSON`이 지원하지 않는 일부 타입에 대해서는 다음과 같이 반환된다.

```typescript
JSON.stringify(undefined) // undefined
JSON.stringify(Symbol('foo')) // undefined
JSON.stringify(() => {}) // undefined
```

모두 `undefined`가 나온다면 그래도 행복할 것 같다. 그러나

### Map, Regex, Set은 빈 JSON

```typescript
JSON.stringify(/foo/) // '{}'
JSON.stringify(new Map()) // '{}'
JSON.stringify(new Set()) //'{}'
```

....?

### Array와 Object 내부에 지원하지 않는 타입이 있는 경우

더 골 때리는 것은 serialize가 가능한 값, 예를 들어 array나 object에서 더 일관성 없이 동작한다는 것이다. `undefined` `Symbol` `Function` 이 배열안에 있으면 `'null'`로 변환된다. 그리고 객체 안에 속성이 있다면 그 속성 전체는 완전히 무시되고 빈 객체 (정확히는 빈 JSON) 가 된다.

```typescript
JSON.stringify([undefined]) // '[null]'
JSON.stringify({ foo: undefined }) // '{}'

JSON.stringify([Symbol()]) // '[null]'
JSON.stringify({ foo: Symbol() }) // '{}'

JSON.stringify([() => {}]) // '[null]'
JSON.stringify({ foo: () => {} }) // '{}'
```

이와 다르게, `Map` `Set` `Regex`가 배열이나 객체 내부에 있다면, 이들은 모두 일관되게 `{}`으로 변환된다. 그리고, 당연히 값도 날아간다.

```typescript
JSON.stringify([/foo/]) // '[{}]'
JSON.stringify({ foo: /foo/ }) // '{"foo":{}}'

JSON.stringify([new Set()]) // '[{}]'
JSON.stringify({ foo: new Set() }) // '{"foo":{}}'

JSON.stringify([new Map()]) // '[{}]'
JSON.stringify({ foo: new Map() }) // '{"foo":{}}'
```

### BigInt와 순환참조는 throw error

여기에 추가로, `BigInt`가 내부에 오게 되면 `TypeError`를 리턴하게 된다.

```typescript
bigint = BigInt(9007199254740991)
JSON.stringify(bigint) //  Uncaught TypeError: Do not know how to serialize a BigInt
```

그리고 우리가 잘 알고 있는 것 처럼, 순환참조를 하는 객체의 경우에도 에러가 난다.

```typescript
const foo = {}
foo.a = foo

JSON.stringify(foo) // Uncaught TypeError: Converting circular structure to JSON
```

한가지 유념에 두어야 할 것은, `BigInt`와 `Cyclic Object` 이 딱 두가지 경우에만 error를 던진다. `JSON.stringify`는 우리가 아는 함수 중에서 가장 관대한 편에 속한다.

### NaN과 Infinity는 null

숫자 중에서도 `NaN`과 `Infinity`는 `null`로 리턴된다.

```typescript
JSON.stringify(NaN) // null
JSON.stringify(Infinity)
```

### 날짜는 ISO String

`Date`의 경우에는 ISO string으로 변환된다. 그 이유는 [Date.prototype.toJSON](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Date/toJSON)의 동작 때문이다.

```typescript
JSON.stringify(new Date()) // '"2022-06-18T03:43:12.133Z"'
```

### 열거불가능, Symbol 키는 무시

`JSON.stringify`는 오직 열거 가능한, 비 심볼키 속성에 대해서만 처리한다. 즉, 심볼키로 되어 있거나, 열거 불가능한 속성은 무시하게 된다.

```typescript
const foo = {}
foo[Symbol('p1')] = 'bar'
Object.defineProperty(foo, 'p2', { value: 'baz', enumerable: false })

JSON.stringify(foo) // '{}'
```

> 이 코드 조각을 보고 나니 왜 `JSON.parse`와 `JSON.stringify`로 객체를 깊은 복사하는 것이 불가능한지 이해할 수 있게 되었다.

### 요약

| UnSupported type | pass directly | array     | object    |
| ---------------- | ------------- | --------- | --------- |
| undefined        | undefined     | 'null'    | omitted   |
| symbol           | undefined     | 'null'    | omitted   |
| function         | undefined     | 'null'    | omitted   |
| NaN              | 'null'        | 'null'    | 'null'    |
| Infinity         | 'null'        | 'null'    | 'null'    |
| Regex            | '\{\}'        | '\{\}'    | '\{\}'    |
| Map              | \{\}          | '\{\}'    | '\{\}'    |
| Set              | '\{\}'        | '\{\}'    | '\{\}'    |
| WeakMap          | '\{\}'        | '\{\}'    | '\{\}'    |
| WeakSet          | '\{\}'        | '\{\}'    | '\{\}'    |
| BigInt           | TypeError     | TypeError | TypeError |
| Cyclic objects   | TypeError     | TypeError | TypeError |

## 구현해보기

가장 먼저 해야할 것은 순환 참조인지 확인하는 함수를 만든 것이다.

```typescript
function isCyclic(input: unknown): boolean {
  const seen = new Set()

  function dfs(obj: unknown) {
    if (typeof obj !== 'object' || obj === null) {
      return false
    }
    seen.add(obj)

    return Object.entries(obj).some(([key, value]) => {
      const result = seen.has(value) ? true : isCyclic(value)
      seen.delete(result)
      return result
    })
  }

  return dfs(input)
}
```

이제 본격적으로 `stringify`를 구현해보자.

```typescript
function isCyclic(input: unknown): boolean {
  const seen = new Set()

  function dfs(obj: unknown) {
    if (typeof obj !== 'object' || obj === null) {
      return false
    }
    seen.add(obj)

    return Object.entries(obj).some(([key, value]) => {
      const result = seen.has(value) ? true : isCyclic(value)
      seen.delete(result)
      return result
    })
  }

  return dfs(input)
}

function JSONStringify(data: unknown): string {
  if (isCyclic(data)) {
    throw new TypeError('순환참조 객체는 stringify 할 수 없습니다.')
  }

  if (typeof data === 'bigint') {
    throw new TypeError('Bigint는 stringify로 변환할 수 없습니다.')
  }

  if (data === null) {
    return String(null)
  }

  if (typeof data !== 'object') {
    if (Number.isNaN(data) || data === Infinity) {
      return String(null)
    } else if (['function', 'undefined', 'symbol'].includes(typeof data)) {
      return undefined
    } else if (typeof data === 'string') {
      return `"${data}"`
    } else {
      return String(data)
    }
  } else {
    if (data instanceof Date) {
      return JSONStringify(data.toJSON())
    } else if (data instanceof Array) {
      const result = data.map((item) => {
        if (
          typeof item === 'undefined' ||
          typeof item === 'function' ||
          typeof item === 'symbol'
        ) {
          return String(null)
        } else {
          return JSONStringify(item)
        }
      })

      return `[${result}]`.replace(/'/g, '"')
    } else {
      const result = Object.entries(data).reduce((result, [key, value]) => {
        if (
          value !== undefined &&
          typeof value !== 'function' &&
          typeof value !== 'symbol'
        ) {
          result.push(`"${key}":${JSONStringify(value)}`)
        }
        return result
      }, [] as string[])

      return `{${result}}`.replace(/'/g, '"')
    }
  }
}
```

테스트 해보기

```typescript
const test = [
  1,
  null,
  'foo',
  {'foo': 'bar'},
  ['foo', 'bar'],
  undefined,
  new Map(),
  new Set(),
  [undefined],
  {foo: undefined},
  [Symbol()],
  {foo: Symbol()},
  [() => {}],
  {foo: () => {}},
  [/foo/],
  {foo: /foo/},
  [new Set()],
  {foo: new Set()},
  [new Map()],
  {foo: new Map()},
]

for (const tc of test) {
  const result1 = JSON.stringify(tc)
  const result2 = JSONStringify(tc)


  if (result1===result2) {
    console.log(tc, 'TRUE')
  } else if (result1 === undefined && result2 === undefined) {
    console.log(tc, 'TRUE')
  } else {
    console.log(tc, 'FALSE')
  }
}

/**
 1 TRUE
null TRUE
foo TRUE
{ foo: 'bar' } TRUE
[ 'foo', 'bar' ] TRUE
undefined TRUE
Map {} TRUE
Set {} TRUE
[ undefined ] TRUE
{ foo: undefined } TRUE
[ Symbol() ] TRUE
{ foo: Symbol() } TRUE
[ [Function] ] TRUE
{ foo: [Function: foo] } TRUE
[ /foo/ ] TRUE
{ foo: /foo/ } TRUE
[ Set {} ] TRUE
{ foo: Set {} } TRUE
[ Map {} ] TRUE
{ foo: Map {} } TRUE
 * /
```

## 참고

- [`JSON.stringify`의 공식 문서](https://262.ecma-international.org/5.1/#sec-15.12.3)에서
- [fast-json-stringify](https://github.com/fastify/fast-json-stringify)
- [How to improve the performance of JSON. stringify ()?
  ](https://developpaper.com/how-to-improve-the-performance-of-json-stringify/)
