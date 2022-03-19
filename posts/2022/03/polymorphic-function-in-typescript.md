---
title: '타입스크립트의 함수의 다형성'
tags:
  - typescript
published: true
date: 2022-03-14 19:11:06
description: 'mapped type과 오버로딩, 어떤걸 쓰는게 좋을까?'
---

## Table of Contents

## Introduction

자바스크립트의 경우를 먼저 생각해보자. 자바스크립트는 함수에 넘기는 인수를 다른 타입으로 하거나, 혹은 다른 위치에 넣는 등 함수가 넘겨받는 인수의 구조를 유연하게 작성할 수 있다. 아래 실제 api를 살펴보자.

- [node.js의 filehandle.write](https://nodejs.org/api/fs.html#filehandlewritebuffer-offset-length-position)
  - `filehandle.write(buffer)`
  - `filehandle.write(string)`
- [node-postgress](https://node-postgres.com/features/queries)의 쿼리
  - `client.query('query', (err, res) => ...)`
  - `client.query('query', ['value1', value2'] (err, res) => ...)`

이런 함수의 다향성을 쉽게 타이핑하기 위해서는 어떻게 해야할까?

## `Union`

`Union` 타입을 쓰는 것은 다른 타입의 인수를 허용하는 함수를 작성할 때 가장 먼저 떠오르는 방법일 것이다.

```typescript
declare function foo(a: string | number)
```

여기서 `a`는 `string` `number` 둘다 될 수 있으므로, 다양한 타입의 인수가 필요하다면 `Union`을 쓰는 것은 적절해보인다. 그리고 내부에 타입 가드 함수를 추가하여 함수 내부에서 적절하게 필요한 타입을 좁힐 수 있다.

```typescript
function foo(a: string | number) {
  if (typeof a === 'string') {
    // do something...
  }

  if (typeof a === 'number') {
    // do something...
  }
}
```

그렇다면, 리턴 타입이 인수가 어떤 타입이느냐에 따라 달라진다고 가정해보자. 그렇다면 어떻게 타이핑하는 것이 좋을까? 여기에서는 제네릭 타입을 이용하여 인수를 타이핑 하는 것이 좋을 것이다. 그리고 이를 올바른 리턴 값에 따라 조건부로 타이핑 하면 될 것이다.

이러한 문제를 한번 예시로 들어보자. `int`라고 하는 인수가 온다면 랜덤한 숫자를, `char`라는 인수가 온다면 랜덤한 글자를 반환하는 함수를 작성해보자. 먼저 자바스크립트다.

```typescript
function getRandom(str) {
  if (str === 'int') {
    // generate a random integer
    return Math.floor(Math.random() * 10)
  } else {
    // generate a random char
    return String.fromCharCode(97 + Math.floor(Math.random() * 26))
  }
}
```

이를 타입스크립트에서 적절하게 타이핑 하기 위해서는, 아래와 같은 순서로 작업을 해야 한다.

- 먼저 `str`이 `"int" | "char"`와 같은 유니언 타입으로 선언하고, 리턴 타입을 이에 의존하도록 해야 한다. 이를 위해서는, 우리는 제네릭 타입을 사용해야 한다.
- 앞서 만든 제네릭 조건부 타입을 `GetReturnType` 이라 불리는 타입에 넘겨주어, `T`에 따라 올바른 리턴 타입이 올 수 있도록 해야 한다.

위 두 조건을 구현한 식이다.

```typescript
type GetReturnType<T> = T extends 'char' ? string : T extends 'int' ? number : never

function getRandom<T extends'char' | 'int'>(str: T): GetReturnType<T> {
  if (str === 'int') {
    // generate a random number
    return Math.floor(Math.random() * 10) as GetReturnType<T>
  } else {
    // generate a random char
    return String.fromCharCode(97+Math.floor(Math.random() * 26)) as GetReturnType<T>
}
```

이제, 여기서 한단계 더 뇌절해서 랜덤 `boolean`도 지원한다고 가정해보자.

```typescript
type GetReturnType<T> = T extends 'char'
  ? string
  : T extends 'int'
  ? number
  : T extends 'bool'
  ? boolean
  : never

function getRandom<T extends 'char' | 'int' | 'bool'>(
  str: T,
): GetReturnType<T> {
  if (str === 'int') {
    // generate a random number
    return Math.floor(Math.random() * 10) as GetReturnType<T>
  } else if (str === 'char') {
    // generate a random char
    return String.fromCharCode(
      97 + Math.floor(Math.random() * 26),
    ) as GetReturnType<T>
  } else {
    // generate a random boolean
    return Boolean(Math.round(Math.random())) as GetReturnType<T>
  }
}
```

위 코드에서 볼 수 있듯이, 타입이 하나씩 함수에 추가될 때 마다 확장하는 과정이 부담스러운 것을 볼 수 있다. 다행이도, 이러한 과정은 제네릭 타입을 받는 대신에 아래처럼 객체 타입으로 하면 좀더 쉬워 진다.

```typescript
type ReturnTypeByInputType = {
  int: number
  char: string
  bool: boolean
}

function getRandom<T extends 'char' | 'int' | 'bool'>(
  str: T,
): ReturnTypeByInputType[T] {
  if (str === 'int') {
    // generate a random number
    return Math.floor(Math.random() * 10) as ReturnTypeByInputType[T]
  } else if (str === 'char') {
    // generate a random char
    return String.fromCharCode(
      97 + Math.floor(Math.random() * 26),
    ) as ReturnTypeByInputType[T]
  } else {
    // generate a random boolean
    return Boolean(Math.round(Math.random())) as ReturnTypeByInputType[T]
  }
}
```

`document.querySelector`를 생각해보자. 이 함수는 여러가지 다양한 태그명을 인수로 받고, 그에 맞는 요소를 리턴한다. [타입스크립트의 `lib.dom.d.ts`를 보면 이러한 구현 내용](https://github.com/microsoft/TypeScript/blob/ca00b3248b1af2263d0223d68e792b7ca39abcab/lib/lib.dom.d.ts#L11050-L11052)을 볼 수 있다.

### 그런데 타입 단언은 왜 필요할까?

위 코드를 보면 거추장스럽게 모든 리턴문에 `as ReturnTypeByInputType[T]`가 붙어 있는 것을 볼 수 있다. 이는 타입스크립트 3.5 부터 추가된 리턴 값에 https://www.typescriptlang.org/docs/handbook/2/indexed-access-types.html (여기에서는 `as ReturnTypeByInputType[T]`)를 주기 위해서다. 해당 인덱스에서 선택한 속성 (타입)의 모든 가능한 intersection에 대해서 리턴 타입을 체크해야 하기 때문이다. 위 예제에서는, 리턴 값이 `ReturnTypeByInputType[T]` 모두를 만족하거나, `number & string & boolean`을 모두 만족하는 intersection 타입을 넘겨주거나 (그런 타입은 `never` 뿐이다) 해야 한다. 양쪽 두 조건을 모두 만족하는 것은 `never` 뿐이므로, `as never`로 작성해도 작동한다.

타입 단언은 본질적으로 안전하지 않은 방식이다. 이를 함수 오버로딩으로 해결하는 방식도 있다. 그러나 두개다 사실 그정도로 안전하지는 않다.

## 옵셔널 파라미터

옵셔널 파라미터 또한 매우 일반적인 방식으로, 파라미터를 정의하여 사용할 수 있으며, 정의 되지 않은 파라미터에 대해서는 검사할 필요도 없다.

타입스크립트에서는, `?`를 사용하면 된다.

```typescript
declare function foo(a: string, b?: boolean)
```

여기에서는 결과적으로, `b`는 `boolean | undefined` 형태의 union 타입이 된다.

이러한 옵셔널 파라미터의 제공 여부에 따라서 다른 유형의 값을 리턴하는 패턴도 일반적으로 볼 수 있다.

검색 결과를 비동기로 가져오는 `search` 함수가 있다고 가정해보자. 이 함수는 콜백 함수를 인수로 받는다. 이 콜백 인수가 있으면, 검색 결과를 콜백 함수로 전달한다. 그렇지 않으면 검색 결과를 확인 할 수 있는 `Promise`를 반환한다.

```javascript
function search(query, cb) {
  const res = api(query)
  if (cb) {
    res.then((data) => cb(data))
    return
  }

  return res
}

const p = search('foo') // return a promise
const v = search('foo', (data) => {}) // void
```

타입스크립트에서는, 이 함수를 다음과 같은 과정으로 타이핑 할 수 있다.

- `cb`를 `?`와 함께 옵셔널 파라미터로 지정한다.
- `cb` 의 타입을 제네릭으로 타이핑 한다.
- `extends` 키워드를 사용하여 올바른 타입으로 타이핑 한다.

```typescript
type Callback = (results: Result[]) => void

function search<T extends Callback | undefined = undefined>(
  query: string,
  cb?: T,
): T extends Callback ? void : Promise<Result[]> {
  const res = api(query)

  if (cb) {
    res.then((data) => cb(data))
    return undefined as void & Promise<Result[]>
  }

  return res as void & Promise<Result[]>
}

const p = search('key') // ✅ Promise<Result[]>
const v = search('key', (data) => {}) // ✅ void
```

여기서 확인할 수 있는 사실은 다음과 같다.

- `extends` 라고 하는 조건부 표현을 사용하여 올바른 리턴타입을 정의 했다.
- 타입 단언이 역시나 필요하다.

여기도 타입단언이 추가되면서 꽤나 복잡해졌다. 만약 복잡한 다형성 함수가 필요하다면, 더 나은 다음의 대안을 쓰는게 좋을 수도 있다.

## 함수 오버로드

타입스크립트는 함수 오버로드를 지원하고 있다. 이 타입스크립트의 함수 오버로드는 1.1 부터 볼 수 있던 오래된 기능이다. 그러나 타입스크립트 초기 개발 중에 추가된 다른 기능들과는 다르게, (enum 등) 이 오버로드 기능은 잘 쓰이지 않는 경향이 있다.

아마도 함수 오버로드를 잘 사용하지 않는 이유는 자바스크립트 개발자들에게는 조금 낯선 개념이라 그런게 아닐까 싶다. 자바스크립트에서는, 함수 오버로드가 없다. 자바스크립트는 특정 스코프에서는 특정한 명칭을 가진 하나의 함수만 존재할 수 있다.

그러나 동적 타입 언어세너는, 자바스크립트의 타입 체크가 런타임 중에 일어난다. 이 말은 함수의 인수를 동적으로 우리가 필요한 만큼 가질 수 있으며, 이는 마치 함수 오버로드와 같이 동작한다는 것이다.

### 함수 오버로딩 구현하기

인수가 숫자라면 이를 문자로, 반대로 문자라면 숫자로 리턴하는 함수를 구현한다고 가정해보자. 자바스크립트에서는 아마 이렇게 구현할 것이다.

```typescript
function switchIt(input) {
  if (typeof input === 'string') return Number(input)
  else return String(input)
}
```

이를 앞선 예제와 같은 형식으로 타입스크립트에서 구현한다면 이렇게 될 것이다.

```typescript
function switchIt<T extends string | number>(
  input: T,
): T extends string ? number : string {
  if (typeof input === 'string') {
    return Number(input) as string & number
  } else {
    return String(input) as string & number
  }
}

const num = switchIt('1') // has type number ✅
const str = switchIt(1) // has type string ✅
```

이것을 함수 오버로딩 방식으로 타이핑 할 것이다.

- 먼저 두개의 다른 시그니쳐를 만든다.
- 오버로드한 함수의 구현부를 작성한다.
  - 유니언 타입으로 각 인수의 타입을 받는다.
  - 함수 내부에서는, 타입 가드를 사용하여 적절한 처리를 추가한다.

```typescript
function switchIt_overloaded(input: string): number
function switchIt_overloaded(input: number): string
function switchIt_overloaded(input: number | string): number | string {
  if (typeof input === 'string') {
    return Number(input)
  } else {
    return String(input)
  }
}
```

함수 오버로드를 사용하여,

- 제네릭과 조건부 타입을 제거
- 타입 단언 제거

이 덕분에

- 가독성 향상. 오버로드한 함수가 어떤 타입이 올 수 있는지 명확하게 구별할 수 있다. 또한 인수의 타입과 그에 따른 리턴 타입이 명확하게 분리되어 있다.
- IDE가 오버로드 함수를 더욱 잘 지원할 수 있게 된다.

### 좀더 복잡한 예제

방금 전에 만들었던 검색 함수 예제를 떠올려 보자. 이를 함수 오버로딩으로 구현하면 다음과 같이 처리할 수 있다.

```typescript
type Callback = (results: Result[]) => void

function search_overloaded(term: string): Promise<Result[]>
function search_overloaded(term: string, cb: Callback): void
function search_overloaded(
  term: string,
  cb?: Callback,
): void | Promise<Result[]> {
  const res = api(term)

  if (cb) {
    res.then((data) => cb(data))
    return
  }

  return res
}

const p = search_overloaded('key') // ✅ Promise<Result[]>
const v = search_overloaded('key', (data) => {}) // ✅ void
```

### 이것도 안전하지 않기는 마찬가지

타입 단언은 종종 좋지 않은 코드 (code smell)로 간주되며, 이를 함수 오버로드로 제거하는 것은 좋아보일 수도 있다. 그러나 이것 또한 안전하지 않은 건 매한가지다.

```typescript
function switch_overloaded(input: string): number
function switch_overloaded(input: number): string
function switch_overloaded(input: number | string): number | string {
  if (typeof input === 'string') {
    return input // 그냥 string 리턴함
  } else {
    return input // 그냥 숫자 리턴함
  }
}

const num = switch_overloaded('1') // ❌ ????
const str = switch_overloaded(1) // ❌ ????
```

> [typescript playground에서 보기](https://www.typescriptlang.org/play?#code/GYVwdgxgLglg9mABAZwO4yhAFgfTgNwFMAnAGzgEMATQqgChjAAcQoAuFKYxgcwEoOYEAFsARiQBQoSLAQp0mXARLlqtBs1aCR44gM7cwPKeGjwkaDNjxEylGvUYt2iIWJKIAPgd763urx8jRABvCUQIiJhgRDooAE8mQjgYp1ZEAF4sxAByZC5eHL5Q8Miy4kIoEGIkNKhEAHoGxEAP2sBThqCeREAazsAWRcALVdKIgF9EQlJkQhKy8srq2s16ptaOwGohwATxnoGhxFGJYYkJCAR81xFM+SslW1UHOhyARiLG5sAZcjPhPMQEpMQYZA+AVE6QwfwBFGgIAopFI8UQFE6RxO9XyxAulkUNhU9nUD2Ky3eqK+Pym-06iGB9VBZIhVWhsPhgJIQA)

읭? 올바르지 않은 타입을 리턴해버렸음에도 에러가 나지 않는다. 타입스크립트 컴파일러는 함수 본체의 코드 (오버로드 된)의 함수 시그니처와 대조할 뿐이지, 분기 문에서 어떻게 오버로드를 다루는지는 알 수 없다. 결과적으로, 오버로드 함수 시그니처와 모순된 내부 코드를 작성할 수 있는 위험성이 존재한다.

### 사실 함수 오버로드도 함수타입의 intersection 일 뿐...

함수 오버로드는 intersection 함수 타입에 대한 단지 문법적 설탕일 뿐이다.

```typescript
function switchIt(input: string): number
function switchIt(input: number): string
```

이는 사실 아래와 같다.

```typescript
type F = ((input: string) => number) & ((input: number) => string)

const switchIt_intersection: F = (input) => {
  if (typeof input === 'string') {
    return Number(input)
  } else {
    return String(input)
  }
}

const num = switchIt_intersection(1) // ✅
const str = switchIt_intersection('1') // ✅
```

마찬가지로, `F`도 객체 타입(인터페이스) 형태로 작성할 수도 있다.

```typescript
interface F {
  (input: number): string
  (input: string): number
}
```

## 정리

함수 오버로드를 사용하건, 조건부 타입을 활용한 제네릭 타입을 사용하건, 어떤 것을 사용하든지 이 선택에는 적절한 이유가 있어야 하고, 어느 것도 안전하지 않기 때문에 신중해야 한다.

- 함수의 인수가 여러개가 될 수 있는 경우, 함수 오버로드를 사용하는게 적절할 수 있다.
- 조건 타입에 따른 제네릭 타입은 인수가 리턴 타입에 영향을 미칠 때 적절하게 활용 가능하다. 매핑으로 구현하면 가독성이 눈에 띄게 향상되기 때문이다. 이를 함수 오버로드로 작성하면 매우 장황해질 수 있고, 읽는 사람으로 하여금 혼란을 야기할 수 있다.
