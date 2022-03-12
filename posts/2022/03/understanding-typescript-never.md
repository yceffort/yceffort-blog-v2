---
title: '타입스크립트 타입 never 에 대해 자세히 알아보자'
tags:
  - typescript
published: true
date: 2022-03-12 15:31:40
description: '알쏭달쏭 신기한 타입스크립트와 타입의 세계'
---

## Table of Contents

## `never`란 무엇인가

`never`가 무엇이고 왜 만들어졌는지 이해하기 위해서는, 먼저 타입시스템에서 `타입`이 무엇을 의미하는지 이해해야 한다.

타입은 가능한 값의 집합을 의미한다. 예를 들어서, `string`이라는 타입은 가능한 모든 문자열의 집합을 의미한다. 그러므로 변수에 `string`이라는 타입을 달아둔다는 것은, 이 변수에는 문자열만 할당할 수 있다는 것을 의미한다.

```typescript
let foo: string = 'bar'
foo = 3 // ❌ 3 은 문자열이 아님
```

타입스크립트에서 `never` 는 없는 값의 집합이다. 타입스크립트 이전에 인기가 있었던 flow에서는, 이와 동일한 역할을 하는 `empty`라고 하는 것이 존재한다.

이 집합에는 값이 없기 때문에, `never` 은 어떠한 값도 가질 수 없으며, 여기에는 `any` 타입에 해당하는 값들도 포함된다. 이러한 특징 때문에, `never` 는 `uninhabitable type` `bottom type` 이라고도 불린다.

> 이와 반대로, `top type`은 `unknown`이라고 정의 되어 있다.

https://www.typescriptlang.org/docs/handbook/typescript-in-5-minutes-func.html#other-important-typescript-types

## 왜 `never`가 필요한가?

숫자에서 아무것도 존재하지 않는 것을 표현하기 위해 0이 존재하는 것처럼, 타입 시스템에서도 그 어떤 것도 불가능하다는 것을 나타내는 타입이 필요하다.

여기서 `불가능` 이라는 뜻은 다음과 같은 것을 의미한다.

- 어떤 값도 가질 수 없는 빈 타입
  - 제네릭 및 함수에서 허용되지 않는 파라미터
  - 호환 되지 않는 타입 교차
  - 빈 유니언 타입 (유니언 했지만 아무것도 안되는 경우)
- 실행이 완료되면 caller에게 제어 권한을 반환하지 않는 (혹은 의도된) 함수의 반환 유형 (예: node의 `process.exit()`)
  - `void`와는 다르다. `void`는 함수가 caller에게 아무것도 리턴하지 않는 다는 것을 의미한다.
- rejected된 promise의 fulfill 값
  ```typescript
  const p = Promise.reject('foo') // const p: Promise<never>
  ```

## `never`가 `union`과 `intersection`에서 작동하는 방식

숫자 0 이 덧셈과 곱셈에서 작동하는 것과 비슷하게, `never` 타입도 `union`과 `intersection`에서 특별한 특징을 가지고 있다.

- 0을 덧셈하면 그 값이 그대로 오는 것 처럼, `never`도 union 타입에서는 drop되는 특징을 가지고 있다.

```typescript
type t = never | string // string
```

- 0을 곱셈하면 0이 되어버리는 것처럼, `never`을 intersection type으로 지정하면 `never`가 되어 버린다.

```typescript
type t = never & string // never
```

이러한 두가지 특징은 이후에 알게 될 주요 사례의 기반이 된다.

## `never` 타입은 어떻게 사용할 수 있을까

### 허용할 수 없는 함수 파라미터에 제한을 하는 방법

`never` 타입에는 값을 할당 할 수 없기 때문에, 함수에 올수 있는 다양한 파라미터에 제한을 거는 용도로 사용할 수 있다.

```typescript
// 이 함수는 never만 사용 가능하다.
function fn(input: never) {
  // do something...
}

declare let myNever: never
fn(myNever) // ✅

// never 이외에 다른 값은 타입 에러를 야기한다.
fn() // ❌
fn(1) // ❌
fn('foo') // ❌
declare let myAny: any
fn(myAny)
```

### `switch` `if-else` 문에서 일치 하지 않는 값이 오는 경우

함수가 `never` 타입만 인수로 받는 경우, 함수는 `never`외의 다른 값과 함께 실행 될 수 없다.

이러한 특징을 사용하여, `switch` 문과 `if-else` 문장 내부에서 철저한 일치를 보장할 수 있다.

```typescript
function unknownColor(x: never): never {
  throw new Error('unknown color')
}

type Color = 'red' | 'green' | 'blue'

function getColorName(c: Color): string {
  switch (c) {
    case 'red':
      return 'is red'
    case 'green':
      return 'is green'
    default:
      return unknownColor(c) // 그 외의 string으 불가능하다.
  }
}
```

### 부분적으로 구조적 타이핑을 허용하지 않는 방법

어떤 함수에서, `VariantA`와 `VariantB` 타입의 파라미터만 허용한다고 가정해보자. 하지만 그 이외에 이 두가지 타입의 속성을 모두 갖고 있는 파라미터 (두 타입의 서브타입)는 허용하지 않는 다고 가정해보자.

위와 같은 경우, `VariantA | VariantB` 와 같은 유니언 타입으로 선언할 수도 있다. 그러나 이 경우 타입스크립트는 구조적 타이핑을 기반으로 하고 있기 때문에, 원래 타입보다 더 많은 속성을 가진 객체 타입을 함수에 전달하는 것이 허용된다. (객체 리터럴 제외) 무슨 말인지 아래 예시에서 살펴보자.

```typescript
type VariantA = {
  a: string
}

type VariantB = {
  b: number
}

declare function fn(arg: VariantA | VariantB): void

const input = { a: 'foo', b: 123 }
fn(input) // 타입스크립트는 이 경우 아무런 에러를 내지 않는다.
```

이 경우, `never`를 사용한다면, 일부 구조 타이핑을 방지할 수 있으며, 사용자가 두가지 모든 속성을 가진 객체를 가져오는 것을 방지할 수 있다.

```typescript
type VariantA = {
  a: string
  b?: never
}

type VariantB = {
  b: number
  a?: never
}

declare function fn(arg: VariantA | VariantB): void

const input = { a: 'foo', b: 123 }
fn(input) // ❌ a는 never라서 안댐
```

### 의도하지 않은 api 사용 방지

```typescript
type Read = {}
type Write = {}
declare const toWrite: Write

declare class MyCache<T, R> {
  put(val: T): boolean
  get(): R
}

const cache = new MyCache<Write, Read>()
cache.put(toWrite) // ✅ generic type이기 때문에 가능
```

위 예제에서, `get` 메소드를 통해 데이터를 읽을 수 있는 읽기전용 캐시를 만들고자 한다. 여기 `put` 메소드에 `never`를 활용하면 이러한 코드를 방지할 수 있다.

```typescript
declare class ReadOnlyCache<R> extends MyCache<never, R> {}

const readonlyCache = new ReadOnlyCache<Read>()
readonlyCache.put(data) // ❌
```

### 이론적으로 이 조건부 분기문에 도달할 수 없음을 나타내는 경우

`infer`를 사용하여 조건 부 타입 내부에 또다른 타입을 변수를 만들 때, 모든 `infer` 키워드에 대해 다른 분기를 추가해야 한다.

```typescript
type A = 'foo'
type B = A extends infer C
  ? C extends 'foo'
    ? true
    : false // inside this expression, C represents A
  : never // 여기는 닿을 수가 없다.
```

### 유니언 유형에서 멤버를 필터링

불가능한 분기점을 나타내는 것 이외에도, 조건형 타입에서 원하지 않는 타입을 필터링하고 싶은 경우에도 사용 가능하다.

방금 살펴보았던 것 처럼, union 타입에서 자동으로 제거되지는 않는다. 이처럼 union 타입에서는 `never`는 무용 지물이다.

만약 특정 기준에 따라 union member를 결정하는 유틸리티 타입을 작성하고 싶다면, `never` 가 유용해질 수 있다.

`ExtractTypeByName` 이라고 하는 유틸리티 타입에서 `name` 속성이 `foo`인 멤버를 추출하고, 일치 하지 않는 멤버를 필터링한다고 가정해보자.

```typescript
type Foo = {
  name: 'foo'
  id: number
}

type Bar = {
  name: 'bar'
  id: number
}

type All = Foo | Bar

type ExtractTypeByName<T, G> = T extends { name: G } ? T : never

type ExtractedType = ExtractTypeByName<All, 'foo'> // the result type is Foo
// type ExtractedType = {
//     name: 'foo';
//     id: number;
// }
```

위 타입이 실행되는 순서는 아래와 같다.

```typescript
type ExtractedType = ExtractTypeByName<All, Name>
type ExtractedType = ExtractTypeByName<Foo | Bar, 'foo'>
type ExtractedType =
  | ExtractTypeByName<Foo, 'foo'>
  | ExtractTypeByName<Bar, 'foo'>
```

```typescript
type ExtractedType = Foo extends { name: 'foo' }
  ? Foo
  : never | Bar extends { name: 'foo' }
  ? Bar
  : never

type ExtractedType = Foo | never
type ExtractedType = Foo
```

### mapped type에서 키를 필터링 하는 용도

타입스크립트에서는, 타입은 immutable 하다. 만약 객체 타입에서 속성을 삭제하고 싶다면, 기존 속성을 변환하고 필터링하여 새롭게 생성해야 한다. 이를 위해 매핑된 타입의 키를 조건부로 다시 매핑하면 해당 키가 필터링된다.

```typescript
type Filter<Obj extends Object, ValueType> = {
  [Key in keyof Obj as ValueType extends Obj[Key] ? Key : never]: Obj[Key]
}

interface Foo {
  name: string
  id: number
}

type Filtered = Filter<Foo, string> // {name: string;}
```

### 제어 흐름에서 타입을 좁히고 싶을 때

함수에서 리턴값을 `never`로 타이핑 했다는 사실은, 함수가 실행을 마칠 때 호출자에게 제어 권한을 반환하지 않는 다는 것을 의미한다. 이를 활용하면, 컨트롤 플로우를 제어하여 타입을 좁힐 수 있다.

> 함수가 never를 리턴하는 경우는 여러가지가 있다. exception, loop에 갇히거나, 혹은 `process.exit`

```typescript
function throwError(): never {
  throw new Error()
}

let foo: string | undefined

if (!foo) {
  throwError()
}

foo // string
```

혹은 `||` `??` 키워드로도 가능하다.

```typescript
let foo: string | undefined

const guaranteedFoo = foo ?? throwError() // string
```

### 호환되지 않는 타입의 intersection이 불가능함을 나타내고 싶을 때

호환이 되지 않는 서로다른 타입에 대해 intersection을 표시한다면 `never`가 된다.

```typescript
type t = number & string // never
```

`never`와 intersecting을 했을 때도 마찬가지다.

```typescript
type t = never & number
```

## `never` 타입을 읽는 법 (에러메시지 에서)

아마도 타입스크립트로 개발을 해본 사람이라면, `Type 'number' is not assignable to type 'never'.` 이라는 메시지를 가끔씩 보았을 것이다. 이는 일반적으로 타입스크립트가 여러가지 타입을 intersect하는 과정에서 발생하는 에러다. 이러한 에러는 타입의 안전성을 유지하기 위해서 타입스크립트 컴파일러가 내보내는 경고다.

아래 예제를 살펴보자.

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
    // 랜덤 숫자 생성
    return Math.floor(Math.random() * 10) // ❌ Type 'number' is not assignable to type 'never'.
  } else if (str === 'char') {
    // 랜덤 char 생성
    return String.fromCharCode(
      97 + Math.floor(Math.random() * 26), // ❌ Type 'string' is not assignable to type 'never'.
    )
  } else {
    // 랜덤 boolean 생성
    return Boolean(Math.round(Math.random())) // ❌ Type 'boolean' is not assignable to type 'never'.
  }
}
```

이 함수는 `number`, `string`, `boolean` 을 넘겨 받은 변수에 따라서 리턴하고 싶었던 것 같다. 그러나 각각의 리턴 문에서 타입스크립트는 에러를 뱉는다. 타입스크립트는 프로그램에서 각각 가능한 상태들에 대해 이러한 타입을 좁히도록 도움을 준다. 즉, 여기에서 `ReturnTypeByInputType[T]`는 런타임시에 number가 될수도, string이 될수도, boolean이 될수도 있다는 것을 의미한다.

여기의 리턴 유형이 가능한 모든 `ReturnTypeByInputType[T]`에 할당할 수 있는지 확인할 수 있는 경우에만 타입 안전성을 확보할 수 있다. 이 3가지 타입의 intersection은 무엇일까? 이 세가지 타입은 모두 서로 호환이 되지 않기 때문에 `never`를 반환하게 된다. 그래서 우리는 `never`메시지를 보게된 것이다. 이를 해결하기 위해서는, 타입 assertion이 필요하다.

- `return Math.floor(Math.random() * 10) as ReturnTypeByInputType[T]`
- `return Math.floor(Math.random() * 10) as never`

또다른 예제를 살펴보자.

```typescript
function f1(obj: { a: number; b: string }, key: 'a' | 'b') {
  obj[key] = 1 // Type 'number' is not assignable to type 'never'.
  obj[key] = 'x' // Type 'string' is not assignable to type 'never'.
}
```

`obj[key]` 는 런타임시에 키에 따라서 string이 될수도 number가 될 수도 있다. 타입스크립트는 따라서 key로 올수 있는 모든 값에 대해 동작할 수 있어야 되므로 제한을 두었다. 따라서 여기에서는 `never`로 결정된다.

## never를 확인하는 방법

사실 `never`인지 확인하는 것은 생각보다 쉽지 않다.

```typescript
type IsNever<T> = T extends never ? true : false

type Res = IsNever<never> // never 🧐
```

`IsNever`로 never인지 확인하기 위해 true, false를 리턴하게 헀지만 실상은 저것마저도 `never`가 된다.

https://github.com/microsoft/TypeScript/issues/23182#issuecomment-379094672 의 대답을 요약하자면

- `never`는 빈 uinion이다
- 타입스크립트는 조건 타입내부에 있는 유니온 타입을 자동으로 결정한다
- 여기에서는 빈 uinon이 들어왔으므로, 여기에 조건 타입은 다시 `never`가 된다.

따라서 우리가 생각하는 `IsNever`의 목적을 달성하기 위해서는 아래와 같은 튜플을 이용하는 방식을 취해야 한다.

```typescript
type IsNever<T> = [T] extends [never] ? true : false
type Res1 = IsNever<never> // 'true' ✅
type Res2 = IsNever<number> // 'false' ✅
```

> 사실 타입스크립트 소스코드에 있는 내용이다 https://github.com/microsoft/TypeScript/blob/main/tests/cases/conformance/types/conditional/conditionalTypes1.ts#L212
