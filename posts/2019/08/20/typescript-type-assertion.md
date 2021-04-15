---
title: 타입스크립트 타입 단언
date: 2019-08-20 06:30:52
published: true
tags:
  - javascript
  - react
description: '## 문제의 시작 문제의 시작은
  [여기](/2019/06/17/typescript-type-enum-partial-record/) 였다.  내가 사용하는 코드는 아래와
  같았다.  ```typescript type GlobalColors = "Red" | "Blue" | "Green" |
  "Black";  // 기본값으로 색상을 선언한다. const enu...'
category: javascript
slug: /2019/08/20/typescript-type-assertion/
template: post
---

## 문제의 시작

문제의 시작은 [여기](/2019/06/17/typescript-type-enum-partial-record/) 였다.

내가 사용하는 코드는 아래와 같았다.

```typescript
type GlobalColors = 'Red' | 'Blue' | 'Green' | 'Black'

// 기본값으로 색상을 선언한다.
const enum ConstGlobalColorSet {
  Red = '11, 11, 11',
  Blue = '22, 22, 22',
  Green = '33, 33, 33',
  Black = '44, 44, 44',
}

// red, blue, green, black에 대해서는 글로벌하게 지정해둔 컬러를 사용하되,
// 그밖의 string이 오면 그냥 그 string을 리턴한다
function GetGlobalColor(colorString: GlobalColors | string) {
  return GlobalColorSet[colorString] || colorString
}
```

그러나 다른 프로젝트에서 아래와 같은 에러가 발생했다.

```javascript
// Element implicitly has an 'any' type because index expression is not of type 'number'.
return GlobalColorSet[colorString] || colorString
```

파라미터로오는 `colorString` 이 enum의 키가 아닐 수도 있기 때문에 발생하는 에러 였다. 기존 lint 룰에서는 any를 accept했기 때문에 에러가 발생하지 않았던 것이다.

## 해결

`as` 키워드를 써서 문제를 해결했다.

```typescript
function GetGlobalColor(colorString: GlobalColors | string) {
  return GlobalColorSet[colorString as GlobalColors] || colorString
}
```

## 타입 단언

타입스크립트의 타입 추론은 매우 좋고 강력한 기능이지만, 어쩔수 없이 한계가 존재하는 경우가 더러 있다. 이를 보완하기 위해, 타입 단언은 컴파일러가 실제 런타임에 존재할 변수와 다르게 추론하거나, 너무 보수적으로 추론하는 경우에 개발자가 수동으로 컴파일러에 대해 타입의 힌트를 주는 것이다.

위의 코드로 돌아와 보자.

```typescript
enum ConstGlobalColorSet {
  Red = '11, 11, 11',
  Blue = '22, 22, 22',
  Green = '33, 33, 33',
  Black = '44, 44, 44',
}
```

위코드는 컴파일을 거치고 나면 다음과 같이 해석된다.

```javascript
var ConstGlobalColorSet
;(function (ConstGlobalColorSet) {
  ConstGlobalColorSet['Red'] = '11, 11, 11'
  ConstGlobalColorSet['Blue'] = '22, 22, 22'
  ConstGlobalColorSet['Green'] = '33, 33, 33'
  ConstGlobalColorSet['Black'] = '44, 44, 44'
})(ConstGlobalColorSet || (ConstGlobalColorSet = {}))

// {Red: "11, 11, 11", Blue: "22, 22, 22", Green: "33, 33, 33", Black: "44, 44, 44"}
```

당연하지만 `Red` `Blue` `Green` `Black`에 대해서는 올바르게 리턴할테지만, 다른 string에 대해서는 null을 리턴할 것이다. 즉, 컴파일 에러가 날 일은 없을 것이다. 이런 경우, `as` 키워드를 통해서 타입단언을 해주면 컴파일 에러 없이 사용할 수 있다.

대부분의 경우 `as any`와 같은 치트키로 컴파일 문제를 해결할 수 있다. 그러나 이런 키워드가 득실 거릴수록 타입스크립트로 얻을 수 있는 장점이 사라지기 때문에, 가능한 적게 사용해야 한다.

## 타입선언과 타입 캐스팅의 차이

**타입 단언은 런타임에 영향을 미치지 않는다. 그러나 타입 캐스팅은 컴파일 타임과 런타임 모두 타입을 변경 시킨다. 타입 단언은 컴파일러에서만 타입을 변경 시킨다**

타입 단언은 두가지로 사용될 수 있다.

```typescript
;(colorString as GlobalColors) < GlobalColors > colorString
```

`<Type>`은 리액트의 JSX 문법과 겹치는 느낌이 있어서 보통 `as type`을 더 많이 쓴다.

## 타입 가드

타입 가드는 타입스크립트 컴파일러에 타입 체크를 알려주는 기능이다. 자바스크립트에서는 이런 느낌으로 처리했을 것이다.

```javascript
function doSomething(x: number | string) {
  if (typeof x === 'string') {
    // string 만 들어오게 처리 해줬기 때문에 에러가 날 수 없음
    console.log(x.substr(1))
  }
  x.substr(1) // 에러 날 수도 있음
}

class Foo {
  foo = 123
  common = '123'
}

class Bar {
  bar = 123
  common = '123'
}

function doStuff(arg) {
  if (arg instanceof Foo) {
    console.log(arg.foo) // OK
    console.log(arg.bar) // undefined
  }
  if (arg instanceof Bar) {
    console.log(arg.foo) // undefined
    console.log(arg.bar) // OK
  }

  console.log(arg.common) // OK
  console.log(arg.foo) // undefined?
  console.log(arg.bar) // undefined?
}

doStuff(new Foo())
doStuff(new Bar())
```

이를 타입스크립트에서 처리하려면 어떻게 해야할까?

```typescript
interface A {
  x: number
}
interface B {
  y: string
}

function doStuff(q: A | B) {
  if ('x' in q) {
    // q: A
  } else {
    // q: B
  }
}
```

`in`키워드를 사용하거나 아래 처럼 `is`키워드를 사용할 수도 있다.

```typescript
interface Foo {
  foo: number
  common: string
}

interface Bar {
  bar: number
  common: string
}

/**
 * arg를 Foo라고 타입 가드를 선언
 */
function isFoo(arg: any): arg is Foo {
  return arg.foo !== undefined
}

function doStuff(arg: Foo | Bar) {
  if (isFoo(arg)) {
    console.log(arg.foo) // OK
    console.log(arg.bar) // Error!
  } else {
    console.log(arg.foo) // Error!
    console.log(arg.bar) // OK
  }
}

doStuff({ foo: 123, common: '123' })
doStuff({ bar: 123, common: '123' })
```
