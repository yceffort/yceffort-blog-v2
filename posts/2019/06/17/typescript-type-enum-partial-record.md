---
title: Typescript Type, Enum, Partial, Record로 글로벌 변수 관리하기
date: 2019-06-18 01:20:52
published: true
tags:
  - react
  - javascript
  - typescript
description:
  '## 고민지점 - Global 로 관리하는 Colorset Red, Blue, Green, Black이 있다. - 이
  색들은 각각 지정된 칼라코드가 있다 - 그러나 때로는 그 컬러코드에 맞게 안쓰는 경우도 있다 - 그러나 때로는 저 네개를 다 안쓰고
  1~3개만 쓰는 경우가 있다.  ## Union types  [Union Type](https://www....'
category: react
slug: /2019/06/17/typescript-type-enum-partial-record/
template: post
---

## 고민지점

- Global 로 관리하는 Colorset Red, Blue, Green, Black이 있다.
- 이 색들은 각각 지정된 칼라코드가 있다
- 그러나 때로는 그 컬러코드에 맞게 안쓰는 경우도 있다
- 그러나 때로는 저 네개를 다 안쓰고 1~3개만 쓰는 경우가 있다.

## Union types

[Union Type](https://www.typescriptlang.org/docs/handbook/advanced-types.html#union-types)

어떤 라이브러리에서 받는 파라미터의 값을 number와 string으로 제한한다고 하자. 그렇다면 코드는 아래와 같을 것이다.

```typescript
function numberOrString(parameter: any) {
  if (typeof parameter === 'number') {
    return console.log('this is number')
  }
  if (typeof parameter === 'string') {
    return console.log('this is string')
  }

  throw new Error(`Expecting number of string, but got ${typeof parameter}`)
}
```

물론 이런식으로 처리할수도 있다. 그러나 문제는 컴파일 상에서만 괜찮다는 것이다. any는 typescript에서 어떤 값이든 들어갈 수 있으므로, 이 에러는 런타임상에서만 발생하게 된다.

일반적인 객체지향 코드에서는, 두 타입으로 하나 hierarchy 를 만들어서 처리할 수도 있지만, 약간 그건 과한 느낌이 있기도 하다. 이 코드를 이렇게 처리할 수도 있다.

```typescript
function numberOrString(parameter: string | number) {
  // do something...
}
```

이런 방식을 `Union Type` 이라고 한다. `Union Type`는 하나의 값에 여러가지 타입을 표현할 수 있게 해준다. 사용할 값을 `|`로 구별해서 넣어주면 된다.

처음 문제로 돌아와서, Global한 컬러로 지정하려는 값이 네개 있다고 했다. 이제 이것은 이렇게 처리하면 된다.

```typescript
export type GlobalColors = 'Red' | 'Blue' | 'Green' | 'Black'
```

자 그럼 이것이 어떻게 동작하는지 보자.

```typescript
function getColor(parameter: GlobalColors) {
  console.log(parameter)
}
```

`getColor()`에 `GlobalColors`가 아닌 다른 값을 넣으면

![ts1](../images/ts1.png)

vscode에서 (물론 plugin덕분이지만) 네개의 값만 강제하는 것을 볼 수 있다. 만약 다른 값을 넣는다면 컴파일상에서 에러가 난다.

![ts2](../images/ts2.png)

한 가지 신기 (당연) 한점은, 이 코드는 자바스크립트로 컴파일 되지 않는다는 것이다. 이점은 `enum`이랑 다른데, 암튼 지간에 저건 자바스크립트에서 처리할 수 없는 일이다. 이러한 타입체크는 나중에 컴파일 하게 된다면, `*.d.ts`에서 처리해준다는 것이다.

## enum

`enum`은 열거형이다. 그렇다.

```typescript
enum GlobalColorSet {
  Red,
  Blue,
  Green,
  Black,
}
```

를 컴파일하면

```javascript
var GlobalColorSet
;(function (GlobalColorSet) {
  GlobalColorSet[(GlobalColorSet['Red'] = 0)] = 'Red'
  GlobalColorSet[(GlobalColorSet['Blue'] = 1)] = 'Blue'
  GlobalColorSet[(GlobalColorSet['Green'] = 2)] = 'Green'
  GlobalColorSet[(GlobalColorSet['Black'] = 3)] = 'Black'
})(GlobalColorSet || (GlobalColorSet = {}))
```

로 나온다. 복잡한데, 결과적으로는 아래와 같다.

```javascript
var GlobalColorSet = {
  0: 'Red',
  1: 'Blue',
  2: 'Green',
  3: 'Black',
  Red: 0,
  Blue: 1,
  Green: 2,
  Black: 3,
}
```

일반적인 map과 다르게 key로 값을 얻을 수 있고, 값으로도 key를 얻을 수 있다.

그러나 `enum`은 const로 선언될 경우 컴파일 결과가 조금다르다.

```typescript
const enum ConstGlobalColorSet {
  Red,
  Blue,
  Green,
  Black,
}
```

다른게 아니고, 사실 아무것도 컴파일 되지 않는다. 이는 읽기 전용으로 생성된 객체이기 때문에 (const) 수정할 객체 자체가 생성되지 않는 것이다. 또한 앞서 보았던 값으로 키를 얻는 행위 또한 불가능해진다.

아무튼, 글로벌하게 쓸 색상을 enum으로 선언했다.

```typescript
const enum ConstGlobalColorSet {
  Red = '11, 11, 11',
  Blue = '22, 22, 22',
  Green = '33, 33, 33',
  Black = '44, 44, 44',
}
```

## Record

`Record<K, V>`로 쓰인다. 여기서 K는 key이고, V는 Value다. `keyof Record<K, T>`는 `k`로, `Record<K, T>[K]`는 `T`다. (느낌이 그렇다는 것) 밑에 예시를 보자.

이거는

```typescript
//이거는
type ColorProperties = Record<GlobalColors, string>
//이거와 같다
type ColorProperties = {
  red: string
  blue: string
  green: string
  black: string
}
```

(말보다 코드가 쉽다) 위에 경우와 마찬가지로, js로 컴파일 됐을 때는 위 두 코드는 아무런 js로 변환되지 않는다.

```typescript
colorProp1['purple'] = 1
colorProp1['orange']
```

위의 코드는 당연히 타입스크립트 컴파일에서 에러가 난다.

## Partial

Partial은 key를 옵셔널하게 해준다.

```typescript
// 이거는
let PartialColorProperties = Partial<ColorProperties>
// 이거와 같다.
let PartialColorProperties = { red?: string, blue?: string, green?: string, black?: string }
```

## 정리

```typescript
// 글로벌 색으로 4가지를 선언한다.
export type GlobalColors = 'Red' | 'Blue' | 'Green' | 'Black'

// 기본값으로 색상을 선언한다.
const enum ConstGlobalColorSet {
  Red = '11, 11, 11',
  Blue = '22, 22, 22',
  Green = '33, 33, 33',
  Black = '44, 44, 44',
}

// 기본 색상값외에 다른 색상을 사용하고 싶다면
const CUSTOM_COLORS: Partial<Record<GlobalColors, string>> = {
  gray: '55, 55, 55',
}
```
