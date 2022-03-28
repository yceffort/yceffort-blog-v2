---
title: '내가 타입스크립트에서 Enum을 잘 쓰지 않는 이유'
tags:
  - typescript
  - javascript
published: true
date: 2022-03-28 11:56:45
description: 'enum이 잘못했네'
---

## Table of Contents

## Introduction

[이전 글](/2020/09/typescript-enum-not-treeshaked)에서도 언급했던 것 처럼, enum은 트리쉐이킹이 되지 않기 때문에 (정확히는 번들러가 무엇을 트리쉐이킹 해야할지 알 수 없으므로) 잘 사용하지 않는 다고 언급했었다.

```typescript
enum Direction {
  Up,
  Down,
  Left,
  Right,
}

Direction.Up // 0
Direction.Down // 1
Direction.Left // 2
Direction.Right // 3
```

는 자바스크립트로 아래와 같이 컴파일된다.

```javascript
'use strict'
var Direction
;(function (Direction) {
  Direction[(Direction['Up'] = 0)] = 'Up'
  Direction[(Direction['Down'] = 1)] = 'Down'
  Direction[(Direction['Left'] = 2)] = 'Left'
  Direction[(Direction['Right'] = 3)] = 'Right'
})(Direction || (Direction = {}))
Direction.Up // 0
Direction.Down // 1
Direction.Left // 2
Direction.Right // 3
// { 0: "Up", 1: "Down", 2: "Left", 3: "Right", Up: 0, Down: 1, Left: 2, Right: 3 }
```

[typescript playground](https://www.typescriptlang.org/play?#code/KYOwrgtgBAIglgJ2AYwC5wPYigbwFBRQCqADgDQGwYDuIFhAMsAGar1QBKcA5gBZt4AvnjzwkaTCAB0pKAHo5UAAyjEKdFikwa2BVACMq8RulNW8xQCYj6yVK59UFqAGYgA)

`const enum`을 사용하여 위와 같은 큰 트랜스파일을 없앨 수도 있지만, `--isolatedModules`옵션으로 인하여 별도의 처리가 필요하다고 언급했었다. 만약 그 문제를 넘어간다 하더라도 `enum`은 문제가 없는 것일까?

> [--isloatedModules](https://www.typescriptlang.org/tsconfig#isolatedModules) These limitations can cause runtime problems with some TypeScript features like const enums and namespaces. Setting the isolatedModules flag tells TypeScript to warn you if you write certain code that can’t be correctly interpreted by a single-file transpilation process.

## 숫자형 enum은 예기치 못한 문제를 이르킬 수 있다.

```typescript
enum Direction {
  Up,
  Down,
  Left,
  Right,
}

declare function move(direction: Direction): void

move(100) // ??
```

`Up`, `Down`, `Left`, `Right`가 0, 1, 2, 3 으로 할당되어서 분명 100은 들어가면 안됐을 텐데, 100도 별다른 문제 없이 들어가는 것을 볼 수 있다. 왜 그럴까?

사실 이는 [타입스크립트에서 의도된 동작](https://github.com/microsoft/TypeScript/issues/38294#event-3305063822)이다.

> [Ryan Cavanaugh](https://github.com/RyanCavanaugh)는 타입스크립트 author 아저씨다. [이 issue](https://github.com/microsoft/TypeScript/issues/26362)로 추측컨데, bitwise연산으로 인한 문제로 보인다.

## 문자형 enum의 경우

구조적 타이핑 세계에서, enum은 named type으로 불리기도 한다. 즉, 값이 올바르고 호환 가능하다 할지라도, 문자열 enum이 필요한 함수나 객체에 값을 전달할 수 없다는 뜻이다. 아래 예시를 살펴보자.

```typescript
enum Direction {
  Up = 'Up',
  Down = 'Down',
  Left = 'Left',
  Right = 'Right',
}

declare function move(direction: Direction): void

move('Up') // impossible
move(Direction.Up) // possible
```

이렇듯 문자형 enum과 숫자형 enum의 동작 방식에 차이가 있고, 또 위험성을 안고 있기 때문에 enum 사용을 꺼리는 편이다.

## Enum간의 값 비교도 안됨

```typescript
enum Direction1 {
  Up = 'Up',
  Down = 'Down',
  Left = 'Left',
  Right = 'Right',
}

enum Direction2 {
  Up = 'Up',
  Down = 'Down',
  Left = 'Left',
  Right = 'Right',
}

// This condition will always return 'false' since the types 'Direction1.Up' and 'Direction2.Up' have no overlap.
if (Direction1.Up === Direction2.Up) {
}
```

아무리 같은 값이라 할지라도, enum 내에 있으면 타입스크립트는 이 값을 비교할 수 없기 때문에 false가 리턴된다.

## Union Types을 대신 써보기

우리에겐 union type이 있다.

```typescript
type Direction = 'Up' | 'Down' | 'Left' | 'Right'

declare function move(direction: Direction): void

move('Up') // possible
```

만약 진짜 enum, 즉 숫자형 enum 을 쓰고 싶다면, `const`, 그리고 `as const` 와 함께 `Values<T>`의 헬퍼 타입을 써보는 것도 좋다.

```typescript
const Direction = {
  Up: 0,
  Down: 1,
  Left: 2,
  Right: 3,
} as const

type Values<T> = T[keyof T]

declare function move(direction: Values<typeof Direction>): void

move(Direction.Up) // Ok!
move(0) // Ok!
move(100) // ㅠ_ㅠ
```

- enum의 동작과 다르게 결과물이 어떨지 코드를 통해 명확히 알 수 있음
- 문자열 enum, 숫자형 enum 등으로 바꾼 다고 해서 (값을 바꾼다고해서) 동작에 차이가 발생하지 않음
- 타입 안전성 확보
- enum과 동일한 편의성 제공
