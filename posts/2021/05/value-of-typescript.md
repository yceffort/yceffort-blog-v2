---
title: 'Typescript, 객체의 키와 값 타이핑하기'
tags:
  - typescript
  - javascript
published: true
date: 2021-05-27 20:43:27
description: '아오 피곤해'
---

```javascript
const object = {
  a: 'a',
  b: 'b',
  c: 'c',
}

const value = 'a'

const values = Object.values(object) // a, b, c
const isValid = values.includes(value) // true

if (!isValid) {
  throw new TypeError(`${value} is not one of values, ${values`)
}
```

위 코드에서, `value`가 `a` `b` `c` 중 하나가 아니면 에러가 날 것이다. 이를 타입스크립트에서 타입 가드를 하는 방법을 살펴보자.

## typescript

```typescript
const object = {
  a: 1,
  b: 2,
  c: 3,
}

type objectShape = typeof object
```

여기서 `objectShape`는 아래와 같을 것이다.

```typescript
type objectShape = {
  a: number
  b: number
  c: number
}
```

여기에 `as const` 를 추가해보자.

```typescript
const object = {
  a: 1,
  b: 2,
  c: 3,
} as const

type objectShape = typeof object
```

```typescript
type objectShape = {
  readonly a: 1
  readonly b: 2
  readonly c: 3
}
```

두가지가 바뀐 것을 볼 수 있다. 첫번째로, 모든 속성에 `readonly`가 붙어서 객체의 키 값을 바꿀 수 없게 되었고 두번째로는 `string`이 었던 값이 정확히 값으로 바뀌게 되었다. 이는 모두 `readonly`로 값이 수정되지 않는 다는 것을 확실히 했기 때문이다.

이번엔 키를 추출해보자.

```typescript
type keys = keyof objectShape // "a" | "b" | "c"
```

이러한 키를 추출했으니, 값들도 추출해 낼 수 있다.

```typescript
type values = objetShape[keys] // 1 | 2 | 3
```

## Valueof Generic

```typescript
type objectShape = typeof object
type keys = keyof objectShape
type values = objectShape[keys]
```

이번엔 제네릭으로 돌아가보자.

```typescript
type values = Shape[keyof objectShape]
type ValueOf<T> = T[keyof T]
```

```typescript
const object = {
  a: 1,
  b: 2,
  c: 3,
}

type ValueOf<T> = T[keyof T]
const a: ValueOf<typeof object> = 1
const b: ValueOf<typeof object> = 2
const c: ValueOf<typeof object> = 3
const d: ValueOf<typeof object> = 4 // error Type '4' is not assignable to type 'ValueOf<{ readonly a: 1; readonly b: 2; readonly c: 3; }>
```
