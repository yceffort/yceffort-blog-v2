---
title: 'suppressImplicitAnyIndexErrors 옵션을 키기 전에'
tags:
  - javascript
  - typescript
published: true
date: 2021-05-23 22:26:52
description: 'Don’t give up and use suppressImplicitAnyIndexErrors 이 멋있어서 배껴봄'
---












타입스크립트를 배우고, 본격적으로 사용하면서 부딪히는 가장 최초의 어려움은 바로 이 에러가 아닐 까 싶다.

```bash
Element implicitly has an 'any' type because type '{}' has no index signature.
```

- https://www.typescriptlang.org/tsconfig#suppressImplicitAnyIndexErrors
- https://www.typescriptlang.org/tsconfig#noImplicitAny

보통 이런 에러는 아래와 같은 코드에서 나타난다.

```typescript
for (const [key, value] of Object.entries(obj)) {
  ///...
}
```

`Object.entries` 는 아마도 다음과 같이 타이핑이 되어 있을 것이다.

```typescript
entries(o: {}): [string, any][];
```



```typescript
const test = { a: 'a', b: 'b', c: 'c' }
for (const [k, v] of Object.entries(test)) {
  const value = test[k] // Element implicitly has an 'any' type because index expression is not of type 'number'.ts(7015)
}
```

`k` 객체의 키로 `v`를 추정할 수 없기 때문에 발생하는 문제다.

이럴 때는 아래와 같이 작업해보자.

## 방법 1.

```typescript
type testKey = 'a' | 'b' | 'c'

const test: { [key in testKey]: string } = { a: 'a', b: 'b', c: 'c' }
for (const [k, v] of Object.entries(test)) {
  const value = test[k as testKey]
  console.log(k === v)
}
```

object의 key의 타입을 추론한다음, 이를 설정하는 방법이다.

## 방법 2.

```typescript
function entries<O extends Object>(obj: O): Array<[keyof O, any]> {
  return Object.entries(obj) as Array<[keyof O, any]>
}

for (const [k, v] of entries(test)) {
  const value = test[k]
  console.log(k === v)
}
```

느슨하게 타이핑되어 있는 `Object.entries`를 강력하게 `test` 객체의 형태에 맞 맞춰 타이핑한다.
