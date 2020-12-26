---
title: 'ES2021 미리보기'
tags:
  - javascript
published: true
date: 2020-12-26 18:57:32
description: '2021년엔 쓸만한 개발자가 되길 바라며'
---

https://github.com/tc39/ecma402/milestone/4

## String replaceAll()

String에서 replace를 하기 위해서는 `.replace('origin', 'change')` 를 썼지만, 모든 글자를 바꾸기 위해서는 `gi`를 활용해서 정규식 변환을 했었어야 했다.

```javascript
const a = '123123123'
a.replace('1', 'a') // "a23123123"
a.replace(/1/gi, 'a') // "a23a23a23"
```

이제 `replaceAll()`을 사용하면 된다.

```javascript
a.replaceAll(1, 'a') // "a23a23a23"
a.replace(/1/gi, 'a') === a.replaceAll(1, 'a') //true
```

## 논리적 할당 연산 (Logical Assignment Operator)

말이 조금 어렵지만(?) 밑에 예제를 보 면 알수 있다.

```javascript
let a = true
let b = 2
let c = 3

// 왼쪽이 참이면 a에 b를 할당한다.
a && (a = b)

a &&= b

// 왼쪽이 거짓이면 a에 b를 할당한다.
a || (a = b)
// 위 식은 아래와 같다
a ||= b

// 왼쪽이 nullish (null, undefined) 면 a를 b에 할당한다.
a ?? (a = b)
// 위 식은 아래와 같다
a ??= b
```

## 숫자 연산자 (Numeric Separators)

```javascript
const a = 1000000
const b = 1_000_000
const c = 1_000_000.123_456
const d = 1000000.123456

a === b // true
c === d // true
```

숫제가 길어지면 `,`를 찍는데, 이제 그것이 자바스크립트에서도 가능해진 것이다. 순전히 사람을 위한 기능이라고 보면 될것 같다.

## Promise.any()

`Promise.all()`과는 다르게, 배열중에 하나라도 먼저 끝나는게 있으면 그 결과를 리턴한다.

```javascript
const promise1 = new Promise((resolve) => setTimeout(resolve, 100, 'first'))
const promise2 = new Promise((resolve) => setTimeout(resolve, 300, 'second'))
const promise3 = new Promise((resolve) => setTimeout(resolve, 500, 'third'))

const promises = [promise1, promise2, promise3]

Promise.any(promises).then((value) => console.log(value))
```

만약 이 중에 하나라도 resolve가 되지 않으면, `AggregateError`를 리턴한다. 이 에러는 배열안의 모든 에러를 하나로 합쳐서 보여준다.

## WeakRef

자바스크립트에서는, 객체는 항상 강하게 참조되었다. 이 말은, 참고하고 있는 객체가 존재하는 이상, 절대로 메모리 내에서 객체가 가비지 컬렉팅이 되지 않는다는 것이다.

```javascript
var a, b
a = b = document.querySelector('.someClass')
a = undefined

// b는 여전히 document.querySelector('.someClass')를 참조한다.
```

위 예제에서는, 객체를 메모리에 계속해서 남겨 두고 싶지 않기 때문에, `WeakRef`를 사용하여 캐시나 큰 객체의 매핑을 구현할 수 있다. 사용하지 않는 경우에는, 메모리를 가비지 컬렉팅하여 필요할 때 마다 새로운 캐시를 생성할 수 있다.

```javascript
const x = new WeakRef(document.querySelector('.gatsby-highlight'))
const element = x.deref()
```

https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WeakRef

그러나 이 `WeakRef`는 가능한 사용을 피하라고 언급되어 있다. 가비지 컬렉터가 작동하는 타이밍, 방법, 및 실행 여부는 자바스크립트 엔진 구현에 따라 달려 있기 때문에 자바스크립트 엔진 마다 이 동작이 달라질 수 있기 때문이다.