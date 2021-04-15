---
title: '자바스크립트의 배열, 그리고 이터러블과 이터레이터 (ES6)'
tags:
  - javascript
published: true
date: 2021-02-21 15:54:51
description: '이터러블과 이터레이터 이름이 헷갈림'
---

자바스크립트는 6가지 원시 타입이 있으며, 그 외에는 모두 객체 타입이다. 따라서 배열도 객체 중 하나라고 볼 수 있다. 그렇다면 아래의 객체도 배열이라고 볼 수 있을까?

```javascript
const a = {
  0: 0,
  1: 1,
  2: 2,
  length: 3,
}

a[0] // 0
a[1] // 1
a[2] // 2
a.length //3
```

그렇다면 정확히 배열이라는 것은 무엇일까?

## 배열

### 기본 정보

배열은 아래 4가지 방법으로 생성할 수 있다.

- 배열 리터럴 (`[]`)
- Array 생성자 함수
- [Array.of](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Array/of)
- [Array.from](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Array/from)

```javascript
const a = {
  0: 0,
  1: 1,
  2: 2,
  length: 3,
}

const b = [1, 2, 3]

typeof a // object
typeof b // object

Object.getPrototypeOf(a) === Object.prototype
Object.getPrototypeOf(b) === Array.prototype // true
```

객체와 배열은 기본적으로 다음과 같은 서로다른 특징이 있다.

|                 | Object       | Array           |     |
| --------------- | ------------ | --------------- | --- |
| structure       | key & value  | index & element |     |
| reference       | property key | index           |     |
| order           | X            | O               |     |
| length property | X            | O               |     |

가장 큰 차이로는, 순서와 `length` property 유무다.

### 희소배열

자바스크립트의 배열은, 일반적인 밀집 배열이 아니다.

여기서 밀집 배열이란, 데이터 타입이 통일되어 있으며 서로 메모리 상에서 연속적으로 인접해 있는 배열을 의미한다. 밀집배열은 따라서 데이터에 접근하는게 효율적이고, 빠르다.

그러나 자바스크립트는 희소배열 형태로 되어 있다. 희소배열은, 밀집 배열과 반대로 배열의 요소를 위한 데이터 공간과 크기가 다르고, 연속적으로 밀집되어 있지도 않은 배열을 의미한다. 이경우 당연히 접근하는데 속도는 조금 느리지만, 요소를 삽입하거나 삭제하는 경우에는 더 빠르다.

## 이터레이션 프로토콜

이터테이련 프로토콜이란, 앞서 언급한 배열 처럼, 순회 가능한 자료 구조를 만들기 위하여 ECMAScript 에서 정의한 규악이다.

ES6 이전에는 배열, 문자열, DOM 콜렉션 등이 각자 방법으로 데이터를 순회할 수 있도록 구성되어 있었지만, ES6에 들어서면서 이러한 순회 가능한 데이터 들이 이터레이션 프로토콜을 준수하여 동일하게 동작하게 끔 설계했다. 이러한 이터레이션 프로토콜에는 두가지가 있다.

### 이터러블 프로토콜

어떠한 값들이 루프되는 것과 같은 이터레이션 동작을 정의하거나, 사용자 정의하는 것을 의미한다. 다시 말해, `Symbol.iterator`를 호출하면 이터레이터를 반환하는 것을 이터러블 프로토콜 이라고 한다.

즉, 해당 객체에 프로토타입을 통해서든, 직접 구현을 했든, `Symbol.iterator`를 호출할 수있고 그것이 이터레이터를 반환한다면, 이터러블이다.

```javascript
const a = {
  0: 0,
  1: 1,
  2: 2,
  length: 3,
}

const b = [1, 2, 3]

a[Symbol.iterator] // undefined
b[Symbol.iterator] === Array.prototype[Symbol.iterator] // true
```

### 이터레이터 프로토콜

값들의 순서를 만드는 표준 방법을 의미한다. 객체가 `next()`를 가지고 있고, 그 객체가 `value`와 `done` (`true`, `false`)를 리턴한다면 이터레이터 프로토콜을 준수한 이터레이터다.

## 나만의 이터러블 만들어보기

피보나치 배열을 리턴하는 이터러블을 만들어보자.

```javascript
const fibonacci = function (max) {
  let prev = 0
  let curr = 1

  return {
    [Symbol.iterator]() {
      return {
        next() {
          last = prev + curr
          prev = curr
          curr = last

          return {
            value: curr,
            done: curr >= max,
          }
        },
      }
    },
  }
}

const fibonacci100 = fibonacci(100)

for (const i of fibonacci100) {
  console.log(i) // 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
}

const result = [...fibonacci100] // [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

const iterator = fibonacci(100)[Symbol.iterator]()

console.log(iterator.next()) // { value: 1, done: false }
console.log(iterator.next()) // { value: 2, done: false }
console.log(iterator.next()) // { value: 3, done: false }
console.log(iterator.next()) // { value: 5, done: false }
```

위와 같은 동작은 배열과 유사하다.

```javascript
const a = [1, 2, 3, 5]
const iterator = a[Symbol.iterator]()

console.log(iterator.next()) // { value: 1, done: false }
console.log(iterator.next()) // { value: 2, done: false }
console.log(iterator.next()) // { value: 3, done: false }
console.log(iterator.next()) // { value: 5, done: false }
console.log(iterator.next()) // { value: undefined, done: true}
```

## 이터러블 이면서, 이터레이터인 객체를 리턴

```javascript
const fibonacciFunc = function (max) {
  let prev = 0
  let curr = 1

  return {
    [Symbol.iterator]() {
      // 메소드 함수의 this는 메서드를 호출한 객체에 바인딩 된다.
      // 즉
      /*
            {
              next: [Function: next],
              [Symbol(Symbol.iterator)]: [Function: [Symbol.iterator]]
            }
            */
      return this
    },
    next() {
      last = prev + curr
      prev = curr
      curr = last

      return {
        value: curr,
        done: curr >= max,
      }
    },
  }
}

const fibonacci100 = fibonacciFunc(100)

for (const i of fibonacci100) {
  console.log(i) // 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
}

const result = [...fibonacci100] // [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

const iterator = fibonacciFunc(100)

console.log(iterator.next()) // { value: 1, done: false }
console.log(iterator.next()) // { value: 2, done: false }
console.log(iterator.next()) // { value: 3, done: false }
console.log(iterator.next()) // { value: 5, done: false }
```
