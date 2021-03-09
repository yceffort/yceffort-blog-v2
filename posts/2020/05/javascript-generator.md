---
title: 자바스크립트 제네레이터
tags:
  - javascript
published: true
date: 2020-05-21 07:33:36
description: "## Generator 제네레이터의 개념에 대해 이해하기 전에, 먼저 반복자 (Iterator)에 대해
  알아보자.  ### 0. Iterator  반복자는, 두개의 속성 (`value`와 `done`)을 반환하는 `next()`메소드를 사용하여
  [Iterator protocal](https://developer.mozilla.org/en-US/docs/W..."
category: javascript
slug: /2020/05/javascript-generator/
template: post
---
## Generator

제네레이터의 개념에 대해 이해하기 전에, 먼저 반복자 (Iterator)에 대해 알아보자.

### 0. Iterator

반복자는, 두개의 속성 (`value`와 `done`)을 반환하는 `next()`메소드를 사용하여 [Iterator protocal](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Iteration_protocols#The_iterator_protocol)을 구현한다. 말이 조금 어려운 것 같으니, 조금 쉽게 설명해보자.

예를 들어 `for ... of` 구문에서 자바스크립트 객체들이 loop되는 것과 같은 iteration 동작을 정의하는 것을 허락하는 것이다. `Array`나 `Map`의 경우에는 default iteration 동작이 담겨져 있다. 

더 쉽게 이야기 하자면, object가 [Symbol.iterator](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Symbol/iterator) 키 속성을 가지고 있다는 것을 의미한다. 어떤 객체가 `반복가능`하다면 이 메소드 `@@iterator`가 인수없이 호출이 가능하고, 반환된 iterator는 반복을 통해서 획득한 값을 얻을 때 사용할 수 있다.

```javascript
const hello = ['hello', 'hi']
console.log(hello[Symbol.iterator]) //[Function: values], iterable

const hi = 1
console.log(hi[Symbol.iterator]) // undefined, not iterable
```

`iterable` 프로토콜이 만약 `next()` 메소드를 가지고 있고, 다음과 같은 규칙을 따르고 있다면 `iterator`라고 정의 한다.

- `next`: 아래 두개의 속성을 가진 object를 반환하는 인수가 없는 함수:
  - `done`: (boolean) 작업을 마쳤을 경우 `true` 그렇지 않다면 `false`
  - `value`: (any) `iterator`로 부터 반환되는 모든 자바스크립트 값이며, `done`이 `true`이면 생략 가능하다.

아래의 예시를 살펴보자.

```javascript
const hello = 'hello'
const stringIterator = hello[Symbol.iterator]()

console.log(stringIterator.next()) //{ value: 'h', done: false }
console.log(stringIterator.next()) //{ value: 'e', done: false } 
console.log(stringIterator.next()) //{ value: 'l', done: false }
console.log(stringIterator.next()) //{ value: 'l', done: false }
console.log(stringIterator.next()) //{ value: 'l', done: false }
console.log(stringIterator.next()) //{ value: undefined, done: true }
```

또 이를 활용해서 원하는 iterator를 정의할 수도 있다.

```javascript
var countDown = new Number(5)

countDown[Symbol.iterator] = function() {
  var _count = 0
  var value = +this
  return {
    next() {
      _count++      
      if (_count < value) {        
        return {value: _count, done: false}
      } else {
        return {done: true}
      }
    }
  }
}

for (let i of countDown) {
  console.log(i) // 1, 2, 3, 4
}

console.log([...countDown]) // [ 1, 2, 3, 4 ]
```

### 1. Genenrator

`Generator`는, 하나의 값을 리턴하는 일반적인 함수와는 다르게 결과의 순서를 생성해 내는 함수라고 볼 수 있다. 앞서 설명한 `iterator`와 동일하게, `next()`를 호출하면, `{value: any, done: boolean}`을 리턴한다.

```javascript
function * generator() {
  yield 'hello'
  yield 'hi'
}

for (let i of generator()) {
  console.log(i) // hello, hi
}
```

- `yield`: 제네레이터 함수의 실행을 일시적으로 중지 시키며, 뒤에 오는 표현식을 반환한다. 즉 일반적인 함수의 `return`문 역할을 하면 된다고 본다. 

- `return`: 수행하고 있는 iterator를 종료시키며, `return`뒤의 표현식은 `{value: return, done: true}` 형태로 반환된다.

`Generator`의 형태가 `Iterator`와 비슷한 걸로 보았을때, `Generator`도 `iterable`하다고 볼 수 있다.

한가지 알아 두어야 할 것은, `next()`와 `yield`가 서로 값을 주고 받을 수 있다는 점이다.

```javascript
function *myGen() {
  const x = yield 1;       // x = 10
  const y = yield (x + 1); // y = 20
  const z = yield (y + 2); // z = 30
  return x + y + z;
}

const myItr = myGen();
console.log(myItr.next());   // {value:1, done:false}
console.log(myItr.next(10)); // {value:11, done:false}
console.log(myItr.next(20)); // {value:22, done:false}
console.log(myItr.next(30)); // {value:60, done: true}
```