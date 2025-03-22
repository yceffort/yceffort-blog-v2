---
title: 자바스크립트 스킬을 향상 시킬 10개의 질문
tags:
  - javascript
published: true
date: 2020-06-03 04:07:55
description:
  '## 자바스크립트 스킬을 향상 시킬 10개의 질문 [10 JavaScript Quiz Questions and
  Answers to Sharpen Your
  Skills](https://typeofnan.dev/10-javascript-quiz-questions-and-answers/) 의 질문을
  보고, 답에 대한 해석을 제멋대로 써보았습니다.  ### 1....'
category: javascript
slug: /2020/05/10-javascript-quiz/
template: post
---

## 자바스크립트 스킬을 향상 시킬 10개의 질문

[10 JavaScript Quiz Questions and Answers to Sharpen Your Skills](https://typeofnan.dev/10-javascript-quiz-questions-and-answers/) 의 질문을 보고, 답에 대한 해석을 제멋대로 써보았습니다.

### 1. 배열 정렬 비교

```javascript
const arr1 = ['a', 'b', 'c']
const arr2 = ['b', 'c', 'a']

console.log(
  arr1.sort() === arr1,
  arr2.sort() == arr2,
  arr1.sort() === arr2.sort(),
)
```

당연한 얘기지만, 자바스크립트에서의 비교는 (`==`, `===`), 원시 타입이 아니라면 reference를 참조해서 비교하게 된다. 먼저 첫 번째 부터 알아보자. [sort()](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Array/sort)는 적절하게 정렬한 후에 그 배열 자체를 반환한다고 되어 있다.

> `sort()` 메서드는 배열의 요소를 적절한 위치에 정렬한 후 그 배열을 반환합니다.

따라서 저 `arr1.sort()`의 결과가 맞던지 틀리던지 상관없이, 같은 참조값을 리턴하므로 `true`가 된다.

두 번째도 앞선 이유와 마찬가지로, 정렬의 결과가 어쨌든 간에 - 같은 참조를 하고 있으므로 `true`를 리턴하게 된다. 세 번째 역시, 앞선 이유와 같이 정렬의 결과를 비교하는게 아닌 참조를 비교하게 되므로 `false`가 된다.

만약 자바스크립트의 배열을 비교 하고 싶다면 어떻게 해야할까? [stackoverflow](https://stackoverflow.com/questions/7837456/how-to-compare-arrays-in-javascript)에 미친 답변들이 많지만, `JSON.stringify`를 이용하는게 가장 쉬울 것 같다.

### 2. Set에 Object가 들어 있다면?

```javascript
const mySet = new Set([{a: 1}, {a: 1}])
const result = [...mySet]
console.log(result)
```

[Set](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Set)은 어디까지나 원시값을 비교해서 제거하므로, `{a: 1}`이라는 중복을 제거해주지 않는다. 따라서

`[{a: 1}, {a: 1}]`가 나올 것이다.

### 3. Deep Object.freeze

```javascript
const user = {
  name: 'Joe',
  age: 25,
  pet: {
    type: 'dog',
    name: 'Buttercup',
  },
}

Object.freeze(user)

user.pet.name = 'Daffodil'

console.log(user.pet.name)
```

[Object.freeze](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Object/freeze)는 객체를 동결시키는 메소드다. 동결된 객체는 새로운 속성을 추가하거나, 제거, 삭제를 할 수 없다. 그러나 `freeze`는 얕은 동결만 수행한다. 따라서, Object안에 있는 object에 대해서는 동결이 되지 않는다. 따라서 문제의 결과는 변경된 값을 리턴하게 된다. 하위 객체까지 모두 동결 시키기 위해서는, 재귀함수를 활용해야 할 것이다.

```javascript
function deepFreeze(object) {
  const propNames = Object.getOwnPropertyNames(object)

  for (const name of propNames) {
    const value = object[name]

    // 값이 존재하고, 그값이 object라면 다시 deepFreeze를 수행한다.
    object[name] =
      value && typeof value === 'object' ? deepFreeze(value) : value
  }

  return Object.freeze(object)
}
```

혹은 [deep-freeze](https://github.com/substack/deep-freeze)라이브러리를 써도 된다.

### 4. 프로토타입 상속

```javascript
function Dog(name) {
  this.name = name
  this.speak = function () {
    return 'woof'
  }
}

const dog = new Dog('Pogo')

Dog.prototype.speak = function () {
  return 'arf'
}

console.log(dog.speak())
```

`Dog` 인스턴스가 생성될때마다, `speak` 프로퍼티에는 `woof`를 반환하는 함수가 매번 할당되게 된다. 그 결과, 인터프리터는 이미 `speak`가 프로퍼티에 할당되어 있으므로, `prototype`체인을 보지 않는다. 따라서, prototype으로 할당한 `speak`는 쓰이지 않게 된다.

### 5. Promise.all의 순서

```javascript
const timer = (a) => {
  return new Promise((res) =>
    setTimeout(() => {
      res(a)
    }, Math.random() * 100),
  )
}

const all = Promise.all([timer('first'), timer('second')]).then((data) =>
  console.log(data),
)
```

[Promise.all](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Promise/all)에 관해 물어보는 문제다. `Promise.all`은 배열 안에 있는 모든 Promise가 실행되거나 어느 하나라도 거부되기를 기다린다. 따라서 배열안에 있는 Promise의 완료 순서와는 상관없이 모두 끝나기를 기다리므로, `all`에서의 `data`의 순서는 보장 될 것이다.

### 6. Reduce 계산

```javascript
const arr = [(x) => x * 1, (x) => x * 2, (x) => x * 3, (x) => x * 4]

console.log(arr.reduce((agg, el) => agg + el(agg), 1))
```

```
1 + 1 * 1 = 2
2 + 2 * 2 = 6
6 + 6 * 3 = 24
24 + 24 * 4 = 120
```

### 7. 복수형 (s) 추가하기

```javascript
const notifications = 1

console.log(
  `You have ${notifications} notification${notifications !== 1 && 's'}`,
)
```

`&&`는 false를 리턴할 것이다. 이를 정확히 표현하기 위해서는

```javascript
notification >= 1 ? 's' : ''
```

이 되야 할 것이다.

### 8. 전개 구문과 변수명 변경

```javascript
const arr1 = [{firstName: 'James'}]
const arr2 = [...arr1]
arr2[0].firstName = 'Jonah'

console.log(arr1)
```

전개 구문은 shallow copy를 수행하므로 사실상 `arr2`는 `arr1`을 가리키고 있다. 따라서, arr2를 바꾸는 것은 arr1에도 영향을 미친다.

### 9. 배열 함수에 바인딩

```javascript
const map = ['a', 'b', 'c'].map.bind([1, 2, 3])
map((el) => console.log(el))
```

조금 의외였는데 - 생각해보니 당연한 거였다.

`.map`은 단순히 `Array.prototype.map`을 `this` 값과 함께 호출한 것이다. `bind` `call` `apply`는 함수에서 호출할 `this`를 지정할 수 있으므로, 결과는 1, 2, 3 이 된다.

### 10. Set의 유일성과 순서

```javascript
const arr = [...new Set([3, 1, 2, 3, 4])]
console.log(arr.length, arr[2])
```

Set은 유일성은 보장해주지만, 순서는 보장하지 않는다. 따라서 길이는 4가 될 것이고, 3번째 엘리먼트는, 2가 될 것이다. arr은 `3, 1, 2, 4`

### 정리

- 자바스크립트에서 원시형이 아닌 나머지는 reference 값을 가지고 있는 것일 뿐이므로, 비교를 할 때 주의를 해야 한다.
- 얕은비교, 얕은복사에 대해서 잘 알아두자
- 자바스크립트 내장 함수가 무엇을 리턴하는지 잘 확인해보자.
