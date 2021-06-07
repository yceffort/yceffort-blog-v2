---
title: 'for vs for-in vs forEach vs for-of 무엇으로 자바스크립트 리스트를 돌아야 하나'
tags:
  - javascript
published: true
date: 2021-06-07 23:05:55
description: '사소하고 짧은 생각'
---

자바스크립트에서 배열을 순회하는 방법은 4가지가 있다.

- `for`
- `for-in`
- `forEach`
- `for-of`

각각의 방법을 살펴보면서, 배열을 순회하기 위한 가장 좋은 방법은 무엇인지 나름 결론을 내려본다.

## for

ES1 시절부터 있었던 가장 근-본적인 방법이다.

```javascript
const arr = ['a', 'b', 'c']
arr.prop = 'prop'

for (let i = 0; i < arr.length; i++) {
  const e = arr[i]
  console.log(i, e)
}
// 0 "a"
// 1 "b"
// 2 "c"
```

- 배열의 첫번째 뿐만 아니라 n번째에서 돌 수도 있음
- 단순히 배열을 순회하려는 목적에 비해서 많은 작업이 필요함 (추가적인 변수 선언 및, 증가, 길이 계산 등)

## `for-in`

놀랍게도, `for-in`도 ES1부터 있었던 근-본 방식이다.

```javascript
const arr = ['a', 'b', 'c']
arr.prop = 'prop'

for (const key in arr) {
  console.log(key, typeof key, arr[key])
}

// 0 "string" a
// 1 "string" b
// 2 "string" c
// 3 "string" prop prop
```

위의 코드 실행결과에서 알 수 있듯이, `for-in`을 배열을 순회하는데 쓰는건 별로 좋지 못하다.

- `key` 값만 가져올 수 있음
- `key` 값의 타입에서 볼 수 있다시피, 숫자가 아니고 문자열로 나온다.

```javascript
const arr = [1, 2, 3]
arr[0] === arr['0'] // true
```

> 배열은 `[]` 에서 숫자로도 접근할 수 있기 때문에 객체와 다르다고 생각할 수 있다. 그러나 배열 또한 객체 이므로, `[]`안에 심볼 외의 값이 들어가면 강제로 string으로 변환한다.

- 위에서 볼 수 있는 것 처럼, 모든 enumerable한 키들을 죄다 순회한다.

따라서 `for-in`은 객체가 enumberable한 모든 속성을 순회할 때 사용하는 것이 좋다. 그러나 이 경우에도, 프로토타입 체인을 순회하는 것이 더 낫다.

## `forEach`

ES5에서 추가된 새로운 방법, `Array.prototype.forEach()`이다.

```javascript
const arr = ['a', 'b', 'c']
arr.prop = 'prop'

arr.forEach((e, index) => {
  console.log(e, index)
})

// a 0
// b 1
// c 2
```

꽤 편리한 방법처럼 보인다. 배열의 요소와 인덱스 모두에 접근할 수 있으며, 화살표 함수를 통해서 더욱더 우아하게 코드를 짤 수 있다.

그럼에도, 아래와 같은 단점이 있다.

- `await`을 루프 내부에 쓸 수 없음
- `forEach()` 중간에 루프를 탈출하는 것이 곤란. 다른 문법의 경우엔, `break`로 가능

```javascript
const arr = ['a', 'b', 'c']
arr.forEach((e, index) => {
  console.log(e)
  if (e === 'b') {
    // break Illegal break statement
    return
  }
})
// a
// b
// c
```

아래와 같이 [some](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Array/some)을 사용하면 탈출할 수 있다.

```javascript
const arr = ['a', 'b', 'c']
arr.some((e, index) => {
  if (e === 'b') {
    return true // break
  }

  console.log(e) // falsy한 값이기 때문에 계속 루프를 돈다.
})
```

그러나 이는 `some`을 의도대로 사용하는 것이 아니거니와, 왜 이런 코드를 짰는지 다른 사람들이 이해하기 어려울 것이다.

## `for-of`

ES6에 나온 가장 최신 기능이다.

```javascript
const arr = ['a', 'b', 'c']
arr.prop = 'prop'

for (const e of arr) {
  console.log(e)
}

// a
// b
// c
```

- 모든 루프를 원하는 대로 순회할 수 있다.
- `await`을 사용한 [for-await-of](https://exploringjs.com/impatient-js/ch_async-iteration.html#for-await-of)가 가능하다.
- `break` `continue`를 사용할 수 있다.

`for-of`를 활용하면, 키만 접근하거나, 혹은 키와 값 모두 접근하거나 하는 것이 모두 가능하다.

```javascript
const arr = ['a', 'b', 'c']
for (const key of arr.keys()) {
  console.log(key, typeof key)
}

// 0 "number"
// 1 "number"
// 2 "number"
```

```javascript
const arr = ['a', 'b', 'c']
for (const [key, value] of arr.entries()) {
  console.log(key, value)
}
// 0 "a"
// 1 "b"
// 2 "c"
```

```javascript
const m = new Map().set(1, 1).set(2, 2).set(3, 3)
for (const [key, value] of m) {
  console.log(key, value)
}

// 1 1
// 2 2
// 3 3
```

## 결론

- `for-of`로 다른 순회문에서 할 수 있는 모든 것을 할 수 있어서 가장 좋다.
- 성능에 대한 비교는 사실 의미가 없을 것 같다. (엄밀히 따지면 `forEach`가 제일 느리다.) 그러나 자바스크립트에서 성능이 유의미할 정도로 순회문을 돌아야 한다면, 웹 어셈블리 등 다른 방법을 알아보는 것이 좋다.
- 여담으로, 적어도 프론트엔드 개발에서 `for-of`를 돌아야 하는 일은 거의 없었던 것 같다. 대부분이 `map` `reduce`를 사용해서 해결할 수 있고, 그 쪽이 더 함수형이고 읽기도 간결하다.
