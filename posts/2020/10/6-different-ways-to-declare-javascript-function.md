---
title: '자바스크립트 함수를 선언하는 여섯가지 방법'
tags:
  - javascript
published: true
date: 2020-10-22 23:05:22
description: '거의 모든 것이라고 했지만 사실 그렇진 않음 어그로임'
---

## Table of Contents

## 자바스크립트의 함수는 일급 객체다

일급객체의 정의는 다음과 같다.

1. 모든 요소는 함수의 실제 매개변수가 될 수 있다.
2. 모든 요소는 함수의 반환 값이 될 수 있다.
3. 모든 요소는 할당 명령문의 대상이 될 수 있다.
4. 모든 요소는 동일 비교의 대상이 될 수 있다.

자바스크립트에서 함수는, 다음 모두를 충족 시키므로 일급 객체라고 볼 수 있다.

## 함수를 선언하는 6가지 방법

### 1. named function declaration (명명 함수 선언)

```javascript
function hello() {
  // ...
}
```

가장 대중적인 방법이다. 함수의 이름이 `hello`가 된다. 이미 여러차례 싸질러 놨듯, 호이스팅 되기 때문에 이 함수는 어느 스코프에서든 호출 할 수 있는 함수가 된다.

### 2. anonymous function expression (익명 함수 표현)

```javascript
var hello = function () {
  //...
}
```

이름이 없는 함수를 변수에 담은 방식이다. 이름이 없는 함수긴 한데, 자바스크립트 엔진이 이름을 변수명으로 추정하여 넣는다.

```javascript
var hello = function () {
  //...
}

hello.name

//  > "hello"
hello

// > ƒ () {
//   //...
// }
```

변수 할당은 호이스팅 되지 않으므로, 할당 된 이후에만 실행 가능하다.

### 3. named function expression (명명 함수 표현)

```javascript
var hello = function originalName() {
  // ...
}
```

2와 거의 동일하다. 다른 점은 함수 이름이 명확하게 선언되어 있으므로 JS 엔진에 의해 추론되지 않는 다는 것이다.

### 4. Immediately-invoked expression (즉시 실행 표현)

```javascript
var hello = (function () {
  //...
})()
```

즉시 실행 함수로, 클로져를 활용할 수 있다. 내부 함수는 변수나 다른 함수등을 쓸 수 있지만,이 함수 밖에서는 완전히 캡슐화되어 접근 할 수 없다. 가장 흔해 빠진 예제 중 하나로는 카운터가 있다.

```javascript
var myCounter = (function (initialValue = 0) {
  let count = initialValue
  return function () {
    count++
    return count
  }
})(1)

myCounter() // 2
myCounter() // 3
myCounter() // 4
```

외부 함수에서 넘겨준 1을 가지고, 내부에서 처리를 하여 리턴하고 있다.

### 5. function constructor

```javascript
var hello = new Function()
```

아마도 이런식으로 함수를 쓸일은 거의 없을 것이다.

```javascript
const adder = new Function('a', 'b', 'return a + b')
adder(2, 6)
// 8
```

이는 `eval()`을 사용하는 것과 같기 때문에 굉장히 위험하다. 그리고 이 생성자는 전역 범위로 한정된 함수만 생성할 수 있다.

### 6. arrow function (화살표 함수)

```javascript
var hello = () => {
  //...
}
```

그리고 요즘들어 많이 쓰이고 있는 화살표 함수다. 몇가지 다른게 있다면

- `constructor`로 쓰일 수 없다.
- `prototype`을 가지고 있지 않는다.
- `yield` 키워드를 허용하지 않으므로 generator를 쓸 수 없다.
- `this`도 다르다.

#### 리턴

```javascript
const f1 = (x, y, z) => x + y + z

const f2 = (x, y, z) => {
  return x + y + z
}
```

위 `f1` `f2`는 값이 같다. object를 바로 리턴하려면 괄호를 씌우면 된다.

```javascript
const f3 = (x, y, z) => ({ x, y, z })
```

## this

근데 이 얘기는 내가 너무 많이 떠든거 같다. https://yceffort.kr/2020/05/difference-between-function-and-arrow#1-this%EC%99%80-arguments%EC%9D%98-%EC%B0%A8%EC%9D%B4
