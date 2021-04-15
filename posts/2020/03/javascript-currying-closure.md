---
title: 자바스크립트 커링과 클로져
tags:
  - javascript
published: true
date: 2020-03-05 06:03:40
description: '## 커링 [이
  글](https://www.sitepoint.com/currying-in-functional-javascript/) 에 잘 정리 되어
  있습니다.  Currying은 여러 개의 인자를 가진 함수를 호출 할 경우, 파라미터의 수보다 적은 수의 파라미터를 인자로 받으면 누락된
  파라미터를 인자로 받는 기법을 말한다.  즉 커링은 함수 하나가 n개...'
category: javascript
slug: /2020/03/javascript-currying-closure/
template: post
---

## 커링

[이 글](https://www.sitepoint.com/currying-in-functional-javascript/) 에 잘 정리 되어 있습니다.

Currying은 여러 개의 인자를 가진 함수를 호출 할 경우, 파라미터의 수보다 적은 수의 파라미터를 인자로 받으면 누락된 파라미터를 인자로 받는 기법을 말한다.

즉 커링은 함수 하나가 n개의 인자를 받는 과정을 n개의 함수로 각각의 인자를 받도록 하는 것이다. 부분적으로 적용된 함수를 체인으로 계속 생성해 결과적으로 값을 처리하도록 하는 것이 그 본질이다.

```javascript
function add(a) {
  console.log(`a of add: ${a}`)
  return function (b) {
    console.log(`a: ${a} / b: ${b}`)
    return a + b
  }
}

add(1)(2) // 이 결과는 3이다.
```

`add(1)`을 javascript 콘솔에서 찍어보면 아래와 같이 나온다.

```javascript
ƒ (b) {
    console.log(`a: ${a} / b: ${b}`)
    return a + b
  }
```

먼저 `add(1)`이 실행되면 위의 함수를 리턴한다. 저 함수 내에서 a 는 1로 기억되고 있다. 그리고 `add(1)`의 결과를 바로 그 다음인 2 파라미터와 함께 바로 실행했다. 그 결과 3이 되었다.

```javascript
var add1 = add(1)
add1(2) // 3
add1(3) // 4
```

`add1`이 선언된 순간, 이 함수가 리턴하는 익명함수는 클로저가 되었다. 이 익명함수에서는, a가 정의된적은 없지만 클로저는 그 함수가 실행된 환경을 기억하고 있으므로 1을 기억하고 익명함수에 계속해서 a = 1이라는 사실을 가지고 함수를 실행하게 된다.

물론 더욱 복잡하게 만들 수도 있다.

```javascript
function add(a) {
  return function (b) {
    return function (c) {
      return a + b + c
    }
  }
}

add(1)(2)(3) // 6
```

위를 화살표 함수로 쓴다면 아래와 같다.

```javascript
const add = (a) => (b) => (c) => a + b + c
```

화살표 함수를 쓰니까 더 간결해졌다.
