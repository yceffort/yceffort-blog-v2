---
title: 재밌는 자바스크립트 면접 문제
tags:
  - javascript
published: true
date: 2020-09-23 23:43:15
description: '뭔가 이상한 자바스크립트 면접 문제는 재밌다. 단 내가 구직 중이 아닐 때만.'
category: javascript
template: post
---

## 문제 1.

```javascript
function foo() {
  let a = (b = 0)
  a++
  return a
}

foo()

typeof a // ??
typeof b // ??
```

### 정답

일단 `a`는 함수내에 선언된 `let`변수 이고, 전역 scope에 선언되지 않았기 때문에 당연히 undefined다.

문제는 `b`인데, 이게 전역변수에 선언 되었을까, 아니면 앞에 let이 있었으니까 `b`도 `let` scope를 따라갈까? 정답은 전역 객체를 따라간다는 것이다. 따라서 `b`는 `number`타입이 된다. 위 코드는 아래와 같다.

```javascript
function foo() {
  let a
  window.b = 0
  a = window.b
  a++
  return a
}
```

## 문제 2.

```javascript
const clothes = ['jacket', 't-shirt']
clothes.length = 0

clothes[0] // ??
```

### 정답

정답을 알기 위해서는, 배열 객체의 `length`의 동작을 알아야 한다.

http://www.ecma-international.org/ecma-262/6.0/#sec-properties-of-array-instances-length

> The length property of an Array instance is a data property whose value is always numerically greater than the name of every configurable own property whose name is an array index.

> The length property initially has the attributes { [[Writable]]: true, [[Enumerable]]: false, [[Configurable]]: false }.

> NOTE Reducing the value of the length property has the side-effect of deleting own array elements whose array index is between the old and new length values. However, non-configurable properties can not be deleted. Attempting to set the length property of an Array object to a value that is numerically less than or equal to the largest numeric own property name of an existing non-configurable array indexed property of the array will result in the length being set to a numeric value that is one greater than that non-configurable numeric own property name.

배열의 길이를 원래 값보다 줄이는 행위는 배열의 요소를 줄이는 역할을 한다. (????) 기존 길이에서 새로운 줄어든 길이만큼 뒤에서 배열의 요소들이 사라진다. 반대로 큰 값을 넣으면 `empty`가 그 만큼 들어간다. 따라서 정답은 undefined다.

## 문제 3

```javascript
const length = 4
const numbers = []
for (var i = 0; i < length; i++);
{
  //잘보면 여기에 ; 가 들어가 있다.
  numbers.push(i + 1)
}

numbers // ??
```

### 정답

`;`은 null statement다. null statement란 empty statment 이며, 이는 곧 아무것도 하지 않는 다는 것을 의미한다. 위 코드는 아래와 같다.

```javascript
const length = 4
const numbers = []
var i
for (i = 0; i < length; i++) {
  // 암것도 안한다.
}
{
  // 그냥 스코프를 생성하는 블록일 뿐
  numbers.push(i + 1)
}

numbers
;[5]
```

## 문제 4

```javascript
function arrayFromValue(item) {
  return
  ;[item]
}

arrayFromValue(10) // ???
```

### 정답

단순히 return 뒤애 새줄이 생겼고, 그 뒤에 `[item]`이 존재한다. 이 경우, 자바스크립트는 return 뒤에 자동으로 `;`을 붙여버리게 된다. 따라서 그 뒷줄에 있는 코드는 아무런 역할을 하지 않는다. (물론 lint에 걸리는게 정상이겠지만) 따라서 답은 `undefined`다.

## 문제 5

```javascript
let i
for (i = 0; i < 3; i++) {
  const log = () => {
    console.log(i)
  }
  setTimeout(log, 100)
}
```

### 정답

자바스크립트 인터뷰 문제를 공부해보다보면 수도없이 나오는 클로저 문제다.

1. for문이 세번돌고, 매번 그때마다 `i`의 값을 확인하는 `log`함수가 생성된다. 그리고 그 `log`를 실행하는 `setTimeOut`이 태스크 큐 대기열에 들어가게 된다.
2. for문이 끝나면, `i`는 3이 되어 있다.
3. for문이 종료 된 뒤에 `setTimeOut`을 실행하려고 `i`를 참조하면, `i`는 3이 되어있다.

따라서 정답은 3이 세번 나온다 이다.

```javascript
for (let i = 0; i < 3; i++) {
  const log = () => {
    console.log(i)
  }
  setTimeout(log, 1000)
}
```

## 문제 6

```javascript
0.1 + 0.2 === 0.3
```

### 정답

정답은 `false`다 .

```javascript
0.1 + 0.2 // 0.30000000000000004
```

이에 대한 흥미로운 사이트가 있는데, 바로 https://0.30000000000000004.com/ 이다. ㅋㅋㅋㅋㅋㅋㅋㅋ

소수점도 2진법으로 코딩되어 있는데, 0.1은 정확히 0.1과 같은 수가 아니고, 2진법으로 가장 가까운 0.1이 표현되어 있는 것이다. 정확히는, 자바스크립트는 숫자를 모두 64비트 IEEE 754 형식으로 다루는데, 콘솔에 입력해서 0.1이 제대로 보이는 것은, 그것을 앞선 형식에 따라 2진법으로 바꾸고, 다시 그 결과를 10진법으로 보여주는 것이다.

## 문제 7

```javascript
myVar // ???
myConst // ???

var myVar = 'value'
const myConst = 3.14
```

### 정답

https://yceffort.kr/2020/05/var-let-const-hoisting/ 에서도 다뤘듯이, `let`과 `const`는 TDZ에 들어가게 된다. 따라서 첫번째 줄에서는 undefined, 2번째 줄에서는 에러가 날 것이다.

```javascript
var myVar // TDZ에 들어가지않고, 호이스팅 된다.
const myConst // error TDZ에 들어갔으며, const의 경우에는 초기화가 되어야 한다. 따라서
// 따라서 Identifier 'myConst' has already been declared 에러가 날 것이다.
```

5, 6, 7은 솔직히 많이 봐서 별로 재미가 없지만, 앞에 4문제는 좀 재밌었다.

더 변태 같은 문제를 원하면 이전 포스트 https://yceffort.kr/2020/05/10-javascript-quiz/ 를 참고해보자.
