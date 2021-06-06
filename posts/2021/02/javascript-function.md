---
title: '자바스크립트 함수의 모든 것'
tags:
  - javascript
published: false
date: 2021-02-22 23:48:36
description: '모든 것이라고 했지만 모든 것이 아닐 수도 있다'
---

자바스크립트에서 함수는 중요하다. 자바스크립트의 함수는 일급객체다. 함수를 정의함으로서 일련의 특정작업을 쉽게 처리할 수 있다.

> Functions are one of the fundamental building blocks in JavaScript. A function is a JavaScript procedure — a set of statements that performs a task or calculates a value. To use a function, you must define it somewhere in the scope from which you wish to call it.

## 함수는 객체다. 그러나 일반적인 객체와 차이점이 있다.

자바스크립트에서는 여섯가지 원시타입을 제외하고는 모두 객체이기 때문에, 함수 역시 객체다. 일반적인 객체와 가장 큰 차이점이라고 한다면

- 일반 객체는 호출할 수 없지만, 함수 객체는 호출할 수 있다.
- 함수 객체만 가지고 있는 프로퍼티가 있다.
  - `arguments`: 전달된 인수정보를 갖고 있다.
  - `caller`: 표준은 아니다 까먹자
  - `length`: 매개변수의 개수
  - `name`: 함수의 이름
  - `__proto__`: 프로토타입에 접근할 수 있는 프로퍼티
  - `prototype`: 생성자 함수로 호출 할수 있는 함수 객체. 근데 이건 함수선언에 따라서 없을 수도 있다.

가 있다.

## 함수를 정의하는 법

### 함수 선언문

```javascript
function add(x, y) {
  return x + y
}
```

함수 선언문은 함수 호이스팅이 발생하므로, 선언하기 이전이라도 실행할 수 있다.

### 함수 표현식

```javascript
var add = function (x, y) {
  return x + y
}
```

이 경우는 함수 선언문과는 다르게, 변수 호이스팅이 일어나기 때문에 선언하기 이전에 접근 할 수는 있지만, 실행은 안된다.

### Function 생성자

```javascript
var add = new Function('x', 'y', 'return x+y')
```

다시한번 말하지만 함수도 객체이기 때문에 생성자를 통해서 함수를 생성할 수 있다.

### 화살표 함수

```javascript
var add = (x + y) => x + y
```

## 함수의 구분

ES6 이전에는, 함수는 다양하게 호출 될 수 있었다.
