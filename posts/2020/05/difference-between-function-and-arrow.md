---
title: javascript 일반 함수와 화살표 함수의 차이
date: 2020-05-19 06:39:42
published: true
tags:
  - javascript
description:
  'ES6에서부터 생긴 `arrow function`은 일반적으로 `()=>{}`의 모양을 하고 있으며, 동작도
  비슷해보인다. 하지만 이 두 선언방식은 두가지 분명한 차이를 가지고 있다. 하지만 그전에 this를 알아야 한다.'
---

ES6에서부터 생긴 `arrow function`은 일반적으로 `()=>{}`의 모양을 하고 있으며, 동작도 비슷해보인다. 하지만 이 두 선언방식은 두가지 분명한 차이를 가지고 있다.

하지만 그전에 this를 알아야 한다.

## 0. this

`this` 는 현재 실행 문맥을 뜻한다. 즉, 호출한 것이 누구냐는 것이다.

```javascript
console.log(this === window) // true
```

위 코드에서는 `console.log`를 호출한 것이 window이기 때문에 true를 리턴한다.

```javascript
const hello = {
  hi: function () {
    console.log(this === window)
    console.log(this)
  },
}
hello.hi() // false
```

위 코드에서 `this`는 `hi`다. 즉, `this`는 현재 실행 문맥을 의미한다.

## 1. this와 arguments의 차이

화살표 함수는 `this`와 `arguments`를 바인딩하지 않는다. 그 대신, 일반적인 `this`와 `arguments`와 동일한 범위를 가지고 있다.

```javascript
function createObject() {
  console.log('Inside `createObject`:', this.foo, this)
  return {
    foo: 42,
    bar: function () {
      console.log('Inside `bar`:', this.foo, this)
    },
  }
}

createObject.call({ foo: 21 }).bar()
```

위 함수의 결과는

```
Inside `createObject`: 21, {foo: 21}
Inside `bar`: 42, bar: f
```

가 된다. 위 `console.log`에서의 this는 `call`인자로 넘겨준 `{foo:21}`이고, `bar`가 호출되는 시점에서 `this`는 `bar`이다. `bar`시점에서 `this.foo`는, `{foo:42}`가 된다.

그러나 화살표 함수에서는 약간 다르다.

```javascript
function createObject() {
  console.log('Inside `createObject`:', this.foo, this)
  return {
    foo: 42,
    bar: () => console.log('Inside `bar`:', this.foo, this),
  }
}

createObject.call({ foo: 21 }).bar()
```

결과는

```
Inside `createObject`: 21, {foo: 21}
Inside `bar`: 21, {foo: 21}
```

즉, 화살표 함수안에서의 `this`는 `createObject`안의 `this`를 따르게 된다. 이는 화살표 함수가 현재 환경의 `this`를 따르게 하고 싶을 때 유용하다는 뜻이다.

## 2. 화살표 함수는 `new`로 호출할수 없다.

es2015에서는 `callable`한 것과 `constructable`한 것과의 차이를 두고 있다. 어떤 함수가 `constructable`하다면, 이는 `new`로 호출되어야 한다. ex) `new User()` 그리고 만약 함수가 `callable`하다면, 이 함수는 `new`없이도 호출이 되어야 한다 .ex) 일반적인 함수 호출

일반적인 함수의 경우 `callable`하며 `constructable`하다. 그러나 화살표 함수는 오로지 `callable`할 뿐이다. 반대로 `class`의 경우에는 오로지 `constructable`할 뿐이다.

## 정리

서로 바꿔서 쓸 수 있는 경우

- `this`, `arguments`를 쓰지 않는 경우
- `bind(this)`를 사용하는 경우

서로 바꿔쓸 수 없는 경우

- `constructable` 함수
- `prototype`에 추가된 함수나 메소드
- `arguments`를 함수의 인자로 사용하는 경우
