---
title: 자바스크립트의 this
date: 2021-03-14 14:55:58
tags:
  - javascript
published: false
description: '더이상 this에 대해서 묻지 마세요'
---

`this`는 함수의 호출자를 의미하며, 자바스크립트에서의 `this`는 함수를 호출 할 때 마다, 함수가 어떻게 호출되었는지에 따라서 바인딩되는 객체가 결정된다. 정적 방식이 아닌, 호출하는 방식에 따라 동적으로 결정된다는 이유 때문에, 많은 사람들이 혼란을 겪고 있는 것 같다.

## Table of Contents

## 1. 화살표 함수의 `this`는 언제나 선언된 화살표 함수 시점의 상위 스코프이다. 그리고 `bind` `call` `apply`로 화살표 함수의 `this`를 바꿀 수 없다.

```javascript
const outerThis = this

const f = () => {
  console.log(this === outerThis) // true
}
```

```javascript
f.bind({ foo: 'bar' })()
f.call({ foo: 'bar' })
f.call({ foo: 'bar' })
// 화살표 함수에 임의로 this를 바인딩 하는 행위는 언제나 무시된다.
```

```javascript
const obj = { f }
obj.f() // true: 부모객체인 obj가 무시되고 언제나 this는 outerThis다.
```

`선언된 화살표 함수 시점의 상위 스코프` 임을 알 수 있는 또다른 예제를 살펴보자.

```javascript
class Foo {
  bar = () => {
    console.log(this) // 언제나 Foo의 instance를 가리킨다. {Foo}가 나온다.
  }
}
```

이방식은 리액트 컴포넌트에 이벤트 리스너를 다는 방식에서 유용하게 사용될 수 있다.

```javascript
class Foo {
  bar = (() => {
    const outerThis = this
    return () => {
      console.log(this === outerThis) // true
    }
  })()
}

// 위와 아래는 같다.

class Foo {
  constructor() {
    const outerThis = this
    this.bar = () => {
      console.log(this === outerThis) // true
    }
  }
}
```

## 2. 그게 아니라면, `this`는 `new` 선언으로 생성된 인스턴스를 따른다.

```javascript
class Foo {
  constructor() {
    console.log(
      this.constructor === Object.create(MyClass.prototype).constructor,
    )
  }
}
```
