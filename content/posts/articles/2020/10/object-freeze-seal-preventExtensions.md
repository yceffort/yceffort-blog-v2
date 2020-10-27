---
title: 'Object.freeze(), Object.seal(), Object.preventExtensions()의 차이'
tags:
  - javascript
published: true
date: 2020-10-27 23:58:46
description: 'ECMAScript 5부터 있었는데 몰랐음'
---

ECMAScript 5 스펙 중에 아래와 같은 것이 있다.

- `Object.freeze()`
- `Object.seal()`
- `Object.preventExtensions()`

얼핏보면 이름까지 비슷해보이는 세 메소드의 차이를 알기 위해서는, 객체의 구조에 대한 기본적인 지식이 있어야 한다.

## 객체의 구조

자바스크립트에서 객체는 특정 속성 또는 동작, 메소드를 포함할 수 있는 데이터 유형이다. 이러한 속성은 변경, 삭제, 혹은 새로운 속성 값을 추가할 수도 있다. 여기에는 두가지 유형이 있다.

- Data Properties: 객체 내부에 정의 되어 있는 일반적인 속성을 의미한다.
- Accessor Properties: 접근자 속성이라고도 하며, 객체의 값을 설정하거나 가져올 수 있는, getter 와 setter라고 보면된다. 이들은 `get` `set` 으로 네이밍 되어 있다.

```javascript
let person = {
  firstName: 'yongchan',
  lastName: 'Kim',

  get fullName() {
    return `${this.firstName} ${this.lastName}`
  },

  set fullName(name) {
    ;[this.firstName, this.lastName] = name.split(' ')
  },
}

person.fullName = 'yc effort'

console.log(person.firstName) // yc
console.log(person.lastName) // effort
```

우리가 생성하는 모든 객체는, 자바스크립트 객체 생성자의 속성을 상속 받게 된다. 그 중 하나가 `Object.prototype`이다. `prototype`속성을 활용해서, 존재하는 모든 객체에 새로운 속성을 추가할 수 있다.

```javascript
let person = {
  firstName: 'yongchan',
  lastName: 'Kim',
```

위에서 예를 든 이 객체에서, 각각의 속성은 다음과 같은 메타데이터를 가지고 있다.

- `enumerable` (boolean): true 라면 loop를 돌아서 확인 가능하다.

```javascript
let obj = {
  x: 1,
  y: 2,
}

Object.defineProperty(obj, 'x', {
  enumerable: false, // false
  configurable: true,
  writable: true,
  value: 1,
})

Object.defineProperty(obj, 'y', {
  enumerable: true,
  configurable: true,
  writable: true,
  value: 2,
})

Object.keys(obj) // ['y'] 만 뜬다 띠요오오오오옹
```

- `configurable` (boolean): true 라면 재 설정이 가능하다.

```javascript
let obj = {
  x: 1,
  y: 2,
}

Object.defineProperty(obj, 'x', {
  enumerable: true,
  configurable: false, // false
  writable: true,
  value: 1,
})

delete obj.x // false 가 뜨면서 삭제가 안됨
delete obj.y // true 가 리턴되고 삭제도됨
```

- `writable` (boolean): true라면 값이 변경 될 수 있다.

```javascript
let obj = {
  x: 1,
  y: 2,
}

Object.defineProperty(obj, 'x', {
  enumerable: true,
  configurable: true,
  writable: false, // false
  value: 1,
})

obj.x = 100 // 100 이 리턴되긴 하는데 수정은 안되있음
obj.y = 100 // 100 이 리턴되며 수정도 되있음
```

반대로, 접근자 속성은 값을 가지고 있지 않다. 이들은 `get` `set` 함수를 가지고 있다.

- `get`
- `set`
- `enumerable`
- `configurable`

값이 없기 때문에, `writeable`은 존재하지 않는다.

```javascript
let obj = {
  x: 1,
  y: 2,
}
```

## Object.freeze()

- 속성을 추가할 수 없다.
- 존재하는 속성을 삭제할 수 없다.
- 변경할 수 없다.
- 속성에 대해 `configurable`을 변경할 수도 없다. `writable` `configurable`는 false로 되어 있다.
- prototype도 변경할 수 없다.
- `freeze()` 되어 있는 객체에 변경하려고 하는 시도는 모두 에러를 내뱉는다.
- `Object.isFrozen()`으로 확인이 가능하다.

## Object.seal()

- 속성을 추가할 수 없다.
- 존재하는 속성을 삭제할 수도 없다.
- 존재하는 속성에 대해 `reconfigure`할 수 없다.
- 데이터 속성을 접근자 속성으로 바꾸거나, 그 반대로도 불가능하다.
- 그러나 존재하는 값에 대해서 수정은 가능하다.
- 또한 존재하는 값에 대해서 추가가 가능하다.
- 위 아래 두 메소드와는 다르게, seal은 봉인한 객체를 리턴하므로, 해당 객체를 써야 한다.

```javascript
let obj = {
  x: 1,
  y: 2,
  z: {
    a: 1,
    b: 2,
  },
}

let sealedObj = Object.seal(obj)
sealedObj.x = 100 // 100 으로 변경된다.
sealedObj.z.c = 300 // 가능.
sealedObj.a = 100 // 이건 안됨
delete sealedObj.x // 불가능
```

## Object.preventExtensions()

전달 받은 객체를 더 이상 확장이 불가능한 상태로 만든다. 더 이상 새 속성을 추가할 수가 없다. 상위 집합 객체에서 기능을 상속한다.

```javascript
let obj = {
  x: 1,
  y: 2,
  z: {
    a: 1,
    b: 2,
  },
}

Object.preventExtensions(obj)
obj.x = 100 // 100 으로 변경된다.
obj.z.c = 3 // 가능
delete obj.z // 가능. 왜냐면 확장만 막기 떄문.
```

갑자기 이 글을 쓴 이유는 https://v8.dev/blog/react-cliff 이것 떄문이다. 다음에 계속 🤔