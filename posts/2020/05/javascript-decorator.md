---
title: 자바스크립트 데코레이터
tags:
  - javascript
published: true
date: 2020-05-20 07:33:36
description:
  '## 데코레이터 ### 0. 설명자  데코레이터에 대해 시작하기 전에, 설명자(Descriptor)에 대해
  알아보자.  설명자란, 객체의 프로퍼티가 쓰기가 가능한지, 그리고 열거가 가능한지 여부를 나타낸다. 그리고 설명자를 구현하기 위해서는,
  [Object.getOwnPropertyDescriptor(obj, propName)](https://develo...'
category: javascript
slug: /2020/05/javascript-decorator/
template: post
---

## 데코레이터

### 0. 설명자

데코레이터에 대해 시작하기 전에, 설명자(Descriptor)에 대해 알아보자.

설명자란, 객체의 프로퍼티가 쓰기가 가능한지, 그리고 열거가 가능한지 여부를 나타낸다. 그리고 설명자를 구현하기 위해서는, [Object.getOwnPropertyDescriptor(obj, propName)](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Object/getOwnPropertyDescriptor) 를 사용해야 한다. 아래의 예를 살펴보자.

```javascript
const hello = {
  get hi() {
    return 'hello'
  },
  number: 42,
}

console.log(Object.getOwnPropertyDescriptor(hello, 'hi'))
console.log(Object.getOwnPropertyDescriptor(hello, 'number'))

Object.defineProperties(hello, {
  hell: {
    value: 1,
    enumerable: false,
    configurable: false,
    writable: false,
  },
})
console.log(Object.getOwnPropertyDescriptor(hello, 'hell'))
```

```json
{
  get: [Function: get hi],
  set: undefined,
  enumerable: true,
  configurable: true
},
{ value: 42, writable: true, enumerable: true, configurable: true },
{ value: 1, writable: false, enumerable: false, configurable: false }
```

- `writable`은 객체의 프로퍼티가 쓰기 가능한지의 여부다.

```javascript
hello.number = 41
console.log(hello.number) //41 로 바꼈다.

hello.hell = 10
console.log(hello.hell) // 1이 리턴되며, 바뀌지가 않는다.
```

- `enumberable`은 객체의 프로퍼티가 열거 가능한지의 여부이며, false라면 `Object.keys`, `Object.values`, `Object.entries`등에서도 해당 프로퍼티를 볼 수 없다.

```javascript
console.log(Object.keys(hello)) // [ 'hi', 'number' ] 가 뜨며, hell은 안보인다 ㅠㅠ
```

- `configurable`은 해당 프로퍼티가 `defineProperty`를 통해 설정될 수 있는지 여부이며, false라면 `defineProperty`로 해당 객체를 설정할 수가 없다.

```javascript
// 다시 define 해보자
Object.defineProperties(hello, {
  hell: {
    value: true,
    enumerable: true,
    configurable: true,
    writable: true,
  },
}) // TypeError: Cannot redefine property: hell
```

- 위에서 본 것처럼 `getter`와 `setter`도 있는데, 이는 주로 동적으로 계산한 값을 반환하는 프로퍼티에 접근하거나, 메소드 호출을 하지 않고도 내부 변수에 접근해야 하는 경우 등에 사용한다.

### 1. 데코레이터

데코레이터는 클래스의 프로퍼티 / 메소드 / 클래스 자체를 수정하는데 사용되는 자바스크립트 함수다. `@xxx`로 작성 될 수 있으며 수정할 프로퍼티 / 메소드 / 클래스 윗줄에 추가해주면 된다. 이는 적용된 메소드가 호출되거나, 인스턴스가 만들어지는 것과 같은 런타임에 실행된다.

#### 클래스 데코레이터

클래스 위에 선언되어, 클래스 자체를 수정하는 예제.

```typescript
// 클래스의 constructor를 덮어쓴다.
function setName(name: string) {
  return <T extends {new (...args: any[]): {}}>(constructor: T) => {
    return class extends constructor {
      name = name
    }
  }
}

@setName('trump')
class President {
  name: string

  constructor(name: string) {
    this.name = name
  }

  sayHello() {
    console.log(`hello, ${this.name}`)
  }
}

const t = new President('obama')
console.log(t.sayHello()) // hello, trump
```

#### 메소드 데코레이터

아래 예제에서는 메소드에 데코레이터가 쓰였으며, 메소드에 logger를 달거나, readOnly등의 속성으로 `writable`을 손쉽게 막을 수 있다.

```typescript
function readOnly(isReadOnly: boolean) {
  return function (
    target: Person,
    propName: string,
    description: PropertyDescriptor,
  ) {
    description.writable = isReadOnly
  }
}

const logger =
  (message: string) =>
  (target: Person, propName: string, description: PropertyDescriptor) => {
    const value = description.value

    description.value = function (...args: any) {
      console.log('LOG >>>', message)
      return value.apply(this, args)
    }
  }

class Person {
  name: string

  constructor(name: string) {
    this.name = name
  }

  @logger('Say hello name')
  @readOnly(false)
  sayHello() {
    return `hello, ${this.name}`
  }
}

const trump = new Person('trump')

console.log(trump.sayHello())
// LOG >>> Say hello name
/// hello, trump

trump.sayHello = () => 'hi xxx'
// Cannot assign to read only property 'sayHello' of object '#<Person>'
```

#### 접근자 데코레이터 (Access Decortaor)

접근자 데코레이터는, 접근자를 선언하기 바로 직전에 선언된다. 이를 이용해 접근자의 정의를 관찰, 수정, 교체 하는 등에도 사용할 수 있다.

```typescript
function configurable(value: boolean) {
  return function (
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor,
  ) {
    descriptor.configurable = value
  }
}

class Person {
  private name: string
  constructor(name: string) {
    this.name = name
  }

  @configurable(false)
  get hi() {
    return this.name
  }
}
```

이 밖에도 프로퍼티 데코레이터, 매개변수 데코레이터 등이 있다. 이 두가지 케이스는, `reflect-metadata`를 사용해야 하고, 아직 공식적으로 ECMA에 채택되지도 않았다.
