---
title: ECMAScript 스펙 이해하기 (1)
tags:
  - javascript
  - web
published: false
date: 2020-07-05 03:23:12
description: "[Understanding the ECMAScript spec, part
  1](https://v8.dev/blog/understanding-ecmascript-part-1)을 번역했습니다.  ```toc
  tight: true, from-heading: 2 to-heading: 3 ```  ## 서문  자바스크립트에 대해 어느 정도 알고 있다고
  하더라도, ..."
category: javascript
slug: /2020/07/작업중-understanding-ecmascript/
template: post
---
[Understanding the ECMAScript spec, part 1](https://v8.dev/blog/understanding-ecmascript-part-1)을 번역했습니다. 

```toc
tight: true,
from-heading: 2
to-heading: 3
```

## 서문

자바스크립트에 대해 어느 정도 알고 있다고 하더라도, [ECMAScript 언어 명세 또는 ECMAScript spec 요약](https://tc39.es/ecma262/)을 읽는 것은 꽤 부담스러운 일이다. 적어도 나는 처음에 그럤다.

구체적인 예시에서 시작해서 스펙을 이해하도록 하자. 다음 코드는 [Object.prototype.hasOwnProperty](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/hasOwnProperty)의 사용을 보여준다.

```javascript
const o = { foo: 1 };
o.hasOwnProperty('foo'); // true
o.hasOwnProperty('bar'); // false
```

위 예제에서는 `o` 객체가 `hasOwnProperty`라는 프로퍼티를 가지고 있지 않다. 여기에서 prototype 체인을 통해서 찾게 된다. 여기에서 `o`의 프로토타입은 `object.prototype`임을 알 수 있다.

`Object.prototype.hasOwnproperty`가 어떻게 동작하는지 의사 코드를 작성해서 알아보자.

```
[object.prototype.hasownproperty](https://tc39.es/ecma262/#sec-object.prototype.hasownproperty)

`hasOwnProperty`가 `v`라는 인수와 함께 실행된다면, 아래와 같은 절차를 거치게 된다.

1. Let P be ? ToPropertyKey(V).
2. Let O be ? ToObject(this value).
3. Return ? HasOwnProperty(O, P).
```

그리고

```
[HasOwnProperty(O, P)](https://tc39.es/ecma262#sec-hasownproperty)

abstract operation인 HasOwnProperty 는 Object가 명시된 속성 키를 가지고 있는지를 반환한다. 여기에서는 boolean이 리턴된다. 이 동작은 O, P와 함께 수행되는데, O는 객체, 그리고 P는 속성 키 값이다. 이 abstract operation은 다음 절차를 거친다.

1. Assert: Type(O) is Object.
2. Assert: IsPropertyKey(P) is true.
3. Let desc be ? O.[[GetOwnProperty]](P).
4. If desc is undefined, return false.
5. Return true.
```

`abstract operation`는 무엇일까? `[[]]` 안에 있는 것은 무엇을 의미하는 것일까? 함수 앞에 `a ? `는 무엇일까? `assert`의 의미는 무엇일까?

## 언어의 타입과 명세의 타입

익숙한 것 부터 알아가자. `undefined`, `true` `false`는 이미 자바스크립트로에서 이미 보던 것이다. 이들은 모두 [language value](https://tc39.es/ecma262/#sec-ecmascript-language-types) 이며, 언어 타입의 값 (values of language types) 이며 이는 스펙에도 명시되어 있다.

스펙에서 내부에서도 `language values`를 사용한다. 예를 들어, 내부 데이터 타입은 필드를 가지고 있으며, 이에 가능한 값으로 `true`와 `false`를 정해 둔다. 반대로 자바스크립트 엔진은 일반적으로, 내부에 language values를 사용하지 않는다. 예를 들어 자바스크립트 엔진이 C++로 쓰여진 경우, 일반적으로 C++의 참과 거짓을 사용한다. (이는 자바스크립트 내부에서 정의한 true false가 아닌, C++의 true false 다.)

언어의 타입 외에도, 스펙은 명세의 타입도 사용한다. 그러나 자바스크립트에서는 이를 사용하지 않는다. 자바스크립트 엔진은 이 것들을 구현할 필요가 없다. 이 글에서는, Record라고 하는 명세 타입에 대해서 알게 될 것이다.

## Abstract Operation

[Abstract Operation](https://tc39.es/ecma262/#sec-abstract-operations)이란 ECMA 스펙에서 정의한 함수다. 이들은 명세를 간결하게 작성할 목적으로 정의된다. 자바스크립트 엔진은 엔진 내부에 이들을 별도의 기능으로 구현할 필요가 없다. 이것들은 자바스크립트에서 직접 호출될 수 없다.

## 인터널 슬롯과 인터널 메소드

[인터널 슬롯과 인터널 메소드 모두](https://tc39.es/ecma262/#sec-object-internal-methods-and-internal-slots) `[[ ]]`안에 있는 것을 가리키는 용어다.

인터널 슬롯은 자바스크립트 객체의 데이터 멤버이거나 특정 타입을 의미한다. 이들은 객체의 상태를 저장하는데 사용된다. 인터널 메소드는 자바스크립트 객체의 멤버 함수다.

얘를 들어, 모든 자바스크립트 객체는 인터널 슬롯 `[[Prototype]]`을, 그리고 인터널 메소드인 `[[GetOwnProperty]]`를 가지고 있다. 

인터널 슬롯과 인터널 메소드는 모두 자바스크립트에서 접근 가능한 것이 아니다. 아까 앞서 언급한 `[[prototype]]`이나 `[[GetOwnProperty]]` 모두 호출하거나 사용할 수는 없다. 자바스크립트 엔진은 자체적으로 내부 사용을 위해서 구현할 수는 있지만, 그럴 필요는 없다.


```
[[GetOwnProperty]](P)
[GetOwnProperty](https://tc39.es/ecma262/#sec-ordinary-object-internal-methods-and-internal-slots-getownproperty-p)

O의 인터널 메소드 `[[GetOwnProperty]]`가 P 속성과 함꼐 호출된다면, 다음과 같은 절차를 거치게 된다.

1. Return ! OrdinaryGetOwnProperty(O, P).
```

(`!`가 무엇을 의미하는지는 다음 챕터에서 다룬다.)

`OrdinaryGetOwnProperty`는 어떤 객체와도 연관되지 않기 때문에 인터널 메소드가 아니다. 대신, 이것이 작동하는 객체에서는 매개 변수로 전달된다.

`OrdinaryGetOwnProperty`가 `Ordinary`한 이유는 일반적인 객체에서 작동하기 때문이다. ECMASCript 오브젝트는 `ordinary` 또는 `exotic`이 될 수 있다. Ordinary 객체는 필수 인터널 메소드라고 불리는 일련의 기본 동작을 거쳐야 한다. 만약 이러한 작업을 거치지 않으면 `exotic`이 된다.

가장 잘 알려진 `exotic`객체는 `Array`다. `Array`의 `length` 프로퍼티는 일반적이지 않은 형태를 띄고 있다. `length` 프로퍼티는 `Array`에서 엘리먼트를 제거하면서 설정할 수 있기 때문이다.

필소 인터널 메소드는 [여기](https://tc39.es/ecma262/#table-5)에 정의되어 있다.

## Completion Records

`!`와 `?`를 사용하는 이류를 알기 위해서는, [Completion Records](https://tc39.es/ecma262/#sec-completion-record-specification-type)에 대해서 이해 해야 된다.

Completion Record란 특정 타입을 의미한다. (혹은 스펙의 목적만을 의미하기도 한다.) 자바스크립트 엔진은 이와 맞는 인터널 데이터 타입을 가지고 있을 필요는 없다. 

Completion Record는 정해진 네임필드 셋을 가지고 있는 데이터 타입의 `record`다

https://v8.dev/blog/understanding-ecmascript-part-1 뭔가 번역과 이해가 나의 수준을 넘어선듯.. 🤔