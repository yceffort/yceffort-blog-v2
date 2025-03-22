---
title: ECMAScript 명세 읽어보기 (2)
tags:
  - javascript
published: false
date: 2020-09-28 20:25:18
description: '조금은 이해가 될지도?'
category: javascript
template: post
---

우리는 자바스크립트에서 속성이 프로토타입 체인을 통해 찾는 것을 알고 있다.

```javascript
const o1 = {foo: 99}
const o2 = {}
Object.setPrototypeOf(o2, o1)
o2.foo
// → 99
```

## 프로토타입 체인은 어떻게 구현되어 있을까?

프로토타입 체인이 어떻게 구현되어 있는지 살펴보기 위해서는, [Object Internal Methods](https://tc39.es/ecma262/#sec-object-internal-methods-and-internal-slots)에 대해 알아보는 것이 좋다. 여기에는 `[[GetOwnProperty]]`와 `[[Get]]`이 존재하는 것을 볼 수 있다. 여기서 살펴볼 것이 `[[Get]]`이다. 애석하게도, [속성 설명자 명세 타입(Property Descriptor specification type)](https://tc39.es/ecma262/#sec-property-descriptor-specification-type)에도 `[[Get]]` 필드가 존재하기 때문에, 이를 잘 구별해야 한다.

`[[Get]]` 은 굉장히 중요한 인터널 메소드다. 일반적인 객체들은 필수 인터널 메소드의 기본 구현을 따른다. 그러나 `Exotic` 객체의 경우에는, 인터널 메소드 `[[Get]]`을 기본 구현대신에 자체적으로 구현할 수 있다. 이 글에서는, 일반적인 객체에 대해서만 다룬다.

`[[Get]]`의 기본 구현을 살펴보자.

> When the [[Get]] internal method of O is called with property key P and ECMAScript language value Receiver, the following steps are taken:
>
> Return ? OrdinaryGet(O, P, Receiver).

여기에서 `Receiver`가 접근자 속성 값의 getter 함수를 호출하는데 사용되는 값임을 알 수 있다.

`OrdinaryGet`은 아래와 같이 정의 되어 있다.

> When the abstract operation OrdinaryGet is called with Object O, property key P, and ECMAScript language value Receiver, the following steps are taken:

1. Assert: IsPropertyKey(P) is true.
2. Let desc be ? O.[[GetOwnProperty]](P).
3. If desc is undefined, then
   1. Let parent be ? O.[[GetPrototypeOf]]().
   2. If parent is null, return undefined.
   3. Return ? parent.[[Get]](P, Receiver).
4. If IsDataDescriptor(desc) is true, return desc.[[Value]].
5. Assert: IsAccessorDescriptor(desc) is true.
6. Let getter be desc.[[Get]].
7. If getter is undefined, return undefined.
8. Return ? Call(getter, Receiver).

프로토타입 체인이 시작되는 곳이 바로 3단계 부터다. 만약 자신의 속성중에서 속성을 찾지 못한다면, 프로토타입의 `[[Get]]` 메소드를 호출하게 되는데, 이는 다시금 `OrdinaryGet`을 호출하는 것이다. (위의 2단계 코드) 그렇게 해서 찾지 못하면, 또 다시 호출하게 되고, 이를 더 이상 프로토타입이 존재하지 않을때 까지 반복하게 된다.

`o2.foo`를 접근하는 방식이 어떻게 이루어지는지 살펴보자. `O`를 `o2`로, `P`를 `foo`로 하여 `OrdinaryGet`을 호출하게 된다. `O.[[GetOwnProperty]]("foo")`는, `o2`가 `foo`라고 하는 속성이 없기 때문에 `undefined`를 리턴하게 된다. 따라서 3번째 단계에서 다시 부모인 `o1`에서 시도하게 된다. (부모는 null이 아니므로) 부모인 `o1`가 `foo`를 가지고 있다면, 이 값을 리턴하게 될 것이다.

여기서 부모인 `o1`은 일반적인 객체이므로, `[[Get]]` 메소드를, `OrdinaryGet`을 통해 호출하게 되며, 위와 같은 작업을 반복하게 될 것이다.

[속성 설명자 (Property Descriptor)](https://tc39.es/ecma262/#sec-property-descriptor-specification-type)는 명세 타입이다. 데이터 속성 설명자(Data Property Descriptor)는 `[[Value]]` 필드에 있는 속성 값을 저장해둔다. 접근 속성 설명자(Accessor Property Descriptors)는 `[[Get]]` 또는 `[[Set]]`에 있는 접근 함수를 저장해 둔다. 이 경우에는, `foo`와 관련있는 속성 설명자는 데이터 속성 설명자다.

2단계의 `desc` 데이터 속성 설명자는 `undefined`가 아니다. 따라서 3단계를 거치지 않고 4단계로 넘어간다. 속성 설명자는 데이터 속성설명자이므로, `[[Value]]`에 있는 99를 리턴하게 되고, 4단계에서 끝나게 된다.

## Receiver는 무엇이고 어디에서 왔을까?

`Receiver`는 8번째 단계에 있는 접근자 속성에만 쓰이는 파라미터다. 접근자 속성의 getter 함수가 호출 될 때 this 를 넘겨주는 역할을 한다.

`OrdinaryGet`은 재귀로 호출되는 동안, 원본 `Receiver`를 넘기게 된다. (`3-3`) `Receiver`가 어디서 오는지 살펴보자.

`[[Get]]`이 호출되는 것을 찾다보면, `Reference`에서 동작하는 abstract operation인 `GetValue`를 찾을 수 있게 된다. `Reference`란 명세 타입으로, 기본 값, 참조 명, 엄격한 참조 플래그로 구성되어 있다. `o2.foo`의 경우에는 기본 값은 `o2`이며, 참조명은 String인 `foo`, 그리고 엄격한 참조 플래그는 `false`이다.

### 왜 Reference는 Record가 아닌가

Reference는 Record일 것 같지만, 그렇지 않다. `Reference`에는 3가지 컴포넌트가 포함 되어 있는데, 세 개의 명명된 필드로 똑같이 표현될 수 있다. `Reference`는 역사적인 이유 때문에 `Record`로 분류되지 않는다.

## `GetValue`로 돌아가서

`GetValue`는 어떻게 되어있는지 살펴보자.

1. ReturnIfAbrupt(V).
2. If Type(V) is not Reference, return V.
3. Let base be GetBase(V).
4. If IsUnresolvableReference(V) is true, throw a ReferenceError exception.
5. If IsPropertyReference(V) is true, then
   1. If HasPrimitiveBase(V) is true, then
      1. Assert: In this case, base will never be undefined or null.
      2. Set base to ! ToObject(base).
   2. Return ? base.[[Get]](GetReferencedName(V), GetThisValue(V)).
6. Else,
   1. Assert: base is an Environment Record.
   2. Return ? base.GetBindingValue(GetReferencedName(V), IsStrictReference(V))

우리 예제에서, Reference는 `o2.foo`인데, 이는 속성 참조이기도 하다. 따라서 우리는 5단계로 넘어간다. `o2`는 원시값이 아니므로, 5-1 단계를 타지 않는다.

따라서 5-2단계에서 `[[Get]]`을 호출한다. `Receiver`는 `GetThisValue(V)`를 넘긴다. 이 경우에, 단순히 `Reference`의 기본 값이다.

### `GetThisValue(V)`

1. Assert: IsPropertyReference(V) is true.
2. If IsSuperReference(V) is true, then
   1. Return the value of the thisValue component of the reference V.
3. Return GetBase(V).
