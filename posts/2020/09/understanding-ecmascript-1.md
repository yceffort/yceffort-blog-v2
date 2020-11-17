---
title: ECMAScript 명세 읽어보기 (1)
tags:
  - javascript
published: true
date: 2020-09-24 18:49:14
description: '가끔 문서를 볼 때 마다 도망쳤던 그 곳'
category: javascript
template: post
---

## Table of Contents

## 서두

```javascript
const o = { foo: 1 }
o.hasOwnProperty('foo') // true
o.hasOwnProperty('bar') // false
```

자바스크립트에 대한 모든 지식이 다 없다는 가정하에, `o`에는 분명 `hasOwnProperty`라는 속성이 없다는 것을 알 수 있다. 이를 찾기 위해서는 프로토타입 체인을 타고 올라가야 한다. `o`의 프로토타입은 `Object.prototype`이다.

`Object.prototype.hasOwnProperty`가 어떻게 작동되는지 알기 위해서, 이제 문서의 내용을 보자.

https://tc39.es/ecma262/#sec-object.prototype.hasownproperty

> When the hasOwnProperty method is called with argument V, the following steps are taken:
>
> 1. Let P be ? ToPropertyKey(V).
> 2. Let O be ? ToObject(this value).
> 3. Return ? HasOwnProperty(O, P).
>
> NOTE
> The ordering of steps 1 and 2 is chosen to ensure that any exception that would have been thrown by step 1 in previous editions of this specification will continue to be thrown even if the this value is undefined or null.

그리고 `hasOwnProperty(O, P)`를 따라가면

> The abstract operation HasOwnProperty takes arguments O (an Object) and P (a property key) and returns a completion record which, if its [[Type]] is normal, has a [[Value]] which is a Boolean. It is used to determine whether an object has an own property with the specified property key. It performs the following steps when called:

> 1. Assert: Type(O) is Object.
> 2. Assert: IsPropertyKey(P) is true.
> 3. Let desc be ? O.[[GetOwnProperty]](P).
> 4. If desc is undefined, return false.
> 5. Return true.

여기서 이제 몇가지 질문들이 생긴다.

- `abstract operation`?
- `[[]]` 안에 있는 것은 무엇일까?
- 함수 앞에 `?`는 무엇일까?
- `asserts`는 무슨 뜻일까?

## 언어 타입과 명세 타입

이 문서에는 `undefined` `true` `false`와 같이 이미 익숙한 개념들이 존재한다. 이들은 [언어 값](https://tc39.es/ecma262/#sec-ecmascript-language-types)이라고하며, 언어 타입의 값을 의미하며 이들은 명세에도 나타나 있다.

이러한 언어의 값에는 내부적으로, `true`와 `false` 같은 값이 존재한다. 반대로, 자바스크립트 엔진이 일반적으로 이해하지 못하는 값이 존재할 수 있다. 예를 들어, 자바스크립트 엔진이 C++로 작성되어 있다고 생각해본다면, C++의 `true` `false` 또한 존재할 것이다. (그리고 이들은 정확히 자바스크립트 값과 매칭되지 않는다.)

이러한 언어타입에 추가로, 명세타입이란 것이 존재한다. 이러한 명세타입은 자바스크립트 언어에는 없지만, 문서(명세)에는 존재하는 타입을 의미한다. 자바스크립트 엔진이 이러한 것들을 구현할 필요가 없다. 본 아티클에서는, 이러한 명세 타입중의 하나로 `Record`에 대해 알아볼 것이다.

## Abstract Operation

[Abstract Operation](https://tc39.es/ecma262/#sec-abstract-operations)이란 ECMA 스펙에서 정의한 함수다. 이들은 명세를 간결하게 작성할 목적으로 정의된다. 자바스크립트 엔진은 엔진 내부에 이들을 별도의 기능으로 구현할 필요가 없다. 이것들은 자바스크립트에서 직접 호출될 수 없다.

## 인터널 슬롯과 인터널 메소드

[인터널 슬롯과 인터널 메소드](https://tc39.es/ecma262/#sec-object-internal-methods-and-internal-slots)는 `[[]]`안에 있는 이름을 의미한다.

- 인터널 슬롯은 자바스크립트 객체의 데이터 멤버이거나 특정 타입을 의미한다. 이들은 객체의 상태를 저장하는데 사용된다.
- 인터널 메소드는 자바스크립트 객체의 멤버 함수다.

얘를 들어, 모든 자바스크립트 객체는 인터널 슬롯 `[[Prototype]]`을, 그리고 인터널 메소드인 `[[GetOwnProperty]]`를 가지고 있다.

인터널 슬롯과 인터널 메소드는 모두 자바스크립트에서 접근 가능한 것이 아니다. 예를 들어서 `o.[[prototype]]`이나 `o.[[GetOwnProperty]]()` 등을 할수가 없다. 자바스크립트 엔진은 자체적으로 내부 사용을 위해서 구현할 수는 있지만, 그럴 필요는 없다.

때때로, 인터널 메소드는 비슷한 이름을 가진 abstract operation에 작업을 위임하기도한다. 그 예가 바로 `[[GetOwnProperty]]`다.

> `[[GetOwnProperty]](P)`

> When the [[GetOwnProperty]] internal method of O is called with property key P, the following steps are taken:

> 1. Return ! OrdinaryGetOwnProperty(O, P).

`OrdinaryGetOwnProperty`는 어떤 객체와도 관련된 것이 아니기 때문에 인터널 메소드가 아니다. 대신, 객채는 여기에 파라미터로 넘어가서 작동을 하게 된다.

`OrdinaryGetOwnProperty`는 일반적인 객체에서 작동되기 때문에 `ordinary`라고 불리운다. ECMAScript 객체는 `ordinary`하거나 `exotic`할 수 있다. `Ordinary` 객체는 필수 인터널 메소드라고 하는 일련의 메소드들의 기본 동작을 갖추고 있어야 한다. 만약에 이러한 기본동작이 없다면, `Exotic`한 것이다.

가장 잘 알려진 `exotic` 객체가 바로 `Array`이다. array는 `length`라는 속성을 가지고 있는데, 이는 일반적인 방식으로 작동하지 않는다. `length`에 값을 부여하여 `array`의 객체를 삭제할 수가 있기 때문이다.

[필수 인터널 메소드의 목록은 다음과 같다.](https://tc39.es/ecma262/#table-5)

## Completion Records

`!`와 `?`를 사용하는 이유를 알기 위해서는, [Completion Records](https://tc39.es/ecma262/#sec-completion-record-specification-type)에 대해서 이해 해야 된다.

Completion Record는 명세 타입이다. (명세의 목적으로만 쓰인다) 자바스크립트 엔진은 이를 실행하는 인터널 데이터 타입을 가질 필요가 없다.

`Completion Record`는 정해진 필드 목록을 가진 `record`이다. 여기서 정해진 필드란 아래 세개를 의미한다.

| 이름         | 설명                                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------ |
| `[[Type]]`   | `normal` `break` `continue` `return` `throw` 중 하나. `normal` 외 모든 것들은 비정상적인 종료다. |
| `[[Value]]`  | 종료로 인해 만들어진 값. 함수의 return 을 통해 나온 값이나, exception을 의미한다.                |
| `[[Target]]` | Used for directed control transfers                                                              |

모든 abstract operation은 암묵적으로 `Completion Record`를 리턴한다. 단순히 Boolean을 리턴하는 abstract operation이라고 할지라도, 암묵적으로 `Completion Record`의 `normal` 타입으로 래핑되어 있다.

만약 exception이 발생하면, `[[Type]]`이 `throw`로, `[[Value]]`로는 exception 객체가 있는 Completion Record가 리턴되었다는 것을 의미한다.

[ReturnIfAbrupt(argument)](https://tc39.es/ecma262/#sec-returnifabrupt) 란 다음 을 의미한다.

1. `argument`가 예외라면, `argment`를 리턴한다.
2. `argument`를 `argument`로 설정한다. `[[Value]]`

즉, 비정상적으로 종료되었을 경우, 즉시 리턴을 하게 된다. 그렇지 않고 정상적인 케이스의 경우에는, `Completion Record`에서 값을 추출한다.

`ReturnIfAbrupt`가 함수 호출과 비슷해보이지만, 사실은 그렇지 않다. `ReturnIfAbrupt`는 다음과 같이 사용될 수 있다.

> 1. obj가 Foo() 라고 가정하자. (obj는 `Completion Record`다.)
> 2. ReturnIfAbrupt(obj)
> 3. Bar(obj) 만약 여기까지 도달했다면, obj는 Completion Record에서 추출된 값이다.

자 이제, `?`로 돌아오자. `? Foo()`는 사실 `ReturnIfAbrupt((Foo()))` 와 동일하다. 이 말인 즉슨 매번 오류 처리 코드를 명시적으로 작성할 필요가 없다. 를 의미한다.

`Let Val be ! Foo()`은 다음을 의미한다.

1. val 은 Foo() 이다.
2. val은 비정상 종료가 아니라고 가정한다.
3. val을 `val.[[Value]]` 로 설정한다.

그래서, 결과적으로 `Object.prototype.hasOwnProperty`의 명세는 아래와 같이 해석 가능하다.

### `Object.prototype.hasOwnProperty` 명세

1. Let P be ? ToPropertyKey(V).
2. Let O be ? ToObject(this value).
3. Return ? HasOwnProperty(O, P).

### `Object.prototype.hasOwnProperty` 해석

1. P는 `ToPropertyKey(V)`다.
2. P가 비정상 종료를 한다면, 그대로 P를 리턴한다.
3. P를 `P.[[Value]]` 로 설정한다.
4. O는 `ToObject(this value)` 다.
5. O가 비정상 종료를 한다면, 그대로 O를 리턴한다.
6. O를 `O.[[Value]]`로 설정한다.
7. temp는 `HasOwnProperty(O, P)`다.
8. temp가 비정상 종료를 한다면, temp를 리턴한다.
9. temp를 `temp.[[Value]]`로 설정한다.
10. NormalCompletion(temp)를 리턴한다.

그리고 `hasOwnProperty`는 아래와 같다.

### `hasOwnProperty` 명세

1. Assert: Type(O) is Object.
2. Assert: IsPropertyKey(P) is true.
3. Let desc be ? O.[[GetOwnProperty]](P).
4. If desc is undefined, return false.
5. Return true.

### `hasOwnProperty` 해석

1. 가정: `Type(O)`는 객체다.
2. 가정: `IsPropertyKey(P)`는 참이다.
3. `desc`는 `O.[[GetOwnProperty]](P)`이다.
4. `desc`가 비정상적으로 끝나면, `desc`를 리턴한다.
5. `desc`를 `desc.[[Value]]`로 설정한다.
6. `desc`가 `undefined`면, `NormalCompletion(false)`를 리턴한다.
7. `NormalCompletion(true)`를 리턴한다.

인터널 메소드 `[[GetOwnProperty]]`를 `!` 없이 표현하면 아래와 같다.

`O.[[GetOwnProperty]]`

1. `temp`는 `OrdinaryGetOwnProperty(O, P)`다.
2. `temp`는 비정상 종료되지 않는다고 가정한다.
3. `temp`는 `temp.[[Value]]`다.
4. `NormalCompletion(temp)`를 리턴한다.

여기에서는 `temp`라고 하는 완전히 새로운 변수를 만들어서 다른 것과 충돌 되지 않도록 하였다.

또한 `return`문이 Completion Record 가 아닌 다른 것을 리턴할때, 암묵적으로 이것이 `NormalCompletion` 으로 감싸져서 리턴된다는 사실을 이용했다.

그렇다면 `Return ? Foo()`는 무엇일까?

1. `temp` 는 `Foo()`다.
2. `temp`가 비정상적으로 종료되면, temp를 리턴한다.
3. `temp`를 `temp.[[Value]]`로 설정한다.
4. `NormalCompletion(temp)`를 리턴한다.

결국 이는 `Return Foo()`와 같으며, 비정상/정상 케이스에서 모두 동일하게 작동한다.

`Return ? Foo()` 는 단순히 `Foo`가 Completion Record를 반환한다는 것을 보다 명확하게 표현하기 위한 장치일 뿐이다.

## Asserts

명세 내의 `Asserts`는 (이 문서에서는 가정이라고 번역) 알고리즘의 불변의 조건을 강조하는 것이다. 이는 오로지 명확성을 위해 추가한 것으로, 구현을 위해서 무언가 추가를 해야하는 것은 아니다. 구현상에서는 이를 확인할 필요가 없다.

## 다음으로

`abstract operation`은 밑에 그림에서 보다시피, 다른 `abstract operation` 위임하는 경우가 있다. 그러나 이 글을 바탕으로 이들이 무엇을 하는지 알아낼 수 있어야 한다. 우리는 또한 또다른 명세 유형인, `Property Descriptors`에 대해 알아볼 것이다.
