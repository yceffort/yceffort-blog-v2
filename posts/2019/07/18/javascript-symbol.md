---
title: Javascript Symbol
date: 2019-07-18 07:03:36
published: true
tags:
  - javascript
description: "## Javascript Primitive 기존에 자바스크립트는 6가지의 primitive가 있었다.  - Object
  - string - number - boolean - null - undefined  그러나 es6가 들어서면서 `symbol`이라는 7번째
  primitive가 추가되었다.  ## Symbol  ```javascript const hel..."
category: javascript
slug: /2019/07/18/javascript-symbol/
template: post
---
## Javascript Primitive

기존에 자바스크립트는 6가지의 primitive가 있었다.

- Object
- string
- number
- boolean
- null
- undefined

그러나 es6가 들어서면서 `symbol`이라는 7번째 primitive가 추가되었다.

## Symbol

```javascript
const helloSymbol = Symbol();
const hiSymbol = Symbol();
```

새로운 심볼 값을 생성했다. 이 심볼로 생성한 값은 변경할 수 없으므로 const에 할당에도 상관없다. 그리고 이렇게 생성된 심볼 값은 프로그램 내에서 유일함을 보장해 준다.

```javascript
let obj = {};
obj[helloSymbol] = "hello";
obj[hiSymbol] = "hi";
console.log(obj);
```

```
{Symbol(): "hello", Symbol(): "hi"}
```

물론 문자열이나 숫자를 key로 사용할 수 있지만, symbol은 유일함을 보장해주기 때문에 이렇게 키값으로 사용할 수 있다.

```javascript
const welcomeSymbol = Symbol("환영합니다");
console.log(welcomeSymbol);
```

```
Symbol(환영합니다)
```

`Symbol`안에 있는 문자열은 일종의 주석으로 보면 될 것 같다.

## 예제

```javascript
const isBlocked = Symbol("is blocked element?");

if (element[isBlocked]) {
  openElement(element);
} else {
  element[isBlocked] = true;
}
```

`element`는 `isBlocked`라는 심볼을 키로 갖는 object다. 문자열이나 숫자가 아닌 심볼을 key로 갖는 속성이다. 이는 유일성을 보장해주기 때문에 다른 키들과의 충돌을 방지할 수 있다. 다만 `obj.name`과 같이 dot을 이용해서 접근할 수 없다. 반드시 `[]`를 활용해서 접근해야 한다.

한가지 주의 해야할 것은 `isBlocked` 심볼 값이 스코프 내에 존재 할 때만 이러한 행위가 가능하다는 것이다. 어떤 모듈이 심볼을 스스로 만드는 경우, 해당 모듈은 해당 심볼을 모든 객체에 적용할 수 있다. **즉 다른 속성과의 충돌을 걱정할 필요가 없다.**

그리고 심볼 키는 이러한 충돌을 방지하기 위해서 만들어 진 것이므로, 일반적인 javascript 객체 조사는 `Symbol`을 무시한다. 무슨 소리냐면...

```javascript
let 메시 = {};
메시["영문명"] = "Lionel Messi";
메시["별명"] = "라이오넬 멧시";
const Nationality = Symbol("선수의 국적");
메시[Nationality] = "칠레";

for (let i in 메시) {
  console.log(i);
}

for (let i of Object.keys(메시)) {
  console.log(i);
}

for (let i of Object.getOwnPropertyNames(메시)) {
  console.log(i);
}
```

```
"영문명"
"별명"
"영문명"
"별명"
"영문명"
"별명"
```

이처럼 심볼 `Nationality` 키는 일반적인 상황에서 모두 무시 되는 것을 볼 수 있다. 물론 이를 조회하는 방법도 있다.

```javascript
Object.getOwnPropertySymbols(메시);
```

```
[Symbol(선수의 국적)]
```

혹은 심볼을 포함해서 모든 키를 조회하고 싶다면

```javascript
Reflect.ownKeys(메시);
```

```
["영문명", "별명", Symbol(선수의 국적)]
```

`Reflect.ownKeys`를 활용하면 된다.

## 심볼의 특징

- 일단 생성되면 변경되지 않는다
- 속성을 부여할 수 없다
- object의 key 값으로 사용할 수 있다.
- 모든 심볼은 고유하다. 주석이 동일하다 하더라도 일단 생성되면 다르게 구별된다.
- 문자열로 자동으로 변환되지 않는다.

```javascript
const newSymbol = Symbol(
  "this symbol"
)`symbol is ${newSymbol}`//Uncaught TypeError: Cannot convert a Symbol value to a string
`symbol is ${String(newSymbol)} ${newSymbol.toString()}`;
// symbol is Symbol(this symbol) Symbol(this symbol)
```

## 심볼을 갖는 방법

- `Symbol()`을 호출한다. 이는 호출할 때 마다 새롭고 고유한 심볼을 만들어 준다.
- `Symbol.for(string)`을 호출한다. 이 메소드는 `Symbol Registry`라는 심볼목록을 참조하여 리턴하는데, 앞서와는 다르게 심볼 목록을 공유한다. `Symbol.for('호날도')`를 계속해서 호출한다면, 매번 같은 심볼을 리턴한다. 이는 심볼이 공유 되어야 하는 상황에서 유용하다.
- `Symbol.length` 처럼 표준에 정의된 심볼을 가져오는 법
