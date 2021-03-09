---
title: '자바스크립트의 제네레이터와 regeneratorRuntime'
tags:
  - javascript
published: true
date: 2021-03-09 20:43:10
description: '아직도 자바스크립트 산을 기어 올라가는 중'
---

[이 전에 generator에 대해서 설명한 적이 있다.](https://yceffort.kr/2020/05/javascript-generator) 이번 포스팅에서는 제네레이터의 설명보다는, 이와 관련된 개념적인 이해와 제네레이터를 사용하기 위해 폴리필로 쓰이는 regeneratorRuntime에 대한 이야기를 해보려고한다.

## 블로킹 하지 않는 다는 것

아마도 '논 블로킹' 자바스크립트 코드를 짜는 것의 중요성을 들어 본적이 있을 것이다. Http 요청이나 데이터베이스 접근 같은 같은 I/O작업이 있을 경우, 일반적으로 콜백이나 프로미스를 사용하여 이를 처리한다. 블로킹이 일어나는 작업을 처리할 경우, 프로그램 전체가 마비되는 끔찍한 일을 마주할 수 있다. 만약 모든 사용자들이 시스템과 상호작용 하기 위해 자리가 빌 때 까지 대기해야 된다고 생각해보자. 🤬

또 하나 다른 이야기 해보자면, 자바스크립트 프로그램이 무한 루프에 들어가서 망해버리는 것이다. `node -e 'while(true){}` 를 실행해보자. 컴퓨터가 맛탱이가 가서 재시작이 필요할 것이다. (물론 어디까지나 개념적인 이야기 이다.)

이런 저런 배경지식들로 비춰봤을 때, 어떻게 es6의 제네레이터가 함수의 실행 중간에 호출을 '중지' 했다가 다시 미래에 '재개' 될 수 있냐는 물음이 생길 것이다. 또한 제네레이터 내에서 무한루프를 도는 코드도 멀쩡히 돌아가는 것을 볼 수 있다.

```javascript
const fibonacci = (function* () {
    let [prev, current] = [0, 1]

    while (true) {
        ;[prev, current] = [current, prev + current]
        yield current // 현재 값을 내보낸다.
    }
})()

const a = fibonacci.next()
console.log(a) // { value: 1, done: false }

const b = fibonacci.next()
console.log(b) // { value: 2, done: false }
```

>. 뭔가,, 이러나고 이씀,,,

얼핏 보면 무언가 자바스크립트의 새로운 버전이 나타나서 처리하는 것 같지만, 그렇지 않다. `regenerator`와 `babel`을 사용한다면, 일반 es5에서도 쉽게 이 코드를 사용할 수 있다.

```javascript
"use strict";

var fibonacci = /*#__PURE__*/regeneratorRuntime.mark(function _callee() {
  var prev, current, _ref;

  return regeneratorRuntime.wrap(function _callee$(_context) {
    while (1) {
      switch (_context.prev = _context.next) {
        case 0:
          prev = 0, current = 1;

        case 1:
          if (!true) {
            _context.next = 10;
            break;
          }

          ;
          _ref = [current, prev + current];
          prev = _ref[0];
          current = _ref[1];
          _context.next = 8;
          return current;

        case 8:
          _context.next = 1;
          break;

        case 10:
        case "end":
          return _context.stop();
      }
    }
  }, _callee);
})();
var a = fibonacci.next();
console.log(a); // { value: 1, done: false }

var b = fibonacci.next();
console.log(b); // { value: 2, done: false }
```

> 우리의 바벨과 regeneratorRuntime느님이 generator를 처리해주고 계시는 보습

![몬가 일어나고 잇음](https://mblogthumb-phinf.pstatic.net/MjAxOTA3MDFfMzAw/MDAxNTYxOTcxNzY2Mjg2.HubJEeou7vpe0OfwuPEbTCff66c4wvZJU0eMPpG9nqog.hagLHvBobHoeMq0JKRY0KVYVNbMjDE9-n1YyujKdw5kg.JPEG.ordo1194/1561971764732.jpg?type=w800)

일단, 제네레이터에 대한 설명은 아래에 자세히 나와있다.

- https://nodeschool.io/ko/
- https://yceffort.kr/2020/05/javascript-generator
- https://github.com/isRuslan/learn-generators

## 게으른 결과

간단한 예제로 시작해보자. 뭔가 내가 일련의 값들을 가지고 무언가를 해야 한다고 상상해보자. 이를 배열로 만들어서 사용하는 방법도 있을 것이다. 하지만 그 길이가 무한대라면? 배열로는 처리할 수 없다. 그러나 제네레이터로 가능하다.

```javascript
function* generateRandoms(max) {
  max = max || 1;

  while (true) {
    let newMax = yield Math.random() * max;
    if (newMax !== undefined) {
      max = newMax;
    }
  }
}
```

`function`뒤에 있는 `*`가 일반적인 함수와는 다른 제네레이터 함수임을 알려주고 있다. 다른 중요한 부분은 `yield`키워드다. 일반적인 함수는 `return`을 쓰지만, 제네레이터는 `yield`다. 제네레이터 함수는 결과를 `yield` 한 곳으로 넘겨준다.

우리는 이 함수가 의도하는 바가 *다음 값을 요청할 때마다, 0과 max 사이의 랜덤한 값을 리턴한다. 이를 프로그램이 끝날 때까지 반복한다* 라는 것을 알 수 있다. 여기서 프로그램이 끝날때란, 내 컴퓨터를 박살내거나 지구 종말이 오는 때를 의미한다. break나 return이 없는 `while(true)`란 그런 것이다.

이 처럼, 우리는 제네레이터를 통해서 값을 '요청' 할 때 딱 받을 수 있다. 이것은 매우 중요하다. 만약 그렇지 않으면, 무한히 증가하는 배열이 모든 메모리를 잡아먹을 수 있기 대문이다. 우리는 값을 `iterator`를 사용해서 얻을 수 있고, 이는 제네레이터 함수를 호출할 때 받을 수 있다.

```javascript
var iterator = generateRandoms();

console.log(iterator.next()); // {value: 0.8768122791044803, done: false}
console.log(iterator.next()); // {value: 0.06359353223017372, done: false}
```

제네레이터는 또한 양방향 커뮤니케이션을 지원한다. `let newMax = yield Math.random() * max;` 그리고 제네레이터는 누군가 사용하지 않는다면 중지된 상태로 남아있게 되며, 누군가 다음 값을 요청할 때 다시 활성화 된다. 만약 `iterator.next`를 호출해서 값을 넘기게 된다면, 그 값을 바탕으로 새로운 결과를 알려주게 된다.

```javascript
console.log(iterator.next(1000)); // {value: 368.0289602019955, done: false}
console.log(iterator.next(2000)); // {value: 21.21145723376827, done: false}
```

## es5의 제네레이터

제네레이터가 어떻게 동작하는지 알기 위해서는, es5로 어떻게 번역되는지를 살펴볼 필요가 있다. 이는 https://babeljs.io/repl 에서 쉽게 가능하다.

```javascript
"use strict";

var _marked = /*#__PURE__*/regeneratorRuntime.mark(generateRandoms);

function generateRandoms(max) {
  var newMax;
  return regeneratorRuntime.wrap(function generateRandoms$(_context) {
    while (1) {
      switch (_context.prev = _context.next) {
        case 0:
          max = max || 1;

        case 1:
          if (!true) {
            _context.next = 8;
            break;
          }

          _context.next = 4;
          return Math.random() * max;

        case 4:
          newMax = _context.sent;

          if (newMax !== undefined) {
            max = newMax;
          }

          _context.next = 1;
          break;

        case 8:
        case "end":
          return _context.stop();
      }
    }
  }, _marked);
}
```

보시다시피, 제네레이터 함수는 `switch` 블록으로 다시 쓰여졌음을 알 수 있다. 그리고 이것이 제네레이터가 동작하는 것에 대한 힌트다. 제네레이터를 일종의 루프안에 있는 상태관리 머신으로 볼 수 있으며, 이는 우리가 어떻게 상호작용 하느냐에 따라 달라진다. `_context`는 현재 상태값을 가지고 있으며, 어떤 case 문이 실행되어야 하는지도 정해준다.

위 코드를 이해하는 쉬운방법은, `case`문을 라인넘버라 보고, `_context.next`를 `GOTO` 문으로 보는 것이다. 

- `case 0`: `max`를 초기화 하고 `case 1`로 간다.
- `case 1`: 랜덤 값을 `yield`하고, 다음번에 실행한다면 4번으로 간다. 
- `case 4`: iterator가 값을 보내줬는지 (`_context.sent`) 확인하고, 그렇다면 `max`를 갱신한다. 그리고 `GOTO 1`로 해서, 다음 랜덤 값을 생성한다.

이것이 `블로킹 하지 않는다` 라는 룰을 준수하면서, 제네레이터가 무한히 루프를 돌면서도 중지되고 재개될 수 있는지를 나타내는 원리다.

## `(!true)`?

한가지 이상한 코드가 있다.

```javascript
if (!true) {
  _context.next = 8;
  break;
}
```
여기선 무슨일이 일어나는 걸까? 이는 우리의 `while(true)`가 어떻게 다시 쓰이는지를 나타낸다. 상태 머신이 루프 할 때 마다 매번 끝이 났는지를 확인한다. 이 예제에서는 절대 그럴 수 없지만, 간혹 제네레이터에 종료절이 필요할 때가 있다. 그럴 때 제네레이터를 멈추는 것이 `case 8`이다. 즉, 제네레이터가 종료하는 경우가 생기게 된다면 `(!true)`대신 종료에 대한 조건이 생길 것이다.,

## 이터레이터의 로컬 상태

한 가지 더 흥미로운 것은, 제네레이터가 어떻게 각 이터레이터의 local state를 보관하고 있는 지다. `newMax`는 `regeneratorRuntime.wrap` 스코프 밖에서 클로져 형태로 존재하므로, `iterator.next()`가 호출되도 계속 값을 유지하고 있을 수 있다. `randomNumbers()` 호출로 새로운 이터레이터가 만들어지면, 또다른 클로져가 만들어진다. 이는 어떻게 각 이터레이터가 동일한 제네레이터를 사용하여 영향을 주지 않고 자신의 상태값을 가지고 있을 수 있는지 보여준다.

## 코드 내부

`switch` 코드 내부도 사실, `regeneratorRuntime.wrap`와 `regeneratorRuntime.mark`에 의해 래핑된 것을 볼 수 있다. 이 코드는 https://github.com/facebook/regenerator 에서 만들어진 모듈로, es5에서도 es6의 제네레이터 함수가 올바르게 동짝 할 수 있도록 도와주는 코드다.

`regeneratorRuntime`에는 많은 흥미로운 코드가 있지만, 먼저 우리는 `Suspended Start`에서 제네레이터의 수명이 시작되는 것을 볼 수 있다.

https://github.com/facebook/regenerator/blob/0c2aba1af78be03da05de96b6c69f231b85993dc/packages/regenerator-runtime/runtime.js#L243-L246

```javascript
function makeInvokeMethod(innerFn, self, context) {
  var state = GenStateSuspendedStart;

  return function invoke(method, arg)  {
    // ...
  }
}
```

여기에서는 단순히 함수를 만들고 리턴한다. 그 말인 즉, `var iterator = generateRandoms()`를 하더라도, `generatorRandoms` 내부에서는 사실 처음 값을 요청할 때 까지는 내부의 어떤 것도 실제로 실행되지 않는다.

제네레이터 함수의 `iterator.next()`를 호출하면, 아래의 코드가 실행된다.

```javascript
var record = tryCatch(innerFn, self, context);
```

https://github.com/facebook/regenerator/blob/0c2aba1af78be03da05de96b6c69f231b85993dc/packages/regenerator-runtime/runtime.js#L293

만약 결과가 `throw`가 아니고 일반적인 `return`이라면, 이 결과를 이터러블 하도록 `{value, done}`으로 감싼다. 그리고 종료 여부에 따라서 상태를 `GenStateCompleted` 나 `GenStateSuspendedYield`로 세팅해둔다. 우리 코드의 경우, 종료는 없으므로 `GenStateSuspendedYield` 상태가 될 것이다.

```javascript
  if (record.type === "normal") {
    // If an exception is thrown from innerFn, we leave state ===
    // GenStateExecuting and loop back for another invocation.
    state = context.done
      ? GenStateCompleted
      : GenStateSuspendedYield;

    if (record.arg === ContinueSentinel) {
      continue;
    }

    return {
      value: record.arg,
      done: context.done
    };
```

https://github.com/facebook/regenerator/blob/0c2aba1af78be03da05de96b6c69f231b85993dc/packages/regenerator-runtime/runtime.js#L294-L308


## 결론

우리는 간단히 제네레이터를 활용하여 잠재적으로 무한한 값의 시퀀스를 만드는 코드를 만들었고, 이는 게으르게 (원하는 때에) 사용될 수 있다. 이는 `regeneratorRuntime`을 활용한다면 구형 브라우저에서도 지금 바로 사용할 수 있는 코드다. 