---
title: '비동기 리소스 (async resources)와 비동기 훅 (async hooks) 이해하기'
tags:
  - javascript
  - nodejs
published: true
date: 2021-08-14 00:20:56
description: '비동기로 불타는 금요일'
---

## Table of Contents

## Introduction

실제 우리가 사용하는 nodejs 애플리케이션은 비동기 작업, 그리고 이로 인해 만들어지고 없어지기를 반복하는 비동기 리소스 등으로 인해 매우 복잡하게 운영되고 있을 수도 있다. 때문에 코드에서 이러한 비동기 리소스의 라이프 사이클을 확인하는 기능은 애플리케이션에 대하나 통찰력, 그리고 실제 실행 가능한 성능 및 잠재적인 최적화 정보를 제공할 수 있기 때문에 매우 유용할 수 있다.

이러한 고급기능을 사용하기 위해 [AsyncListener](https://github.com/nodejs/node-v0.x-archive/pull/6011)라 든가 [async_wrap](https://github.com/nodejs/node-v0.x-archive/commit/709fc160e5) 와 같은 것들을 통해 많은 시도가 있었다. 그리고 감사하게도, 이제는 비동기 리소스의 라이프 사이클을 추적할 수 있는, 매우 성숙하나 기능인 `async_hooks`을 갖게 되었다. 지금 2021-08-14 00:26:43 을 기준으로도 여전히 실험 단계 이긴 하지만 (nodejs 16.6.2 기준), 꽤 많이 다듬어졌고, 형태를 갖추고 있다.

`async_hooks`는 다양한 기능을 가지고 있지만, 그 중에 가장 흥미로운 것은 파일 읽기, http 요청, http server 생성, 데이터 베이스에 쿼리 수행과 같은 응용프로그램에서 자주 수행하는 작업에서 어떤 일들이 벌어지는지 쉽게 이해할 수 있다는 것이다.

같이 보면 좋은 글 들

- https://nodejs.org/api/async_hooks.html
- https://itnext.io/a-pragmatic-overview-of-async-hooks-api-in-node-js-e514b31460e9

## 비동기 리소스의 생명주기

비동기 리소스는 비동기 작업의 일부로 생성된다. 비동기 리소스는 비동기 작업을 추적하는데 사용되는 객체에 불과하다. 따라서 작업이 완료되면 자연스럽게 실행되는 콜백과 연결된다. 비동기 리소스가 이 용도에 맞게 작동되면, 다른 객체와 마찬가지로 가비지 컬렉팅되어 사라진다.

가장 간단한 예제로 `setTimeout`을 들 수 있다. `setTimeout`은 비동기 리소스인 `Timeout`을 리턴하는데, 이는 타이머를 함수의 리턴 값으로 추적하기 위해 사용된다. nodejs repl에서 `setTimeout`을 호출해보자.

```bash
> setTimeout(()=>{}, 1000)
Timeout {
  _idleTimeout: 1000,
  _idlePrev: [TimersList],
  _idleNext: [TimersList],
  _idleStart: 6720,
  _onTimeout: [Function (anonymous)],
  _timerArgs: undefined,
  _repeat: null,
  _destroyed: false,
  [Symbol(refed)]: true,
  [Symbol(kHasPrimitive)]: false,
  [Symbol(asyncId)]: 26,
  [Symbol(triggerId)]: 5
}
```

이 `Timeout` 객체에는 다음과 같은 정보가 담겨있다.

- `timeout` 값인 `_idleTimeout`
- Timer callback `_onTimeout`
- `timer`와 `interval`인지 를 구분하는 값인 `_repeat`
- 현재 `timeout`이 활성화 중인지 여부를 나타내는 `_destroyed`
- ....

일반적인 비동기 리소스의 생명주기는 다음과 같다.

1. 생성됨
2. 콜백이 실행됨
3. 없어짐

`async_hook`을 사용하면, 콜백 함수를 붙일 수 있는 `hooks`을 제공하여, 위 생명주기의 여러 단계를 살펴볼 수 있다. `hook`에는 `init` `before` `after` `destroy`와 같은 네가지 타입이 있다. 이 단계는 위 생명주기에서 다음 과 같은 순서로 실행된다.

1. 생성됨 `init()`
2. `before()` 콜백이 실행됨 `after()`
3. 없어짐 `destroyed()`

비동기 리소스가 얼마나 지속 되느냐에 따라 비동기 리소스 콜백은 0번에서 여러번까지도 실행될 수 있다. 따라서 특정 비동기 리소스의 hook이 여러번 실행되거나, 혹은 실행이 아예 안될 수도 있다. 위 예제에서 `setTimeout`는 한번씩 호출되지만, `setInterval`을 사용하면 `init`뒤에 여러번 반복해서 실행될 수 있다.

예를 들어, `setTimeout`과 `setInterval`은 모두 `Timeout`이라고 불리는 비동기 리소스를 만든다. 하지만 아래와 같은 차이가 있다.

```bash
> setInterval(()=>{}, 1000)
Timeout {
  _idleTimeout: 1000,
  _idlePrev: [TimersList],
  _idleNext: [TimersList],
  _idleStart: 23081,
  _onTimeout: [Function (anonymous)],
  _timerArgs: undefined,
  _repeat: 1000,
  _destroyed: false,
  [Symbol(refed)]: true,
  [Symbol(kHasPrimitive)]: false,
  [Symbol(asyncId)]: 138,
  [Symbol(triggerId)]: 5
}
```

`_repeat` 속성에 숫자값 1000이 들어가 있어서 걔쏙해서 반복될 것임을 알 수 있다.

`Promise`의 경우에는 조금 다른데, 여기엔 `promiseResolve`라는 훅이 있어 `resolved`나 `rejected` 직후에 실행된다. 아래 순서를 보자.

1. 생성됨 `init()`
2. Promise가 resolve되거나 reject됨 `promiseResolve()`
3. `before()` 콜백이 실행됨 `after()`
4. 없어짐 `destroy()`

## 실제 애플리케이션에서의 비동기 리소스

실제 우리가 사용하는 애플리케이션에서는, 비동기 리소스는 그 생명주기 동안 많은 async hooks을 트리거 할 수 있다. 아래 http request 핸들러를 살펴보자.

```javascript
app.post("/user", (req, res) => {
  db(req.body, (err, stored) => {
    if (err) {
      return res.sendStatus(500)
    }

    notifyUpstream(stored, (err) =. {
      if (err) {
        return res.sendStatus(500)
      }
      res.sendStatus(201)
    })

    logger.log('stored in database')
  })
})
```

이 http request handler가 하는 작업은 아래와 같다.

- `http` 를 통해서 데이터를 받음
- 데이터베이스에 저장
- `http`를 통해 업스트림 서비스에 알림
- 메시지 로깅

위 네가지 작업이 4개의 비동기 리소스를 생성한다고 가정해보자.

- `DB Operation` 작업은 `HTTP Client Request`와 `Logging` 보다는 오래 걸리지 않을 것이다. 왜냐면 `storeInDb` 함수가 `notifyUpstream`과 `logger.log`의 작업이 완료 될 때 까지 기다리지 않기 떄문이다.
- `Logging`도 마찬가지로 비동기 작업인데, 비동기 리소스를 만들기 때문이다. 다만 이 리소스는 다른 것에 비해 생명주기가 짧다.
- `Incoming HTTP Request`는 가장 마지막에 없어지는 리소스가 될 것이다. `notifyUpstream`가 완료되고 응답이 완전히 전송된 후에만 완료되기 때문이다.

## 타이머를 쓰는 실제 예제

이제 비동기 리소스의 라이프 사이클을 이론적으로 몇가지 살펴보았으므로, 몇가지 실제 사례를 살펴보자. 이 데모에서는 async_hooks가 사용된 몇가지 코드 예제를 사용할 것이다. 

### 타이머 설정하기

```javascript
const { logger } = require("./setup");
logger.clearLog();

setTimeout(() => {
  logger.write("timer callback");
}, 1000);
```

```bash
    (asyncId: 2) INIT (Timeout) (triggerAsyncId=1) (resource=Timeout)
    (asyncId: 2) BEFORE
timer callback
    (asyncId: 2) AFTER
    (asyncId: 2) DESTROY

```

이 결과에 따르면

- 비동기 리소스 `Timeout`은 `setTimeout`이 호출되었을 떄 초기화 되었다.
- `1000ms`가 만료되기 전에, `before` async hook이 타이머 콜백이 실행되기 직전에 실행되었다.
- Timer 콜백이 실행되고, `timer callback`이 로깅 되었다.
- Timer 콜백이 실행된 이후에, `after` async hook이 실행되었다.
- `Timeout` 리소스가 사라지기 직전에 `destroy` async hook이 실행되었다.