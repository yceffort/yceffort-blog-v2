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

### `setTimeout`

```javascript
const { logger } = require('./setup')
logger.clearLog()

setTimeout(() => {
  logger.write('timer callback')
}, 1000)
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

### nested `setTimeout`

```javascript
const { logger } = require('./setup')
logger.clearLog()

setTimeout(() => {
  logger.write('outer timer callback')
  setTimeout(() => {
    logger.write('inner timer callback')
  }, 1000)
}, 1000)
```

```bash
    (asyncId: 2) INIT (Timeout) (triggerAsyncId=1) (resource=Timeout)
    (asyncId: 2) BEFORE
outer timer callback
        (asyncId: 3) INIT (Timeout) (triggerAsyncId=2) (resource=Timeout)
    (asyncId: 2) AFTER
    (asyncId: 2) DESTROY
        (asyncId: 3) BEFORE
inner timer callback
        (asyncId: 3) AFTER
        (asyncId: 3) DESTROY
```

이 예제에서는, 바깥 쪽 Timeout 리소스가 트리거 되면, 또다른 Timeout 리소스를 트리거 한다. 내부 타이머의 Timeout리소스가 `triggerAsyncId` 2를 가지고 있고, 이는 외부 Timeout 리소스의 `asyncId`임을 할 수 있다. 이로 미루어보아 내부 타이머가 외부 타이머의 트리거로 실행되었음을 알 수 있다.

그러나, 사실은 외부 Timer리소스가 내부 Timer 리소스보다 먼저 없어졌다고 보는 것이 맞다. 그 이유는 외부 타이버가 내부 타이머의 실행을 기다리거나, 콜백일 실행되는 것을 기다리지 않기 떄문이다.

### clear `setTimeout`

```javascript
const { logger } = require('./setup')
logger.clearLog()

clearTimeout(
  setTimeout(() => {
    logger.write('timer callback')
  }, 1000),
)
```

```bash
    (asyncId: 2) INIT (Timeout) (triggerAsyncId=1) (resource=Timeout)
    (asyncId: 2) DESTROY
```

이 예제에서는, `BEFORE` 나 `AFTER`의 존재를 확인할 수는 없다. 왜냐하면 타이머가 즉시 제거 되었으며, 콜백 역시 `clearTimeout`의 호출로 인해 실행될 기회를 잃어버렸기 때문이다. 따라서, `before` `after` hook은 실행되지 않았다.

### `setInterval`

```javascript
const { logger } = require('./setup')
logger.clearLog()

let count = 0
let interval = null
interval = setInterval(() => {
  logger.write(`callback executed`)
  if (++count >= 3) {
    clearInterval(interval)
  }
}, 1000)
```

```bash
    (asyncId: 2) INIT (Timeout) (triggerAsyncId=1) (resource=Timeout)
    (asyncId: 2) BEFORE
callback executed
    (asyncId: 2) AFTER
    (asyncId: 2) BEFORE
callback executed
    (asyncId: 2) AFTER
    (asyncId: 2) BEFORE
callback executed
    (asyncId: 2) AFTER
    (asyncId: 2) DESTROY
```

`setTimeout`과 유사하게, `Timeout` 비동기 리소스를 생성한다. 그러나 `Timeout`는 `setInterval`로 만들어졌기 때문에 여기에서는 지속적인 비동기 리소스로 볼 수 있다. 지속적인 비동기 리소스의 경우 `before` `after`가 반복해서 호출될 수 있다. 이 예제에서는 3번 정도 호출하도록 되어있으므로, `before` `after`도 각각 3번씩 호출된다. `clearInterval`를 하게 된다면, `destroy`가 초훌되고 종료된다.

## Custom 비동기 리소스를 활용한 실질 적인 예제

지금까지는 NodeJS의 비동기 리소스인 `Timeout` 객체에 대해서만 다뤘다. `async_hooks` 모듈은 자바스크립트 내장 API인 `AsyncResource` 클래스를 사용하여 사용자가 직접 비동기 리소스를 만들어 쓸 수 있도록 도와준다.

### 자동으로 사라지는 Custom 비동기 리소스

```javascript
const { logger } = require('./setup')
const { AsyncResource, executionAsyncId } = require('async_hooks')
logger.clearLog()

class DBQuery extends AsyncResource {
  constructor(query) {
    super('DBQUERY', {
      triggerAsyncId: executionAsyncId(),
      requireManualDestroy: false, // This defaults to false even if not provided
    })
    this.query = query
  }

  executeQuery(callback) {
    this.runInAsyncScope(callback, null)
  }
}

const dbquery = new DBQuery()
dbquery.executeQuery(() => {
  logger.write('query executed!')
})

setTimeout(() => {
  // wait until the DBQuery instance is garbage collected...
}, 9999999)
```

- `requireManualDestroy`를 false로 지정해두었다. 리소스에 대한 `destroy` hook이 있다면, 리소스가 가비지 콜렉팅 될때 해당 hook이 자동으로 실행되어야 한다. 이 작업은 nodejs내부에서 직접 수행된다. 그리고 이 작업은 v8내부에 있는 리소스 객체인 **Weak Callback**이라고 하는 destroy hook에 등록되어 실행된다.
- 이 코드의 실행이 끝나면, 애플리케이션이 계속 살아 있게 하는 코드가 실행된다. 그 이유에 대해서는 나중에 설명한다.

이 코드를 NodeJS Cli 플래그인 `--trace-gc`와 함께 실행하면, 카비지 콜렉션 로그도 함께 볼 수 있다.

```bash
$ ode --trace-gc custom-async-resource-auto-destroy.js
[4848:0x60e5a70]       36 ms: Scavenge 2.5 (3.0) -> 2.1 (4.0) MB, 0.8 / 0.0 ms  (average mu = 1.000, current mu = 1.000) allocation failure
[4848:0x60e5a70]       53 ms: Scavenge 2.6 (4.5) -> 2.4 (5.3) MB, 1.1 / 0.0 ms  (average mu = 1.000, current mu = 1.000) task
[4848:0x60e5a70]     8163 ms: Mark-sweep (reduce) 2.4 (7.3) -> 1.8 (7.3) MB, 0.9 / 0.0 ms  (+ 1.4 ms in 9 steps since start of marking, biggest step 0.3 ms, walltime since start of marking 3 ms) (average mu = 1.000, current mu = 1.000) finalize incremental marking via task GC in old space requested
[4848:0x60e5a70]     8768 ms: Mark-sweep (reduce) 1.8 (4.3) -> 1.8 (4.8) MB, 2.4 / 0.0 ms  (+ 1.8 ms in 9 steps since start of marking, biggest step 0.4 ms, walltime since start of marking 4 ms) (average mu = 0.993, current mu = 0.993) finalize incremental marking via task GC in old space requested
```

```bash
    (asyncId: 2) INIT (DBQUERY) (triggerAsyncId=1) (resource=DBQuery)
    (asyncId: 2) BEFORE
query executed!
    (asyncId: 2) AFTER
    (asyncId: 3) INIT (Timeout) (triggerAsyncId=1) (resource=Timeout)
    (asyncId: 2) DESTROY
```

해당 리소스는 `destroy` 훅이 실행된 순간 [Mark-Sweep](https://v8.dev/blog/trash-talk#major-gc)이 트리거 되어 즉시 가비지 콜렉팅 되었다.

`setTimeout` 콜백은 `dbquery`객체를 사용하지 않으므로, `setTimeout`이 실행된 이후에는 `dbquery`에 대한 참조가 없어 가비지 콜렉팅이 수행된다.

만약 마지막에 타이머가 없다면, 애플리케이션은 즉시 종료되어 가비지 콜렉팅이 수행될 시간조차 없어질 것이다. 따라서, `destroy` 훅은 실행되지 않는다.

이번에는, `setTimeout`안에서 `dbquery`를 참조하는 코드를 작성해보자.

```javascript
const { logger } = require('./setup')
const { AsyncResource, executionAsyncId } = require('async_hooks')
logger.clearLog()

class DBQuery extends AsyncResource {
  constructor(query) {
    super('DBQUERY', {
      triggerAsyncId: executionAsyncId(),
      requireManualDestroy: false, // This defaults to false even if not provided
    })
    this.query = query
  }

  executeQuery(callback) {
    this.runInAsyncScope(callback, null)
  }
}

const dbquery = new DBQuery()
dbquery.executeQuery(() => {
  logger.write('query executed!')
})

setTimeout(() => {
  // Keep a reference to dbquery so that it won't be garbage collected
  console.log(dbquery.asyncId())
}, 9999999)
```

```bash
$ node --trace-gc custom-async-resource-auto-destroy-nogc.js
[10609:0x483c100]       31 ms: Scavenge 2.4 (3.0) -> 2.0 (4.0) MB, 0.7 / 0.0 ms  (average mu = 1.000, current mu = 1.000) allocation failure
[10609:0x483c100]       47 ms: Scavenge 2.6 (4.3) -> 2.4 (5.0) MB, 1.0 / 0.0 ms  (average mu = 1.000, current mu = 1.000) task
[10609:0x483c100]     8153 ms: Mark-sweep (reduce) 2.4 (7.0) -> 1.8 (7.0) MB, 0.9 / 0.0 ms  (+ 1.4 ms in 7 steps since start of marking, biggest step 0.4 ms, walltime since start of marking 3 ms) (average mu = 1.000, current mu = 1.000) finalize incremental marking via task GC in old space requested
[10609:0x483c100]     8758 ms: Mark-sweep (reduce) 1.8 (4.0) -> 1.8 (4.5) MB, 2.8 / 0.0 ms  (+ 1.6 ms in 8 steps since start of marking, biggest step 0.4 ms, walltime since start of marking 5 ms) (average mu = 0.993, current mu = 0.993) finalize incremental marking via task GC in old space requested
```

```bash
    (asyncId: 2) INIT (DBQUERY) (triggerAsyncId=1) (resource=DBQuery)
    (asyncId: 2) BEFORE
query executed!
    (asyncId: 2) AFTER
    (asyncId: 3) INIT (Timeout) (triggerAsyncId=1) (resource=Timeout)
```

가비지 콜렉션이 실행중이라 할지라도,리소스가 가비지 콜렉팅 되지 않았기 때문에 `destroy` 훅이 실행되지 않는 다는 것을 알 수 있다. 그 이유는 `setTimeout`안에 `dbquery` 객체의 참조가 유지되어 있어 `dequery`가 가비지 콜렉팅 되지 않도록 하고 있기 때문이다.

### 수동으로 destroy 되는 비동기 리소스

`requireManualDestroy`가 `true`가 되면 `destroy` 훅이 자동으로 실행되지 않고, 비동기 리소스에서 `emitDestroy()`를 호출해서 수동으로 제거해야 한다.

```javascript
const { logger } = require('./setup')
const { AsyncResource, executionAsyncId } = require('async_hooks')
logger.clearLog()

class DBQuery extends AsyncResource {
  constructor(query) {
    super('DBQUERY', {
      triggerAsyncId: executionAsyncId(),
      requireManualDestroy: true,
    })
    this.query = query
  }

  executeQuery(callback) {
    this.runInAsyncScope(callback, null)
  }

  destroy() {
    this.emitDestroy()
  }
}

const dbquery = new DBQuery()
dbquery.executeQuery(() => {
  logger.write('query executed!')
})
dbquery.destroy()

// Wait until the resource is manually destroyed
setTimeout(() => {}, 9999999)
```

```bash
$ node --trace-gc custom-async-resource-manual-destroy.js
[12212:0x48c00f0]       30 ms: Scavenge 2.4 (3.0) -> 2.0 (4.0) MB, 0.8 / 0.0 ms  (average mu = 1.000, current mu = 1.000) allocation failure
[12212:0x48c00f0]       48 ms: Scavenge 2.6 (4.3) -> 2.4 (5.3) MB, 0.9 / 0.0 ms  (average mu = 1.000, current mu = 1.000) task
```

```bash
    (asyncId: 2) INIT (DBQUERY) (triggerAsyncId=1) (resource=DBQuery)
    (asyncId: 2) BEFORE
query executed!
    (asyncId: 2) AFTER
    (asyncId: 3) INIT (Timeout) (triggerAsyncId=1) (resource=Timeout)
    (asyncId: 2) DESTROY
```

보이는 것처럼, `destroy` 훅은 `emitDestroy()`가 호출된 직후에 바로 실행되어졌다. 만약 `dbquery.destroy()`를 주석처리 하거나 없앤다면, `destroy` 훅은 객체가 가비지 콜렉팅 되어도 실행되지 않을 것이다.

## 마치며

이 외에도 HTTP request, 파일 시스템 접근, 암호화 작업과 같은 다른 유형의 비동기 작업이 있을 수 있다. 더 많은 예제를 [여기](https://github.com/deepal/async-hooks-demo)에서 살펴보자.
