---
title: 더 빠른 async function 과 promises
tags:
  - javascript
  - nodejs
published: false
date: 2020-07-16 08:17:28
description: '[Faster async functions and
  promises](https://v8.dev/blog/fast-async)을 번역 요약한 글입니다. ```toc from-heading: 2
  to-heading: 3 ```  자바스크립트의 비동기 처리는 예전부터 특별히 빠르지 않다는 비판을 많이 받아 왔다. 설상가상으로,
  자바스크립트 애플리케이션 (특히 ...'
category: javascript
slug: /2020/07/faster-async-functions-and-promises/
template: post
---

[Faster async functions and promises](https://v8.dev/blog/fast-async)을 번역 요약한 글입니다.

## Table of Contents

자바스크립트의 비동기 처리는 예전부터 특별히 빠르지 않다는 비판을 많이 받아 왔다. 설상가상으로, 자바스크립트 애플리케이션 (특히 Node.js 서버) 에서 비동기 프로그래밍이 있을 때 라이브로 디버깅 하는 것은 결코 쉬운일이 아니다. 다행히도, 이러한 흐름에 변화가 있었다. 이 아티클에서는 V8에서 비동기 성능과 Promise를 최적화 한 방법을 살펴보고, 비동기 코드에 대한 디버깅 경험을 향상 시킨 방법에 대해서 소개한다.

<iframe width="640px" height="360px" src="https://www.youtube.com/embed/DFP5DKDQfOc" frameBorder="0" allow="autoplay; encrypted-media" allowFullScreen></iframe>

## 비동기 프로그래밍에 대한 새로운 접근법

### 콜백에서 Promise로, 그리고 async로

Promise가 자바스크립트에 등장하기 전까지, 특시 Node.js에서는 콜백 기반 API 들은 비동기 코드 처리를 위해 사용되었다. 예로 아래 코드를 살펴보자.

```javascript
function handler(done) {
  validateParams((error) => {
    if (error) return done(error)
    dbQuery((error, dbResults) => {
      if (error) return done(error)
      serviceCall(dbResults, (error, serviceResults) => {
        console.log(result)
        done(error, serviceResults)
      })
    })
  })
}
```

이러한 패턴은 모두가 아는 것처럼 이른바 `콜백 지옥` 이라고 불리우는, 콜백이 매우 중첩되어 있는 코드로, 이는 코드의 가독성과 유지보수성을 매우 떨어뜨린다.

운이 좋게도, Promise가 자바스크립트의 일부가 되면서, 위에서의 코드를 조금더 우아하고 유지보수 가능하도록 작성할 수 있다.

```javascript
function handler() {
  return validateParams()
    .then(dbQuery)
    .then(serviceCall)
    .then((result) => {
      console.log(result)
      return result
    })
}
```

더 최근 부터는, async 함수의 도움을 받을 수 있다. 위 비동기 코드는 더욱더 동기 코드처럼 작성할 수 있다.

```javascript
async function handler() {
  await validateParams()
  const dbResults = await dbQuery()
  const results = await serviceCall(dbResults)
  console.log(results)
  return results
}
```

async 함수를 사용하면, 코드는 간결해지고, 데이터의 흐름과 제어권을 보기 더 쉬워 지며, 여전히 비동기 코드로 작성할 수 있다.

> 자바스크립트의 실행은 여전히 단일 스레드에서 이뤄지며, async function이 뭔가 물리적 스레드를 새로 만들어서 처리하는 것이 아님을 명심하자.

### 이벤트 리스너 콜백에서 async iteration으로

Node.js에서 흔히 볼 수 있는 또다른 비동기 패러다임은 [ReadableStreams](https://nodejs.org/api/stream.html#stream_readable_streams)이다. 예를 들면 아래와 같다.

```javascript
const http = require('http')

http
  .createServer((req, res) => {
    let body = ''
    req.setEncoding('utf8')
    // data를 받아오려면 콜백함수에 접근해야함.
    req.on('data', (chunk) => {
      // 콜백함수에서 data에 접근
      body += chunk
    })
    // 종료 처리 역시 콜백 함수내에서 이뤄져야 함.
    req.on('end', () => {
      res.write(body)
      res.end()
    })
  })
  .listen(1337)
```

이 코드가 조금 읽기 어려울 수 있다. `data`는 콜백 내에서만 처리할 수 있는 `chunck`로 처리되며, 스트림 종료 처리는 콜백 내부에서 발생한다. 함수가 즉시 종료되고 실제 동작은 콜백에서 이뤄져야 한다는 사실을 인지하지 못하면 버그를 만들기 쉽다.

운 좋게도, ES2018 부터 [async iteration](https://2ality.com/2016/10/asynchronous-iteration.html) 이 도입되었고, 위 코드는 아래처럼 단순화 시킬 수 있다.

```javascript
http
  .createServer(async (req, res) => {
    try {
      let body = ''
      req.setEncoding('utf8')
      for await (const chunk of req) {
        body += chunk
      }
      res.write(body)
      res.end()
    } catch {
      res.statusCode = 500
      res.end()
    }
  })
  .listen(1337)
```

`data` `end` 라고 명명된 두개의 서로 다른 콜백내에서 데이터 처리 로직을 넣는 대신에, 모든 것을 한가지 async 함수 안에 넣어두고, 새롭게 만들어진 `for await...of`를 사용하여 chunk를 비동기적으로 순회하였다. 그리고 추가로 `try-catch`블록을 작성하여 `unhandledRejection` 에러를 처리하였다.

## Async의 성능 향상

v8 개발진은 v8 5.5에서 v8 6.8에 이르기까지 비동기 코드에 대한 성능 향상을 이뤄 왔다. 프로그래머들이 속도에 대한 걱정 없이 새로운 프로그래밍 패러다임을 안정적으로 쓸 수 있도록 제공하였다.

![doxbee benchmark](https://v8.dev/_img/fast-async/doxbee-benchmark.svg)

위 벤치마크는 무거운 Promise 작업을 수행한 코드다. 위 차트에서는 낮을수록 겅능이 더 좋은 것으로 볼 수 있다.

`parrallel 벤치마트에서는`Promise.all()에 대한 성능을 테스트 하였다.

![parallel benchmark](https://v8.dev/_img/fast-async/parallel-benchmark.svg)

성능이 8배 가까이 좋아진 것을 알 수 있다. 이것보다 V8 팀은 최적화가 실제 사용자 코드에 어떻게 영향을 미치는지 확인할 필요가 있다.

![real world](https://v8.dev/_img/fast-async/http-benchmarks.svg)

위 차트는 몇몇 유명한 HTTP 미들웨어 프레임워크를 대상으로 무거운 promise와 async를 테스트한 결과다. 이 차트는 초당 요청 속도 처리를 나타낸 것으로, 그래프가 높을 수록 성능이 좋은 것이다.

이러한 성능 향상에는 다음 세가지 요소의 도움을 받은 것이다.

- 새로운 최적화 컴파일러인 [TurboFan](https://v8.dev/docs/turbofan)
- 새로운 가비지 콜렉터 [Orinoco](https://v8.dev/blog/orinoco)
- `await`이 마이크로 틱에서 스킵되던 Node.js의 버그

TurboFan의 출시, 그리고 메인스레드에서 분리된 가비지 컬렉터 등이 성능에 도움을 주었지만, Node.js 8에서 일부 경우에 마이크로 틱을 건너뛰는 버그를 해결했다는 점도 한몫을 했다. 이 버그는 의도치 않은 스펙 위반으로 시작되었지만, 나중에 최적화 아이디어를 주게 되었다. 일단 이 버그가 무엇인지 살펴보자.

```javascript
const p = Promise.resolve()

;(async () => {
  await p
  console.log('after:await')
})()

p.then(() => console.log('tick:a')).then(() => console.log('tick:b'))
```

위 코드에서는 promise `p`를 만들고, `await`결과를 기다린다. 그리고 이 `p`에는 두가지 핸들러가 걸려있다. 이 코드의 실행 순서는 어떻게 될까?

![await bug](https://v8.dev/_img/fast-async/await-bug-node-8.svg)

이러한 결과가 직관적으로 보이긴 하지만, 스펙에 따르면 이 결과는 정확하지 않다. Node.js 10에서는 체인으로 연결된 핸들러를 먼저 실행하고, 그 이후에 비동기를 실행하는 계속하도록 변경되었다.

![nodejs10 no longer has the await bug](https://v8.dev/_img/fast-async/await-bug-node-10.svg)

이 정확한 결과는 즉각적으로 이해 되지는 않으며, 실제로 자바스크립트 개발자에게는 놀라운 일이었다. 따라서 설명을 들을 만 한다. Promise와 async의 세계로 뛰어들기 전에, 몇가지 기초를 살펴보자.

### Tasks와 Microtasks

자바스크립트의 상위 구조에는 tasks와 microtasks가 있다. tasks 는 I/O 및 타이머와 같은 이벤트를 처리하고 한번에 하나씩 실행한다. 마이크로 태스크는 async/await 및 promise에 주어진 비동기 실행을 구현하고, 각 태스크가 종료될때 마다 실행한다. 실행이 이벤트 루프로 돌아가기전에 마이크로 태스크 대기열은 항상 비워진다.

![마이크로태스크와 태스크의 차이](https://v8.dev/_img/fast-async/microtasks-vs-tasks.svg)

더 자세한 내용을 알고 싶다면, [이글](https://jakearchibald.com/2015/tasks-microtasks-queues-and-schedules/)을 확인해보면 된다.

### async 함수

MDN에 따르면, async 함수는 결과를 반환하는 암묵적인 promise를 이용해 비동기적으로 작동하는 함수다. async 함수는 비동기 코드를 동기 코드 처럼 보이게 위한 것으로, 개발자로 부터 비동기 처리의 복잡성 일부를 숨긴다.

가장 간단한 async 함수는 아래와 같다.

```javascript
async function computeAnswer() {
  return 42
}
```

만약 호출된다면 promise를 리턴하며, 이 함수의 값을 다른 어떤 promise로도 얻을 수 있다.

```javascript
const p = computeAnswer()
// → Promise

p.then(console.log)
// prints 42 on the next turn
```

이제 다음에 마이크로 태크스가 실행되면, promise `p`의 값을 얻을 수 있다. 다시 말해서, 아까 작성했던 코드는 `Promise.resolve`를 사용하는 것과 의미상 동등하다.

```javascript
function computeAnswer() {
  return Promise.resolve(42)
}
```

async 함수의 진정한 힘은 `await` 으로 부터 온다. `await`은 promise가 resolve 될 동안 함수 실행을 중단 시키며, 실행이 완료되면 다시 재개 한다. `await`의 값은 promise 실행의 결과다. 아래 예를 살펴보자.

```javascript
async function fetchStatus(url) {
  const response = await fetch(url)
  return response.status
}
```

`fetchStatus`의 실행은 `await`에서 중지되며, `await`의 `fetch` promise가 실행되고 난뒤에 재개된다. 이를 채이닝 헤핸들러를 사용하면 아래와 같다고 볼 수 있다.

```javascript
function fetchStatus(url) {
  return fetch(url).then((response) => response.status)
}
```

여기서 핸들러는 `async` 함수 다음에 있는 `await` 코드가 포함되어 있다.

보통 Promise는 await을 거치지만, 실제로는 임의의 자바스크립트 값에서 대기할 수 있다. 즉 `await`의 값이 실제로 `promise`가 아니더라도, promise로 변환된다.

```javascript
async function foo() {
  const v = await 42
  return v
}

const p = foo()
// → Promise

p.then(console.log)
// prints `42` eventually
```

더 흥미롭게도 `await`은 어떠한 `thenable`과도 사용할 있다. (then method에 있는 어아무 객체) 그리고 이것이 실제 promise인지 여부도 상관이 없다. 따라서 실제로 임의의 시간을 보내는 `sleep`을 비동기 `sleep`처럼 표현할 수도 있다.

```javascript
class Sleep {
  // sleep 시간을 받는다.
  constructor(timeout) {
    this.timeout = timeout
  }
  // 임의로 then이라고 불리우는 함수를 만들었다.
  then(resolve, reject) {
    const startTime = Date.now()
    setTimeout(() => resolve(Date.now() - startTime), this.timeout)
  }
}

;(async () => {
  // await 과 쓰면 실제 비동기 처럼 동작한다.
  const actualTime = await new Sleep(1000)
  console.log(actualTime)
})()
```

## await 동작의 이해

V8에서 `await`이 어떻게 처리되는지 이해하기 위해서는, 스펙을 살펴볼 필요가 있다. 아래 코드를 기준으로 살펴보자.

```javascript
async function foo(v) {
  const w = await v
  return w
}
```

위 함수가 호출되면, 파라미터 `v`를 `promise`로 감싸고, `promise`가 resolve 될 때 까지 async 함수의 동작을 멈춘다. 이 동작 이후에는, 함수의 실행이 다시 재개 되고, `w`에는 `promise`의 결과 값을 할당 받게 된다. 바로 이 값이 async 함수의 결과 값으로 리턴되게 된다.
