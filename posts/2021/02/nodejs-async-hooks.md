---
title: 'Nodejs의 Async hook'
tags:
  - javascript
  - nodejs
published: false
date: 2021-02-03 20:57:53
description: '2021년 2월 현재 Experimental 상태에 있는 기능 (v15.8 기준)'
---

Nodejs의 `async_hook` 은 비동기로 실행되는 리소스에 대해서 추적이 용이하도록 만들어진 훅이다.

## 사용법

- `init`: 특정 비동기 리소스가 초기화 될 때 호출된다.
- `before` `after`: 비동기 리소스가 실행되기 직전, 혹은 그 이후에 실행된다.
- `destroy`: 콜백함수에서 무엇을 하던지간에, 비동기 리소스가 끝나면 실행된다.
- `promiseResolve`: `Promise` 가 `resolve` 함수를 호출하면, 훅이 이 함수를 호출한다.

설명만으로는 완전히 이해가 어려우니까 예제를 살펴보자.

```javascript
const fs = require('fs')
const async_hooks = require('async_hooks')

// 콘솔에 기록한다
// 그러나 단순히 console.log()를 쓰지 않은 이유는,
// console.log도 마찬가지로 비동기로 동작하기 때문이다.
// 따라서 여기에 console.log를 쓴다면 계속해서 `init`을 트리거 해서 무한 루프가 돌 것이다.
const writeSomething = (phase, more) => {
  fs.writeSync(
    1,
    `Phase: "${phase}", Exec. Id: ${async_hooks.executionAsyncId()} ${
      more ? ', ' + more : ''
    }\n`,
  )
}

// hook을 만들고 각각을 정의한다.
const timeoutHook = async_hooks.createHook({
  init(asyncId, type, triggerAsyncId) {
    writeSomething(
      'Init',
      `asyncId: ${asyncId}, type: "${type}", triggerAsyncId: ${triggerAsyncId}`,
    )
  },
  before(asyncId) {
    writeSomething('Before', `asyncId: ${asyncId}`)
  },
  destroy(asyncId) {
    writeSomething('Destroy', `asyncId: ${asyncId}`)
  },
  after(asyncId) {
    writeSomething('After', `asyncId: ${asyncId}`)
  },
})
timeoutHook.enable()

writeSomething('Before call')

// Set the timeout
setTimeout(() => {
  writeSomething('Exec. Timeout')
}, 1000)
```

실행 결과는 아래와 같다.

```bash
Phase: "Before call", Exec. Id: 1
Phase: "Init", Exec. Id: 1 , asyncId: 2, type: "Timeout", triggerAsyncId: 1
Phase: "Before", Exec. Id: 2 , asyncId: 2
Phase: "Exec. Timeout", Exec. Id: 2
Phase: "After", Exec. Id: 2 , asyncId: 2
Phase: "Destroy", Exec. Id: 0 , asyncId: 2
```

모니터링 툴이나 이미 사용중인 로그 추적 툴에 데이터를 입력하는 것을 추적하기 위해 굉장히 유용하게 쓰일 수 있다.

## Promise 예제

위에 있던 코드 예제에서, `setTimeout`대신에, 아래의 코드를 넣어보자.

```javascript
const calcPow = async (n, exp) => {
  writeSomething('Exec. Promise')

  return Math.pow(n, exp)
}

;(async () => {
  await calcPow(3, 4)
})()
```

```bash
Phase: "Init", Exec. Id: 1 , asyncId: 2, type: "PROMISE", triggerAsyncId: 1
Phase: "Init", Exec. Id: 1 , asyncId: 3, type: "PROMISE", triggerAsyncId: 1
Phase: "Exec. Promise", Exec. Id: 1
Phase: "Init", Exec. Id: 1 , asyncId: 4, type: "PROMISE", triggerAsyncId: 3
Phase: "Before", Exec. Id: 4 , asyncId: 4
Phase: "After", Exec. Id: 4 , asyncId: 4
```

우리의 예제에서는 `Promise`가 두개 밖에 없지만, 어쩐지 결과에서는 `Init`이 4번 호출되었다. 이는 nodejs 팀이 버전 12에서 부터 비동기 실행 성능을 향상 시키기 위한 작업의 결과물이다. 자세한 내용은 [여기](https://v8.dev/blog/fast-async)에 있다.

그것을 제외하고는, 실행이 우리의 예측대로 흘러갔따.

## 프로파일링

이 훅을 활용하면, 비동기 함수가 실행되는데 소요된 시간을 측정할 수 있다.
