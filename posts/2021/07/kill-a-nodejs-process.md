---
title: 'Nodejs 프로세스를 종료시키는 방법'
tags:
  - javascript
  - nodejs
published: true
date: 2021-07-16 17:28:35
description: '이사하느라 힘들었습니다.'
---

nodejs 프로세스가 종료되는 상황으로는 여러가지가 있다. 에러가 발생하는 케이스와 같이 사전에 예방할 수 있는 경우가 있고, 혹은 메모리 부족과 시스템 오류와 같은 예방할 수 없는 것이 있다. 이 Process Global은 Event Emitter 인스턴스이며, graceful exit가 실행되면, 종료 이벤트를 발생 (emit) 한다. 그러면 애플리케이션 코드가 이 이 벤트를 수신하여 마지막 순간에 동기로 일어나는 정리 작업을 할 수 있다.

다음은 프로세스 종료를 의도적으로 발생시킬 수 있는 몇가지 방법이다.

| Operation                   | 예시                       |
| --------------------------- | -------------------------- |
| 수동 프로세스 종료          | `process.exit(1)`          |
| Uncaught exception          | `throw new Error()`        |
| Unhandled promise rejection | `Promise.reject()`         |
| error event 무시            | `EventEmitter#emit('error')` |
| Unhandled Signals           | `$ kill <PROCESS_ID>`      |

이러한 오류 중 대부분은 `uncaught errors` `unhandled rejects`와 같이 실수로 발생되는 경우도 있지만, 이 들 중 일부는 프로세스를 직접 종료하기 위해 만들어 진 것이다.

## Process Exit

`process.exit(code)`는 프로세스를 종료하기 위한 가장 간단한 도구다. 프로세스의 수명이 다하여 종료시켜도 되는 경우에 스크립트를 작성할 떄 매우 유용하다. 이 코드는 선택사항이며, 기본값은 0 이고 0에서 255까지 선택 가능하다. 0은 성공적인 프로세스 실행을 나타내는 반면, 0이 아닌 숫자는 사고가 발생했다는 것을 나타낸다. 이러한 값은 다양한 외부 툴에서 사용된다. 예를 들어, 테스트를 실행 할 때, 0이 아니면 테스트가 실패한 것이다.

`process.exit`가 직접 실행되면, 콘솔에는 암묵적으로 텍스트가 출력되지 않는다. 오류를 알리기 위해 이 메서드를 호출하는 경우, 사용자가 직접 오류를 찍어야 한다.

```bash
$ node -e "process.exit(42)"
$ echo $?
```

이 경우, shell이 종료를 나타내긴 했지만, nodejs 애플리케이션에서는 이 메시지가 출력되지 않았다. 이렇게 되면 사용자는 무슨 일이 일어났는지를 알지 못한다. 따라서 아래와 같이 종료시키는 것이 좋다.

```javascript
function checkConfig(config) {
  if (!config.host) {
    console.error("Configuration is missing 'host' parameter!")
    process.exit(1)
  }
}
```

사용자는 이 경우 명확하게 이해할 수 있다. 콘솔에 에러가 찍히고, 사용자는 이 상황에 대해 이해하고 해걸할 수 있다.

`process.exit()`는 매우 강력한 도구다. 하지만 재사용 가능한 라이브러리에 이 코드를 사용해서는 안된다.라이브러리에서 오류가 발생하면 애플리케이션이 오류를 어떻게 할지 결정할 수 있도록 오류를 생성해야 한다.

## Exceptions, Rejections, 그리고 Emitted Error

`process.exit()`는 시작/설정단계에서 사용할 수 있는 강력한 도구인반면, 실행 단계에서는 다른 툴을 사용해야한다. 예를 들어, 애플리케이션이 http 요청을 처리할 때 발생하는 오류는 프로세스를 종료하지 않고 오류 응답만 반환해야 한다. 오류가 발생한 위치에 대한 정보를 노출하는 것도 필요하다. 따라서 여기서 던져진 오류 객체가 유용하다.

`Error` 클래스의 인스턴스에는 스택 추적 및 메시지 문자열과 같이 오류의 원인을 파악하는데 유용한 메타데이터가 포함되어 있다. `Error` 클래스를 기반으로 사용자가 고유의 애플리케이션 `Error` 클래스를 만들어서 확장해서 사용하는 것이 일반적이다. `Error`를 인스턴스화하는 것 자체로는 부수효과가 없다. (=별일이 일어나지 않는다.) 오류가 발생하기 위해서는, 이 `Error` 클래스를 던져야 한다.

에러는 `throw` 키워드를 사용해서 던지거나, 특정 논리적인 오류가 발행할때 나타난다. 이러한 상황이 나타나면 현재 스택은 `unwinds`가 된다. 이 뜻은 각 함수가 `try...catch` 가 감싸는 문구를 만날 때까지 종료됨을 의미한다. 만약 `try...catch`를 만나지 못한다면, 이 는 uncaught된 에러로 간주한다.

`throw` 키워드를 사용하여 `throw new Error('hi')`와 같이 에러를 던지는 것은, 기술적으로 무엇이든 던질 수 있다. 무엇이든 던져지게 되면 이는 예외로 간주된다. 이렇게 던져지는 에러 인스턴스는 이 인스턴스를 기반으로 에러의 속성을 예상할 수 있으므로, 에러 인스턴스를 생성하는 것이 중요하다.

Node.js 라이브러리 내부에서 널리 사용되는 또다른 패턴은, 릴리즈 간에 일관성을 유지하기 위한 `.code` 값을 제공하는 것이다. 일례로 `ERR_INVALID_URI`가 있는데, 사람이 읽을 수 있는 `message`는 바뀔 수 있지만, `.code` 는 바뀌지 않는다.

안타깝게도, 에러를 구분하는 방법 중 또다른 하나는 `.message` 프로퍼티를 사용하는 것인데, 이는 위험하고 오류가 발생하기 쉽다. Node.js에서는 모든 라이브러리에서 오류를 완벽하게 구분할 수 있는 방법은 없다.

uncaught 에러가 스택에 던져지면, 콘솔에 찍히고 프로세스가 종료되며, 종료 상태값은 1이다. 이러한 예외의 예제를 살펴보자.

```bash
/tmp/foo.js:1
throw new TypeError('invalid foo');
^
Error: invalid foo
    at Object.<anonymous> (/tmp/foo.js:2:11)
    ... TRUNCATED ...
    at internal/main/run_main_module.js:17:47
```

`process` 글로벌은 Event Emitter로 `uncapturedException` 이벤트를 수신하여 uncaught 에러를 처리하는데 사용한다.

```javascript
const logger = require('./lib/logger.js')
process.on('uncaughtException', (error) => {
  logger.send('An uncaught exception has occured', error, () => {
    console.error(error)
    process.exit(1)
  })
})
```

Promise Rejection은 에러를 던지는 것과 유사하다. Promise에서 `reject()` 메서드가 호출되거나, 비동기 함수내에서 에러가 던져지는 경우 사용된다.

```javascript
Promise.reject(new Error('oh no'))

;(async () => {
  throw new Error('oh no')
})()
```

```
(node:52298) UnhandledPromiseRejectionWarning: Error: oh no
    at Object.<anonymous> (/tmp/reject.js:1:16)
    ... TRUNCATED ...
    at internal/main/run_main_module.js:17:47
(node:52298) UnhandledPromiseRejectionWarning: Unhandled promise
  rejection. This error originated either by throwing inside of an
  async function without a catch block, or by rejecting a promise
  which was not handled with .catch().
```

`uncaught exception`와는 다르게, 이러한 거부로 인해 node.js v14 에서는 크래쉬하지 않는다. 그러나, 그 이후 버전부터는 프로세스가 크래쉬된다. 또한 , 이 이벤트는 다음과 같이 캐치할 수 있다.

```javascript
process.on('unhandledRejection', (reason, promise) => {})
```

Event Emitter는 nodejs에서 흔한 패턴으로, 라이브러리와 애플리케이션 등에서 기본 클래스에서 확장한 많은 객체들이 존재한다.

Event Emitter가 `error` 이벤트를 발생시켰는데 여기에 아무런 리스너가 없다면, Emitter가 내보낸 인수를 던진다. 그렇게 되면 에러가 나서 프로세스가 종료된다.

```
events.js:306
    throw err; // Unhandled 'error' event
    ^
Error [ERR_UNHANDLED_ERROR]: Unhandled error. (undefined)
    at EventEmitter.emit (events.js:304:17)
    at Object.<anonymous> (/tmp/foo.js:1:40)
    ... TRUNCATED ...
    at internal/main/run_main_module.js:17:47 {
  code: 'ERR_UNHANDLED_ERROR',
  context: undefined
}
```

작업하는 Event Emitter 인스턴스에서 에러 이벤트를 수신하여, 애플리케이션이 멈추지 않고 이벤트를 정상적으로 처리할 수 있도록 해야 한다.

## Signal

시그널은 운영체제에서 하나의 프로그램에서 다른 프로그램으로 작은 숫자 메시지를 보내기 위해 제공하는 메커니즘이다. 이러한 숫자는 상수 문자열로 참조되는 경우가 많다. 예를 들어, 시그널 `SIGKILL`은 숫자 9의 시그널을 나타낸다.

운영 체제에 따라 서로다른 시그널이 정의될 수 있지만, 아래 목록은 일반적으로 범용이다.

| 이름    | 숫자 | handleable | Node.js 동작 | 목적                                 |
| ------- | ---- | ---------- | ------------ | ------------------------------------ |
| SIGUP   | 1    | YES        | 종료         | 부모 터미널이 종료된 경우            |
| SIGINT  | 2    | YES        | 종료         | `Ctrl + C`로 터미널에 간섭하는 경우 |
| SIGQUIT | 3    | YES        | 종료         | `Ctrl + D`로 터미널을 끝내려는 경우  |
| SIGKILL | 9    | NO         | 종료         | 프로세스가 강제로 죽는 경우          |
| SIGUSR1 | 10   | YES        | 디버거 시작  | 사용자 정의 시그널 1                 |
| SIGUSR2 | 12   | YES        | 종료         | 사용자 정의 시그널 2                 |
| SIGUSR1 | 10   | YES        | 종료         | 정상종료                             |
| SIGUSR1 | 19   | NO         | 종료         | 프로세스가 강제로 멈추는 경우        |

프로그램에서 이러한 시그널 처리를 구현할 수 있도록 한 경우, Handleable이 YES 다. NO로 표시되어 있는 경우 처리할 수 없다. Node.js 동작은 신호가 수신되었을 때 Node.js 프로그램의 기본작업을 나타낸다. 마지막 열은, 일반적으로 어떻게 사용되는지 알려준다.

Node.js에서 이러한 시그널을 수신하기 위해서는, 아래처럼 `process`객체에 이벤트 리스너를 달면 된다.

```javascript
#!/usr/bin/env node
console.log(`Process ID: ${process.pid}`);
process.on('SIGHUP', () => console.log('Received: SIGHUP'));
process.on('SIGINT', () => console.log('Received: SIGINT'));
setTimeout(() => {}, 5 * 60 * 1000); // keep process alive
```

이 프로그램을 터미널에서 실행하고, `Ctrl+C`를 해보면, 프로세스가 죽지 않는다. 그 대신, `SIGINT`시그널을 받는다. 다른 터미널 창으로 가서, 프로세스 ID 값을 기준으로 

```bash
$ kill -s SIGHUP <PROCESS_ID>
```

를 실행하면, 이는 한 프로그램이 다른 프로그램으로 신호를 보낼 수 있다는 것을 알 수 있다. 이전 터미널에서 실행중인 node.js 프로그램이 SIGHUP  신호를 수신하여 인쇄한다.

눈치챘을 수도 있지만, Node.js 는 다른 프로그램에도 명령을 전송할 수 있다.

```bash
$ node -e "process.kill(<PROCESS_ID>, 'SIGHUP')"
```

이는 첫번쨰 프로그램에 `SIGHUP`를 표시하게 한다. 만약, 해당 프로세스를 종료 시키고 싶다면 아래 명령어를 통해서 `SIGKILL` 시그널을 보내면 된다.

```bash
$ kill -9 <PROCESS_ID>
```

이 시점에서, 애플리케이션은 종료된다.

이러한 시그널은 정상 종료 처리 이벤트를 처리하기 위해 Node.js 애플리케이션에서 많이 사용된다. 예를 들어 쿠버네틱스의 pod가 종료되면 애플리케이션에 `SIGTERM` 신호를 보낸다음, 30초 타이머를 시작한다. 그러면 애플리케이션이 30초 내에 정상적으로 종료되면서 연결을 닫고, 데이터를 저장할 수 있다. 타이머 이후에도 프로세스가 활성화 되어있으면 쿠버네틱스가 `SIGKILL`을 보낸다.

https://thomashunter.name/posts/2021-03-08-the-death-of-a-nodejs-process