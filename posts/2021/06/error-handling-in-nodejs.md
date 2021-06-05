---
title: 'Nodejs에서 올바르게 에러 처리하기'
tags:
  - javascript
  - nodejs
published: true
date: 2021-06-05 20:56:10
description: 'SSR을 다루면서 에러처리에 대해 고민했던 나날들😑'
---

Nodejs에서 비동기 프로그래밍을 처음 접해보는 개발자들은, 종종 에러를 제대로 처리하는 방법에 대해서 혼돈을 느끼곤 한다. 비동기 처리의 경우 에러가 제대로 잡히지 않는 경우도 있다. 뭐 어쨌든, 애플리케이션에서 에러를 올바르게 처리했고, 그 에러를 성공적으로 발견했다고 가정하자. 그 다음에 중요한 것은, 방금 잡은 에러에 대해서 어떻게 해야하냐는 것이다. 그냥 로그만 남길까? 그 위로 에러를 한번 더 던져야 하나? 이 에러가 어디서 끝나야 하나? 그렇다면 어떻게? 만약 애플리케이션이 HTTP 요청을 하는 동안, 에러가 났고 이를 잡았다면, 해당 에러를 요청한 사람에게 보여줘야 하나?? 여기에는 수 많은 질문이 있을 수 있다.

먼저, 애플리케이션이 REST API를 다루고 있고, 네트워크를 통해 하나이상의 다른 서비스와 통신하는 nodejs 기반 마이크로 서비스라고 가정해보자. 그렇다면, 여기에서 우리가 다뤄야 하는 것은 무엇일까?

- 가능한 모든 에러의 결과물에 대해서 예측 가능해야 한다.
- 수동으로 개입ㅈㅇㄱ 하지 않아도 심각한 에러에서 스스로 복구 될 수 있어야 한다.
- HTTP 요청을 처리하는 동안 발생한 오류는, 클라이언트가 이를 기반으로 작업을 수행하는데 도움이 될 수 있는 최소한의 정보와 함께 클라이언트에 전달되어야 한다.
- 오류의 근본적인 원인을 쉽게 추적할 수 있고 디버깅하기도 쉬워야 한다.

## 1. 비동기 에러를 정확히 잡아야 한다

비동기 코드를 작성하는데 익숙하지 않다면, 비동기 상황에서 발생하는 에러를 처리하는 코드를 작성하기 어려울 수 있다. 일반적으로, 이를 처리하는 세가지 정도의 패턴이 있다.

- callback: [error-first callback](https://nodejs.org/api/errors.html#errors_error_first_callbacks) 접근법. 이 상황 에서는 `try-catch`가 별로 도움이 안된다.

```javascript
function myAsyncFunction(callback) {
  setTimeout(() => {
    callback(new Error('oops'))
  }, 1000)
}

myAsyncFunction((err) => {
  if (err) {
    // handle error
  } else {
    // happy path
  }
})
```

- promises와 promise callback을 사용하는 방법

```javascript
function myAsyncFunction() {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      reject(new Error('oops'))
    }, 1000)
  })
}

myAsyncFunction()
  .then(() => {
    // happy path
  })
  .catch((err) => {
    // handle error
  })
```

- `async-await`와 resolve promise (혹은 ES6 generator & yield)

```javascript
function myAsyncFunction() {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      reject(new Error('oops'))
    }, 1000)
  })
}

;(async () => {
  try {
    await myAsyncFunction()
    // happy path
  } catch (err) {
    // handle error
  }
})()
```

`await`을 사용하게 되면 조금 시나리오가 다르다. 다음 두 가지 예를 보자.

```javascript
// Example 1
try {
  return await myAsyncFunction()
} catch (err) {
  // await을 사용했기 때문에 여기서 함수에서 에러가 나면 잡히게 된다.
}

// Example 2
try {
  return myAsyncFunction()
} catch (err) {
  // promise가 resolve된게 아니고, 단순히 리턴이 되어버렸기 때문에 여기에 닿을 수가 없다.
}
```

따라서 비동기 함수에서 에러를 처리하려고 할 때는 조심해야 한다.

## 2. uncaught exception이나 unhandled rejections에 대해 적절하게 처리해야 한다.

진짜 열심히 코딩을 해서 대부분의 잠재적인 오류 시나리오를 쫀쫀하게 처리했다고 하더라고, 내가 예상했던 시나리오를 벗어나는 에러가 발생할 수 있다. 이러한 시나리오를 일괄적으로 파악해서 처리할 수 있다. 프로세스 객체에서 내보내는 두가지 이벤트 `uncaughtException`, `unhandledRejection`를 사용하면 가능하다. 그러나 이를 적절하게 활용하지 않는다면 예기치 못한 상황이 발생 될 수 있다.

`uncaughtException`, `unhandledRejection`는 애플리케이션이 지속할 수 없는 상황을 의미한다. 여기에 리스너를 달 때는, 아래 사항을 조심해야 한다.

- 에러에 대해 명확하게 로그를 남겨서 추후에 원인 파악을 할 수 있게 할 것 (로그 관리 시스템 또는 APM 서버에 전송)
- 애플리케이션을 강제로 종료하여, 프로세스 매니저나 도커 오케스트레이터가 이를 대체할 프로세스를 시작 할 수 있도록 한다.

위 두 상황에서 프로세스를 종료하지 않고 계속 애플리케이션을 실행한다면, 애플리케이션이 멈추거나 예기치못한 동작을 빚을 수 있다.

```javascript
process.on('uncaughtException', (err) => {
  logger.fatal('an uncaught exception detected', err)
  process.exit(-1)
})

process.on('unhandledRejection', (err) => {
  logger.fatal('an unhandled rejection detected', err)
  process.exit(-1)
})
```

## 3. 에러 마스킹

대부분의 개발자들이 저지르는 또하나의 실수중 하나는 오류를 마스킹 하여 콜스택 아래의 호출자가 오류가 발생했음을 인식하지 못하게 하는 것이다. 경우에 따라서 이렇게 해야하는 경우도 있지만, 무지성으로 이 작업을 수행해버리면 오류를 추적하고 진단하는 것이 불가능하게 되어 애플리케이션의 심각한 다운타임으로 이어진다.

```javascript
function processUsers() {
    try {
        const body = await client.get('http://example.com/users');
        const users = body.users || [];
        // do something with users
    } catch (err) {
       // handle error
       // client.get에서 에러가 나든 users처리하다 에러가 나든 상관없다면 이렇게 해도된다.
       // 그러나 이렇게 여러 경우의 수를 묶어버리면 에러에 대한 문제를 정확히 찾기 어렵다.
    }
}
```

이러한 코드는 에러에 대한 로그처리를 다른 곳에서 진행했고, 현재 함수로 부터 더이상 에러가 올라가도 되지 않아도 된다는 자신감이 있을 떄만 해야 한다. (HTTP 요청 에러가 클라이언트로 가지 말아야 한다거나) 그렇지 않으면, 어떠한 유형의 오류가 발생했는지 정확히 확인하고, 아래 호출자에서 무엇이 잘못되었는지 정확히 알 수 있도록 에러를 던져야 한다.

## 4. 제네릭 에러를 정확한 에러로 변환해야 한다.

애플리케이션이 에러의 유형에 따라 다른 행동을 해야 하는 경우, 에러 객체를 특정 에러 객체로 변환하는 것이 중요하다.

```javascript
if (err instanceof AuthenticationError) {
  return res.status(401).send('not authenticated')
}

if (err instanceof UnauthorizedError) {
  return res.status(403).send('forbidden')
}

if (err instanceof InvalidInputError) {
  return res.status(400).send('bad request')
}

if (err instanceof DuplicateKeyError) {
  return res.status(409).send('conflict. entity already exists')
}

// Generic error
return res.status(500).send('internal error occurred')
```

자바스크립트의 `Error`는 매우 제네릭하다.따라서 에러를 명확히 하기 위해서는, `error.message` `error.code` `error.stack` 속성을 확인해봐야 한다. 그러나 이는 꽤 규모있는 애플리케이션을 처리하는 경우에는 불편할 수 있다. Nodejs에는 `TypeError` `SyntaxError` `RangeError`와 같은 [구체적인 에러](https://nodejs.org/dist/latest-v12.x/docs/api/errors.html#errors_class_assertionerror)들이 몇가지 있다. 그러나 모든 경우에 이들을 재사용할 수 있는 건 아니다.

이를 위해서는, 나만의 에러 유형을 정의하고 적시에 올바른 에러를 던져야 한다.

```javascript
class UserServiceError extends Error {
  constructor(...args) {
    super(...args)
    this.code = 'ERR_USER_SERVICE'
    this.name = 'UserServiceError'
    this.stack = `${this.message}\n${new Error().stack}`
  }
}

class InvalidInputError extends Error {
  constructor(...args) {
    super(...args)
    this.code = 'ERR_INVALID_INPUT'
    this.name = 'InvalidInputError'
    this.stack = `${this.message}\n${new Error().stack}`
  }
}

async function getUser(userId) {
  if (!userId) throw new InvalidInputError('userId is not provided')

  try {
    return getUserFromApi(userId)
  } catch (err) {
    throw new UserServiceError(err.message)
  }
}
```

이렇게 처리하면, 다른 개발자들에게 에러 코드 목록을 보여주거나, 에러가 발생할 때 마다 코드를 확인하여 처리할 필요가 없다.

## 5. 외부 서비스의 예기치 못한 상황에 대처하기

만약 외부 서비스를 사용해야 하는 경우가 있다면, 가능한 잘못될 수 있는 모든 시나리오에 대처할 필요가 있다.

```javascript
function processUsers() {
    try {
        const body = await client.get('http://example.com/users');
        const users = body.users || [];
        // do something with users
    } catch (err) {
       // handle error
    }
}
```

위 예제에서, 사용자 목록을 불러오기 위해서는, api가 성공 응답(200)에서만 객체를 반환한다고 가정한다. 결과가 있다면 베열이 될 수도 있고, 없다면 null이 될 수도 있는 users 속성이 있다고 가정해보자.

만약 api 개발자들이 `body.users`가 외 다른 곳에서 결과가 오도록 객체의 응답구조를 변경한다면? 애플리케이션은 기본값 `[]`를 사용하여 계속 실행되며, 어떤일이 발생했는지 알 수 없게 된다.

항상 다른 서비스를 사용할 때는 엄격하게 대처할 필요가 있다. 비정상적인 방법으로 계속 서비스 하는 것보다, 애플리케이션이 빠르게 실패하는 것이 좋다. 이렇게 하면 잠재적으로 발생할 수 있는 문제를 빠르게 식별할 수 있고, 데이터 손상이나 불일치 등을 막을 수 있다.

## 6. 에러 별로 적절한 로그 레벨을 사용하기

적절한 로그 레벨을 선택하여 로그를 남기는 것은 중요하다. 모든 로그 라이브러리는 일반적으로 서로 다른 레벨의 로그를 기록할 수 있고, 각 수준의 로그를 다른 대상 (`stdout` `syslog` `file`) 으로 보낼 수 있다. 이 작업을 제대로 수행하기 위해서는, 메시지의 중요도에 따라 올바른 로그레벨을 선택해야 한다.

- `debug`: 심각하지 않는 메시지. 후에 디버그를 위해서 필요한 경우
- `info`: 성공(또는 실패하지 않는) 작업을 식별하는데 필요한 정보성 메시지
- `warn`: 즉각적인 액션이 필요하지 않은 경고성 메시지. 그러나 추후에 디버깅을 위해서 필요한 경우
- `error`: 즉각적인 액션이 필요한 모든 에러. 이 에러를 무시할 경우 심각한 시나리오로 이어지는 경우
- `fatal`: 서비스 중단 과 같은 중요 구성요소의 장애를 나타내는 모든 오류

위와 같은 규칙을 엄격히 준수한다면, 잘못된 경보가 울리지 않는 상태에서 중요하거나 필요한 문제를 즉시 식별할 수 있다.
