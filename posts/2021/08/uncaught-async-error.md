---
title: 'uncaught async error를 올바르게 처리하기'
tags:
  - javascript
published: true
date: 2021-08-23 13:21:41
description: 'async가 있으면 함수 실행이 뒤로 넘어간다니까요?'
---

## Async IIFE

먼저, 즉시 실행 함수내에서 에러를 던지고 이 에러를 잡아보자.

```javascript
try {
  ;(() => {
    throw new Error('error')
  })()
} catch (e) {
  console.log(e) // caught
}
```

무사히(?) 에러가 잡히는 모습을 볼 수 있다.

하지만 여기에 `async` 키워드를 추가하면 어떻게 될까?

```javascript
try {
  ;(async () => {
    throw new Error('err') // uncaught
  })()
} catch (e) {
  console.log(e)
}
```

같은 코드에 `async`만 추가했을 뿐인데, 에러가 잡히지 않는 모습이다. 왜 그럴까?

동기 코드에서는, 에러가 동기로 발생하기 때문에, `try...catch` 문에서 잡을 수 있었다. 단순하게 이야기하면, 프로그램 실행이 `try...catch`를 벗어나지 않기 때문에 에러를 잡을 수 있었던 것이다.

하지만 비동기 함수의 경우는 다르다. 여기서 동기 작업이라 함은 단순히 `Promise`객체를 만들고 이를 함수의 마지막에 실행하는 것 뿐이다. `try...catch` 문구는 에러가 던져지는 시점에서는 이미 끝나있고, 따라서 여기에서 잡히지 않는다.

따라서 이를 해결 하기 위해서는, 아래 두 가지 방법으로 해결이 가능하다.

```javascript
;(async () => {
  throw new Error('err')
})().catch((e) => {
  console.log(e) // caught
})
```

```javascript
;(async () => {
  try {
    throw new Error('err')
  } catch (e) {
    console.log(e) // caught
  }
})()
```

요것은 https://yceffort.kr/2021/02/run-await-return-return-await 이것과 좀 비슷하다.

## Async forEach

또 한가지 다른 것은 async `forEach`다. 아래 코드는 앞서 이야기한 것 처럼 동기 코드이기 때문에 에러가 잘 잡힌다.

```javascript
try {
  ;[1, 2, 3].forEach((index) => {
    throw new Error(`err ${index}`)
  })
} catch (e) {
  console.log(e) // caught
}
```

그러나 역시 이 것도 비동기로 바꾸게 되면 에러가 잡히지 않게 된다.

```javascript
try {
  ;[1, 2, 3].forEach(async (index) => {
    throw new Error(`err ${index}`)
  })
} catch (e) {
  console.log(e)
}
```

```bash
Uncaught (in promise) Error: err 1
Uncaught (in promise) Error: err 2
Uncaught (in promise) Error: err 3
```

이 경우에는 `await Promise.all`을 사용한다. 그런데 여기서 조금 다른게 있다. `map`을 썼을 때와 `forEach`를 썼을 때 차이다.

`forEach`

```javascript
try {
  await Promise.all(
    [1, 2, 3].forEach(async (index) => {
      throw new Error(`err ${index}`)
    }),
  )
} catch (e) {
  console.log(e) // undefined is not iterable (cannot read property Symbol(Symbol.iterator))
}
```

`map`

```javascript
try {
  await Promise.all(
    [1, 2, 3].map(async (index) => {
      throw new Error(`err ${index}`)
    }),
  )
} catch (e) {
  console.log(e) // caught Error: err 1 이후 루프를 돌지 않음
}
```

어떤일이 일어나는지 정확히 알기 위해, `console.log`를 추가해 보자.

```javascript
try {
  await Promise.all(
    [1, 2, 3].forEach(async (index) => {
      console.log('forEach', index)
      throw new Error(`err ${index}`)
    }),
  )
} catch (e) {
  console.log(e) // undefined is not iterable (cannot read property Symbol(Symbol.iterator))
}
```

```
forEach 1
forEach 2
forEach 3
TypeError: undefined is not iterable (cannot read property Symbol(Symbol.iterator))
    at Function.all (<anonymous>)
    at <anonymous>:2:16
```

`forEach`는 `break`가 없다. 즉 중간에 도망갈 수 없는 loop 구문이다. 따라서 exception 유무와 상관없이 다 돌게 된다. 그러므로 `Promise.all`을 사용해야 하는 상황에서는 일반적으로 `forEach`대신 `map`을 쓴다.

- https://262.ecma-international.org/6.0/#sec-array.prototype.foreach

> There is no way to stop or break a forEach() loop other than by throwing an exception. If you need such behavior, the forEach() method is the wrong tool.

> `return false`를 쓰면 forEach를 나올 수 있다는 포스팅도 종종 보이는데, 사실 이건 엄밀히 말하면 그렇게 보이는 것 뿐이다.

```javascript
function hello() {
  ;[1, 2, 3].forEach((index) => {
    console.log(`${index} 도는 중`)
    return false
  })
}
```

```bash
1 도는 중
2 도는 중
3 도는 중
```

```javascript
try {
  await Promise.all(
    [1, 2, 3].map(async (index) => {
      console.log('forEach', index)
      throw new Error(`err ${index}`)
    }),
  )
} catch (e) {
  console.log(e) // undefined is not iterable (cannot read property Symbol(Symbol.iterator))
}
```

## Promise Chaining

비동기 함수는 비동기 작업을 수행하기 위하여 Promise에 의존한다. 따라서, `.then(onSuccess, onError)` 콜백에서도 비동기 함수를 사용할 수 있다.

> 이와 관련된 포스팅: https://yceffort.kr/2021/07/promise-then-f-f-vs-promise-catch

아래 코드에서는 에러가 잡히지 않지만

```javascript
Promise.resolve().then(
  /*onSuccess*/ () => {
    throw new Error('err') // uncaught
  },
  /*onError*/ (e) => {
    console.log(e)
  },
)
```

별도로 이렇게 `catch` 문이 빠져 있다면 잡을 수 있게 된다.

```javascript
Promise.resolve()
  .then(
    /*onSuccess*/ () => {
      throw new Error('err')
    },
  )
  .catch(
    /*onError*/ (e) => {
      console.log(e) // caught
    },
  )
```

## Early Init

잡히지 않는 예외의 또다른 케잇스는 promise와 await을 분리하여 병렬로 실행하는 것이다. `await`은 `async` 함수의 실행만을 중지해서 실행하므로, 이경우 병렬화가 일어나버리게 된다. 아래 예제를 살펴보자.

```javascript
const wait = (ms) => new Promise((res) => setTimeout(res, ms))

;(async () => {
  try {
    const p1 = wait(3000).then(() => {
      throw new Error('err')
    }) // uncaught
    await wait(2000).then(() => {
      throw new Error('err2')
    }) // caught
    await p1
  } catch (e) {
    console.log(e)
  }
})()
```

이 경우에는 두 개의 `await`을 모두 기다리지 않는다. 하나에서 error가 나버리면, `try...catch`로 해당 에러를 잡아버리고, 그 다음으로 넘어가버리게 된다. 따라서 나머지 하나의 에러는 잡히지 않게 된다.

```bash
Error: err2
Uncaught (in promise) Error: err
```

이 경우에도, 마찬가지로 `Promise.all`을 통해서 문제를 해결할 수 있다.

```javascript
;(async () => {
  try {
    const p1 = wait(3000).then(() => {
      throw new Error('err')
    })
    await Promise.all([
      wait(2000).then(() => {
        throw new Error('err2')
      }), // p1
      p1,
    ])
  } catch (e) {
    console.log(e)
  }
})()
```

## 이벤트 리스너

이벤트 리스너와 같이 콜백에서도 종종 unhandled exception이 발생하곤 한다. 이 경우에는 동기나 비동기나 별다른 차이가 없다. 따라서 적절하게 `try...catch`를 사용하면 된다.

```javascript
document.querySelector('button').addEventListener('click', async () => {
  throw new Error('err') // uncaught
})
```

```javascript
document.querySelector('button').addEventListener('click', () => {
  throw new Error('err') // uncaught
})
```

## Promise Constructor

Promise Constructor 내부에서 동기로 에러가 발생하면 다음과 같이 잘 잡을 수 있따.

```javascript
new Promise(() => {
  throw new Error('err')
}).catch((e) => {
  console.log(e) // caught
})
```

그러나, 여기에서도 비동기로 에러가 발생할 경우에는 잡히지 않게 된다.

```javascript
new Promise(() => {
  setTimeout(() => {
    throw new Error('err') // uncaught
  }, 0)
}).catch((e) => {
  console.log(e)
})
```

여기에서는 `resolve`와 `reject`를 적절하게 사용해주는 것이 좋다.

아래 처럼 하게 되면, `setTimeout()`은 이미 태스크 큐 뒤로 넘어가서 실행되기 때문에 에러가 잡히지 않게 된다.

```javascript
new Promise((res, rej) => {
  setTimeout(() => {
    // 1
    connection.query('SELECT ...', (err, results) => {
      // 2
      if (err) {
        rej(err)
      } else {
        const r = transformResult(results) // 3
        res(r)
      }
    })
  }, 1000)
})
```

대신,

```javascript
new Promise((res, rej) => {
  setTimeout(res, 1000) // 1 비동기로 넘긴다
})
  .then(() => {
    connection.query('SELECT ...', (err, results) => {
      // 2 넘긴 다음에 쿼리 실행
      if (err) {
        rej(err)
      } else {
        res(results)
      }
    })
  })
  .then((results) => transformResult(results)) // 3 해당 쿼리에 대한 적절한 `then`처리
```

이렇게 되면 모든 오류가 체인으로 전파되어 `.catch`나 `await`이 적절하게 처리할 수 있게 된다.
