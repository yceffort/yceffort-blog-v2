---
title: 'uncaught async error를 올바르게 처리하기'
tags:
  - javascript
published: true
date: 2021-08-23 13:21:41
description: 'async 하나면 되던 것도 안된다니까요?'
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

요것은 https://yceffort.kr/2021/02/run-await-return-return-await 이것과 좀 비슷한 느낌이다.

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
try{
	await Promise.all([1,2,3].forEach(async (index) => {
		throw new Error(`err ${index}`)
	}));
}catch(e) {
	console.log(e); // undefined is not iterable (cannot read property Symbol(Symbol.iterator))
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
try{
	await Promise.all([1,2,3].forEach(async (index) => {
    console.log('forEach', index)
		throw new Error(`err ${index}`)
	}));
}catch(e) {
	console.log(e); // undefined is not iterable (cannot read property Symbol(Symbol.iterator))
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


`forEach`는 `break`가 없다. 즉 중간에 도망갈 수 없는 loop 구문이다. 따라서 for문을 다 돌게 되고, 

- https://262.ecma-international.org/6.0/#sec-array.prototype.foreach