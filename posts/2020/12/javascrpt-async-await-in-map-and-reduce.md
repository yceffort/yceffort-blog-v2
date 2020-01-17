---
title: 'map과 reduce에서 async await 사용하기'
tags:
  - javascript
published: true
date: 2020-12-22 20:44:19
description: '당연한거 아님?'
---

```javascript
function sayHello(name) {
  return new Promise((resolve, reject) => {
    setTimeout(() => resolve(`Hello, ${name}`), 2000)
  })
}

const message1 = await sayHello('yceffort')
console.log(message1)
```

요런 비동기 함수가 있고, 이를 map으로 처리한다고 가정해보자.

```javascript
const names = [
  'yceffort1',
  'yceffort2',
  'yceffort3',
  'yceffort4',
  'yceffort5',
  'yceffort6',
]

const messages = names.map(async (name) => await sayHello(name))
console.table(messages)
```

이렇게 하면 될 것 같지만?

```
(6) [Promise, Promise, Promise, Promise, Promise, Promise]
0: Promise {<pending>}
1: Promise {<pending>}
2: Promise {<pending>}
3: Promise {<pending>}
4: Promise {<pending>}
5: Promise {<pending>}
```

아쉽게도 모든 결과가 pending으로 뜬다. `await`은 `Promise` 객체만 기다려 주기 때문에 그런 것으로 보인다. 반변에 우리가 넘긴 것은 `list`다.

따라서 이를 정상적으로 실행하기 위해서는 `Promise.all`을 사용해야 한다.

```javascript
const promiseMessages = await Promise.all(
  names.map(async (name) => await sayHello(name)),
)
console.log(promiseMessages)
```

```javascript
;[
  'Hello, yceffort1',
  'Hello, yceffort2',
  'Hello, yceffort3',
  'Hello, yceffort4',
  'Hello, yceffort5',
  'Hello, yceffort6',
]
```

그렇다면 `reduce`의 경우에는 어떻게 처리하면 좋을까?

```javascript
const oddMessages = names.reduce(async (prev, current, index) => {
  if (index % 2 > 0) {
    return [...prev, await sayHello(current)]
  } else {
    return prev
  }
}, [])
```

이렇게 하면 당연히 안될 것이다. 여기에서 `prev`는 기존의 값이 아닌 `Promise`일 것이다.

```javascript
const oddMessages = await names.reduce(async (prev, current, index) => {
  const prevResult = await prev.then()
  if (index % 2 === 0) {
    const result = await sayHello(current)
    return Promise.resolve([...prevResult, result])
  } else {
    return Promise.resolve(prevResult)
  }
}, Promise.resolve([]))
```

기존에 있던 모든 return을 `Promise.resolve`로 감싸고, 이전에 넘어온 `prev`는 `then`처리를 했다.

```javascript
;['Hello, yceffort1', 'Hello, yceffort3', 'Hello, yceffort5']
```
