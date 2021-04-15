---
title: Javascript - Promise
date: 2019-05-21 11:42:41
published: true
tags:
  - javascript
description: '## Promise ```javascript new Promise(executor) ```  `executor`는
  `resolve`및 `reject` 인수를 전달할 실행함수를 의미한다. 실행함수는 `resolve`와 `reject`를 받아 즉시 실행된다.
  실행함수는 보통 비동기 작업을 시작한 후, 모든 작업을 끝내면 `resolve`를 호출해서 `Prom...'
category: javascript
slug: /2019/05/21/javascript-promise/
template: post
---

## Promise

```javascript
new Promise(executor)
```

`executor`는 `resolve`및 `reject` 인수를 전달할 실행함수를 의미한다. 실행함수는 `resolve`와 `reject`를 받아 즉시 실행된다. 실행함수는 보통 비동기 작업을 시작한 후, 모든 작업을 끝내면 `resolve`를 호출해서 `Promise`를 이행하고, 오류가 발생한 경우 `reject`를 호출해 거부된다.

`Promise`는 다음 중 하나의 상태를 가진다.

- 대기(pending): 이행되거나 거부되지 않는 초기 상태
- 이행(fullfiled): 연산이 성공적으로 완료됨
- 거부(rejected): 연산이 실패

![promise](https://mdn.mozillademos.org/files/8633/promises.png)

### Promise.all(iterable)

`iterable`내에 모든 프로미스를 이행하는데, 대신 어떤 프로미스가 거부를 하게 되면 즉시 거부하는 프로미스를 반환한다. 모든 프로미스가 이행되는 경우, 프로미스가 결정한 값을 순서대로 배열로 반환한다.

### Promise.race(iterable)

`iterable`내에 가장 빠르게 이행/거부 한 값을 반환한다.

### Promise.reject()

주어진 이유로 거부하는 Promise 객체를 반환한다

### Promise.resolve()

주어진 값으로 이행하는 Promise를 반환한다. `then`이 있는 경우, 반환된 프로미스는 `then`을 따라가고 마지막 상태를 취한다.

```javascript
function myAsyncFunction(url) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('GET', url)
    xhr.onload = () => resolve(xhr.responseText)
    xhr.onerror = () => reject(xhr.statusText)
    xhr.send()
  })
}
```
