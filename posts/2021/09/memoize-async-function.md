---
title: '비동기 함수 memoize 하는 방법'
tags:
  - javascript
published: true
date: 2021-09-08 22:06:46
description: 'memo, useMemo, useCallback, 그리고...'
---

## Introduction

메모이제이션은 프로그래밍에 있어 유용한 개념 중 하나다. 한번 실행하는데 비용이나 시간이 많이 드는 계산을 두 번 이상 동일하게 하는 것을 방지할 수 있다. 동기 함수에 메모이제이션 하는 것은 비교적 간단하다. 하지만 비동기 함수에서 메모이제이션을 어떻게 적용하는게 좋을까?

## 메모이제이션

일단 가장 간단한 순수 함수를 메모이제이션을 하는 것을 살펴보자.

```javascript
function getSquare(x) {
  return x * x
}
```

이를 메오이제이션 하기 위해, 예를 들어 아래와 같은 방법을 사용할 수 있다.

```javascript
const memo = {}

function getSquare(x) {
  if (memo.hasOwnProperty(x)) {
    return memo[x]
  }
  memo[x] = x * x
  return memo[x]
}
```

간단하게 몇줄 만으로도 메모이제이션을 할 수 있게 되었다.

그러나 위 방법은 굉장히 조악하므로, `memoize` 함수를 만들어 보자. 이 함수는 첫번째 인수로는 순수함수를, 두번째 함수로는 `getKey()` 함수 (첫번째 인수의 함수의 유니크 키를 리턴할 수 있는 함수)를 받아서 결과값을 메모이제이션 시킨다.

```javascript
function memoize(fn, getKey) {
  const memo = {}
  return function memoized(...args) {
    const key = getKey(...args)
    if (memo.hasOwnProperty(key)) return memo[key]

    memo[key] = fn.apply(this, args)
    return memo[key]
  }
}
```

이를 적용시켜 보자.

```javascript
const memoGetSquare = memoize(getSquare, (num) => num)
memoGetSquare(10) // 100
memoGetSquare(10) // 100 두번째 부터는 계산하지 않고 있던 값을 그냥 리턴한다.
```

## 비동기 함수 메모이제이션 하기

`expensiveOperation(key)`라는 비동기 함수가 있다고 가정해보자. 이 함수는 굉장히 시간/비용이 많이 드는 작업을 하고, 결과 값을 반환하면 콜백함수를 실행한다.

```javascript
// 비동기 작업을 실행하고 결과에 따라 콜백을 수행
expensiveOperation(key, (data) => {
  // Do something
})
```

이걸 메모이제이션 한다고 가정해본다면...?

```javascript
const memo = {}

function memoExpensiveOperation(key, callback) {
  if (memo.hasOwnProperty(key)) {
    callback(memo[key])
    return
  }

  expensiveOperation(key, (data) => {
    memo[key] = data
    callback(data)
  })
}
```

간단해보인다. 🤔 그러나 이 함수는 한가지 문제가 존재한다. `a`라는 인수를 받아 실행하는 과정에서, 또 똑같이 `a`라는 인수의 요청이 들어오면 어떻게 될까? 이 경우 첫번째 실행기 끝나지 않았다면, 두번째 함수도 마찬가지로 실행되어 버리기 때문에 중복해서 호출될 것이다. 우리는 이렇게 동시에 실행 되기 보다는, 어쨌든 빨리 끝난 함수의 결과값을 받아다가 실행하길 원할 것이다.

```javascript
const memo = {}
const progressQueues = {}

function memoExpensiveOperation(key, callback) {
  // 메모에 값이 있다면, 해당 콜백을 그냥 바로 실행
  if (memo.hasOwnProperty(key)) {
    callback(memo[key])
    return
  }

  if (!progressQueues.hasOwnProperty(key)) {
    // queue에 해당 키로 실행 중인 것이 없다면, 콜백을 배열 형태로 넣는다.
    progressQueues[key] = [callback]
  } else {
    // queue에 실행 중인게 있다면, 콜백을 배열에 추가해서 넣는다.
    progressQueues[key].push(callback)
    return
  }

  expensiveOperation(key, (data) => {
    // 결과를 메모이즈
    memo[key] = data
    // 줄줄이 있던 콜백 모두 실행
    for (const cb of progressQueues[key]) {
      cb(data)
    }
    // 큐 처리
    delete progressQueue[key]
  })
}
```

이를 앞선 예시 처럼 헬퍼 형태로 만들어 보자.

```javascript
function memoizeAsync(fn, getKey) {
  const memo = {},
    progressQueues = {}

  return function memoized(...allArgs) {
    const callback = allArgs[allArgs.length - 1]
    const args = allArgs.slice(0, -1)
    const key = getKey(...args)

    if (memo.hasOwnProperty(key)) {
      callback(memo[key])
      return
    }

    if (!progressQueues.hasOwnProperty(key)) {
      progressQueues[key] = [callback]
    } else {
      progressQueues[key].push(callback)
      return
    }

    fn.call(this, ...args, (data) => {
      memo[key] = data
      for (let callback of progressQueues[key]) {
        callback(data)
      }
      delete progressQueue[key]
    })
  }
}

// USAGE
const memoExpensiveOperation = memoizeAsync(expensiveOperation, (key) => key)
```

## Promises

이번에는 `processData(key)`라는 함수가 `key`를 인수로 받고, promise를 리턴하는 모습을 상상해보자. 그리고 이를 메모이제이션 해보자.

가장 간단하게 하는 방법은 아래와 같을 것이다.

```javascript
const memo = {}
function memoProcessData(key) {
  if (memo.hasOwnProperty(key)) {
    return memo[key]
  }

  memo[key] = processData(key) // memoize the promise for key
  return memo[key]
}
```

앞서 언급했던 `memoize` 함수와 별반 다를게 없다. 그렇다면 리턴하는 Promise 값을 메모이제이션하려면 어떻게 해야할까?

```javascript
const memo = {},
  progressQueues = {}

function memoProcessData(key) {
  return new Promise((resolve, reject) => {
    // 메모이제이션 된 값이 있다면 리턴
    if (memo.hasOwnProperty(key)) {
      resolve(memo[key])
      return
    }

    if (!progressQueues.hasOwnProperty(key)) {
      // queue에 해당 키로 실행 중인 것이 없다면, 콜백을 배열 형태로 넣는다.
      progressQueues[key] = [[resolve, reject]]
    } else {
      // queue에 실행 중인게 있다면, 콜백을 배열에 추가해서 넣는다.
      progressQueues[key].push([resolve, reject])
      return
    }

    processData(key)
      .then((data) => {
        // 리턴된 값 메모이제이션
        memo[key] = data // memoize the returned data
        // resolve 실행
        for (let [resolver] of progressQueues[key]) resolver(data)
      })
      .catch((error) => {
        // reject 실행
        for (let [, rejector] of progressQueues[key]) rejector(error)
      })
      .finally(() => {
        // clean up progressQueues
        delete progressQueues[key]
      })
  })
}
```

## 보완할 것

메모이제이션을 위해 `memo`라는 객체를 사용하기 때문에, 다양한 인수로 호출이 많아 질 수록, 이 객체의 크기가 갈수록 커질 것이다. 이러한 상황을 방지하기 위해 [Least Recently Used, aka LRU](<https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)>)와 같은 캐싱 정책을 사용할 수도 있을 것이다. 이는 메모이제이션에 드는 메모리에 대한 문제까지도 해결할 수 있을 것이다.
