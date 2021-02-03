---
title: 'no return, await, return, await return 의 차이'
tags:
  - javascript
published: true
date: 2021-02-03 17:27:03
description: 'try catch 블록에서는 동작이 다르네'
---

50% 확률로 reject 되는 아래와 같은 함수가 있다고 가정해보자.

```javascript
async function resolveOrReject() {
  // 1초를 그냥 기다린다.
  await new Promise((_) => setTimeout(_, 1000))

  // 50% 확률로 true false
  const randomResult = Boolean(Math.round(Math.random()))

  if (randomResult) {
    return 'good'
  } else {
    throw Error('bad')
  }
}
```

## 그냥 호출

```javascript
async function a() {
  try {
    resolveOrReject()
  } catch (e) {
    return 'caught!'
  }
}
```

이 경우 `a()`는 1초를 기다리지 않고 언제나 `undefined`를 `fulfilled` (이행) 할 것이다. 우리가 여기에서 `await` 하거나 껼과를 기다리지 않으므로, 실패했을 경우에 대해서 처리를 할 수가 없다. 대부분 이런 코드는 실수일것이다.

## await

```javascript
async function b() {
  try {
    await resolveOrReject()
  } catch (e) {
    return 'caught!'
  }
}
```

이 함수의 경우에는 항상 1초를 기다리며, `undefined`가 오거나 `caught`가 올 것이다. `resolveOrReject`의 결과를 기다리기 때문에, 실패 했을 경우 `catch` 블록을 실행할 수 있게 되었다. 그러나 실패하지 않았을 경우에는 이 값을 가지고 아무것도 하지 않는다.

## return

```javascript
async function c() {
  try {
    return resolveOrReject()
  } catch (e) {
    return 'caught'
  }
}
```

이 경우에도 마찬가지로 1초를 기다리며, 성공할 경우 `good` 이 오고, 실패할 경우에는 `caught`가 오는 것이 아니고 그냥 에러가 던져진다. 따라서 실패할 경우엔 `catch` 블록에 들어가지 않는다는 것을 알 수 있다.

## return await

```javascript
async function d() {
  try {
    return await resolveOrReject()
  } catch (e) {
    return 'caught'
  }
}
```

1초를 기다리며, `good`이 오거나 `caught`가 오게 된다. `resolveOrReject`의 결과를 기다리므로, 에러가 났을 때는 `catch` 블록으로 , 이행이 되었을 경우 정상적으로 결과를 리턴한다.

위 함수를 나누면 이렇게 볼 수 있다.

```javascript
async function d() {
  try {
    // resolveOrReject 의 결과를 기다리며, 결과를 변수에 넣는다.
    const result = await resolveOrReject()
    // 만약 위에서 에러가 던져졌다면, catch 블록으로 넘어간다.
    // 그렇지 않으면, 결과를 리턴한다.
    return result
  } catch (e) {
    return 'caught'
  }
}
```

**이 것은 어디까지나 `try ... catch` 블록에서만 유효하다.** 그 외의 영역에서는 `return await`은 의미하다. `async` 함수 내에 있는 `return await`은 프로미스를 기다렸다가 결과가 나올떄까지, 현재의 함수를 콜스택에 넣어두며, 외부 promise가 resolve 되기전에 추가로 마이크로 태스크가 생기게 된다. 따라서 `try ... catch` 블록이 아니라면 단순히 `return something`으로 처리하면 된다.

https://github.com/eslint/eslint/blob/master/docs/rules/no-return-await.md
