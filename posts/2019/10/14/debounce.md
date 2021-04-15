---
title: typescript debounce
date: 2019-10-14 06:17:00
published: true
tags:
  - typescript
description: '> Creates a debounced function that delays invoking func until
  after wait milliseconds have elapsed since the last time the debounced
  function was invoked. The debounced function comes with a cancel m...'
category: typescript
slug: /2019/10/14/debounce/
template: post
---

> Creates a debounced function that delays invoking func until after wait milliseconds have elapsed since the last time the debounced function was invoked. The debounced function comes with a cancel method to cancel delayed func invocations and a flush method to immediately invoke them. Provide options to indicate whether func should be invoked on the leading and/or trailing edge of the wait timeout. The func is invoked with the last arguments provided to the debounced function. Subsequent calls to the debounced function return the result of the last func invocation.

[출처](https://lodash.com/docs/4.17.15#debounce)

디바운스는 과다한 이벤트 로직이 실행되는 것을 방지하는 함수로, 호출이 반복되는 동안에는 반복해서 로직이 실행되는 것을 막고, 설정한 시간이 지나고 나서야 로직이 실행하게 하는 함수다.

```typescript
export function debounce<Params extends any[]>(
  func: (...args: Params) => any,
  timeout: number,
): (...args: Params) => void {
  let timer: NodeJS.Timeout
  return (...args: Params) => {
    clearTimeout(timer)
    timer = setTimeout(() => {
      func(...args)
    }, timeout)
  }
}
```

즉, 반복되는 이벤트가 계속해서 실행될 때, 매번 그 이벤트를 실행하는 것이 아니라, timeout 만큼의 시간이 흐른뒤에, 이전의 이벤트를 무시하고 이벤트 하나만 실행하는 것이다.
