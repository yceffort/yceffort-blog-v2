---
title: '자바스크립트로 메모이제이션 구현하기'
tags:
  - javascript
published: true
date: 2020-11-23 22:47:01
description: '까먹지 않게 기억해두기'
---

```javascript
const memoize = (func) => {
  // 메모이제이션을 위한 클로져 생성

  // 메모이제이션 값을 저장해둔다.
  const results = {}

  return (...args) => {
    // 파라미터로 메모이제이션 키 생성
    const memoKey = JSON.stringify(args)

    // 결과가 없으면 메모이제이션 값을 넣어둔다.
    if (!results[memoKey]) {
      results[memoKey] = func(...args)
    }

    // 메모이제이션 값을 리턴
    return results[memoKey]
  }
}
```
