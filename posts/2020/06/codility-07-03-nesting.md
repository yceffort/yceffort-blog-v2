---
title: Codility - Nesting
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-25 12:25:19
description:
  "## Nesting ### 문제  `(`와 `)`로 이루어진 문자열이 있다. 이 문자열의 `(` `)` 짝이 맞게
  이루어져 있는지 확인하라.  ### 풀이  ```javascript function solution(S) {     const split =
  S.split('')     const stack = []     for (let i of split..."
category: algorithm
slug: /2020/06/codility-07-03-nesting/
template: post
---

## Nesting

### 문제

`(`와 `)`로 이루어진 문자열이 있다. 이 문자열의 `(` `)` 짝이 맞게 이루어져 있는지 확인하라.

### 풀이

```javascript
function solution(S) {
  const split = S.split('')
  const stack = []
  for (let i of split) {
    // 여는 괄호라면 스택에 하나씩 넣는다
    if (i === '(') {
      stack.push(true)
      // 닫는 괄호라면
    } else {
      // 닫는괄호인데 여는괄호가 없다면 이미 글러먹었다
      if (stack.length === 0) {
        return 0
        // 하나 있으면 꺼낸다
      } else {
        stack.pop()
      }
    }
  }

  // 스택이 깔끔하게 비어있으면 1을 리턴한다.
  return stack.length === 0 ? 1 : 0
}
```

https://app.codility.com/demo/results/training8N66E9-5ZY/
