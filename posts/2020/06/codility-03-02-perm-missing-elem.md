---
title: Codility - Perm missing elem
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 05:25:19
description:
  '## 3-2 Perm Missing Elem ### 문제  길이 N으로 이루어진 배열 A은, 1부터 N+1 의 숫자로
  이루어져 있다. 여기에서 빠진 숫자를 찾아라.  ``` A[0] = 2 A[1] = 3 A[2] = 1 A[3] = 5  4 가 누락되어
  있으므로, 정답은 4 다. ```  ### 풀이  ```javascript function solut...'
category: algorithm
slug: /2020/06/codility-03-02-perm-missing-elem/
template: post
---

## 3-2 Perm Missing Elem

### 문제

길이 N으로 이루어진 배열 A은, 1부터 N+1 의 숫자로 이루어져 있다. 여기에서 빠진 숫자를 찾아라.

```
A[0] = 2
A[1] = 3
A[2] = 1
A[3] = 5

4 가 누락되어 있으므로, 정답은 4 다.
```

### 풀이

```javascript
function solution(A) {
  if (!A.length) {
    return 1
  }

  // 사이즈
  const size = A.length
  // 한개를 빼먹었으므로, 최대 숫자는 한개를 더 갔을 것이다.
  // 한개를 더 간 숫자들의 합을 구한다.
  let sum = ((size + 1) * (size + 2)) / 2

  // 거기에서 모든 배열을 하나씩 빼면 없는 숫자가 나올 것이다.
  for (let i = 0; i <= size - 1; i++) {
    sum -= A[i]
  }

  return sum
}
```

### 해설

![sum of n](https://i.stack.imgur.com/qYmeo.gif)

https://app.codility.com/demo/results/trainingS58ZMJ-NBP/
