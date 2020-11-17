---
title: Codility - Perm Check
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 05:25:19
description: "## Perm Check ### 문제  길이 N인 배열이 주어져 있고, 안에는 서로 다른 숫자가 들어가 있다. 이 서로
  다른 숫자가 연속하는 숫자면 true, 아니라면 false를 리턴하라.  ``` A[0] = 4 A[1] = 1 A[2] = 3 A[3] =
  2  는 1을 리턴하면 된다. ```  ``` A[0] = 4 A[1] = 1 A[2] = 3 ..."
category: algorithm
slug: /2020/06/codility-04-04-permcheck/
template: post
---
## Perm Check

### 문제

길이 N인 배열이 주어져 있고, 안에는 서로 다른 숫자가 들어가 있다. 이 서로 다른 숫자가 연속하는 숫자면 true, 아니라면 false를 리턴하라.

```
A[0] = 4
A[1] = 1
A[2] = 3
A[3] = 2

는 1을 리턴하면 된다.
```

```
A[0] = 4
A[1] = 1
A[2] = 3

는 false를 리턴하면 된다.
```

### 풀이

```javascript
function solution(A) {
  // 정렬
  const sorted = A.sort((a, b) => a - b)
  for (let i=0; i < sorted.length; i++) {
      if (i + 1 !== sorted[i]) {
          return 0
      }
  }
  return 1
}
```



https://app.codility.com/demo/results/training3V3SZS-VUU/