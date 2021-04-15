---
title: Codility - Triangle
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 12:25:19
description:
  '## Triangle ### 문제  길이 N의 배열 A가 주어진다.   (P, Q, R)은 삼각형이 될 수 있는데,
  이는   - 0 ≤ P < Q < R < N   - A[P] + A[Q] > A[R] - A[Q] + A[R] > A[P] - A[R] +
  A[P] > A[Q]  라는 조건을 만족 하기 때문이다.  ``` A[0] = 10     A[1] ...'
category: algorithm
slug: /2020/06/codility-06-04-triangle/
template: post
---

## Triangle

### 문제

길이 N의 배열 A가 주어진다.

(P, Q, R)은 삼각형이 될 수 있는데, 이는

- 0 ≤ P < Q < R < N
- A[P] + A[Q] > A[R]
- A[Q] + A[R] > A[P]
- A[R] + A[P] > A[Q]

라는 조건을 만족 하기 때문이다.

```
A[0] = 10
A[1] = 2
A[2] = 5
A[3] = 1
A[4] = 8
A[5] = 20

은 0, 2, 4 (10, 5, 8)로 삼각형을 만들 수 있으므로 1을 리턴한다. 그러나 만들 수 없다면 0을 리턴한다.
```

주어진 A 배열에서 삼각형을 만들 수 있는 3개의 조합이 존재하는지 확인해서, 존재한다면 1을, 아니라면 0을 리턴해라.

### 풀이

```javascript
function solution(A) {
  const sorted = A.sort((a, b) => a - b)

  for (let i = 0; i < sorted.length - 2; i++) {
    const a = sorted[i]
    const b = sorted[i + 1]
    const c = sorted[i + 2]

    if (a + b > c && b + c > a && a + c > b) {
      return 1
    }
  }

  return 0
}
```

https://app.codility.com/demo/results/training6R2NPK-JJH/
