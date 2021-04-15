---
title: Codility - Min Avg Two Slice
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 05:25:19
description:
  '## Min Avg Two Slice ### 문제  길이가 N인 비어있지 않은 배열 A가 주어진다. 한쌍의 숫자 P,
  Q의 범위는 `0 <= P < Q < N` 다. 주어진 P와 Q로 A배열을 slice한다. (최소 2개이상의 요소가 있어야 한다.) (P,
  Q)는 `A[P] + A[P + 1] + ... + A[Q]`이며, (P, Q)의 평균은 `(A[P...'
category: algorithm
slug: /2020/06/codility-05-03-min-avg-two-slice/
template: post
---

## Min Avg Two Slice

### 문제

길이가 N인 비어있지 않은 배열 A가 주어진다. 한쌍의 숫자 P, Q의 범위는 `0 <= P < Q < N` 다. 주어진 P와 Q로 A배열을 slice한다. (최소 2개이상의 요소가 있어야 한다.) (P, Q)는 `A[P] + A[P + 1] + ... + A[Q]`이며, (P, Q)의 평균은 `(A[P] + A[P + 1] + ... + A[Q]) / (Q − P + 1)`다. 평균이 최소가 되는 P의 값을 구하라.

```
A가 아래와 같이 이루어져있다고 가정하자.
A[0] = 4
A[1] = 2
A[2] = 2
A[3] = 5
A[4] = 1
A[5] = 5
A[6] = 8

slice (1, 2), 평균은 (2 + 2) / 2 = 2;
slice (3, 4), 평균은 (5 + 1) / 2 = 3;
slice (1, 4), 평균은 (2 + 2 + 5 + 1) / 4 = 2.5.
```

### 풀이

```javascript
function solution(A) {
  let min = Number.MAX_SAFE_INTEGER
  let minIndex = 0
  for (let i = 0; i < A.length - 1; i++) {
    let twoSum = (A[i] + A[i + 1]) / 2

    if (min > twoSum) {
      min = twoSum
      minIndex = i
    }

    if (i + 2 <= A.length - 1) {
      let threeSum = (A[i] + A[i + 1] + A[i + 2]) / 3

      if (min > threeSum) {
        min = threeSum
        minIndex = i
      }
    }
  }

  return minIndex
}
```

### 해설

처음에 고민을 많이 했는데, 문제에 힌트가 있었다. 예시에서 2개의 평균, 4개의 평균을 구하는 예제를 보여주었는데, 4개이상의 요소의 평균의 최소값은 2~3개 내에서 결정된 다는 사실이다. 그 사실만 인지하게 되면, 쉽게 풀수 있는 문제다.

https://app.codility.com/demo/results/trainingPXJM9C-P26/
