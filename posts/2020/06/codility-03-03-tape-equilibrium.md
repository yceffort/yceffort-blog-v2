---
title: Codility - Tape Equilibrium
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 05:25:19
description:
  '## Tape Equilibrium ### 문제  길이 N의 배열을 임의로 두개로 쪼개고, 이렇게 해서 생긴 두배열의
  합을 각각 구할때, 이 서로 두합의 차이가 가장 작은 경우를 구하라.  ``` A[0] = 3 A[1] = 1 A[2] = 2 A[3] =
  4 A[4] = 3 이경우 네가지로 쪼갤 수 있는데  P = 1, difference = |3 − ...'
category: algorithm
slug: /2020/06/codility-03-03-tape-equilibrium/
template: post
---

## Tape Equilibrium

### 문제

길이 N의 배열을 임의로 두개로 쪼개고, 이렇게 해서 생긴 두배열의 합을 각각 구할때, 이 서로 두합의 차이가 가장 작은 경우를 구하라.

```
A[0] = 3
A[1] = 1
A[2] = 2
A[3] = 4
A[4] = 3
이경우 네가지로 쪼갤 수 있는데

P = 1, difference = |3 − 10| = 7
P = 2, difference = |4 − 9| = 5
P = 3, difference = |6 − 7| = 1
P = 4, difference = |10 − 3| = 7

여기서 답은 1이다
```

### 풀이

```javascript
function solution(A) {
  // 좌측 SUM
  let leftSum = 0
  // 우측 SUM
  let rightSum = A.reduce((a, b) => a + b, 0)

  // 아직 답은 없음
  let answer = null

  // 배열을 순회하면서
  for (let i = 0; i < A.length - 1; i++) {
    // 왼쪽 SUM은 하나씩 추가
    leftSum += A[i]
    // 오른쪽 SUM은 하나씩 제거
    rightSum -= A[i]
    // 둘의 차이 계산
    const diff = Math.abs(leftSum - rightSum)
    // 둘의 차이가 하나도 계산이 안되어 있거나, 현재 값보다 차이가 적다면 갱신
    if (answer === null || answer > diff) {
      answer = diff
    }
  }
  return answer
}
```

https://app.codility.com/demo/results/trainingRC4CVP-VPY/
