---
title: Codility - Missing Integer
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-24 05:25:19
description:
  '## Missing Integer ### 문제  주어진 배열 A에 빠져 있는 가장 작은 양의 정수를 구하시오  ```
  A=[1, 3, 6, 4, 1, 2] 이라면 답은 5 A=[1, 2, 3] 이라면 답은 4 A=[-1, -3] 이라면 답은 1
  ```   ### 풀이  ```javascript function solution(A) {     // 배열 길...'
category: algorithm
slug: /2020/06/codility-04-03-missing-integer/
template: post
---

## Missing Integer

### 문제

주어진 배열 A에 빠져 있는 가장 작은 양의 정수를 구하시오

```
A=[1, 3, 6, 4, 1, 2] 이라면 답은 5
A=[1, 2, 3] 이라면 답은 4
A=[-1, -3] 이라면 답은 1
```

### 풀이

```javascript
function solution(A) {
  // 배열 길이 만큼 false로 채워진 체커를 생성
  const checker = Array(A.length).fill(false)

  // A를 돌면서 양의 정수라면 해당 checker의 index를 true로 바꿔준다.
  for (let i = 0; i < A.length; i++) {
    if (A[i] > 0) {
      checker[A[i] - 1] = true
    }
  }

  // 가장 가까운 false위치를 찾는다.
  const index = checker.indexOf(false)
  // 없으면 모든 수가 다 차있는 것이므로 길이 + 1, 아니라면 해당 index + 1을 리턴한다.
  return index === -1 ? checker.length + 1 : index + 1
}
```

### 해설

문제와 관련된 해설은 아니고, indexOf는 일반적인 배열을 for

https://app.codility.com/demo/results/training8EH9VG-2JS/
