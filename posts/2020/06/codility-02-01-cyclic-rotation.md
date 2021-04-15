---
title: Codility - Cyclic Rotation
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 05:25:19
description:
  '## 2-1 Cyclic Rotation ### 문제  배열 A가 주어지고 이를 K번 각 배열의 요소를 오른쪽으로
  이동시켰을 때, 그 결과를 리턴하시오.  ``` A = [3, 8, 9, 7, 6] K = 3  [3, 8, 9, 7, 6] -> [6,
  3, 8, 9, 7] [6, 3, 8, 9, 7] -> [7, 6, 3, 8, 9] [7, 6, 3, 8...'
category: algorithm
slug: /2020/06/codility-02-01-cyclic-rotation/
template: post
---

## 2-1 Cyclic Rotation

### 문제

배열 A가 주어지고 이를 K번 각 배열의 요소를 오른쪽으로 이동시켰을 때, 그 결과를 리턴하시오.

```
A = [3, 8, 9, 7, 6]
K = 3

[3, 8, 9, 7, 6] -> [6, 3, 8, 9, 7]
[6, 3, 8, 9, 7] -> [7, 6, 3, 8, 9]
[7, 6, 3, 8, 9] -> [9, 7, 6, 3, 8]
```

### 풀이

```javascript
function solution(A, K) {
  // 오른쪽으로 움직여야 하는 횟수
  const sliceTimes = K % A.length

  // 회전할 필요가 없거나, 회전을 배열 길이 만큼 한다면 그냥 배열을 리턴한다.
  if (sliceTimes === 0 || sliceTimes === A.length) {
    return A
  }

  // 배열을 잘 잘라서 리턴한다.
  return [
    ...A.slice(A.length - sliceTimes),
    ...A.slice(0, A.length - sliceTimes),
  ]
}
```

### 해설

문제와 관련이 없지만, [splice](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/splice)는 원본 배열에 영향을 미치고, [slice](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/slice)는 원본 배열에 영향을 미치지 않는다.

https://app.codility.com/demo/results/trainingYUK5SH-UVZ/
