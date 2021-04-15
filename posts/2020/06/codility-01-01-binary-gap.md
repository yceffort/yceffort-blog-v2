---
title: Codility - Binary Gap
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 05:25:19
description:
  '## 1-1 Binary Gap ### 문제  숫자 N을 이진수로 바꿨을때, 1과 1사이에 있는 0의 개수가 가장 많이
  연속해 있는 0의 개수를 구하라.  ``` 9는 이진수로 바꿀 경우 1001, 이경우 0의 최대 개수는 2. 529는 이진수로 바꿀 경우
  1000010001, 이경우 0의 최대 개수는 3. 20은 이진수로 바꿀 경우 10100, 이 경우...'
category: algorithm
slug: /2020/06/codility-01-01-binary-gap/
template: post
---

## 1-1 Binary Gap

### 문제

숫자 N을 이진수로 바꿨을때, 1과 1사이에 있는 0의 개수가 가장 많이 연속해 있는 0의 개수를 구하라.

```
9는 이진수로 바꿀 경우 1001, 이경우 0의 최대 개수는 2.
529는 이진수로 바꿀 경우 1000010001, 이경우 0의 최대 개수는 3.
20은 이진수로 바꿀 경우 10100, 이 경우 0의 최대 개수는 1. (100은 1로 둘러 쌓여 있지 않음)
15는 이진수로 바꿀경우 1111 이므로, 0의 개수는 0개
```

### 풀이

```javascript
function solution(N) {
  // 2진법으로 변환
  const binary = N.toString(2)
  // 1로 쪼갠다.
  const splitted = binary.split(1)

  // 두개 이하로 쪼개질 경우, 1이 한개 밖에 없으므로 값은 0
  if (splitted.length <= 2) {
    return 0
  } else {
    // 마지막 나누기는 의미가 없다.
    // '' 이거나 1로 막혀있지 않다면 숫자가 나올 것이므로
    splitted.pop()
    // 제일
    return splitted.reduce((p, c) => (c.length > p ? c.length : p), 0)
  }
}
```

### 해설

배열의 마지막 요소를 `pop`하는 것만 기억하면 될 듯.

https://app.codility.com/demo/results/training3C4SN2-EXK/
