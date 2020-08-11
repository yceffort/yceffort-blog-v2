---
title: Codility - Count div
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 05:25:19
description: "## Count Div ### 문제  A와 A보다 같거나 큰 B, 그리고 K가 주어질 때, A와 B사이에 K로 나누면
  나머지가 0인 숫자의 개수를 구하라.  ``` A=6 B=11 K=2 6, 8, 10 이 있으므로, 정답은 3 이다. ```  ###
  풀이  ```javascript function solution(A, B, K) {     return ..."
category: algorithm
slug: /2020/06/codility-05-01-count-div/
template: post
---
## Count Div

### 문제

A와 A보다 같거나 큰 B, 그리고 K가 주어질 때, A와 B사이에 K로 나누면 나머지가 0인 숫자의 개수를 구하라.

```
A=6
B=11
K=2
6, 8, 10 이 있으므로, 정답은 3 이다.
```

### 풀이

```javascript
function solution(A, B, K) {
    return Math.floor(B / K) - Math.floor(A / K) + (A % K === 0 ? 1 : 0)
}
```

https://app.codility.com/demo/results/training7NC844-ZWB/