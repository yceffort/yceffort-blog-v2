---
title: Codility - Distinct
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 12:25:19
description: "## Distinct ### 문제  배열 A안에 unique한 숫자가 몇 개 있는지 리턴하라.  ###
  풀이  ```javascript function solution(A) {     return [...new Set(A)].length }
  ```  Set을 활용하면 쉽게 풀 수 있다. Set이 아니더라도 object등을 활용해보면 된다.   https:..."
category: algorithm
slug: /2020/06/codility-06-01-distinct/
template: post
---
## Distinct

### 문제

배열 A안에 unique한 숫자가 몇 개 있는지 리턴하라.

### 풀이

```javascript
function solution(A) {
    return [...new Set(A)].length
}
```

Set을 활용하면 쉽게 풀 수 있다. Set이 아니더라도 object등을 활용해보면 된다.


https://app.codility.com/demo/results/training4VKM6Q-5SX/