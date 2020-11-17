---
title: Codility - Frog Jump
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 05:25:19
description: "## 3-1 Frog Jump ### 문제  개구리가 X에서 Y까지 뛰어야 하고, 한번에 D 만큼 점프 할 수 있을
  때, 몇번을 뛰어야 하는가?  ### 풀이  ```javascript function solution(X, Y, D) {     return
  Math.ceil((Y - X) / D) } ```  https://app.codility.com/..."
category: algorithm
slug: /2020/06/codility-03-01-frog-jump/
template: post
---
## 3-1 Frog Jump

### 문제

개구리가 X에서 Y까지 뛰어야 하고, 한번에 D 만큼 점프 할 수 있을 때, 몇번을 뛰어야 하는가?

### 풀이

```javascript
function solution(X, Y, D) {
    return Math.ceil((Y - X) / D)
}
```

https://app.codility.com/demo/results/trainingHC62NP-TRW/