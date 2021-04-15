---
title: Codility - Stone Wall
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-25 12:25:19
description:
  '## StoneWall ### 문제  돌은 N미터 길이를 가지고 있으며, 두께는 모두 일정하다. 배얼에 돌 높이가
  주어져 있으며, 아래와 같이 해석할 수 있다.  - H[i]: 왼쪽에서 오른쪽으로 벽의 높이 - H[0]: 벽 왼쪽 끝의 높이 -
  H[N-1]: 벽 마지막 끝의 높이  ``` H[0] = 8    H[1] = 8    H[2] = 5 H[3]...'
category: algorithm
slug: /2020/06/codility-07-04-stone-wall/
template: post
---

## StoneWall

### 문제

돌은 N미터 길이를 가지고 있으며, 두께는 모두 일정하다. 배얼에 돌 높이가 주어져 있으며, 아래와 같이 해석할 수 있다.

- H[i]: 왼쪽에서 오른쪽으로 벽의 높이
- H[0]: 벽 왼쪽 끝의 높이
- H[N-1]: 벽 마지막 끝의 높이

```
H[0] = 8    H[1] = 8    H[2] = 5
H[3] = 7    H[4] = 9    H[5] = 8
H[6] = 7    H[7] = 4    H[8] = 8
```

는 7을 리턴해야 하는데, 그 이유는 아래와 같다.

![stone-wall](https://codility-frontend-prod.s3.amazonaws.com/media/task_static/stone_wall/static/images/auto/4f1cef49cc46d451e88109d449ab7975.png)

### 풀이

```javascript
function solution(H) {
  const stack = []
  let count = 0

  for (let i = 0; i < H.length; i++) {
    // 베이스가 돌을 찾는다.
    // 베이스가 될 돌은 무조건 하나 있어야 하고
    // 현재 쌓으려는 돌 위치보다 낮아야 한다.
    while (stack.length > 0 && stack[stack.length - 1] > H[i]) {
      stack.pop()
    }

    // 돌 명단이 비어있거나, 새로 쌓아야 할 돌이 스택의 마지막 돌 보다 높다면 새로 쌓는다.
    if (stack.length === 0 || stack[stack.length - 1] < H[i]) {
      // 새로 쌓고 지금 높이를 리턴한다.
      stack.push(H[i])
      count += 1
    }
  }

  return count
}
```

https://app.codility.com/demo/results/trainingTEZQDK-37Z/
