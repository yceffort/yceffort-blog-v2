---
title: Codility - Passing Cars
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 05:25:19
description: "## Passing Cars ### 문제  N의 길이로 이루어진 배열 A는 0과 1로 이루어져 있는데, 0과 1은 각각
  다음과 같은 의미를 가지고 있다.  - 0은 차가 동쪽으로 간다 - 1은 차가 서쪽으로 간다  이 때 동쪽으로 간 차와 서쪽으로 간 차를
  짝지을 수 있는 개수를 구하라. 단 먼저 동쪽으로 간차와 그 이후에 서쪽으로 간 차만 짝 지을 수 ..."
category: algorithm
slug: /2020/06/codility-05-04-passing-cars/
template: post
---
## Passing Cars

### 문제

N의 길이로 이루어진 배열 A는 0과 1로 이루어져 있는데, 0과 1은 각각 다음과 같은 의미를 가지고 있다.

- 0은 차가 동쪽으로 간다
- 1은 차가 서쪽으로 간다

이 때 동쪽으로 간 차와 서쪽으로 간 차를 짝지을 수 있는 개수를 구하라. 단 먼저 동쪽으로 간차와 그 이후에 서쪽으로 간 차만 짝 지을 수 있다.

```
A배열이 아래와 같이 주어져 있다면
A[0] = 0
A[1] = 1
A[2] = 0
A[3] = 1
A[4] = 1

짝 지을 수 있는 경우의 수는 (0, 1), (0, 3), (0, 4), (2, 3), (2, 4).

5가지다.
```

단 짝의 개수가 1,000,000,000개를 넘어가면 그냥 -1을 리턴한다.


### 풀이

```javascript
// you can write to stdout for debugging purposes, e.g.
// console.log('this is a debug message');

function solution(A) {
    
    // 동쪽으로 간차의 개수를 센다
    let east = 0
    // 결과
    let passing = 0
    
    for (let i of A) {
      // 동쪽으로 간 차를 센다.
      if (i === 0) {
          east += 1
      } else {
        // 서쪽으로 간 차가 나타난다면, 현재 동쪽으로 간 차 개수만큼 더한다.
        // 현재 동쪽으로 간 차 개수만큼 짝이 될 수 있기 때문이다.
        passing += east
      }
    }
    
    if (passing > 1000000000) {
        return -1
    }
    
    return passing
}
```

https://app.codility.com/demo/results/trainingXFPYT4-R3D/