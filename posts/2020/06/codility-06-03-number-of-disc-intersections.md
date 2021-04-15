---
title: Codility - Number of Disc Intersections
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 12:25:19
description:
  '## Number of Disc Intersections ### 문제  N개의 디스크가 존재하고, 디스크는 각각 0~
  N-1의 번호를 가진다. 이는 A라는 배열에서 표현되는데, `A[N]` 는 해당 디스크의 반경을 의미한다.   ``` A[0] = 1
  A[1] = 5 A[2] = 2 A[3] = 1 A[4] = 4 A[5] = 0 ```  ![discs]...'
category: algorithm
slug: /2020/06/codility-06-03-number-of-disc-intersections/
template: post
---

## Number of Disc Intersections

### 문제

N개의 디스크가 존재하고, 디스크는 각각 0~ N-1의 번호를 가진다. 이는 A라는 배열에서 표현되는데, `A[N]` 는 해당 디스크의 반경을 의미한다.

```
A[0] = 1
A[1] = 5
A[2] = 2
A[3] = 1
A[4] = 4
A[5] = 0
```

![discs](https://codility-frontend-prod.s3.amazonaws.com/media/task_static/number_of_disc_intersections/static/images/auto/0eed8918b13a735f4e396c9a87182a38.png)

이 때, 교차하는 디스크의 수를 구하라.

### 풀이

```javascript
function solution(A) {
  const length = A.length
  let intersections = 0

  // 시작 점과 끝점을 저장하는 새로운 배열을 만든다.
  // [ [ -1, 1 ], [ -4, 6 ], [ 0, 4 ], [ 2, 4 ], [ 0, 8 ], [ 5, 5 ] ]
  const info = A.map((disc, index) => [index - disc, index + disc])

  // 이를 시작점이 작은 순대로 배열한다.
  // [ [ -4, 6 ], [ -1, 1 ], [ 0, 4 ], [ 0, 8 ], [ 2, 4 ], [ 5, 5 ] ]
  const sorted = info.sort((a, b) => a[0] - b[0])

  // 가장 바깥에 있는 원부터 돈다
  for (let i = 0; i < sorted.length; i++) {
    // const targetStart = sorted[i][0]
    const targetEnd = sorted[i][1]

    // 그 다음 것 부터 돈다
    for (let j = i + 1; j < sorted.length; j++) {
      const compareStart = sorted[j][0]
      // const compareEnd = sorted[j][1]

      // 겹치는 경우에만
      if (compareStart <= targetEnd) {
        intersections += 1
        // 겹치는 횟수가 특정 횟수를 넘어가면 -1을 리턴하랜다.
        if (intersections > 10000000) {
          return -1
        }
      } else {
        // 시작점 순으로 정렬했으므로, 이 이후는 안봐도 안겹친다. 따라서 break
        break
      }
    }
  }

  return intersections
}
```

하나씩 고민해보자. 1과 5과 교차하기 위해서는 어떻게 해야할까? 둘 사이의 거리는 일단 4다. 두 반지름 (반경)의 합이 4를 넘어야 한다. for 문을 돌면서 둘 사이의 차이와 반지름의 합을 계산해서 구해보면 될까?

라고 했지만 타임아웃 에러가 났다. for문을 이중으로 돌아야 하기 때문에 `O(N^2)`복잡도가 나온다. break가 없어서 빼도 박도 못한다.

둘이 겹치는지 안겹치는지 확인해볼 수 있는 방법이 있을까? 각 그리는 원마다 시작점과 끝점을 `[s, e]` 식으로 저장해 두었다가, 비교 대상의 끝점과 시작점이 겹치는지 확인해보면 될 것이다.

https://app.codility.com/demo/results/training86S6RE-49X/
