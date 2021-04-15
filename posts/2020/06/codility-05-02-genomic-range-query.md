---
title: Codility - Genomic Range Query
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 05:25:19
description:
  '## Genomic Range Query ### 문제  DNA는 A, C, G, T로 구성되어 있는데, 이는 각각 1,
  2, 3, 4를 가르킨다. 이러한 DNA를 리턴하는 S가 있고, 배열의 길이가 같은 P와 Q가 있다.  ``` S=CAGCCTA P=[2,
  5, 0] Q=[4, 5, 6]  각 0번째 요소는 2, 4다. 2번째 ~ 4번째 DNA는 GCC...'
category: algorithm
slug: /2020/06/codility-05-02-genomic-range-query/
template: post
---

## Genomic Range Query

### 문제

DNA는 A, C, G, T로 구성되어 있는데, 이는 각각 1, 2, 3, 4를 가르킨다. 이러한 DNA를 리턴하는 S가 있고, 배열의 길이가 같은 P와 Q가 있다.

```
S=CAGCCTA
P=[2, 5, 0]
Q=[4, 5, 6]
```

각 0번째 요소는 2, 4다.
2번째 ~ 4번째 DNA는 GCC이고, 여기서 제일 작은 값은 C, 즉 2를 리턴한다.

각 1번째 요소는 5, 5다.
T 밖에 없으므로 2를 리턴한다.

각 2번째 요소는 0, 6이다.
CAGCCT이고, 가장 작은 값은 A다. 즉 1을 리턴한다.

답은 `[2, 4, 1]` 이다.

### 풀이

```javascript
function solution(S, P, Q) {
  const answers = []
  for (let i = 0; i < P.length; i++) {
    const slice = S.slice(P[i], Q[i] + 1)

    if (slice.indexOf('A') !== -1) {
      answers.push(1)
    } else if (slice.indexOf('C') !== -1) {
      answers.push(2)
    } else if (slice.indexOf('G') !== -1) {
      answers.push(3)
    } else if (slice.indexOf('T') !== -1) {
      answers.push(4)
    }
  }

  return answers
}
```

- slice를 하고, 최초에는 이를 sort해서 테스트 하려니 timeout이 났다.
- slice를 하고, slice한 문자열을 돌면서 체크하니 역시 timeout이 났다.
- slice 된 문자열에 그냥 indexOf를 하기로 했다.
