---
title: Codility - Max Counters
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-24 05:25:19
description:
  '## Max Counters ### 문제  숫자 N이 주어진다. 이 숫자 N은 모든 요소가 0인 길이 N인 배열을
  의미한다. 그리고 배열 A가 존재한다.   ``` 숫자 N이 5로 주어지고, 배열 A는 [3, 4, 4, 6, 1, 4, 4] 라고
  가정하자.  초기 값 [0, 0, 0, 0 0] A[0] = 3, 3번째 (3-1번째) 요소의 크기를 1 늘린...'
category: algorithm
slug: /2020/06/codility-04-02-max-counters/
template: post
---

## Max Counters

### 문제

숫자 N이 주어진다. 이 숫자 N은 모든 요소가 0인 길이 N인 배열을 의미한다. 그리고 배열 A가 존재한다.

```
숫자 N이 5로 주어지고, 배열 A는 [3, 4, 4, 6, 1, 4, 4] 라고 가정하자.

초기 값 [0, 0, 0, 0 0]
A[0] = 3, 3번째 (3-1번째) 요소의 크기를 1 늘린다.
- [0, 0, 1, 0, 0]
A[0] = 4, 4번째 (4-1번째) 요소의 크기를 1 늘린다.
- [0, 0, 1, 1, 0]
A[0] = 4, 4번째 (4-1번째) 요소의 크기를 1 늘린다.
- [0, 0, 1, 2, 0]
A[0] = 6, 모든 숫자의 크기를 현재 가장 큰 값으로 맞춘다.
- [2, 2, 2, 2, 2]
....

최종 결과는
- [3, 2, 2, 4, 2]
```

### 풀이

```javascript
function solution(N, A) {
  // 0으로 초기화된 길이 N의 배열을 만든다.
  const array = Array(N).fill(0)
  // 배열 내 최대 값
  let max = 0
  // 마지막 max counter의 기준이 되었던 수
  let maxCounter = 0
  for (let i = 0; i < A.length; i++) {
    // 모든 숫자를 올려야 하는 경우
    if (A[i] > N) {
      // maxCounter에 다같이 올라가는 최대 숫자를 저장해 둔다.
      maxCounter = max

      // 하나씩만 올리면 되는 경우
    } else {
      // 현재 숫자가 maxCounter보다 작을 경우 maxCounter로 초기화 한다.
      if (array[A[i] - 1] < maxCounter) {
        array[A[i] - 1] = maxCounter
      }

      // 그리고 +1을 한다
      array[A[i] - 1] += 1

      // 이렇게 새롭게 세팅된 숫자가 배열의 최대값인지 확인한다.
      if (max < array[A[i] - 1]) {
        max = array[A[i] - 1]
      }
    }
  }

  // 배열의 값이 maxCounter보다 작다면 그 값으로 리턴한다.
  return array.map((i) => (i < maxCounter ? maxCounter : i))
}
```

## 해설

처음에는 저 maxCounter액션을 array를 map을 돌면서 +1 을 해줬더니 timeout 에러가 났다. 그도 그럴 것이 안그래도 배열을 n회 순환하는데, +1 을하면서 또 순환하면 복잡도가 `O(N^2)`이 될 것이기 때문이다. 그래서 최대한 배열을 한번에 순환하는 방식으로 해결하고자 노력했다.

https://app.codility.com/demo/results/training7YGA6S-4D7/
