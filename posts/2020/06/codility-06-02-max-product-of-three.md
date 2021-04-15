---
title: Codility - Max Product of Three
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 12:25:19
description:
  '## Max Product of Three ### 문제  길이 N인 배열 A가 주어졌을때, 임의로 세개의 숫자를 곱했을
  때 가장 큰 값을 만들 수 있는 배열의 Index를 리턴해라.  ``` A[0] = -3 A[1] = 1 A[2] = 2 A[3] = -2
  A[4] = 5 A[5] = 6  2, 4, 5번째를 곱하면 60을 만들수 있고 이것이 가장 큰 ...'
category: algorithm
slug: /2020/06/codility-06-02-max-product-of-three/
template: post
---

## Max Product of Three

### 문제

길이 N인 배열 A가 주어졌을때, 임의로 세개의 숫자를 곱했을 때 가장 큰 값을 만들 수 있는 배열의 Index를 리턴해라.

```
A[0] = -3
A[1] = 1
A[2] = 2
A[3] = -2
A[4] = 5
A[5] = 6

2, 4, 5번째를 곱하면 60을 만들수 있고 이것이 가장 큰 경우 이므로, 60을 리턴하면된다.
```

### 풀이

```javascript
function solution(A) {
  const sorted = A.sort((a, b) => a - b)
  const size = sorted.length

  let biggest = 0

  biggest = sorted[size - 1] * sorted[size - 2] * sorted[size - 3]

  if (sorted[0] < 0 && sorted[1] < 0 && sorted[size - 1] > 0) {
    const possible = sorted[0] * sorted[1] * sorted[size - 1]

    if (possible > biggest) {
      biggest = possible
    }
  }

  return biggest
}
```

하나 조심해야 할 것은, 두 개의 음수 \* 한개의 양수 조합으로도 큰 수를 만들 수 있다는 것이다. 따라서 무조건 큰 값 세개를 곱해버리면 안된다. 따라서 가장 큰 수를 만들 수 있는 경우의 수는 아래와 같다.

- 숫자가 큰 순서대로 세개를 곱하거나
- 음수이하의 가장 작은 숫자 두개를 곱하고 가장 큰 양수를 곱하거나

또 하나 조심해야할 것은, `sort()`다. 그냥 아무 함수 없이 sort했더니, 제대로 정렬하지 못했다. (졸아서 그런거라고 치자)

> compareFunction이 제공되지 않으면 요소를 문자열로 변환하고 유니 코드 코드 포인트 순서로 문자열을 비교하여 정렬됩니다. 예를 들어 "바나나"는 "체리"앞에옵니다. 숫자 정렬에서는 9가 80보다 앞에 오지만 숫자는 문자열로 변환되기 때문에 "80"은 유니 코드 순서에서 "9"앞에옵니다.

한국 말이 더 어렵다 (....)

> If compareFunction is not supplied, all non-undefined array elements are sorted by converting them to strings and comparing strings in UTF-16 code units order. For example, "banana" comes before "cherry". In a numeric sort, 9 comes before 80, but because numbers are converted to strings, "80" comes before "9" in the Unicode order. All undefined elements are sorted to the end of the array.

비교 함수가 정의 되지 않는다면, 모든 요소를 string으로 변환하고 이 string을 UTF-16 코드로 변환해서 비교 한다는 것이다.

```javascript
const a = [-1, -10, -9, -5, -7]
const sorted = a.sort()
console.log(sorted) // [-1, -10, -5, -7, -9]]
```

숫자 비교 시에는 절대 compareFunction을 비우지말자 -

[출처](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Array/sort)

https://app.codility.com/demo/results/trainingCGB84G-ST8/
