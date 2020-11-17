---
title: Javascript Reduce
date: 2019-07-22 08:36:28
published: true
tags:
  - javascript
description: "멍청이라 그런지 `reduce` 함수가 잘 이해 되지 않았다. ## Reduce  ```javascript const
  list = [1, 2, 3, 4, 5]; const initValue = 10; const totalSum =
  list.reduce(   (accumulator, currentValue, currentIndex, array) => {  ..."
category: javascript
slug: /2019/07/22/javascript-reduce/
template: post
---
멍청이라 그런지 `reduce` 함수가 잘 이해 되지 않았다.

## Reduce

```javascript
const list = [1, 2, 3, 4, 5];
const initValue = 10;
const totalSum = list.reduce(
  (accumulator, currentValue, currentIndex, array) => {
    return accumulator + currentValue;
  },
  initValue
);
```

```
25
```

- `currentValue`: 처리할 현재 요소
- `currentIndex` (optional): 처리할 요소의 인덱스
- `accumulator`: 콜백의 반환값을 계속해서 누적한다. 이 예제에서는 처음엔 `1`, 그 다음엔 `1 + currentValue`, 그 다음엔 `(1 + currentValue) + currentValue` 가 될 것이다.
- `array` (optional): `reduce`를 호출한 배열, 여기서는 `list = [1, 2, 3, 4, 5]`이 될 것이다.
- `initValue` (optional): `reduce`의 최초 값. 없으면 배열의 0번째 값이 된다. 이 예제에서는 `initValue`값이 10 이라서, 최종결과는 `10 + (1 + 2 ... + 5)` 이 될 것이다.

| call | accumulator | currentValue | currentIndex | array       | return |
| ---- | ----------- | ------------ | ------------ | ----------- | ------ |
| 1st  | 10          | 1            | 0            | [1,2,3,4,5] | 11     |
| 2nd  | 11          | 2            | 1            | [1,2,3,4,5] | 13     |
| 3rd  | 13          | 3            | 2            | [1,2,3,4,5] | 16     |
| 4th  | 16          | 4            | 3            | [1,2,3,4,5] | 20     |
| 5th  | 20          | 5            | 4            | [1,2,3,4,5] | 25     |

## 중첩 배열 펼치기

```javascript
const complicatedList = [[0, 1], [2, 3], [4], [5, 6]];
complicatedList.reduce(
  (accumulator, currentValue) => accumulator.concat(currentValue),
  []
);
```

```
[0, 1, 2, 3, 4, 5, 6]
```

이보다 더 괴랄한 array의 경우에도 재귀를 사용하여 가능하다.

```javascript
const moreComplicatedList = [[0, 1], [[[2, 3]]], [[4, 5]], 6];

const flatten = function(arr, result = []) {
  for (let i = 0, length = arr.length; i < length; i++) {
    const value = arr[i];
    if (Array.isArray(value)) {
      flatten(value, result);
    } else {
      result.push(value);
    }
  }
  return result;
};

flatten(moreComplicatedList);
```

```
[0, 1, 2, 3, 4, 5, 6]
```
