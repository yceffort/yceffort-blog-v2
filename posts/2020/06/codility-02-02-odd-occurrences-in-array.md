---
title: Codility - Odd Occurrences in array
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-23 05:25:19
description: "## 2-2 Odd Occurrences in array ### 문제  숫자로 이뤄진 배열에서 홀수 번 등장하는 숫자를
  찾아서 리턴해라.  ``` A[0] = 9  A[1] = 3  A[2] = 9 A[3] = 3  A[4] = 9  A[5] = 7 A[6]
  = 9  7은 한번만 등장하므로 7을 리턴해야 한다. ```  ### 풀이  ```javascri..."
category: algorithm
slug: /2020/06/codility-02-02-odd-occurrences-in-array/
template: post
---
## 2-2 Odd Occurrences in array

### 문제

숫자로 이뤄진 배열에서 홀수 번 등장하는 숫자를 찾아서 리턴해라.

```
A[0] = 9  A[1] = 3  A[2] = 9
A[3] = 3  A[4] = 9  A[5] = 7
A[6] = 9

7은 한번만 등장하므로 7을 리턴해야 한다.
```

### 풀이

```javascript
function solution(A) {
    // 등장하는 숫자를 저장한다.    
    const map = {}
    
    // 숫자를 순회한다
    for (let i=0; i <= A.length - 1; i++) {
        const target = A[i]
        // 해당 숫자가 존재한다면 해당 키를 제거한다.
        if (map[target]) {
            delete map[target]
        // 해당 숫자가 존재하지 않는다면 (처음등장했다면) 추가한다.
        } else {
            map[target] = true
        }
    }
    
    // 리턴한다.
    return +Object.keys(map)[0]
}
```

### 해설

믿거나 말거나 성능은 `O(N) or O(N*log(N))` 가 나왔는데, 생각보다 key-value객체에 key값으로 접근하는 속도가 빠른듯.

https://app.codility.com/demo/results/trainingBF4N34-BBK/