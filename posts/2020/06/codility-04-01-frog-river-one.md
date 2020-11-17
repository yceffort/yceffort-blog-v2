---
title: Codility - Frog River One
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-24 05:25:19
description: "## Frog River One ### 문제  개구리가 X 까지 가고 싶은데, X까지 가기 위해서는 1부터 X를 모두
  지나가야 한다. 예를 들어보자.   ``` 이렇게 배열이 주어져 있고  A[0] = 1 A[1] = 3 A[2] = 1 A[3] = 4
  A[4] = 2 A[5] = 3 A[6] = 5 A[7] = 4  5까지 가고 싶다고 가정했을때, A[..."
category: algorithm
slug: /2020/06/codility-04-01-frog-river-one/
template: post
---
## Frog River One

### 문제

개구리가 X 까지 가고 싶은데, X까지 가기 위해서는 1부터 X를 모두 지나가야 한다. 예를 들어보자.


```
이렇게 배열이 주어져 있고

A[0] = 1
A[1] = 3
A[2] = 1
A[3] = 4
A[4] = 2
A[5] = 3
A[6] = 5
A[7] = 4

5까지 가고 싶다고 가정했을때, A[6]까지는 1~5의 지점이 모두 존재하기 때문에 갈수 있으며, 해당 인덱스인 6을 리턴한다. 
하지만 갈 수 없을 경우 -1을 리턴하면 된다.
```

### 풀이

```javascript
function solution(X, A) {
    
    // X 까지 가고 싶다면, 1~X 까지의 숫자가 모두 존재해야한다.
    let sum = (X * (X + 1)) / 2
    
    // 등장했던 숫자인지 아닌지 판별한다.
    // 이럴거면 굳이 Set을 안써도 되긴하네
    const appear = new Set()

    for (let i=0; i < A.length; i++) {    
        const target = A[i]
        
        // 전에 없던 숫자만
        if (!appear.has(target)) {
            // 숫자를 노출 목록(?) 에 더하고
            appear.add(target)

            // 합계에서 하나씩 뺸다
            sum -= target
            
            // 0 이 된다면 바로 그 index다
            if (sum === 0) {
                return i
            }

            // 만약 0 보다 작아진다면 이미 글러먹었으므로 리턴한다.
            if (sum < 0) {
              return -1
            }
        }
    }
    
    return -1
}
```

## 해설

배열 관련 문제가 나온다면 항상 최초 한번의 순회로 어떻게 잘 승부 볼 수 있을지 고민해보자

https://app.codility.com/demo/results/trainingKDYYTF-8AU/