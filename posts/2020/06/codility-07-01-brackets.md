---
title: Codility - Brackets
tags:
  - algorithm
  - javascript
published: true
date: 2020-06-25 08:58:02
description:
  '## Brackets ### 문제  문자열 S가 주어지고, S는 다음 경우 일 때 참을 반환해야 한다.  - S가
  비어있는 경우 - `(U)` or `[U]` or `{U}` 의 형태로 괄호안에 문자열이 있는 경우 - 괄호가 짝이 맞게 닫혀있는
  경우  예를 들어  `{[()()]}`는 괄호가 알맞게 들어있지만, `([)()]`는 그렇지 못하다. (짝은 맞...'
category: algorithm
slug: /2020/06/codility-07-01-brackets/
template: post
---

## Brackets

### 문제

문자열 S가 주어지고, S는 다음 경우 일 때 참을 반환해야 한다.

- S가 비어있는 경우
- `(U)` or `[U]` or `{U}` 의 형태로 괄호안에 문자열이 있는 경우
- 괄호가 짝이 맞게 닫혀있는 경우

예를 들어

`{[()()]}`는 괄호가 알맞게 들어있지만, `([)()]`는 그렇지 못하다. (짝은 맞지만 잘못닫혀있음)괄호가 올바르게 형성되어 있는 경우 1, 아니면 0을 리턴하자.

### 풀이

```javascript
function solution(S) {
  const splited = S.split('')

  const stack = []

  for (let i of splited) {
    // 여는 거
    if (i === '{' || i === '[' || i === '(') {
      stack.push(i)
    } else {
      if (stack.size === 0) return 0

      // 닫는 것이라면 가장 최근에 열었던 것이랑 비교 한다.
      const pop = stack.pop()

      if (i === ')') {
        if (pop !== '(') {
          return 0
        }
      }

      if (i === '}') {
        if (pop !== '{') {
          return 0
        }
      }

      if (i === ']') {
        if (pop !== '[') {
          return 0
        }
      }
    }
  }

  return stack.length === 0 ? 1 : 0
}
```

여는 괄호라면 stack에 넣고, 닫는 괄호라면 스택에 맨지막 괄호와 비교해서 적절한 괄호인지 확인한다.

https://app.codility.com/demo/results/training9MREFG-CYW/
