---
title: 나만의 eslint 룰 만들어보기'
tags:
  - javascript
  - eslint
published: true
date: 2022-06-17 12:19:04
description: ''
---

## Table of Contents

## Introduction

react@17 이 업데이트 되면서 더이상 `jsx, tsx` 파일에 `import React`를 할 필요가 없어졌다. [참고](https://ko.reactjs.org/blog/2020/09/22/introducing-the-new-jsx-transform.html) 이를 사용함으로써 여러가지 이점이 있지만, 무엇보다 번들 사이즈가 줄어든 다는 장점이 가장 크다. (아주 작은 정도지만)

그러나 기존 react@16 기반의 코드에서 저 `import React from 'react'` 코드를 모두 제거하기란 쉽지 않다. `import React from 'react'`를 모두 찾고 검색해서 지우는 방법도 있겠지만, 저 사이에 무엇이라도 껴 있다면, (`import React, { MouseEvent } from 'react'` 와 같이) 이 방법도 소용이 없다. 그래서 어떻게 해결할까 고민하던 중, `eslint`가 있으니 이를 활용하면 쉽게 해결할 수 있지 않을까 하는 아이디어가 떠올랐다.

## `no-restricted-imports`

아마도 대부분의 프로젝트에서는 eslint를 사용 중일 것이다. 그래서 eslint에 있는 기본 룰인 [no-restricted-imports](https://eslint.org/docs/latest/rules/no-restricted-imports)를 사용해서 해결해보자.

```javascript
module.exports = {
  rules: {
    'react/react-in-jsx-scope': ['off'],
    'no-restricted-imports': [
      'error',
      {
        paths: [
          {
            name: 'react',
            importNames: ['default'],
            message: "import React from 'react' makes bundle size larger.",
          },
        ],
      },
    ],
  },
}
```

`react` 라는 import가 있고, 이 importNames이 기본값 (`React`)일 경우 에러메시지를 띄우는 방법이다. 이방법을 활용하면 같은 원리로 [트리쉐이킹이 안되는 `lodash`](/2021/08/javascript-tree-shaking#%EB%AC%B4%EC%97%87%EC%9D%84-%ED%95%B4%EC%95%BC%ED%95%A0%EC%A7%80-%EA%B0%90%EC%9D%B4-%EC%98%A4%EC%A7%80-%EC%95%8A%EC%9D%84-%EB%95%8C) import 하는 것을 막을 수 있다.

> 하지만 아쉽게도 이 방법은 자동으로 fix 까지 해주지 않는다. 물론 자동으로 import를 해서 fix 할 수도 있겠지만, 그것보다는 개발자가 직접 수정하는 것이 더 안전할 것이다.

## eslint 룰 만들기?

이 방법으로 문제를 해결하긴 했지만, 갑자기 궁금했졌다. 내가 직접 관련된 문제를 해결할 수 있는 rules을 만들어 볼 순 없을까? 🧐

### eslint 동작 방식 이해

eslint 의 동작방식을 이해하기 위해서 알아야 하는 단 한가지는 바로 [AST](/2021/05/ast-for-javascript)다. 이 글을 요약해서 설명하자면, AST는 우리가 작성한 코드를 기반으로 트리 구조의 데이터 스트럭쳐를 만들어 낸다. 즉, eslint 는 코드를 AST를 활용해서 트리구조를 만든 다음, 여기에서 지적하고 싶은 코드를 만들어서 룰로 저장하는 것이다.

### 간단한 예제

먼저, 한 글자 짜리 변수를 막는 룰을 만든다고 가정해보자. https://astexplorer.net/ 에서 변수 선언문 트리를 만들면, 아래와 같은 결과를 얻을 수 있다.

```javascript
const hello = 'world'
```
