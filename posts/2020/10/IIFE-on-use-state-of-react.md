---
title: '리액트의 useState와 lazy initialization'
tags:
  - react
  - javascript
published: true
date: 2020-10-18 19:19:27
description: '리액트 최적화의 길은 멀고도 험하다'
---

흔히 쓰고 있는 `useState`를 다시 살펴보자.

```javascript
const [count, setCount] = useState(
  Number.parseInt(window.localStorage.getItem(cacheKey)),
)
```

```javascript
const [count, setCount] = useState(() =>
  Number.parseInt(window.localStorage.getItem(cacheKey)),
)
```

두 코드의 차이는, useState에 직접 변수를 넣는가와 즉시실행 화살표 함수를 넣느냐의 차이다.

`useState`에 직접적인 값 대신에 함수를 넘기는 것을 [게으른 초기화(lazy)](https://reactjs.org/docs/hooks-reference.html#lazy-initial-state)라고 한다. react 공식 문서에서는, 이러한 게으른 초기화를 초기 값이 복잡한 연산을 포함할 때 사용하라고 되어 있다. 게으른 초기화 함수는 **오직 state가 처음 만들어 질 때만 실행된다.** 이 후에 다시 리렌더링이 된다면, 이 함수의 실행은 무시된다.

다시 말해, `useState`는 그 함수가 처음 렌더링 될 때 작동하며, 이는 `count` state의 초기값을 만든다. `setCount`가 실행되면, 전체 함수가 다시 실행되며, `count`의 값을 업데이트 한다. 이는 `count`의 값이 변경될 때 마다 리 렌더링을 발생시킨다. 다시말해, 이 초기 값은 다시 쓰일 일이 없게 된다.

따라서, 리 렌더링이 발생할 때 마다 `localStorage`의 값을 읽지만, 오직 우리는 딱 최초 렌더링 시에만 해당 값이 필요하므로, 이는 필요없는 계산을 계속해서 하게 되는 것이다. 두 번째 예제에서는 게으른 초기화를 하기 때문에 불필요한 계산을 막게된다.

말이 조금 어렵다면, 아래 예제를 살펴보자.

예제1

```javascript
const Counter = () => {
  const initialValue = Number.parseInt(window.localStorage.getItem(cacheKey))
  const [count, setCount] = useState(initialValue)
  // ...
```

이제 매번 리렌더링 하여 `Counter`함수가 호출 될 때 마다, `localStorage`의 값을 계속해서 가져오는 것을 볼 수 있다.

예제2

```javascript
// Example 2
const Counter = () => {
  const [count, setCount] = useState(function() {
    return Number.parseInt(window.localStorage.getItem(cacheKey)),
  })

  // ...
}
```

이 더분째 예제에서는 이전 예제와는 다르게 딱 한번 초기화 할 때 한 번만 우리가 원하는 대로 호출하는 것을 볼 수 있다.

그렇다면 모든 값들을 게으른 초기화로 처리하면 어떨까?

```javascript
// 원시 값 리턴
const Counter = () => {
  const [count, setCount] = useState(() => 0)

  // ...
}
```

```javascript
// prop 또는 이미 존재하는 변수의 값을 리턴
const Counter = ({ initialCount }) => {
  const [count, setCount] = useState(() => initialCount)

  // ...
}
```

각각 초기 값이 간단한 값이거나 이미 계산된 값인 경우이다. 비록 함수가 게으른 초기화로 인해 한번만 호출되지만, 여전히 함수를 만드는 비용이 존재한다. 그리고 함수를 만드는 비용이 보통 변수를 생성하거나 단순히 값을 넘기는 비용보다는 크다. 이는 과한 최적화의 예다.

그렇다면 언제 게으른 최적화를 써야할까? 이는 상황에 따라 다르다. 문서에서는 '비싼 비용의 계산' 이 필요할 때 쓰라고 되어있다. 앞선 예와 같이 `localStorage` 의 접근, `map`, `filter`,`find` 등의 배열을 조작하는 것들이 그 예가 될 수 있다. 일반적으로, 함수를 통해서 값을 구해야한다면, 이는 비싼 비용이 드는 계산이며, 게으른 초기화를 하는게 좋을 수도 있다. `new Date()`도 마찬가지로.
