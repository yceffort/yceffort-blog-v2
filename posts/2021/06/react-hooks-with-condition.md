---
title: 'React Hooks위에 조건문을 추가할 수 없는 이유'
tags:
  - javascript
  - react
published: false
date: 2021-06-10 18:21:19
description: '처음 접했을 때 범할 수 있는 실수'
---

리액트를 처음 접해서 작업을 했을때, 아래와 같은 실수를 해본 경험이 있다.

```tsx
import React, { useState } from 'react'

export default function Hello({ effect }: { effect: boolean }) {
  const [state, setState] = useState(false)

  console.log('effect', effect)

  if (effect) {
    const [newState, setNewState] = useState(false)
  }

  return <>하이 {newState}</>
}
```

비록 `effect`가 `true`로 넘어와서 조건문을 탄다 할지라도, `newState`는 아래와 같은 경고문과 함께 사용할 수가 없다.

```
Error: newState is not defined
```

좀 더 나아가서, [eslint-plugin-react-hooks](https://github.com/facebook/react/tree/master/packages/eslint-plugin-react-hooks)을 쓰다보면, 아예 에러까지 난다.

> React Hook "useState" is called conditionally. React Hooks must be called in the exact same order in every component render.eslintreact-hooks/rules-of-hooks

- https://reactjs.org/docs/hooks-rules.html#eslint-plugin
- https://github.com/facebook/react/blob/aecb3b6d114e8fafddf6982133737198e8ea7cb3/packages/eslint-plugin-react-hooks/src/RulesOfHooks.js#L455-L465

왜 hook은 조건문 내부에서 쓰지 못하는 걸까?

## 훅이 호출되면 어떤일이 생길까

`useState` `useEffect`와 같은 훅 함수들은 두 단계로 상태값을 저장한다는 점에서 흥미롭다.

- global state에 있는 값을 조정한다.
- 컴포넌트 라이프 사이클에 따라서 실행되지만, 매 실행시에 마다 같은 코드가 실행되는 것이 아니다.

먼저 아래의 간단한 예제를 보자.

```jsx
function Component() {
  const [first, setFirst] = useState('first')
  const [second, setSecond] = useState('second')
  return <>안녕?</>
}
```

리액트 컴포넌트가 마운트되면, 컴포넌트의 인스턴스에 연결된 상태를 생성한다. 이 상태는 컴포넌트가 마운트되어 있는 동안 호출된 모든 훅의 링크드 리스트를 저장한다.

따라서, React가 Component 내부의 `use*` 함수를 호출하게 되면, 링크드 리스트를 만들게 된다.

```json
{
  value: "first",
  next: {
    value: "second",
    next: null, // End of the linked list
  },
};
```

컴포넌트 초기화 단계에서, 리액트의 코드는 아래와 같다.

```javascript
function render(Component) {
  // use* 함수가 호출될 때마다 변경되는 글로벌 변수 값
  global.currentComponentHooks = null
  // 가장 마지막으로 마운트된 훅을 추적
  global.lastMountedHook = null

  const children = Component()
  // [...]
}
```

`useState`의 내부를 상상해본다면, 아래와 같은 모습일 것이다.

```javascript
function useState(value) {
  const hook = {
    value: value,
    next: null,
  }
  if (global.currentComponentHooks === null) {
    // 글로벌 훅이 없다면 이 훅을 등록해버림
    global.currentComponentHooks = hook
  } else {
    // 그렇지 않다면, 글로벌훅의 마지막으로 등록된 훅의 다음에 등록
    global.lastMountedHook.next = hook
  }
  // 글로벌 가장 마지막 훅을 현재 훅으로 교체
  global.lastMountedHook = hook
  // [...]
  return [value, updateState] // 리턴
}
```

위의 상태로 미뤄 보았을때, 앞선 예제 컴포넌트는 아래와 같은 모습으로 추정해볼 수 있다.

```jsx
function Component() {
  // null
  const [first, setFirst] = useState('first')
  // {value 'first', next: null}
  const [second, setSecond] = useState('second')
  // {value 'first', next: {value: 'second', next: null}}
  return /*...*/
}
```

## 업데이트

이제 `set*`함수로 값을 갱신했다고 가정해보자. 마운트 시점에서는, 여러개의 훅 함수를 링크드 리스트에 쌓아두었다. 업데이트가 이뤄진다면 링크드 리스트를 순회하여 저장된 값을 각각 읽게 된다.

리액트는, 컴포넌트를 다시 렌더링하기 전에 링크드리스트의 루트를 가리키는 포인터를 만들어야 한다.

```javascript
function update(Component) {
  global.currentHook = global.currentComponentHooks // {value 'first', next: {value: 'second', next: null}}
  const children = Component()
  // global.currentHook === null
}
```

이 말인 즉슨, `useState` 함수의 내부도 반드시 바껴야 한다는 것을 의미한다.

```javascript
// 더이상 초기값이 필요하지 않다
function useState(/*value*/) {
  const hook = global.currentHook
  global.currentHook = hook.next // 포인터를 다음 훅으로 이동시킨다
  return [hook.value, updateState]
}
```

그리고 우리가 만든 컴포넌트에서는 이렇게 보이게 된다.

```javascript
function Component() {
  // {value 'updated', next: {value: 'second', next: null}}
  const [first, setFirst] = useState('first') // first === 'updated' our mutated state
  // {value 'second', next: null}
  const [second, setSecond] = useState('second') // second === 'second' the unchanged initial state
  return /*...*/
}
```

## 그래서 왜, 훅에 조건문을 달 수 없는가?

만약에 이러한 훅 동작 외부에 조건문을 달면 어떻게 될까?

```javascript
function Component({ doEffect }) {
  const [first, setFirst] = useState(0)
  if (doEffect) {
    useEffect(/*...*/)
  }
  const [second, setSecond] = useState(0)
}
```

만약 해당 컴포넌트의 `doEffect`가 `false`로 넘어왔다고 가정해보자.

```javascript
function Component({ doEffect }) {
  // {value: 0, next: {value: 0, next: null}}
  const [first, setFirst] = useState(0)
  if (doEffect) {
    // {value: 0, next: null}
    useEffect(/*...*/) // ⚠️ Wrong hook here
  }
  // null
  const [second, setSecond] = useState(0) // ⚠️ No hook left!!
}
```

`doEffect`의 값에 따라 링크드리스트의 연결이 끊어지면서 이후의 훅을 등록할 수가 없게 된다. 이를 수정하기 위해서는, `useEffect` 내부에서 조건문을 타야 한다.

```javascript
useEffect(() => {
  if (doEffect) {
    // Do your magic
  }
}, [doEffect])
```
