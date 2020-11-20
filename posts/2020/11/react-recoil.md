---
title: React를 위한 상태관리 라이브러리, Recoil
tags:
  - javascript
  - react
published: true
date: 2020-11-16 22:28:12
description: '상태관리 춘추전국시대'
---

`Redux`, `MobX`등은 이제 리액트 프로젝트를 만든다면 필수로 같이 쓰게 되는 상태 관리 라이브러리 들 중 하나가 된 것 같다. 상태 관리 라이브러리의 필요성은 알지만 서도, 무분별하게 상태 관리 라이브러리를 설치해서 무조건 쓰는 것에 대해 나 또한 그다지 긍정적이지는 않다.

- https://medium.com/@dan_abramov/you-might-not-need-redux-be46360cf367
- https://dev.to/g_abud/why-i-quit-redux-1knl
- https://hackernoon.com/goodbye-redux-26e6a27b3a0b

개인적으로는 리액트에서 제공하는 `useState` `useContext` 등으로 충분하다고 생각하지만서도, 이미 대세가 되어버린 상태관리 라이브러리의 시대에 나 또한 흐름을 따라 갈 수 밖에 없다.

이 와중에 React에서 만든 `Recoil`이라고 하는 상태 관리 라이브러리가 나왔다. 과연 이 라이브러리가 다른 상태관리 라이브러리랑은 무엇이 다른지, 또 정말 쓸 만한지 고민해보자.

앞서서 리액트의 `Context`만으로 충분할 것 같다고 말한 것과는 다르게, 리액트에서는 `Context`의 한계에 대해서 명백히 인식하고 있는 것 같다.

> My personal summary is that new context is ready to be used for low frequency unlikely updates (like locale/theme). It's also good to use it in the same way as old context was used. I.e. for static values and then propagate updates through subscriptions. It's not ready to be used as a replacement for all Flux-like state propagation.

https://github.com/facebook/react/issues/14110#issuecomment-448074060

그리고 실제로 React의 Context API를 쓰던 Redux가 성능상의 문제로 인해 이를 철회한 사건도 있었다.

> In v6, we switched from individual components subscribing to the store, to having <Provider> subscribe and components read the store state from React's Context API. This worked, but unfortunately the Context API isn't as optimized for frequent updates as we'd hoped, and our usage patterns led to some folks reporting performance issues in some scenarios.

https://github.com/reduxjs/react-redux/releases/tag/v7.0.1

또한 Provider의 값이 배열이나 객체 인 경우, 여기에서 구조가 조금이라도 바뀌게 된다면 `Context`를 구독하고 있는 하위 모든 컴포넌트가 다시 렌더링되는 참사가 발생된다.

React Context API는 분명 나쁜 API는 아니지만, 그 한계가 어느정도 있다는 것을 알 수 있다. 그렇기 때문에 Facebook 팀에서도 그 한계를 인지하고 Recoil 을 만든게 아닐 까 싶다.

> For reasons of compatibility and simplicity, it's best to use React's built-in state management capabilities rather than external global state. But React has certain limitations:

> - Component state can only be shared by pushing it up to the common ancestor, but this might include a huge tree that then needs to re-render.
> - Context can only store a single value, not an indefinite set of values each with its own consumers.
> - Both of these make it difficult to code-split the top of the tree (where the state has to live) from the leaves of the tree (where the state is used).

## Recoil

### RecoilRoot

`recoil` 의 state를 사용하기 위해서는 부모 트리에 `RecoilRoot`를 선언해야 한다. 가장 좋은 위치는 바로 Root다.

```javascript
import React from 'react'
import { RecoilRoot, atom } from 'recoil'

function App() {
  return (
    <RecoilRoot>
      <Component />
    </RecoilRoot>
  )
}
```

### Atom

`atom`은 state의 조각을 의미한다. `atom`은 어떤 컴포넌트에서든 읽기/쓰기가 가능하다. 컴포넌트는 이 `atom`의 값을 구독하여 읽을 수 있으며, `atom`의 업데이트는 곳 이를 구독하고 있는 모든 컴포넌트의 업데이트를 야기한다.

```javascript
const textState = atom({
  key: 'textState', // unique ID
  default: '', // 기본값
})
```

### useRecoilState

이름에서 느껴지듯이, `useState`와 비슷하게 값과 이를 조작할 수 있는 `setter` 함수를 리턴한다.

```javascript
const [text, setText] = useRecoilState(textState)
```

### useRecoilValue

`useRecoilState`와는 다르게, 오로지 `atom`의 값만 가져올 수 있다.

```javascript
const text = useRecoilValue(textState)
```

### useSetRecoilState

`atom`의 `setter` 만 가져올 수 있다.

```javascript
const setText = useSetRecoilState(textState)
```

### selector

`atom`과 함께 중요한 개념 중 하나다. `Selector`는 상태에서 파생된 데이터다. `get`을 활용하여 `atom`으로 부터 파생된 데이터를 가져올 수 있으며, `set`을 활용하여 하나이상의 atom을 업데이트 할 수 있다.

```typescript
function selector<T>({
  key: string,

  get: ({
    get: GetRecoilValue
  }) => T | Promise<T> | RecoilValue<T>,

  set?: (
    {
      get: GetRecoilValue,
      set: SetRecoilState,
      reset: ResetRecoilState,
    },
    newValue: T | DefaultValue,
  ) => void,

  dangerouslyAllowMutability?: boolean,
})
```

- `key`: 유니크 아이디로, 애플리케이션 전체에서 다른 `selector`나 `atom`과 중복되서는 안된다.
- `get`: 상태로 부터 연산할 수 있는 값이다. 단순히 값이나 `Promise`로 부터 야기되는 비동기 값을 가져올 수 있으며, 또한 같은 타입을 갖는 `atom`이나 `selector` 를 리턴할 수도 있다.
  - get: 다른 `atom` `selector`에서 값을 가져오기 위해 제공되는 함수다. 이 `get`을 거치는 모든 `atom`과 `selector`는 의존성을 가진 것으로 간주된다. 따라서 이 `get`에서 쓰이는 값이 변하게 되면, 이 selector 또한 변하게 된다.
- `set?`: 만약 이 `set`이 설정되면, `selector` 는 쓰기 가능한 `state`를 리턴하게 된다.
  - `get`: 위와 마찬가지로 다른 `atom` `selector`에서 값을 가져오기 위해 제공되는 함수다.
  - `set`: `recoil`의 state 값을 쓰기 위해 제공 되는 함수다. 첫번째 파라미트로는 `Recoil`의 state를, 두번째 파라미터로는 새로운 값을 넘겨주면 된다.
- `dangerouslyAllowMutability`: `selector`는 파생된 상태로 부터의 순수함수 이기 때문에, 의존성의 같은 input이 제공되면 항상 같은 값을 리턴해야 한다. 이 옵션을 오버라이드 하고 싶을 때 쓴다.

예제를 살펴보자.

```javascript
import { atom, selector, useRecoilState, DefaultValue } from 'recoil'

// 화씨 온도를 저장해 두는 atom
const tempFahrenheit = atom({
  key: 'tempFahrenheit',
  default: 32,
})

// 섭씨 온도는 화씨로 부터 파생된다.
const tempCelsius = selector({
  key: 'tempCelsius',
  // 현재 화씨 값을 기준으로 연산하여 화씨 값을 가져온다.
  get: ({ get }) => ((get(tempFahrenheit) - 32) * 5) / 9,
  // 섭씨 값을 설정하면, 화씨 값을 set 한다.
  set: ({ set }, newValue) =>
    set(
      tempFahrenheit,
      newValue instanceof DefaultValue ? newValue : (newValue * 9) / 5 + 32,
    ),
})

function TempCelsius() {
  // selector와 atom 모두 useRecoilState를 활용하여 값을 설정하고 가져오는 것을 알 수 있다.
  const [tempF, setTempF] = useRecoilState(tempFahrenheit)
  const [tempC, setTempC] = useRecoilState(tempCelsius)
  const resetTemp = useResetRecoilState(tempCelsius)

  const addTenCelsius = () => setTempC(tempC + 10)
  const addTenFahrenheit = () => setTempF(tempF + 10)
  const reset = () => resetTemp()

  return (
    <div>
      Temp (Celsius): {tempC}
      <br />
      Temp (Fahrenheit): {tempF}
      <br />
      <button onClick={addTenCelsius}>Add 10 Celsius</button>
      <br />
      <button onClick={addTenFahrenheit}>Add 10 Fahrenheit</button>
      <br />
      <button onClick={reset}>>Reset</button>
    </div>
  )
}
```

## 느낌

- 일단 API가 굉장히 단순하고, hook을 사용하고 있기 때문에 리액트의 hook 생태계에 익숙한 사람들에게 낮은 러닝 커브로 다가올 것 같은 생각이 든다. 또 현재 `state`로 되어 있는 리액트 프로젝트를 굉장히 빠르게 마이그레이션 할 수 있을 것 같다. `<RecoilRoot/>`로 루트 프로젝트를 감싸고, `useState`를 `useRecoilState`로 바꾸면 일단은 된다.
- 컴포넌트가 사용하는 데이터만 별개로 사용할 수 있어서 좋았다.
- `selector` 라는 이름이 주는 혼란함이 있었다. `selector` 인데 `set`이 왜 됨???
- [리액트 동시성 모드가 사용가능해지면 이를 지원할 수도 있다는 언급이 있었다.](https://recoiljs.org/docs/introduction/motivation/) 왜냐하면 [Recoil은 내부적으로 React의 상태를 사용하고 있으며](https://github.com/facebookexperimental/Recoil/blob/55059f54ad1d09bfac8d086316bb18bed9cc2879/src/hooks/Recoil_Hooks.js#L20) 이는 곧 [React에서 내놓을 동시성 모드](https://ko.reactjs.org/docs/concurrent-mode-intro.html)를 지원할 수도 있다는 가능성이 존재한다고 볼 수 있기 때문이다. (실제로 motivation에서 그렇게 이야기 하기도 했고) 사용이 간편하다, Facebook이 만들었다는 것 외에 다른 상태 관리 라이브러리와 다른 가장 큰 차별점 & 그리고 도입을 해야하는 이유가 있다면 바로 이것 때문이 아닐 까 싶다. (물론 아직은 멀었지만)
  > We have the possibility of compatibility with Concurrent Mode and other new React features as they become available.

## 더 알아보기

- https://www.youtube.com/watch?v=_ISAA_Jt9kI&ab_channel=ReactEurope
- https://ui.toast.com/weekly-pick/ko_20200616
