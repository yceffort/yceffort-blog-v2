---
title: '리액트 v18 버전 톺아보기'
tags:
  - react
  - javascript
  - typescript
published: true
date: 2022-04-04 19:02:15
description: '큰거 왔다'
---

## Table of Contents

## Introduction

대규모 애플리케이션에서 버전업을 한다는 것은, 그것도 주로 사용하는 major framework의 major 버전 업을 하는 것은 꽤나 어려운 일이다. 지금도 잘 작동하고 있는 애플리케이션을 왜 업데이트 해야 하는지 개발자 부터 저 높은 어르신 까지 먼저 설득이 필요하다. 허락을 구했다면 breaking change가 있는지 살펴보고 있다면 수정해야 한다. 만약 수정 가이드가 있다면 다행이지만 없다면 코드를 하나씩 살펴보면서 고쳐야 한다. 또 고치는 데서만 끝나는 것이 아니다. regression test도 필요하고, 테스트 만으로는 못미더울 기획자나 QA 테스터 분께서 살펴보는 시간도 필요하다. 이런 저런 이유로 봤을 때 대다수의 많은 프로젝트들이 아직도 구형 버전에 머물러 있는 것은 그리 놀라운 일은 아니다. major 버전업은 누구에게나 피곤한 일이다.

그럼에도 개발자들은 항상 major 버전업에 귀기울일 필요는 있다. major 버전업은 분명 기능적으로든 성능적으로든 좋은 방향이 적용되어 있을 것이고, 이는 개발자들에게 좀 더 나은 개발 경험 내지는 고객들에게 더 좋은 애플리케이션 경험을 제공해 줄 수 있다. 또 새로운 개발자를 유인할 수 있는 좋은 방법이기도 하다. jquery로 되어 있는 웹 애플리케이션과 최신의 자바스크립트 프레임워크와 섹시한 문법(?) 으로 작성되어 있는 웹 애플리케이션, 둘 중에 어떤 것을 개발하고 싶은지 열에 아홉은 후자를 선호할 것이다.

웹 애플리케이션 시장의 큰 파이를 차지하고 있는 react의 18 버전이 나왔다. [공식 블로그 글](https://reactjs.org/blog/2022/03/29/react-v18.html)을 통해서 어떤 것이 변경되어있는지 대략적으로 알 수 있고 또 훌륭하게 정리해놓은 블로그 글도 여기저기 많다. 하지만 조금 더 깊게 공부해보고자 [공식 CHANGELOG](https://github.com/facebook/react/blob/main/CHANGELOG.md#1800-march-29-2022)를 보고, 직접 사용해보고, 요약해 보고자한다.

## New Feature

### React

#### `useId`

`useId`는 클라이언트와 서버간의 hydration의 mismatch를 피하면서 유니크 아이디를 생성할 수 있는 새로운 훅이다. 이는 주로 고유한 `id`가 필요한 접근성 API와 사용되는 컴포넌트에 유용할 것으로 기대된다. 이렇게 하면 React 17 이하에서 이미 존재하고 있는 문제를 해결할 수 있다. 그리고 이는 리액트 18에서 더 중요한데, 그 이유는 새로운 스트리밍 렌더러가 HTML을 순서에 어긋나지 않게 전달해 줄 수 있기 때문이다.

아이디 생성 알고리즘은 [여기](https://github.com/facebook/react/pull/22644)에서 살펴볼 수 있다. 아이디는 기본적으로 트리 내부의 노드의 위치를 나타내는 base 32 문자열이다. 트리가 여러 children으로 분기 될때 마다, 현재 레벨에서 자식 수준을 나타내는 비트를 시퀀스 왼쪽에 추가하게 된다.

```jsx
import Head from 'next/head'
import styles from '../styles/Home.module.css'
import { useId } from 'react'
import Child from '../src/components/child'
import SubChild from '../src/components/SubChild'

export default function Home() {
  const id = useId()
  return (
    <>
      <div className="field">Home: {id}</div>
      <SubChild />
      <SubChild />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
    </>
  )
}
```

```jsx
import { useId } from 'react'

export default function Child() {
  const id = useId()
  return <div>child: {id}</div>
}
```

```jsx
import { useId } from 'react'
import Child from './child'

export default function SubChild() {
  const id = useId()

  return (
    <div>
      Sub Child:{id}
      <Child />
    </div>
  )
}
```

```
Home: :r0:
Sub Child::r1:
child: :r2:
Sub Child::r3:
child: :r4:
child: :r5:
child: :r6:
child: :r7:
child: :r8:
child: :r9:
child: :ra:
child: :rb:
child: :rc:
child: :rd:
child: :re:
child: :rf:
child: :rg:
child: :rh:
```

자세한 알고리즘을 알고 싶다면, 앞서 언급한 PR을 참고하면 도움이 될 것 같다.

#### `startTransition` `useTransition`

이 두 메소드를 사용하면 일부 상태 업데이트를 긴급하지 않은 것 (not urgent)로 표시할 수 있다. 이것으로 표시되지 않은 상태 업데이트는 긴급한 것으로 간주된다. 긴급한 상태 업데이트 (input text 등)가 긴급하지 않은 상태 업데이트 (검색 결과 목록 렌더링)을 중단할 수 있다.

상태 업데이트를 긴급한 것과 긴급하지 않은 것으로 나누어 개발자에게 렌더링 성능을 튜닝하는데 많은 자유를 주었다고 볼 수 있다.

```javascript
function App() {
  const [resource, setResource] = useState(initialResource)
  const [startTransition, isPending] = useTransition({ timeoutMs: 3000 })
  return (
    <>
      <button
        disabled={isPending}
        onClick={() => {
          startTransition(() => {
            const nextUserId = getNextId(resource.userId)
            setResource(fetchProfileData(nextUserId))
          })
        }}
      >
        Next
      </button>
      {isPending ? 'Loading...' : null} <ProfilePage resource={resource} />
    </>
  )
}
```

- `startTransition`는 함수로, 리액트에 어떤 상태변화를 지연하고 싶은지 지정할 수 있다.
- `isPending`은 진행 여부로, 트랜지션이 진행중인지 알 수 있다.
- `timeoutMs`로 최대 3초간 이전 화면을 유지한다.

이를 활용하면, 버튼을 눌러도 바로 로딩상태로 전환되는 것이 아니고 이전화면에서 진행상태를 볼 수 있게 된다.

#### `useDeferredValue`

`useDeferredValue`를 사용하면, 트리에서 급하지 않은 부분의 재렌더링을 지연할 수 있다. 이는 `debounce`와 비슷하지만, 몇가지 더 장점이 있다. 고정된 지연시간이 없으므로, 리액트는 첫번째 렌더링이 반영되는 즉시 지연 렌더링을 시도한다. 이 지연된 렌더링은 인터럽트가 가능하며, 사용자 입력을 차단하지 않는다.

```javascript
import { useDeferredValue } from 'react'

const deferredValue = useDeferredValue(value, {
  timeoutMs: 5000,
})
```

`value`의 값이 바뀌어도, 다른 렌더링이 발생하는 동안에는 최대 5000ms가 지연된다. 시간이 다되거나, 렌더링이 완료된다면 `deferredValue`가 변경되면서 상태값이 변하게 될 것이다.

#### `useSyncExternalStore`

`useSyncExternalStore`는 스토어에 대한 업데이트를 강제로 동기화 하여 외부 스토어가 concurrent read를 지원할 수 있도록 하는 새로운 훅이다. 외부 데이터에 대한 원본에 대한 subscription을 필요로 할 때 더이상 `useEffect`가 필요하지 않고, 이는 리액트 외부 상태와 통합되는 모든 라이브러리에 권장된다.

새로운 용어들이 몇개 보인다. 살펴보자

- `External Store`: 외부 스토어라는 것은 우리가 subscribe하는 무언가를 의미한다. 예를 들어 리덕스 스토어, 글로벌 변수, dom 상태 등이 될 수 있다.
- `Internal Store`: `props` `context` `useState` `useReducer` 등 리액트가 관리하는 상태를 의미한다.
- `Tearing`: 시각적인 비일치를 의미한다. 예를 들어, 하나의 상태에 대해 UI가 여러 상태로 보여지고 있는, (= 각 컴포넌트 별로 업데이트 속도가 달라서 발생하는) UI가 찢어진 상태를 의미한다.

사실 리액트 18이전에는, 이러한 문제가 없었다. 그러나 리액트 18부터 도입된 [`concurrent` 렌더링](https://ko.reactjs.org/docs/concurrent-mode-intro.html)이 등장하며서, 렌더링이 렌더링을 잠시 일시중지할 수 있게 되면서 이 문제가 대두되기 시작했다. 일시중지가 발생하는 사이에 업데이트는 렌더링에 사용되는 데이터와 이와 관련된 변경사항을 가져올 수 있게 되었다. 이로 인해 UI는 동일한 데이터에 다른 값을 표시할 수 있게 되버렸다.

> [관련 이슈 살펴보기](https://github.com/reactwg/react-18/discussions/69)

동기 렌더링 시에는, UI는 항상 일관성을 유지할 수 있었다.

![synchronous rendering](https://d33wubrfki0l68.cloudfront.net/dbdfd8eb6f330f77d9b8f53356b5085af6696a48/cec12/images/use_sync_external_store/rendering_before_react_18.png)

그러나 concurrent 렌더링에서는, 초기에는 아래 그림처럼 파란색이다. 리액트는 외부 스토어가 바뀌면서 빨간색으로 업데이트 한다. 리액트는 계속해서 컴포넌트를 빨간색으로 바꾸려고 시도할 것이다. 이 과정에서 발생하는 UI의 불일치를 `tearing`이라고 한다.

![concurrent rendering](https://d33wubrfki0l68.cloudfront.net/3df29b67e19ed60ad572e16fa7e5e5cfed757a93/6140a/images/use_sync_external_store/concurrent_rendering_react_18.png)

이 문제를 해결하기 위해, 처음에는 [리액트 팀에서 `useMutableSource`라는 훅을 만들어](https://github.com/reactjs/rfcs/blob/main/text/0147-use-mutable-source.md) 안전하게 외부의 mutable한 소스를 읽어왔다. 그러나 개발을 시작하면서 [API에 결함이 있다는 것](https://github.com/reactwg/react-18/discussions/84)을 알게 되었고 `useMutableSource`는 사용이 어려워 졌다. 많은 논의 끝에, `useMutableSource`는 `useExternalStore`로 변경되었다.

[useExternalStore](https://github.com/reactwg/react-18/discussions/86)는 리액트 18에서 스토어 내 데이터를 올바르게 가져올 수 있도록 도와준다.

이해를 돕기위해, [이 레포](https://github.com/facebook/react/tree/main/packages/use-sync-external-store)를 방문해보자.

```javascript
import {useSyncExternalStore} from 'react';

  or

// Backwards compatible shim
import {useSyncExternalStore} from 'use-sync-external-store/shim';

//Basic usage. getSnapshot must return a cached/memoized result
useSyncExternalStore(
  subscribe: (callback) => Unsubscribe
  getSnapshot: () => State
) => State

// Selecting a specific field using an inline getSnapshot
const selectedField = useSyncExternalStore(store.subscribe, () => store.getSnapshot().selectedField);
```

`useSyncExternalStore`는 두개의 함수를 인자로 받는다.

- `subscribe`: 등록할 콜백 함수
- `getSnapshot`: 마지막 이후로 subscribe 중인 값이 렌더링된 이후 변경되었는지, 문자여이나 숫자 처럼 immutable한 값인지, 혹은 캐시나 메모된 객체인지 확인하는데 사용된다. 이후, 훅에 의해서 immutable한 값이 반환된다.

`getSnapShot`의 결과로 메모이제이션 된 값을 제공하는 api는 다음과 같다.

```javascript
import { useSyncExternalStoreWithSelector } from 'use-sync-external-store/with-selector'

const selection = useSyncExternalStoreWithSelector(
  store.subscribe,
  store.getSnapshot,
  getServerSnapshot,
  selector,
  isEqual,
)
```

[리액트 Conf에서 이야기한 실제 예제](https://www.youtube.com/watch?t=694&v=oPfSC5bQPR8&feature=youtu.be)에 대해 살펴보자.

```jsx
import React, { useState, useEffect, useCallback, startTransition } from 'react'

// library code

const createStore = (initialState) => {
  let state = initialState
  const getState = () => state
  const listeners = new Set()
  const setState = (fn) => {
    state = fn(state)
    listeners.forEach((l) => l())
  }
  const subscribe = (listener) => {
    listeners.add(listener)
    return () => listeners.delete(listener)
  }
  return { getState, setState, subscribe }
}

const useStore = (store, selector) => {
  const [state, setState] = useState(() => selector(store.getState()))
  useEffect(() => {
    const callback = () => setState(selector(store.getState()))
    const unsubscribe = store.subscribe(callback)
    callback()
    return unsubscribe
  }, [store, selector])
  return state
}

//Application code

const store = createStore({ count: 0, text: 'hello' })

const Counter = () => {
  const count = useStore(
    store,
    useCallback((state) => state.count, []),
  )
  const inc = () => {
    store.setState((prev) => ({ ...prev, count: prev.count + 1 }))
  }
  return (
    <div>
      {count} <button onClick={inc}>+1</button>
    </div>
  )
}

const TextBox = () => {
  const text = useStore(
    store,
    useCallback((state) => state.text, []),
  )
  const setText = (event) => {
    store.setState((prev) => ({ ...prev, text: event.target.value }))
  }
  return (
    <div>
      <input value={text} onChange={setText} className="full-width" />
    </div>
  )
}

const App = () => {
  return (
    <div className="container">
      <Counter />
      <Counter />
      <TextBox />
      <TextBox />
    </div>
  )
}
```

만약 위의 예제 처럼, `startTransition`를 사용하고 있다면, 이는 코드가 `tearing`될 수 있다는 것을 의미한다. 이러한 이슈를 해결하기 위해, `useSyncExternalStore`를 사용할 수 있다.

`useState` `useEffect`를 사용하고 있는 `useStore`를 `useSyncExternalStore`로 변경해보자.

```javascript
import { useSyncExternalStore } from 'react'

const useStore = (store, selector) => {
  return useSyncExternalStore(
    store.subscribe,
    useCallback(() => selector(store.getState(), [store, selector])),
  )
}
```

코드가 훨씬 깔끔해진 것을 볼 수 있다.

그렇다면 어떤 라이브러리들이 이러한 concurrent rendering에 영향을 받을까?

- 렌더링 중에 외부 가변 데이터에 접근하지 않고, react props, state, context 만을 사용하여 정보를 전달하는 컴포넌트와 훅만 가지고 있는 라이브러리라면 영향을 받지 않을 것이다.
- 데이터 fetch, 상태관리, redux, mobx, relay 등은 영향을 받을 것이다. 이는 리액트 외부에 상태를 저장하기 때문이다. concurrent 렌더링 시에는 react가 모르게 렌더링 중에 이러한 값이 업데이트 될 수 있기 때문이다.

#### `useInsertionEffect`

`useInsertionEffect`는 css-in-js 라이브러리가 렌더링 도중에 스타일을 삽입할 때 성능 문제를 해결할 수 있는 새로운 훅이다. css-in-js 라이브러리를 사용하지 않는다면 사용할 필요가 없다. 이 훅은 dom이 한번 mutate된 이후에 실행되지만, layout effect가 일어나기전에 새 레이아웃을 한번 읽는다. 이는 리액트 17 이하 버전에 있는 문제를 해결할 수 있으며, 리액트 18에서는 나아가 concurrent 렌더링 중에 브라우저에 리액트가 값을 반환하므로, 레이아웃을 한번더 계산할 수 있는 기회가 생겨 매우 중요하다.

어떻게 보면 `useLayoutEffect`와 비슷한데, 차이가 있다면 DOM 노드에 대한 참조에 엑세스 할 수 있다는 것이다.

클라이언트 사이드에서 `<style>` 태그를 생성해서 삽입할 때는 성능 이슈에 대해 민감하게 살펴보아야 한다. CSS 규칙을 추가하고 삭제한다면 이미 존재하는 모든 노드에 새로운 규칙을 적용하는 것이다. 이는 최적의 방법이 아니므로 많은 문제가 존재한다.

이를 피할 수 있는 방법은 타이밍이다. 리액트가 DOM을 변환한경우, 레이아웃에서 무언가를 읽기전 (`clientWidth`와 같이) 또는 페인트를 위해 브라우저에 값을 전달하기 전에 DOM에 대한 다른 변경과 동일한 타이밍에 작업을 하면 된다.

```jsx
function useCSS(rule) {
  useInsertionEffect(() => {
    if (!isInserted.has(rule)) {
      isInserted.add(rule)
      document.head.appendChild(getStyleForRule(rule))
    }
  })
  return rule
}
function Component() {
  let className = useCSS(rule)
  return <div className={className} />
}
```

이는 `useLayoutEffect`와 마찬가지로 서버에서 실행되지는 않는다.

### React DOM Client

`react-dom/client`에 새로운 API가 추가되었다.

#### `createRoot`

렌더링 또는 언마운트할 루트를 만드는 새로운 메소드다. `ReactDOM.render`대신 사용하며, 리액트 18의 새로운 기능은 이 것 없이 동작 하지 않는다.

**before**

```jsx
import ReactDOM from 'react-dom'
import App from 'App'

const container = document.getElementById('root')

ReactDOM.render(<App name="yceffort blog" />, container)

ReactDOM.render(<App name="yceffort post" />, container)
```

**after**

```jsx
import ReactDOM from 'react-dom'
import App from 'App'

const container = document.getElementById('root')

// 루트 생성
const root = ReactDOM.createRoot(container)

// 최초 렌더링
root.render(<App name="yceffort blog" />) // During an update, there is no need to pass the container again
// 업데이트 시에는, container를 다시 넘길 필요가 없다.
root.render(<App name="yceffort post" />)
```

#### `hydrateRoot`

서버사이드 렌더링 애플리케이션에서 hydrate하기 위한 새로운 메소드다. 새로운 React DOM Server API와 함께 `ReactDOM.hydrate` 대신 사용하면 된다. 리액트 18의 새로운 기능은 이와 함께 작동하지 않는다.

**before**

```jsx
import ReactDOM from 'react-dom'
import App from 'App'

const container = document.getElementById('root')

ReactDOM.hydrate(<App name="yceffort blog" />, container)
```

**after**

```jsx
import ReactDOM from 'react-dom'
import App from 'App'

const container = document.getElementById('root')

const root = ReactDOM.hydrateRoot(container, <App name="yceffort blog" />)
```

위 두 메소드 모드 `onRecoverableError`를 옵션으로 받을 수 있는데, 리액트가 렌더링이나 hydration시 에러가 발생하여 리커버리를 시도할 때 logging을 할 수 있는 목적으로 제공된다. 기본값으로 [reportError](https://developer.mozilla.org/en-US/docs/Web/API/reportError)나 구형 브라우저에서는 `console.error`를 쓴다.

### React DOM Server

`react-dom/server`에 새로운 API가 추가되었으며, 이는 서버에서 streaming Suspense를 완벽하게 지원한다.

#### `renderToPipeableStream`

node 환경에서 스트리밍 지원

- `<Suspense>`와 함께 사용 가능
- 콘텐츠가 잠시 사라지는 문제없이 `lazy`와 함께 코드 스플리팅 가능
- 지연된 콘텐츠 블록이 있는 HTML 스트리밍이 나중에 뜰 수 있음

#### `renderToReadableStream`

Cloudflare, deno와 같이 모던 엣지 런타임 환경에서 스트리밍 지원

`renderToString`는 여전히 존재하지만, 사용하는 것이 권장되지는 않는다.

> 이와 관련된 내용은 [내 이전 블로그글](/2022/01/how-react-server-components-work)에서 다룬적이 있으니 참고해보면 좋다.

## Deprecation

- `react-dom`: `ReactDOM.render`
- `react-dom`: `ReactDOM.hydrate`
- `react-dom`: `ReactDOM.unmountComponentAtNode`
- `react-dom`: `ReactDOM.renderSubtreeIntoContainer`
- `react-dom/server`: `ReactDOMServer.renderToNodeStream`

## Breaking Change

### React

#### `Automatic batching`

React Batch 업데이트 방식을 변경하여 자동으로 더 많은 배치를 수행할 수 있도록 성능이 향상되었다. 여기서 `batching`이란 여러 상태 업데이트를 하나의 리렌더링으로 처리하여 성능을 향상시키는 방법이다. 예를 들어, 버튼 하나 클릭이 두개의 상태를 업데이트 (`useState`가 두번 수행) 한다면, 리액트는 이를 하나의 리렌더링으로 처리할 수 있도록 해주는 것을 의미한다.

그러나 리액트는 언데 업데이트를 배치로 처리했는지가 일관성있게 이뤄지고 있지 않았다. 옐르 들어 데이터를 fetch 한 다음, `handleClick` 에서 상태를 업데이트 하는 경우, 리액트는 업데이트를 배치하지 않고 개별 업데이트 두개를 수행하곤 했었다. 그 이유는 브라우저 이벤트 중에는 배치로 일괄 처리 하지만, 이벤트가 이미 처리된 후(콜백)에서 상태를 업데이트 처리하고 있었기 때문이다.

리액트 18에서는, 어디에서 이벤트가 발생했는지와 상관없이 자동으로 모든 업데이트가 배치되어 이뤄진다.

```jsx
function App() {
  const [count, setCount] = useState(0)
  const [flag, setFlag] = useState(false)

  function handleClick() {
    fetchSomething().then(() => {
      // React 18 and later DOES batch these:
      setCount((c) => c + 1)
      setFlag((f) => !f)
      // React will only re-render once at the end (that's batching!)
    })
  }

  return (
    <div>
      <button onClick={handleClick}>Next</button>
      <h1 style={{ color: flag ? 'blue' : 'black' }}>{count}</h1>
    </div>
  )
}
```

만약 이러한 동작을 원치 않는다면 `flushSync`를 쓰면 된다.

```jsx
import { flushSync } from 'react-dom' // Note: react-dom, not react

function handleClick() {
  flushSync(() => {
    setCounter((c) => c + 1)
  })
  // React has updated the DOM by now
  flushSync(() => {
    setFlag((f) => !f)
  })
  // React has updated the DOM by now
}
```

그런데 이게 왜 breaking change 일까?

일단 훅의 경우에는, 대부분의 경우 제대로 배치처리가 자동으로 그냥 작동할 것으로 예상하고 있었다. 그러나 클래스의 경우, 이벤트 내부에서 상태 업데이트를 동기적으로 읽을 수 있는 방법이 있다. 아래 코드를 보자.

```javascript
handleClick = () => {
  setTimeout(() => {
    this.setState(({ count }) => ({ count: count + 1 }))

    // { count: 1, flag: false }

    // 사실은 배치 때문에 // { count: 0, flag: false } 임
    console.log(this.state)

    this.setState(({ flag }) => ({ flag: !flag }))
  })
}
```

리액트 18에서는, 이러한 케이스는 더이상 존재하지 않는다. `setTimeout`에 있는 것 조차 배치로 때려버리기 때문에, 위는 동기적으로 렌더링이 진행되지 않을 것이다.

함수형 컴포넌트의 경우에는, `useState`가 기존 변수를 업데이트하지 않기 때문에 문제가 되지 않을 것이다.

```javascript
function handleClick() {
  setTimeout(() => {
    console.log(count); // 0
    setCount(c => c + 1);
    setCount(c => c + 1);
    setCount(c => c + 1);
    console.log(count); // 0
  }, 1000)
```

#### Stricter Strict Mode

향후에는, 리액트에서는 컴포넌트가 마운트가 해제된 사이에서도 상태를 유지할 수 있는 기능을 제공할 예정이다. 이를 위하여 이번 18에서는 `Strict Mode`에 새로운 개발 모드 전용 체크를 도입했다. 컴포넌트가 재마운트 될 때 마다모든 컴포넌트를 자동으로 마운트 해제하고, 다시 마운트하여 이전 상태를 복원한다. 이로 인해 앱이 깨지면, 기존 상태로 다시 마운트 할 수 있는 컴포넌트를 수정할 때까지 `Strict Mode`를 삭제하는 것이 좋다.

#### 일관된 `useEffect` 타이밍

위에서 언급한 `Automatic Batching`에서 이어지는 맥락이다. 클릭, keydown event와 같은 개별 사용자 입력 에벤트 중에 업데이트가 발생한 경우, 항상 동기식으로 effect 함수를 플러쉬한다. 이전에는 이 기능이 예측가능하거나, 일관적이지 못했다.

#### 엄격해진 hydration 에러

텍스트 애용 누락, 텍스트 내용 불일치 등은 이제 경고 대신 오류로 처리된다. 리액트는 서버 마으컵을 일치시키기 위해 클라이언트 노드에 삽입이나 삭제를 함으로서 개별 노드를 수정해주지 않고, 이제는 트리에서 가장 가까운 `<Suspense>` 바운더리 까지 클라이언트 렌더링으로 돌아간다. 이를 통해 hydration 트리의 일관성을 확보하고, 불일치로 인해 발생할 수 있는 잠재적인 보안 문제를 해결할 수 있다.

#### Suspense 가 이제 항상 일관되게 적용됨

트리에 완전히 추가되기전에, 컴포넌트가 suspend 된 경우, 리액트는 불완전한 상태로 컴포넌트를 추가하거나 effect를 발생시키지 않는다. 대신 리액트는 새 트리를 완전히 버리고 비동기 작업이 완료될 때 가지 기다린 다음, 다시 처음부터 렌더링을 시도한다. 리액트는 브라우저를 차단하지 않고 동시에 렌더링을 재시도 한다.

#### Suspense와 layout effect

트리가 suspend 되었다가 fallback으로 돌아가면, 리앹그는 레이아웃 effect를 정리한 다음, 바운더리 내부의 내용이 다시 표시 될 때 까지 만든다. 이로인해 컴포넌트 라이브러리가 suspense와 함께 사용될때 레이아웃을 올바르게 측정할 수 없었던 문제가 해결된다.

#### 새로운 js 환경 (polyfill 필요)

리액트는 이제 모던 브라우저 기능인 `Promise` `Symbol` `Object.assign`에 의존한다. 최신 브라우저 기능을 제공하지 않거나, 혹은 호환되지 않는 인터넷 익스플로러 등 오래된 브라우저를 지원해야 하는 경우, 애플리케이션에 글로벌 플로필을 추가하는 것을 고려해봐야 한다.

## 눈에 띄는 변화

### React

#### `undefined`도 렌더링 가능

이제 컴포넌트가 `undefined`를 리턴해도 에러를 리턴하지 않는다. jsx에 return 문을 잊지 않도록 linter의 도움을 받는 것을 추천한다.

#### 테스트 시에, `act` 경고가 옵트인 됨

e2e 테스트 시 `act` 경고는 불필요하다. [`opt-in` 개념을 도입](https://github.com/reactwg/react-18/discussions/102)하여 유닛테스트 시에만 이러한 경고문을 받을 수 있도록 구성할 수 있다.

#### No Suppression of `console.log`

strict 모드에서, 각 컴포넌트를 두번씩 렌더링 하면 예끼치않은 사이드 이펙트를 겪을 수 있다. react 17에서는 이러한 로그를 쉽게 읽게 하기 위해 두 렌더링 중에 하나의 `console.log`를 의도적으로 띄우지 않았다. 그러나 [이러한 동작이 혼란스럽다는 의견](https://github.com/facebook/react/issues/21783)이 있어 더이상 경고문을 제거하지 않는다. 대신, `React DevTools`가 설치되어 있다면, 두번째 로그가 회색으로 표시되고, 이를 완전히 없앨 수 있는 옵션이 존재한다.

#### 메모리 사용량 최적화

리액트는 마운트 해제시에 더 많은 내부 필드를 정리하여, 애플리케이션에 존재할 수 있는 메모리 누수로 인한 영향을 줄여주었다.

### React DOM Server

#### `renderToString`

서버에서 suspending이 일어날 경우 더이상 에러가 발생하지 않는다. 대신 가장 가까운 `<Suspense>` 바운더리에 fallback HTML을 내보낸후, 클라이언트 레벨에서 같은 렌더링을 재시도 한다. `renderToString`보다는 `renderToPipableStream` `renderToReadableStream`과 같은 스트리밍 api로 전환하는 것을 추천한다.

#### `renderToStaticMarkup`

서버에서 suspending이 일어날 경우 더이상 에러가 발생하지 않는다. 대신 가장 가까운 `<Suspense>` 바운더리에 fallback HTML을 내보낸후, 클라이언트 레벨에서 같은 렌더링을 재시도 한다.

## All Changes

위 내용을 포함한 모든 변경사항은 [https://github.com/facebook/react/blob/main/CHANGELOG.md#all-changes](https://github.com/facebook/react/blob/main/CHANGELOG.md#all-changes)에 나와 있다.
