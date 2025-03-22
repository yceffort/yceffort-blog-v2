---
title: React Hooks Api (1)
date: 2019-08-09 02:01:07
published: true
tags:
  - react
  - javascript
description:
  '# Hooks API Hook은 react 16.8에서 추가된 개념으로, Hook을 시용하면 class를 갖성하지
  않아도 state관리와 같은 react의 기능을 사용할 수 있다.  ## 기본 Hook  ### useState  ```javascript
  const [state, setState] = useState(initialState); setStat...'
category: react
slug: /2019/08/08/react-hooks-api-1/
template: post
---

# Hooks API

Hook은 react 16.8에서 추가된 개념으로, Hook을 시용하면 class를 갖성하지 않아도 state관리와 같은 react의 기능을 사용할 수 있다.

## 기본 Hook

### useState

```javascript
const [state, setState] = useState(initialState)
setState(newState)
```

상태 유지값, 그리고 그 값을 수정하는 함수를 반환한다. 이전의 `state`값을 받아다가 수정할 수도 있다.

```javascript
function Counter({initialCount}) {
  const [count, setCount] = useState(initialCount)
  return (
    <>
      Count: {count}
      <button onClick={() => setCount(initialCount)}>Reset</button>
      <button onClick={() => setCount((prevCount) => prevCount + 1)}>+</button>
      <button onClick={() => setCount((prevCount) => prevCount - 1)}>-</button>
    </>
  )
}
```

동일한 값으로 갱신하는 경우(`Object.is`) 값이 업데이트 하지 않고 처리를 종료한다.

### useEffect

```javascript
useEffect(didUpdate)
```

화면에 렌더링이 완료된 이후에 수행한다. 또한, 컴포넌트가 화면에서 제거 될 때 정리 해야할 리소스도 선언할 수 있다.

```javascript
useEffect(() => {
  const subscription = props.source.subscribe()
  return () => {
    subscription.unsubscribe()
  }
})
```

`unsubscribe`는 이제 ui에서 컴포넌트를 제거하기 직전에 수행한다. 그리고 만약, 컴포넌트가 여러번 렌더링 된다면 다음 effect가 수행되기 전에 이전 effect가 정리된다.

만약 조건부로 실행하기 위해서는 아래와 같은 방법을 활용할 수도 있다.

```javascript
useEffect(() => {
  const subscription = props.source.subscribe()
  return () => {
    subscription.unsubscribe()
  }
}, [props.source])
```

그럼 이제 `props.source`값이 변경 될때 만 `useEffect`가 발생하게 된다.

### useContext

#### Context

`context`를 이용하면, 매번 일일이 props를 넘겨주지 않아도, 컴포넌트 트리전체에 데이터를 제공할 수 있다. 즉, context는 react 컴포넌트 트리안에서 전역적으로 대이터를 공유할 수 있도록 고안된 방법이다.

```html
<Page user={user} avatarSize={avatarSize} />
  <PageLayout user={user} avatarSize={avatarSize} />
    <NavigationBar user={user} avatarSize={avatarSize} />
      <Link href={user.permalink}>
        <Avatar user={user} size={avatarSize} />
      </Link>
    </NavigatorBar>
  </PageLayout>
</Page>
```

실제로 `user`와 `avatarSize`를 사용하는 곳은 `Link`컴포넌트 인데, page 온갖 컴포넌트를 거치면서 값을 내려주는 것을 볼 수 있다. 이게 더 심해지는 경우, 같은 데이터를 트리안의 여러 레벨의 컴포넌트에게 주어야 할 때도 있다. 이렇게 **데이터가 변할 때 마다 모든 하위 컴포넌트에게 해당 값을 알려주는 것이 `context`이다.**

```javascript
const MyContext = React.createContext(defaultValue)
```

Context객체를 만든다. Context 객체를 구독하고, 컴포넌트를 렌더링 할 때 트리 상위에서 가장 가까이 짝이 맞는 `Provider`로 부터 현재 값을 읽는다. 여기서 선언된 `defaultValue`는 트리안에서 적절한 Provider를 찾지 못했을 때에만 쓰는 값이다.

```html
<MyContext.provider value="{someValue}">
  <SomeComponent />
</MyContext.provider>
```

`Provider`는 context를 구독하는 컴포넌트들에게 context의 변화를 알리는 역할을 한다. `Provider`는 value에 있는 `prop`을 받아서 이 값을 하위 컴포넌트에 전달한다.

```html
<MyContext.Consumer>
  {value => /* context 값을 이용한 렌더링 */}
</MyContext.Consumer>
```

context 변화를 구독하는 React Component다. 반드시 `Context.Consumer`의 자식은 함수여야만 한다. 이 함수는 context의 현재 값을 받고, React 노드를 반환해야 한다.

```javascript
const value = useContext(MyContext)
```

를 사용하면, context객체를 받아서, 현재 context의 값을 반환한다.
