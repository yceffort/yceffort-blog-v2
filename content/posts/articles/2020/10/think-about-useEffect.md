---
title: 'useEffect는 라이프 사이클 메소드가 아니다.'
category: react
tags:
  - react, javascript
published: true
date: 2020-10-02 22:40:20
description: '생각없이 useEffect를 쓰지 말자'
template: post
---

## useEffect는 라이프 사이클 메소드가 아니다

과거 리액트 클래스 컴포넌트에는 `constructor` `componentDidMount` `componentDidUpdate` `componentWillUnmount` 와 같이 리액트 라이프 사이클에 대응할 수 있는 각각의 메소드가 존재했다. 함수형 컴포넌트의 훅으로 넘어오면서, 이러한 라이프 사이클 메소드를 훅으로 각각 대체하려고 하지만 이는 큰 실수다.

결론부터 말하자면, `useEffect`는 라이프 사이클 훅이 아니다. `useEffect`는 app 의 state값을 활용하여 동기적으로 부수효과를 만들 수 있는 메커니즘이다.

> The question is not "when does this effect run" the question is "with which state does this effect synchronize with"

https://twitter.com/ryanflorence/status/1125041041063665666

```javascript
useEffect(fn) // all state
useEffect(fn, []) // no state
useEffect(fn, [these, states])
```

## `eslint-plugin-react-hooks/exhaustive-deps`로 deps를 무시하지마라

물론 기술적으로는 가능하다. 그리고 때로는 사용하는게 좋은 이유가 될 수 있다. 그러나 대부분의 경우에 이를 사용하는 것은 나쁜 생각이며, 잠재적인 버그를 만들 수 있다. 이런 이야기를 했을 때, 많은 사람들이 '컴포넌트를 mount 했을 때만 실행 하고 싶을 때가 있다' 라고 말할 수 있다. 그러나 이는 라이프사이클의 접근법이며, 옳지 못하다. `useEffect`에 deps가 있을 경우, effect 백은 deps에 변화가 있을 때 항상 실행된다. 그 외에는, app의 state 변화로 부터 부수효과와 분리되어 sync가 맞지 않게 된다. (app의 state값과 부수효과가 별개로 돌아가게 된다.)

요약하자면, 이는 버그로 이어질 수 있으므로 해당 룰을 off해서는 안된다.

> app의 state와 상관없이 mount시에 실행되어야만 하는 코드가 얼마나 많겠느냐? 하는 의미로 받아드리면 될 것 같다.

## 하나의 큰 `useEffect`를 만들지 마라.

각각의 `useEffect`는 관심사를 따로 분리해 두어야 한다. 하나의 큰 `useEffect`보다는, 각각의 로직을 분리해두는 것이 훨씬 좋다.

## 불필요한 외부 함수를 만들지 마라.

아래와 같은 코드는, `useEffect`에 두가지 deps를 추가해야 된다.

```javascript
// before. Don't do this!
function DogInfo({ dogId }) {
  const [dog, setDog] = React.useState(null)
  const controllerRef = React.useRef(null)
  const fetchDog = React.useCallback((dogId) => {
    controllerRef.current?.abort()
    controllerRef.current = new AbortController()
    return getDog(dogId, { signal: controller.signal }).then(
      (d) => setDog(d),
      (error) => {
        // handle the error
      },
    )
  }, [])
  React.useEffect(() => {
    fetchDog(dogId)
    return () => controller.current?.abort()
  }, [dogId, fetchDog])
  return <div>{/* render dog's info */}</div>
}
```

위의 코드를 다음과 같이 바꿨다.

```javascript
function DogInfo({ dogId }) {
  const [dog, setDog] = React.useState(null)
  React.useEffect(() => {
    const controller = new AbortController()
    getDog(dogId, { signal: controller.signal }).then(
      (d) => setDog(d),
      (error) => {
        // handle the error
      },
    )
    return () => controller.abort()
  }, [dogId])
  return <div>{/* render dog's info */}</div>
}
```

`useEffect` 밖에서 정의되어 있던 `fetchDog` 함수를 `useEffect` 내부로 가지고 왔다. 이전에는 이것이 외부에 정의되어 있었기 때문에, deps 배열에 추가해야 했다. 또한 이 때문에 무한 루프에 빠지는 것을 방지하기 위하여 memoize를 해야 했다. 또한, controller를 위해 `ref`도 사용했다.

반드시 effect내에서 사용할 함수는 외부가 아닌 내부에서 정의 해야 한다.
