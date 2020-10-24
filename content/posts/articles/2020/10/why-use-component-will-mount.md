---
title: 'useComponentWillMount??'
tags:
  - javascript
published: true
date: 2020-10-23 21:26:18
description: '라이프 사이클의 굴레에서 벗어나'
---

```javascript
useEffect(() => {
  //.. do something
}, []) // empty deps
```

이렇게 의존성이 비어있는 `useEffect`가 `componentDidMount`와 비슷한 타이밍에 동작하는 것이 아니라는 사실은 이런저런 블로그 글에 많이 나와있다. (사실 가장 비슷한건 `useLayoutEffect`다.)

> [] 는 이펙트에 리액트 데이터 흐름에 관여하는 어떠한 값도 사용하지 않겠다는 뜻입니다. 그래서 한 번 적용되어도 안전하다는 뜻이기도 합니다.

https://iqkui.com/ko/a-complete-guide-to-useeffect/

https://yceffort.kr/2020/10/think-about-useEffect

그러나 여전히 많은 사람들이 (나를 포함해서) 라이프 사이클 메소드의 향기에서 벗어나지 못하고 있는 것 같다.

## componentWillMount

https://ko.reactjs.org/docs/react-component.html#unsafe_componentwillmount

`componentWillMount`는 말그대로 컴포넌트가 마운트 되기 직전에 실행되는 라이프 사이클 메소드다. 그러나 이름에서 보이는 것 처럼, deprecated 가 되었고, 얼마전에 나오는 v17에서는 완전히 사라졌다.

https://reactjs.org/blog/2018/03/29/react-v-16-3.html#component-lifecycle-changes

> For example, with the current API, it is too easy to block the initial render with non-essential logic. In part this is because there are too many ways to accomplish a given task, and it can be unclear which is best. We’ve observed that the interrupting behavior of error handling is often not taken into consideration and can result in memory leaks (something that will also impact the upcoming async rendering mode). The current class component API also complicates other efforts, like our work on prototyping a React compiler.

이유인 즉, 렌더링에 필요하지 않은 로직을 렌더링 직전에 (==`componentWillMount`) 넣어서 렌더링을 방해하는 경우가 많아졌다는 것이다. 그리고 이러한 동작은 메모리 유출을 낳는 경우가 많기 때문에 지원을 중단했다고 밝혔다.

그러나 아직까지도 많은 리액트 라이브러리들이 `componentWillMount`에 의존하고 있다.

## useComponentWillMount

근데 그럼에도 불구하고 정말 정말 mount가 되기 직전에 무언가를 해야한다면, 근데 쓰고 있는 컴포넌트가 함수형이라고 한다면 어떻게 해야할까?

```javascript
export const useComponentWillMount = (func) => {
  const willMount = useRef(true)
  if (willMount.current) func()
  willMount.current = false
}
```

`useRef`는 매번 렌더링할 떄 동일한 ref객체를 제공한다. 따라서 func가 딱한번 실행되도록 보장할 수 있다. 그렇다면 이것이 mount되기 직전에 실행된다고 볼 수 있는 이유는 무엇일까?

`useEffect`는 말그대로 부수효과를 발생시키는 hook이기 때문에 렌더링이 된 이후에 실행된다.

그러나 `useRef` 코드에서 함수가 넘어오게 되면, (함수가 호출되는 시점에) 딱 한번 바로 실행할 수 있게 된다. 그리고 이 값은 또한 컴포넌트의 전체 라이프 사이클 내에서 계속해서 유지되는 것을 리액트에서 보장해준다.

> 이 기능은 클래스에서 인스턴스 필드를 사용하는 방법과 유사한 어떤 가변값을 유지하는 데에 편리합니다.

> useRef는 매번 렌더링을 할 때 동일한 ref 객체를 제공한다는 것입니다.

또 재밌는 방법은 이것이었다.

```javascript
export const useComponentWillMount = (func) => {
  useMemo(func, [])
}
```

의존성이 없는 `useMemo`를 쓰게 되면, 함수가 다시 호출 될 일이 없으므로 렌더링 직전에 실행되는 것을 보장할 수 있다.

참고: https://stackoverflow.com/questions/53464595/how-to-use-componentwillmount-in-react-hooks
