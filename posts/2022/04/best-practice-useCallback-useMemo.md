---
title: '리액트의 useCallback useMemo, 정확하게 사용하고 있을까'
tags:
  - javascript
  - react
published: true
date: 2022-04-16 18:51:14
description: '메모이제이션에 대한 고민 🤔'
---

## Table of Contents

## Introduction

리액트 코드를 리뷰하다보면, `useCallback`과 `useMemo`를 정말 많은 곳에 사용하는 것을 발견하게 된다. 일반적으로 두 훅을 쓰게 되는 이유는 컴포넌트에 무언가 함수를 전달할 때 마다 `useCallback`을 사용하는 것 같지만, 이는 문제에 대한 올바른 해결방법이 아니며 오히려 렌더링 시간에 문제를 일으킬 수 있다.

[많은 글](https://goongoguma.github.io/2021/04/26/When-to-useMemo-and-useCallback/)에서 언급된 것처럼 `useMemo`와 `useCallback`을 이용한 최적화의 비용은 공짜가 아니다.

## 문제는 무엇일까

리액트 컴포넌트 트리는 매우 클 수 있다. `React DevTools`를 열고 애플리케이션을 살펴보면, 한번에 많은 컴포넌트가 렌더링 되는 것을 볼 수 있다. 이 과정에서 한두개 정도 불필요한 `useCallback` `useMemo`를 사용하는 것은 별 문제가 되지 않지만, 이 코드가 여기저기 존재한다면 문제가 될 수 있다.

일반적인 오해 중 하나로는, `useCallback`을 사용하면 렌더링중 함수 재생성을 방지할 수 있다는 것인데, 꼭 그런 것 만은 아니다.

`useCallback`은 제공된 deps를 기준으로 반환된 함수 객체를 메모이제이션 하는 것 뿐이다. 즉, 동일한 deps가 제공되면 (참조로 비교) 동일한 함수 객체를 반환한다.

만약 그냥 새로운 함수를 매번 만드는 대신 `useCallback`으로 선언된 함수를 컴포넌트나 훅으로 넘겨주는 경우, `useCallback`을 사용함으로써 새로운 함수를 만들고, 새로운 배열을 만들어서 함수를 실행하고, deps의 동일성을 비교하기 위한 함수와 종속성 집합을 메모리에 저장하게 될 것이다.

이는 단순히 함수를 props로 만들어서 전달하는 것보다 훨씬더 많은 비용이 든다. `useCallback`을 사용하든, 사용하지 않았든 기능적으로는 동일하게 동작했을 것이다.

## props의 참조 동일성은 언제 문제가 될까?

만약 자식 컴포넌트가, `React.memo`를 사용하고 있거나 `React.PureComponent`로 구현되어 있는 경우, 리액트는 props가 정확하게 일치하는 한 부모 컴포넌트가 리렌더링 되더라도 이 자식 컴포넌트를 리렌더링하지 않을 것이다. 만약 다른 모든 props가 참조적으로 동일하지만, 만약 의도치 않게 새 함수 인트선트 또는 객체 인스턴스를 전달하게 되면, 해당 컴포넌트가 다시 리렌더링 된다.

이러한 컴포넌트에는 다시 렌더링하는데 비용이 많이 드는 하위 컴포넌트가 존재할 수 있으므로, `memo`에 대한 이러한 약속을 지키지 않으면 성능이 저하될 수 있다.

이러한 참조 동일성은 `props`가 `useEffect`의 종속성으로 사용되는 경우에도 문제가 될 수 있다. `props`가 변경될 때 마다 이 `useEffect`가 트리거 될 것이다.

## 지켜야할 규칙

### `useMemo`와 `useCallback`을 사용하지 말아야할 경우

1. `host` 컴포넌트에 (`div` `span` `a` `img`...) 전달하는 모든 항목에 대해 쓰지 말아야한다. 리액트는 여기에 함수 참조가 변경되었는지 신경쓰지 않는다. (`ref`, `mergeRefs`는 여기에서 제외된다.)
2. `leaf` 컴포넌트에는 쓰지말아야 한다.
3. `useCallback` `useMemo`의 의존성 배열에 완전히 새로운 객체와 배열을 전달해서는 안된다. 이는 항상 의존성이 같지 않다는 결과를 의미하며, 메모이제이션을 하는데 소용이 없다. `useEffect` `useCallback` `useMemo`의 모든 종속성은 참조 동일성을 확인한다.

```javascript
// dont
const x = [‘hello’];
const cb = useCallback(()={},[prop1,prop2, x])

// dont
const [a, ...rest] = someArray;
const cb = useCallback(()={},[rest]
```

4. 전달하려는 항목이 새로운 참조여도 상관없다면, 사용하지 말아야 한다. 매번 새로운 참조여도 상관없는데, 새로운 참조라면 메모이제이션하는 것이 의미가 없다.

> host 컴포넌트: 호스트 환경 (브라우저 또는 모바일)에 속하는 플랫폼 컴포넌트를 의미한다. DOM 호스트의 경우, `div`, `img`와 같은 요소가 될 수 있다.
> leaf 컴포넌트: DOM에서 다른 컴포넌트를 렌더링하지 않는 컴포넌트 (html 태그만 렌더링하는 컴포넌트)

### `useMemo`와 `useCallback`을 사용해야 하는 경우

1. 하위트리에 많은 Consumer가 있는 값을 Context Provider에 전달해야 하는 경우 `useMemo`를 사용하는 것이 좋다. `<ProductContext.Provider value={{id, name}} >`의 경우, 어떤 이유로든 해당 컴포넌트가 리렌더링 된다면 `id` `name`이 동일하더라도 매번 새로운 참조를 만들어 죄다 리렌더링 될 것이다.
2. 계산 비용이 많이 들고, 사용자의 입력 값이 `map`과 `filter`을 사용했을 때와 같이 이후 렌더링 이후로도 참조적으로 동일할 가능성이 높은 경우, `useMemo`를 사용하는 것이 좋다.
3. `ref` 함수를 부수작용과 함께 전달하거나, `mergeRef-style` 과 같이 wrapper 함수 ref를 만들 때 `useMemo`를 쓰자. ref 함수가 변경이 있을 때마다, 리액트는 과거 값을 `null`로 호출하고 새로운 함수를 호출한다. 이 경우 ref 함수의 이벤트 리스너를 붙이거나 제거하는 등의 불필요한 작업이 일어날 수 있다. 예를 들어, `useIntersectionObserver`가 반환하는 `ref`의 경우 `ref` 콜백 내부에서 observer의 연결이 끊기거나 연결되는 등의 동작이 일어날 수 있다.
4. 자식 컴포넌트에서 `useEffect`가 반복적으로 트리거되는 것을 막고 싶을 때 사용하자.
5. 매우 큰 리액트 트리 구조 내에서, 부모가 리렌더링 되었을 때 이에 다른 렌더링 전파를 막고 싶을 때 사용하자. 자식 컴포넌트가 `React.memo` `React.PureComponent`일 경우, 메모이제이션된 props를 사용하게되면 딱 필요한 부분만 리렌더링 될 것이다.

`React DevTools Profiler`를 사용하면 컴포넌트의 리렌더링 속도가 느린 경우, 상태 변경이 일어났을 때 얼마나 렌더링 시간이 걸렸는지 조사할 수 있다. 이렇게 하면 거대한 계단식 리렌더링을 방지하기 위해 `React.memo`를 사용할 위치를 찾을 수 있고, 필요한 경우 `useCallback` `useMemo`를 사용하여 상태변경을 더 효율적으로 만들 수 있다.
