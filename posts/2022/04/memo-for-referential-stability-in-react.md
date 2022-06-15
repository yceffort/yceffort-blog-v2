---
title: '참조 동일성을 위한 메모이제이션'
tags:
  - javascript
  - react
published: true
date: 2022-04-16 19:42:29
description: '세상에 나쁜 메모이제이션은 없다 🤔'
---

## Table of Contents

## Introduction

리액트에서 `useMemo`나 `useCallback`을 사용해서 언제 메모이제이션을 해야하는지에 대한 논의는 꾸준히 존재해 왔다. 대부분, 메모이제이션을 해야하는 이유로 주장하는 것은 크게 두가지다.

- 복잡한 연산이나 계산을 최적화 하기 위해
- 리렌더간의 객체 참조성을 안전하게 가져가기 위해

사람들이 대부분 `useMemo` `useCallback`을 사용하지 말라고 할 때 언급하는 것은 보통 첫번째를 대부분 언급한다. 그리고 뒤 따라 오는 대답으로는 너무 '일찍 최적화' 하지 말라 라는 답이 온다.

하지만 여기에서 더 흥미로운 것은, 리렌더링 사이에서 객체를 안정화 시키기 위한 두번째 사용사례다. 이는 리액트의 함수형 프로그래밍 모델과 자바스크립트 언어 특징 간의 불일치를 드러낸다.

하지만 먼저, 왜 우리는 객체의 안정적인 참조를 원하는 것일까?

## 불안정한 객체가 새나가는 위험성

대부분의 사람들은 리액트의 함수형 컴포넌트가 어떤식으로 동작하는지 알 것이다. 매번 리액트 함수형 컴포넌트게가 렌더링되면, 해당 함수에 정의 되어 있는 로컬 함수와 커스텀 훅 모두 폐기 되고 처음부터 대시 생성된다. 이렇게 됨으로써 발생하는 성능 문제를 모던 브라우저에서 측정하기란 어렵지만, 리렌더링되면서 생기는 모든 객체는 계속해서 재생성되고 이 참조가 모두 다르게 된다.

이러한 객체가 그냥 일반적인 로컬 변수이고, 메모이즈 되지 않은채로 하위 컴포넌트로만 전달되지 않는다면, 리렌더링간에 발생하는 재생성은 크게 문제가 되지 않을 것이다.

그러나 이것이 커스텀 훅과 같은 재사용가능한 추상화를 생성할때, 불안정한 값을 반환하는 것은 잠재적으로 위험한 일이 될 수 있다. 궁극적으로 , hook, api, 라이브러리의 제작자가, 이를 사용하는 사용자가 불안정한 값을 다음과 같은 종속성 array에 넣을지 여부는 알수가 없다.

- `useEffect`안에 넣어서 이러한 변화를 추적하고자 하는 경우
- `useMemo` `useCallback`의 메모이 제이션 값으로 사용하는 경우
- `React.memo` `React.PureComponent`로 감싸진 자식 컴포넌트에 prop으로 넘기는 경우

이상적인 상황에서는, 훅, api, 라이브러리가 생성하는 이러한 값의 참조는 오직 의미있는 변화가 있을 때만 변경되어야 하며, 반대로 매 리렌더링으로 일어나는 생성시마다 새롭게 만들어져서는 안된다. 사용자가 api 에서 내려오는 값에 변화가 있는지 확인하기 위해 번거롭게 하지 않으려면, 사용자의 손에 넘어가기전에 이러한 값을 안정화시켜줄 필요가 있다.

그 다음으로 알아볼 것은, 리액트에서 어떻게 객체를 안정적으로 만들 것이냐 하는 일이다.

## 값을 안정화 시키는 방법

값을 안정화 시키는 방법에는 리액트에서 두가지가 있다.

- `useMemo` `useCallback`으로 객체를 모두 메모이제이션 하는 방법
- `useRef`를 사용하여 컴포넌트, 훅, api 외부에서 이를 사용할 수 있도록 끌어올리는 방법

위 두가지에 대해 모두 살펴보자.

### 모든 것을 메모이제이션

가장 단순하고 확실한 방법으로, 모든 것을 `useMemo` `useCallback`으로 감싸서 메모이제이션 한다음, 의존성이 변경되지 않는 한 리렌더링 간에 이러한 값의 변화를 막고 재사용할 수 있게 하는 것이다.

많은 사람들이 미리최적화하는것, (aka premature optimization) 이 좋지 않은 관행임을 잘 알고 있음에도 결국엔 모든 것을 메모이제이션 하는 것을 보았다. 물론, 모든 것을 메모이제이션 하는 것은 성능을 해칠 수 있지만, 불안정한 참조가 얘기치 않게 다른 메모이제이션을 파괴하는 것이 더욱 나쁘다.

form을 만들 때 사용하는 [react-hook-form](https://react-hook-form.com/)의 [원칙 중 하나](https://github.com/alibaba/hooks/blob/master/docs/guide/blog/function.en-US.md#principle)로, 훅으로 부터 반환되는 모든 함수를 메모이제이션하는 것을 볼 수 있다.

> https://github.com/react-hook-form/react-hook-form/blob/f7d9805844c5df7a0949d9907936530b3112287f/src/useFieldArray.ts#L332-L350

> `useMemo` `useCallback`은 캐시 삭제의 대상이 될 수도 있다. 리액트는 메모리의 상황이 여의치 않으면, 이러한 캐시된 값들을 날리고 다시 초기화할 수도 있다. 리액트 문서에는 다음과 같은 내용이 존재한다. 이와 관련된 내용이 리액트 공식 문서에 존재한다. https://reactjs.org/docs/hooks-reference.html#usememo

> You may rely on useMemo as a performance optimization, not as a semantic guarantee. In the future, React may choose to “forget” some previously memoized values and recalculate them on next render, e.g. to free memory for offscreen components. Write your code so that it still works without useMemo — and then add it to optimize performance.

> 이와 같은 내용이 걱정된다면, 아래에서 언급할 ref를 사용할 수도있다.

## ref에 모든 것을 저장하기

한가지 많은 사람들이 사용하고 있지 않은 메모이제이션의 대안으로, `useRef`를 사용하여 리렌더링 사이에 동일한 값을 사용하는 방법이 존재한다. 이러한 기법을 사용하고 있는 라이브러리가 [react-table](https://github.com/TanStack/react-table/blob/3ed64a99419d3c122f2f0f5e7138c491a094b349/packages/react-table/src/createTable.tsx#L225-L237) 이다.

이러한 기법이 가능한 이유는, `ref` 내부의 값이 컴포넌트의 `state`와는 다르게 컴포넌트 외부에 저장되어있기 때문이다. [dan abramov가 언급했던 것처럼](https://twitter.com/dan_abramov/status/1099842565631819776) `useRef`도 일종의 `useState`다.

```javascript
// useRef()
useState({ current: initialValue })[0]
```

`state`와는 다르게, `reft의 값은 리렌더링 사이에 파괴되지 않으며, 새로 생성되지도 않는다.

## 불일치를 다룰 수 있는 좋은 방법

위 두개의 해결책은, 모두 앞서 언급했던 리액트의 함수형 프로그래밍 모델과, 자바스크립트 언어가 가지는 비 함수형 언어 사이의 불일치 때문에 발생하는 문제다. 쉽게 설명하자면, 리액트는 매 함수형 컴포넌트 리렌더링 사이에 함수 내부의 모든 로컬 객체를 재생성하는 특징을 가지고 있고, 자바스크립트는 ID나 참조가 아닌 값으로 비교하는 함수형 불변 데이터 구조를 제공하지 않고 있기 때문이다.

앞으로 이를 해결할 수 있는 방법으로 기대해볼 수 있는건 다음과 같다.

- [Javascript Record, Tuple](https://github.com/tc39/proposal-record-tuple)이 실제로 만들어진다면, 참조가 아닌 값과 내용으로 비교할 수 있는 immutable한 데이터 구조가 만들어질 것이다. 물론, 여전히 함수의 동일성에 대해서는 여전히 물음표다.
- [React Forget](https://www.youtube.com/watch?v=lGEMwh32soc) 으로 리액트 컴파일러가 자동으로 메모이제이션 해줄 수도 있다. (이 react forget에 대해서는 여전히 논란이지만)

## 모든 것을 메모이제이션 해야 하는 이유

### 왜 모든 컴포넌트를 memo 해야 하는가?

앞서 언급했던 것 처럼, memo가 필요한 상황은 분명히 존재한다. 그러므로 우리가 할 수 있는 선택지는 두가지다.

- 가끔 `memo`를 쓰기
- 모두 `memo`를 쓰기

첫번째가 물론 가장 이상적이다. `memo`를 사용해야할 때만 찾아서 쓰고, 그렇게 하는 것이다. 그리고 이를 대규모 팀에 적용하기 위해서, 계속해서 우리는 상기 시켜야 한다. 그러나 솔직히 아무리 열심히 작업한다라더라도 이르 제대로 100% 지키기는 어렵다.

그렇다면, 모든 것을 `memo`로 한다음, 잘못된 `memo`를 했을 때 비용은 얼마나 드는가 생각해봐야 한다. 만약 `memo` 컴포넌트를 잘 못 썼다면, 매 리렌더링 시에 여기에서 props에 대한 얕은 비교를 수행할 것이다. 그리고 이에 따른 절차는 아래와 같을 것이다.

1. 렌더링 함수 칠행
2. 모든 콜백을 새로 할당
3. 모든 `useMemo`를 새로 할당
4. 새로운 JSX elements 할당
5. 1 ~ 4를 모든 자식에 반복
6. 리액트 reconciler가 오래된 트리와 새로운 트리 비교

리액트 앱을 프로파일링 해본적이 있다면, 렌더링하는 모든 컴포넌트에 무시할 수 없는 성능적인 영향이 있다는 것을 알 수 있다. 반면 메모의 props 비교는 프로파일링에 거의 나타나지 않는다.

컴포넌트를 다시 불필요하게 렌더링하는 것은 props가 변경되었는지 여부를 불필요하게 테스트하는 것보다 비용이 더 많이 든다. 따라서 불필요한 비교보다는 불필요한 리렌더링을 막는 쪽에 더 심혈을 기울이는게 낫다. 모든 개발자가 실수할 수 있으므로, 이러한 실수를 막기 위한 더 최선의 방법으로 모든 것을 `memo`하는 것이다.

#### memo의 cpu 비용

만약 `memo`가 더 cpu에 무리가 가는 일이라면 어떨까? 경험상 그렇지 않은 것 같다. `memo`로 인한 문제가 프로파일에 뜨는 것을 본 적은 거의없지만, 렌더링이 cpu 시간을 소모하는 것음 매우 일반적이다. 일반적인 문제는, 너무 많은 컴포넌트가 한번에 마운팅 되는 등의 문제다.

#### memo의 메모리 비용

물론 `memo`는 값을 기억해야하는 특성상 메모리 소비가 존재한다. 그러나 이는 리액트에서는 조금 다르다. 리액트는 동작 방식으로 인해 이전 렌더링 결과는 후속 렌더링과 비교하기 위해 항상 유지지되고 있으야 한다. (항상 두개의 값을 가지고 있어야 한다.) 이것이 리액트의 reconciliation 의 기본이다.

> I don't think that it's a great analogy. Doing memoize() on every function would be horrible because you'd have to store the state of the input/output for all the calls. In the React case, React already does that for everything, so it's "free". - https://twitter.com/Vjeux/status/1083902075946205189

### `useCallback`을 모든 콜백함수에 쓰는 이유

`memo`에서 하고 있는 생각과 동일하다. 대부분의 콜백의 경우, 다른 컴포넌트의 `props`로 전달된다. 만약 이를 `useCallback`으로 감싸지 않는다면, `memo`가 깨질 것이다. 간단하다. `memo`가 동작하기 위해서, `useCallback`을 사용한다.

primitive 컴포넌트 전달되는 콜백은 어떤가? 여기에는 `useCallback`이 필요 없나? 그렇다. 그러나 만약 다른 사람이 이를 다른 컴포넌트로 감싼다면, 이 원래 컴포넌트 내부의 콜백을 다시 `useCallback`으로 감쌀까? 아마도 아닐 것이다.

여기에서도 앞선 논리와 동일한 논리가 적용된다. `useCallback`도 마찬가지로 cpu와 메모리를 잡아먹을 것이지만, 무시할 수준일 것이다. 모든 콜백은 메모리 어딘가에 저장될 필요가 있다. 이말인 즉슨, 이들은 언젠가 다시 호출될 수도 있다는 뜻이다.

### 모든 props와 deps에 `useMemo`를 사용하는 이유

새 객체나 배열을 만들 때도 마찬가지다. 이를 `useMemo`로 감싸지 않으면, props를 받는 모든 컴포넌트를 리렌더링할 것이다.

모든 렌더링 사이에 재생성되는 모든 데이터 구조는 deps에 표시함으로써 `useCallback`과 `useMemo`내에서 사용할 수 있다. 기본적으로 이를 메모이제이션 하지 않은 경우, 성능문제를 디버깅할 때 오랜시간을 소비해야 한다.
