---
title: '리액트의 렌더링은 어떻게 일어나는가?'
tags:
  - web
  - javascript
  - react
published: true
date: 2022-04-09 23:12:56
description: '리액트에서 메모이제이션을 언제 해야하는가 고민 하다가 여기까지 왔다'
---

## Table of Contents

## 렌더링이란 무엇인가?

리액트에서 렌더링이란, 컴포넌트가 현재 props와 state의 상태에 기초하여 UI를 어떻게 구성할지 컴포넌트에게 요청하는 작업을 의미한다.

### 렌더링 프로세스 살펴보기

렌더링이 일어나는 동안, 리액트는 컴포넌트의 루트에서 시작하여 아래쪽으로 쭉 훑어 보면서, 업데이트가 필요하다고 플래그가 지정되어 있는 모든 컴포넌트를 찾는다. 만약 플래그가 지정되어 있는 컴포넌트를 만난다면, 클래스 컴포넌트의 경우 `classComponentInstance.render()`를, 함수형 컴포넌트의 경우 `FunctionComponent()`를 호출하고, 렌더링된 결과를 저장한다.

컴포넌트의 렌더링 결과물은 일반적으로 JSX 문법으로 구성되어 있으며, 이는 js가 컴파일되고 배포 준비가 되는순간에 `React.createElement()`를 호출하여 변환된다. `createElement`는 UI 구조를 설명하는 일반적인 JS 객체인 React Element를 리턴한다. 아래 예제를 살펴보자.

```jsx
일반적인 jsx문법
return <SomeComponent a={42} b="testing">Text here</SomeComponent>

// 이것을 호출해서 변환된다.
return React.createElement(SomeComponent, {a: 42, b: "testing"}, "Text Here")

// 호출결과 element를 나타내는 객체로 변환된다.
{type: SomeComponent, props: {a: 42, b: "testing"}, children: ["Text Here"]}
```

전체 컴포넌트에서 이러한 렌더링 결과물을 수집하고, 리액트는 새로운 오브젝트 트리 (가상돔이라고 알려져있는)와 비교하며, 실제 DOM을 의도한 출력처럼 보이게 적용해야 하는 모든 변경 사항을 수집한다. 이렇게 비교하고 계산하는 과정을 리액트에서는 `reconciliation`이라고 한다.

그런 다음, 리액트는 계산된 모든 변경사항을 하나의 동기 시퀀스로 DOM에 적용한다.

### 렌더와 커밋 단계

리액트는 이 단계를 의도적으로 두개로 분류하였다.

- `Render phase`:컴포넌트를 렌더링하고 변경사항을 계산하는 모든 작업
- `Commit phase`: 돔에 변경사항을 적용하는 과정

리액트가 DOM을 커밋페이즈에서 업데이트 한 이후에, 요청된 DOM 노드 및 컴포넌트 인스턴스를 가리키도록 모든 참조를 업데이트 한다. 그런 다음 클래스 라이프 사이클에 있는 `componentDidMount` `componentDidUpdate` 메소드를 호출하고, 리액트 함수형 컴포넌트에서는 `useLayoutEffect`훅을 호출 한다.

리액트는 짧은 timeout을 세팅한 이후에, 이것이 만료되면 `useEffect`를 호출한다. 이러한 단계는 `Passive Effects` 단계라고도 알려져 있다.

이러한 클래스 라이브 사이클 메소드 다이어그램은 [여기](https://projects.wojtekmaj.pl/react-lifecycle-methods-diagram/)에서 확인해 볼 수 있다.

> 이번에 리액트 18에서 나온 `Concurrent Mode`의 경우, 브라우저가 이벤트를 처리할 수 있도록 렌더링 단계에서 작업을 일시 중지 할 수 있다. 리액트는 해당 작업을 나중에 다시시작하거나, 버리거나, 다시 계산할 수 있다. 렌더링이 패스가 된 이후에도, 리액트는 커밋단계를 한단계 동기적으로 실행한다.

여기서 중요한 사실은, **렌더링은 DOM을 업데이트 하는 것과 같은것이 아니고, 컴포넌트는 어떠한 가시적인 변경이 없이도 컴포넌트가 렌더링 될 수 있다는 것** 이다.리액트가 컴포넌트를 렌더링하는 경우

- 컴포넌트는 이전과 같은 렌더링 결과물을 리턴해서, 아무런 변화가 일어나지 않을 수 있다.
- Concurrent Mode에서는, 리액트는 컴포넌트를 렌더링 하는 작업을 여러번 할 수 있지만, 다른 업데이트로 인해 현재 작업이 무효화 되면 매번 렌더링 결과물을 버린다.

## 리액트는 어떻게 렌더링을 다루는가

### 렌더링 순서를 만드는 법

최초 렌더링이 끝난이후에, 리액트가 리렌더링을 queueing 하는 방법에는 여러가지가 있다.

- 클래스 컴포넌트
  - `this.setState()`
  - `this.forceUpdate()`
- 함수형 컴포넌트
  - `useState()`의 setter
  - `useReducer()`의 dispatches
- 기타
  - `ReactDOM.render()`를 호출하는 것 (`forceUpdate`와 동일) (리액트 18에서는 사라짐)

### 일반적인 렌더링 동작

여기에서 우리가 기억해야할 중요한 것이 있다.

**리액트의 기본적인 동작은 부모 컴포넌트가 렌더링되면, 리액트는 모든 자식 컴포넌트를 순차적으로 리렌더링 한다는 것이다.**

예를 들어, `A > B > C > D` 순서의 컴포넌트 트리가 있다고 가정해보자. `B`에 카운터를 올리는 버튼이 있고, 이를 클릭했다고 가정해보자.

1. `B`의 `setState()`가 호출되어, B의 리렌더링이 렌더링 큐로 들어간다.
2. 리액트는 트리 최상단에서 부터 렌더링 패스를 시작한다.
3. `A`는 업데이트가 필요하다고 체크 되어 있지 않을 것이므로, 지나간다.
4. `B`는 업데이트가 필요한 컴포넌트로 체크되어 있으므로, B를 리렌더링 한다. `B`는 `C`를 리턴한다.
5. `C`는 원래 업데이트가 필요 한것으로 간주되어 있지 않았다. 그러나, 부모인 `B`가 렌더링 되었으므로, 리액트는 그 하위 컴포넌트인 `C`를 렌더링 한다. `C`는 `D`를 리턴한다.
6. `D`도 마찬가지로 렌더링이 필요하다고 체크되어 있지 않았지만, `C`가 렌더링된 관계로, 그 자식인 `D`도 렌더링 한다.

즉

**컴포넌트를 렌더링 하는 작업은, 기본적으로, 하위에 있는 모든 컴포넌트 또한 렌더링 하게 된다.**

또한

**일반적인 렌더링의 경우, 리액트는 `props`가 변경되어 있는지 신경쓰지 않는다. 부모 컴포넌트가 렌더링 되어 있기 때문에, 자식 컴포넌트도 무조건 리렌더링 된다.**

즉, 루트에서 `setState()`를 호출한다는 것은, 기본적으로, 컴포넌트 트리에 있는 모든 컴포넌트를 렌더링 한다는 것을 의미한다. 이제 트리의 대부분의 컴포넌트가 동일한 렌더링 결과물을 반환할 가능성이 높기 때문에, 리액트는 DOM을 변경할 필요가 없다. 그러나 리액트는 여전히 컴포넌트에게 렌더링을 요청하고, 이 렌더링 결과물을 비교하는 작업을 요구한다. 두가지 모두 시간과 노력이 필요하다.

한가지 기억해둬야 할 것은, 렌더링이 꼭 나쁜 것만은 아니라는 것이다. 단지 리액트가 실제로 DOM을 변경해야 하는지 여부를 확인하는 것일 뿐이다.

### 리액트 렌더링 규칙

리액트 렌더링의 중요한 규칙 중 하나는 **렌더링은 '순수' 해야하고 '부수작용' 이 없어야ㅑ한다는 것** 이다. 근데 이는 매우 복잡하고 어려운데, 왜냐하면 대다수의 부수 작용이 왜 이러났는지 뚜렷하지 못하고, 어떤 것도 망가 뜨리지 않기 때문이다. 예를 들어, 엄밀히 말하면 `console.log()`도 부수작업을 야기하지만, 그 어떤 것도 망가 뜨리지 않는다. `prop` 가 변경되는 것은 명백한 부수효과 이며, 이는 무언가를 망가 뜨릴 수 있다. 렌더링 중간에 ajax 호출 또한 부수효과를 일으키고, 이는 요청의 종류에 따라서 명백하게 앱에 예기치 못한 결과를 야기할 수 있다.

[Rules of React](https://gist.github.com/sebmarkbage/75f0838967cd003cd7f9ab938eb1958f)라는 글이 있다. 이 글에서는, 렌더링을 표함한 다양한 리액트의 라이프 사이클 메소드의 동작과, 어떠한 동작이 '순수' 한지, 혹은 안전한지를 나타내고 있다. 요약하자면

렌더링 로직이 할 수 없는 것

- 존재하는 변수나 객체를 변경해서는 안된다.
- `Math.random()` `Date.now()`와 같은 래덤 값을 생성할 수 없다.
- 네트워크 요청을 할 수 없다.
- `state`를 업데이트

렌더링 로직은

- 렌더링 도중에 새롭게 만들어진 객체를 변경
- 에러 던지기
- 아직 만들어지지 않은 데이터를 lazy 초기화 하는일 (캐시 같은)

### 컴포넌트 메타데이터와 파이버

리액트는 애플리케이션에 존재하는 모든 현재 컴포넌트 인스턴스를 추적하는 내부 데이터 구조를 가지고 있다. 이 데이터 구조의 핵심적인 부분은, 다음 과 같은 메타데이터 필드를 포함하고 있는 Fiber라고 불리는 객체다.

- 컴포넌트 트리의 특정 시점에서 렌더링 해야하는 컴포넌트 타입의 유형
- 이 컴포넌트와 관련된 prop, state의 상태
- 부모, 형제, 자식 컴포넌트에 대한 포인터
- 리액트가 렌더링 프로세스를 추적하는데 사용되는 기타 메타데이터

리액트 17의 `fiber` 타입은 [여기](https://github.com/facebook/react/blob/v17.0.0/packages/react-reconciler/src/ReactFiber.new.js#L47-L174)에서 볼 수 있다.

렌더링 패스 동안, 리액트는 fiber 객체의 트리를 순회하고, 새로운 렌더링 결과를 계산한 결과로 나온 업데이트 된 트리를 생성한다.

**`fiber` 객체는 실제 컴포넌트 prop과 state 값을 저장하고 있다.** 컴포넌트에서 `prop`와 `state`의 값을 꺼내서 쓴다는 것은, 사실 리액트는 이러한 값을 fiber 객체에 있는 것으로 전달해준다. 사실, 클래스 컴포넌트의 경우, 리액트는 컴포넌트를 렌더링 하기 직전에 [`componentInstance.props = newProps`를 통해서 복사본을 저장](https://github.com/facebook/react/blob/v17.0.0/packages/react-reconciler/src/ReactFiberClassComponent.new.js#L1038-L1042)해준다. `this.props`가 존재한다는 것은, 리액트가 내부 데이터 구조의 참조를 복사해 두었다는 뜻이기도 하다. 즉, 컴포넌트라는 것은 리액트 fiber 객체를 보여주는 일종의 외관이라고 볼 수 있다.

비슷하게, [리액트 훅의 작동 또한 해당 컴포넌트의 fiber 객체에 연결된 링크드 리스트 형태로 저장하는 방식](https://www.swyx.io/getting-closure-on-hooks/)으로 동작한다. 리액트가 함수형 컴포넌트를 렌덜이하면, fiber에 연결된 후의 링크드 리스트롤 가져오며, [다른 훅을 호출할 떄마다 훅에 저장된 적절한 값을 반환한다.](https://github.com/facebook/react/blob/v17.0.0/packages/react-reconciler/src/ReactFiberHooks.new.js#L795)

부모 컴포넌트가 렌더링되어 자식 컴포넌트가 주어진다면, 리액트는 fiber 객체를 만들어 이 컴포넌트의 인스턴스를 추적한다. 클래스 컴포넌트의 경우, [`const instance = new YourComponentType(props)` 가 호출되고](https://github.com/facebook/react/blob/v17.0.0/packages/react-reconciler/src/ReactFiberClassComponent.new.js#L653) 새로운 컴포넌트 인스턴스를 fiber 객체에 저장한다. 함수형 컴포넌트의 경우에는, [YourComponentType(props)](https://github.com/facebook/react/blob/v17.0.0/packages/react-reconciler/src/ReactFiberHooks.new.js#L405)를 호출한다.

### 컴포넌트 타입과 재조정 (`Reconciliation`)

[재조정 페이지에 언급되어 있는 것](https://reactjs.org/docs/reconciliation.html#elements-of-different-types) 처럼, 리액트는 기존 컴포넌트 트리오 ㅏDOM 구조를 가능한 많이 재사용함으로써 리렌더링의 효율성을 추구한다. 동일한 유형의 컴포넌트, 또는 HTML 노드를 트리의 동일한 위치에 렌더링하도록 리액트에 요청하게 되면, 리액트는 해당 컴포넌트 또는 HTML 노드를 만드는 대신에 해당 업데이트만 적용한다. 즉, 리액트에 해당 컴포넌트 타입을 같은 위치에 렌더링 하도록 계속 요청이 있다면, 리액트는 계속 컴포넌트의 인스턴스를 유지한다는 뜻이다. 클래스 컴포넌트의 경우, 실제 컴포넌트의 실제 인스턴스와 동일한 인스턴스를 사용한다. 함수형 컴포넌트는, 클래스와 같은 느낌의 인스턴스는 없지만, `<MyFunctionComponent />` 가 보여지고 활성화 상태로 유지되고 있다는 관점에서 인스턴스를 나타내는 것으로 볼수도 있다.

그렇다면, 리액트는 어떻게 결과물이 실제로 변경된 시기와 방법을 알 수 있을까?

리액트 렌더링 로직은 elements를 그들의 `type` 필드를 기준으로 먼저 비교하는데, 이 때 `===`를 사용한다. 만약 지정된 element가 `<div>`에서 `<span>`으로, 또는 `<ComponentA />`에서 `<ComponentB />`로 변경된 경우, 전체 트리가 변경되었다고 가정하여 비교 프로세스의 속도를 높인다. 결과적으로 리액트는 모든 DOM노드를 포함한 기존 컴포넌트 트리를 삭제하고 새로운 컴포넌트 인스턴스를 처음부터 다시 만든다.

즉, 렌더링 동안에는 절대로 새로운 컴포넌트 타입을 만들어서는 안된다. 새로운 컴포넌트 타입을 만들다면, 이는 참조가 다르고, 이는 리액트가 차위 컴포넌트 트리를 모두 파괴하고 새로운 트리를 만들게 된다.

코드로 설명하자면,

```jsx
function ParentComponent() {
  // 이는 매번 새로운 컴포넌트 참조를 만들게 된다.
  function ChildComponent() {}

  return <ChildComponent />
}
```

대신에

```jsx
// 컴포넌트 타입 참조가 한번 딱 만들어진다.

function ChildComponent() {}

function ParentComponent() {
  return <ChildComponent />
}
```

를 사용하자.

### `key`와 `Reconciliation`

또한가지, 리액틀가 컴포넌트 인스턴스를 식별하는 방법으로 `key` prop이 있다. `key`는 실제 컴포넌트로 전달되는 요소는 아니다. 리액트는 이를 활용해 컴포넌트 타입의 특정 인스턴스를 구별하는데 사용할 수 있는 고유한 식별자로 사용한다.

아마도 `key`를 가장 많이 사용하는 경우는 리스트를 렌더링 할 때 일 것이다. `key`는 목록의 순서변경, 추가, 삭제와 같은 방식으로 변경될 수 있는 데이터를 렌더링하는 경우에 매우 중요하다. **여기서 중요하다는 것은 고유한 값을 사용해야 한다는 것이다. 고유한 값을 사용할 수 없는 최후의 수단으로, 배열의 인덱스를 사용해야 한다.**

왜 중요한지 한번 살펴보자. `<TodoListItem />` 컴포넌트 10개를 렌더링하고, 이를 키로 index를 사용하여 `0..9`를 할당했다. 이제, `6`, `7`을 지우고, 새롭게 3개를 추가해서 이제 키가 `0..10`이 되었다. 리액트는 이 때 단순히 하나만 추가하고 마는데, 리액트가 보기엔 10개에서 11개로 늘어난 차이밖에 없기 떄문이다. 리액트는 이제 기존에 있던 컴포넌트와 DOM 노드를 재활용할 것이다. 그러나 이 뜻은, `<TodoListItem key={6} />`가 8로 넘겨받은 props를 사용하여 렌더링 할 것이다. 컴포넌트 인스턴스는 살아있지만, 이전과 다른 데이터 객체를 기반으로 하고 있다. 이는 효과가 있을 수도 있지만, 예기치 못한 문제가 발생할 수 있다. 또한 기존 목록의 아이템이 이전과 다른 데이터를 표시해야 하기 때문에, 리액트는 텍스트와 다른 DOM내용을 변경하기 위해 목록의 아이템중 몇개에 업데이트를 적용해야 한다. 그러나, 목록의 아이템이 사실상 변한 것이 아니므로 업데이트가 필요하지 않는 것으로 간주된다.

대신에 `key={todo.id}`와 같은 것으로 처리했다면, 리액트는 올바르게 2개의 아이템을 지우고 3개를 추가할 것이다. 이는 두개의 컴포넌트 인스턴스와 DOM노드를 지우고, 새롭게 3개의 컴포넌트 인스턴스, DOM노드를 만드는 것을 의미한다.

`key`는 리스트에 있는 컴포넌트의 인스턴스를 식별하는데 유용하다. **어떤 리액트 컴포넌트에든 `key`를 추가하여 식별자를 부여할 수 있고, `key`를 변경하는 것은 리액트가 오래된 컴포넌트 인스턴스를 없애고, 새로운 DOM을 만든다는 것을 의미한다.** 일반적인 유즈케이스는 앞서 언급한 리스트의 경우이다. `<Form key={selectedItem.id}>`을 렌더링하면 선택한 항목이 변경될 때 리액트가 form을 삭제하고 다시 생성하므로, form의 오래된 상태 문제를 방지할 수도 있다.

### 렌더링 배치와 타이밍

기본적으로, `setState()`를 호출하는 것은 리액트가 새로운 렌더링 패스를 시작한다는 뜻이고, 이는 동기적으로 실행되어 리턴된다. 이에 추가적으로, 리액트는 렌더링 배치 형태의 최적화를 자동으로 실행한다. 여기서 말하는 렌더링 배치란, `setState()`에 대한 여러 호출로 인해 하나의 렌더 패스가 대기열에 저장되어 실행되는 것을 말하며, 일반적으로 약간의 지연이 발생한다.

[리액트 문서에서 언급하는 것 중 하나는 `state` 업데이트는 비동기 적일 수 있다는 사실](https://reactjs.org/docs/state-and-lifecycle.html#state-updates-may-be-asynchronous)이다. 특히 리액트는 리액트 이벤트 핸들러에서 발생하는 상태 업데이트를 자동으로 일괄적으로 처리한다. 리액트 이벤트 핸들러는, 일반적인 리액트 애플리케이션에서 매우 큰부분을 차지하기 때문에, 이는 주어진 앱의 대부분의 상태 업데이트가 실제로 일괄적으로 처리된다는 것을 의미한다.

리액트는 이벤트 핸들러를 `instability_batchedUpdates` 라고 하는 내부 함수로 래핑하여 이벤트 핸들러르르 렌더링 한다. 리액트는 `instability_batchedUpdates`가 실행중일 때, 대기중인 모든 상태 업데이트를 추적한 다음에, 단일 렌더링 경로로 적용한다. 리액트는 지정된 이벤트에 대해서 어떤 핸들러를 호출해야하는지 이미 정확하게 알고 있기 때문에, 이벤트 핸들러에서 사용하는 이방법은 매우 잘 먹힌다.

개념적으로, 리액트가 내부적으로 한느 일을 다음과 같은 의사 코드로 상상해볼 수 있다.

```javascript
// 진짜 이렇게 코드가 돌아간다는 건 아님
function internalHandleEvent(e) {
  const userProvidedEventHandler = findEventHandler(e)

  let batchedUpdates = []

  unstable_batchedUpdates(() => {
    // 이 안에 대기중인 모든 업데이트가 일괄 처리된 업데이트로 푸쉬될 것이다
    userProvidedEventHandler(e)
  })

  renderWithQueuedStateUpdates(batchedUpdates)
}
```

그러나 이는 실제 즉시 콜스택 외부에 대기중인 상태 업데이트와 함께 배치되지 않는 다는 것을 의미한다. 아래 예제를 살펴보자.

```javascript
const [counter, setCounter] = useState(0)

const onClick = async () => {
  setCounter(0)
  setCounter(1)

  const data = await fetchSomeData()

  setCounter(2)
  setCounter(3)
}
```

이는 세개의 렌더링 패스를 실행할 것이다. 먼저 `setCounter(0)` `setCounter(1)`를 함께 배치할 것이다. 이는 둘다 원래 이벤트 핸들러의 콜 스택 중에 발생하므로, 둘다 `unstable_batchedUpdates`의 호출 내에서 발생할 것이기 때문이다.

그러나 `setCounter(2)`는 `await` 이후에 실행된다. 즉 원래 동기식 콜 스택이 완료되고, 이 함수의 후반부는 완전히 다른 이벤트 루프 콜 스택에서 훨씬 나중에 실행될 것이다. 그 때문에, 리액트는 전체 렌더링 패스를 `setCounter(2)` 호출의 마지막 단계로 동기적으로 실행하고, 렌더링 패스를 완료 한 이후에, `setCounter(2)`에서 리턴할 것이다. 이와 유사한 동작이 `setCounter(3)`에서도 마찬가지 형태로 일어날 것이다.

커밋단계의 라이프사이클 메소드에는 `componentDidMount` `componentDidUpdate` `useLayoutEffect`와 같은 몇가지 추가 적인 엣지 케이스가 존재한다. 이는 주로 브라우저가 페인팅을 하기전에 렌더링 후 추가 로직을 수행할 수 있도록 하기 위해 존재한다. 일반적인 사용사례는 다음과 같다.

- 불완전한 일부 데이터로 컴포넌트를 최초 렌더링
- 커밋 단계 라이프 사이클에서, DOM 노드의 실제 크기를 `ref`를 통해 측정하고자 할 때
- 해당 측정을 기준으로 일부 컴포넌트의 상태 설정
- 업데이트된 데이터를 기준으로 즉시 리렌더링

이러한 사용사례에서, 초기의 부분 렌더링된 UI가 사용자에게 절대로 표시되지 않도록 하고, 최종 UI 만 나타날 수 있게 한다. 브라우저는 수정중인 DOM 구조를 다시 계산하지 자바스크립트는 여전히 실행중이고,이벤트 루프를 차단하는 동안에는 실제로 화면에 아무것도 페인팅하지 않는다. 그러므로, `div.innerHTML = "a"`, `div.innerHTML="b"`와 같은 작업을 수행하면 `a`는 나타나지 않고 `b`만 나타날 것이다.

이 때문에 리액트는 항상 커밋 단계 라이프사이클에서 렌더링을 동기로 실행한다. 이렇게 하면 부분적인 렌더링을 무시하고 최동 단계의 렌더링 내용만 화면에 표시할 수 있다.

마지막으로, 모든 `useEffect` 콜백이 완료되면 `useEffect` 콜백의 상태 업데이트가 대기열에 저장되고, `Passive Effects` 단계가 끝나면 플러시된다.

`unstable_batchedUpdates`API가 public 하게 export 되는 것에 주목할 필요가 있다. 그러나

- 이름에서 알 수 있듯이, `불안정`으로 표시되고, React API에서 공식으로 지원하는 부분은 아니다.
- 그러나 리액트 팀은 `불안정`한 api 치고는 가장 안전적이며, 페이스북의 코드 절반이 이에 의존하고 있다고 이야기 했다.
- `react` 패키지에서 export 되는 다른 React의 핵심 API와는 다르게, `unstable_batchedUpdates`는 reconciler에 특화된 API로 리액트 패키지의 일부가 아니다. 대신에, 이는 `react-dom` `react-native`에서 export 된다. 즉, `react-three-fiber`나 `ink`와 같은 다른 reconciler와는 다르게 `unstable_batchedUpdates`를 export 하지 않을 가능성이 크다.

리액트 18에서 소개된 Concurrent 모드에서는, 리액트는 모든 업데이트를 배치로 실행한다.

> [Automatic Batching에 대하여 알아보기](/2022/04/react-18-changelog#automatic-batching)

### 렌더 동작의 엣지 케이스

리액트에서 개발중인 `<StrictMode >` 태그 내부에서는 컴포넌트를 이중으로 렌더링 한다. 즉, 렌더링 로직이 실행되는 횟수가 커밋된 렌더링 패스의 횟수와 동일하지 않으며, 렌더링을 수행하는 동안 `console.log()`문에 의존하여 발생한 렌더링의 수를 셀 수 없다. 대신 `React DevTools Profiler`를 사용하여 추적을 캡쳐하고, 전체적으로 커밋된 렌더링 갯수를 세거나, `useEffect` 훅 또는 `componentDidMount` `componentDidUpdate` 라이프 사이클에서 로깅을 추가하는 방법을 사용해야 한다. 이렇게 하면 실제로 렌더링 패스를 완료하고 이를 커밋한 경우에만 로그가 찍힌다.

정상적인 상황에서는 절대로 실제 렌더링 로직에서 상태 업데이트를 대기열에 넣어서는 안된다. 즉, 클릭이 발생할 때 `setState()`를 호출하는 콜백을 사용하는 것은 괜찮지만, 실제 렌더링 동작의 일부로 `setState()`를 호출하는 것은 안된다.

그러나 여기에는 한가지 예외가 있다. 함수 컴포넌트는 렌더링하는 동안 `setState()`를 직접호출할 수 있지만, 이는 조건부로 수행되고 컴포넌트가 렌더링될 때 마다 실행되지 않는다. 이것은 클래스 컴포넌트의 `getDerivedStateFromProps`와 동등하게 작동한다. 렌더링 하는 동안 함수 컴포넌트가 상태 업데이트를 대기열에 밀어 넣어두면, 리액트는 즉시 상태 업데이트를 적용하고 해당 컴포넌트 중 하나를 동기화 하여 다시 렌더링 한 후 계속 진행한다. 컴포넌트가 상태 업데이트를 무한하게 queueing하고 리액트가 다시 렌더링을 하도록 강제하는 경우, 리액트는 최대 50회까지 만 실행한 후에 이 무한반복을 끊어버리고 오류를 발생 시킨다. 이 기법은 `useEffect` 내부에 `setState()`호출과 리렌더링을 하지 않고 prop 값을 기준으로 state의 값을 강제로 업데이트 할 때 사용할 수 있다.

```jsx
function ScrollView({ row }) {
  const [isScrollingDown, setIsScrollingDown] = useState(false)
  const [prevRow, setPrevRow] = useState(null)

  // 조건부로 prop 값을 기준으로 바로 state를 업데이트 때릴 수 있음
  if (row !== prevRow) {
    setIsScrollingDown(prevRow !== null && row > prevRow)
    setPrevRow(row)
  }

  return `Scrolling down: ${isScrollingDown}`
}
```

## 렌더링 성능 향상시키기

렌더링은 리액트의 동작 방식에서 일반적으로 예상할 수 있는 부분이지만, 렌더링 작업이 때때로 낭비될 수 있다는 것도 사실이다. 컴포넌트의 렌더링 출력이 변경되지 않았고, DOM의 해당 부분을 업데이트할 필요가 없다면 해당 컴포넌트를 렌덜이 태우는 것은 정말로 시간낭비다.

리액트 컴포넌트 렌더링 결과물은 항상 현재 props와 state의 상태를 기반으로 결정되어야 한다. 따라서 props와 state가 변경되지 않았음을 미리 알고 있다면 렌더링 결과물은 동일 할 것이고, 이 컴포넌트에 대해 변경이 필요하지 않고 렌더링 작업을 건너 뛸 수 도 있다는 것에 대해서도 알아야 한다.

일반적으로 소프트웨어 성능을 개선하는 건 두가지 접근법이 존재한다.

- 동일한 작업을 가능한 더 빨리 수행하는 것
- 더 적게 작업하는 것

리액트에서 렌더링을 최적화하는 것은 주로 컴포넌트 렌더링을 적절하게 건너뛰어서 작업량을 줄이는 것이다.

### 컴포넌트 렌더링 최적화 기법

리액트는 컴포넌트 렌더링을 생략할 수 있는 세가지 API를 제공한다.

- [React.Component.shouldComponentUpdate](https://reactjs.org/docs/react-component.html#shouldcomponentupdate): 클래스 컴포넌트의 옵셔널 라이프 사이클 메소드로, false를 리턴하면 리액트는 컴포넌트 렌더링을 건너뛴다. 이 메소드 내부에는 `boolean`을 리턴할 어떤 로직이든 집어넣을 수 있지만, 가장 일반적인 방법은 컴포넌트의 prop와 state가 변경되었는지 확인하고, 변경되지 않았을 때 false를 리턴하는 것이다.
- [React.PureComponent](https://reactjs.org/docs/react-api.html#reactpurecomponent): `shouldComponentUpdate`를 구현할때 props와 state를 비교하는 것이 가장 일반적인 방법이므로, `PureComponent` 를 base class로 구현하면 `Component` + `shouldComponentUpdate`를 사용하는 것과 같은 효과를 볼 수 있다.
- [React.Memo()](https://reactjs.org/docs/react-api.html#reactmemo): 내장 고차 컴포넌트 타입으로, 컴포넌트 타입을 인수로 받고, 새롭게 래핑된 컴포넌트를 리턴된다. 래퍼 컴포넌트의 기본 동작은 `props`의 변경이 있는지 확인하고, 변경된 `props`가 없다면 다시 렌더링 하지 못하게 하는 것이다. 함수 컴포넌트와 클래스 컴포넌트는 모두 이 것을 사용하여 래핑 할 수 있다.

이 모든 기법은 `shallow equality (얕은 비교)`를 사용한다. 즉 서로 다른 객체에 있는 모든 개별 필드를 검사하여 객체의 내용이 같은지 다른지 확인한다. 다시말해, `obj1.a === obj2.a && obj1.b === obj2.b && ........`를 수행하는 것이다. 이는 자바스크립트 엔진에서 매우 간단한 작업인 `===`를 사용하므로 매우 빠르게 끝난다. 그러므로, 세가지 방법은 모두 같은 방법론을 사용하는 것이다. `const shouldRender = !shallowEqual(newProps, prevProps)`

여기에 잘 알려지지 않은 기법도 하나 더 있다. 리액트 컴포넌트가 렌더링 결과물을 지난번과 정확히 동일한 참조를 반환한다면, 리액트는 해당 하위 컴포넌트를 렌더링하는 것을 건너 뛴다. 이 기술을 구현하는 방법은 대략 두가지 정도가 있다.

- 결과물에 `props.children`이 있다면, 이 컴포넌트가 상태 업데이트를 수행해도 element는 동일할 것이다.
- 일부 Element를 `useMemo()`로 감싸면, 종속성이 변경될 때 까지 동일하게 유지된다.

아래 코드를 살펴보자.

```jsx
// 상태가 업데이트되도 props.children은 다시렌더링 되지 않는다.
function SomeProvider({ children }) {
  const [counter, setCounter] = useState(0)

  return (
    <div>
      <button onClick={() => setCounter(counter + 1)}>Count: {counter}</button>
      <OtherChildComponent />
      {children}
    </div>
  )
}

function OptimizedParent() {
  const [counter1, setCounter1] = useState(0)
  const [counter2, setCounter2] = useState(0)

  const memoizedElement = useMemo(() => {
    // counter2가 업데이트되도 같은 참조를 반환하므로, counter1이 변경되지 않는 한 같은 참조를 리턴할 것이다.
    return <ExpensiveChildComponent />
  }, [counter1])

  return (
    <div>
      <button onClick={() => setCounter1(counter1 + 1)}>
        Counter 1: {counter1}
      </button>
      <button onClick={() => setCounter1(counter2 + 1)}>
        Counter 2: {counter2}
      </button>
      {memoizedElement}
    </div>
  )
}
```

이러한 모든 기법들에서, 컴포넌트 렌더링을 건더뛰면 리액트는 마찬가지로 하위 트리의 전체 렌더링을 건너뛰어 이는 "재귀적으로 자식을 렌더링" 하는 동작을 중지하게 된다.

### 새로운 props의 참조가 렌더링 최적화에 어떻게 영향을 미치는가?

앞서 보앗듯, **기본적으로 리액트는 중첩된 컴포넌트의 props가 변경되지 않았더라도 다시 렌더링을 수행한다.** 이는 하위 컴포넌트에 새로운 참조를 props로 전달하는 것 또한 문제가 되지 않는다는 것을 의미한다. 왜냐하면 같은 props가 오던 상관없이 렌더링을 할 것이기 때문이다. 아래 예제를 살펴보자.

```jsx
// ParentComponent가 렌더링될때마다, 하위 자식 컴포넌트의 props는 변경되지 않았지만 그것과 상관없이 계속 리렌더링 된다.
function ParentComponent() {
  const onClick = () => {
    console.log('Button clicked')
  }

  const data = { a: 1, b: 2 }

  return <NormalChildComponent onClick={onClick} data={data} />
}
```

`ParentComponent`가 매번 렌더링 될 때 마다, 매번 새로운 `onClick` 함수의 참조와 새로운 `data` 객체 참조를 만들어서, 이를 props로 자식 컴포넌트에 넘겨줄 것이다. (함수가 화살표건 일반 함수건, 어쩄거나 새로운 함수 참조가 생긴다는 사실에는 변함이 없다.)

이는 또한 `<div/>`나 `<button/>`를 `React.memo()`래핑하는 것 처럼, 호스트 컴포넌트에 대해 렌더링을 최적화 하는 것이 별 의미가 없다는 것을 뜻한다. 이러하나 기본 컴포넌트 하위에 하위 컴포넌트가 없으므로 렌더링 프로세스는 여기서 중지되버리고 말 것이다.

하지만, **하위 컴포넌트가 props가 변경되었는지 확인하여 렌더링을 최적화 하려는 경우, 새 props를 전달하면 하위 컴포넌트가 렌더링을 수행하게 된다.** 새 prop 참조가 실제로 새로운 데이터인 경우에 이방법이 유용하다. 그러나 상위 컴포넌트가 단순히 콜백 함수를 전달하는 수준이면 어떻게 될까?

```jsx
const MemoizedChildComponent = React.memo(ChildComponent)

function ParentComponent() {
  const onClick = () => {
    console.log('Button clicked')
  }

  const data = { a: 1, b: 2 }

  return <MemoizedChildComponent onClick={onClick} data={data} />
}
```

이제, `ParentComponent`가 렌더링 될 때 마다 `MemoizedChildComponent`는 해당 props 가 새로운 참조로 변경되었음을 확인하고 다시 렌더링을 수행한다. `onClick` 함수와 데이터 객체의 값이 변하지 않았음에도!

이러한 과정을 요약하자면

- `MemoizedChildComponent`는 렌더링을 건너뛰고 싶었지만, 항상 다시 렌더링 될 것이다.
- 새로운 참조가 계속해서 생기기 때문에 `props`의 변화를 비교하는 것은 무의미한 일이다.

비슷하게,

```jsx
function Component() {
  return (
    <MemoizedChild>
      <OtherComponent />
    </MemoizedChild>
  )
}
```

도, `props.children`이 항상 새로운 참조를 가리키기 때문에 항상 자식 컴포넌트를 새로 렌더링 할 것이다.
