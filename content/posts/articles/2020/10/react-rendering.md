---
title: '리액트 렌더링에 대한 이해'
tags:
  - react
  - javascript
published: false
date: 2020-10-29 22:45:09
description: '리액트 재조정 알고리즘 (reconciliation) 에 대해서도 생각해볼 필요가 있다'
---

렌더링은 비단 리액트 뿐만 아니라 모든 프론트엔드 개발에 있어서 가장 중요한 부분 중 하나다. 리액트의 `render()`는, 클래스 컴포넌트에서 유일하게 빠져서는 안되는 메소드 이며, 브라우저 윈도우에 렌더링을 담당하고 있다.

## `render()`

`render()`는 사용자가 호출할 수 없다. 리액트 컴포넌트의 라이프 사이클의 일부분이며, 리액트에 의해 다양한 이유로 호출된다. 일반적으로 리액트 컴포넌트가 처음 초기화 되거나, state값이 변경되었을 때 호출된다. 이 메서드는 arguments를 받지 않으며, `jsx.element`를 리턴한다. 그리고 이 `jsx.element`는 브라우저 윈도우에 HTML 구조로 그려지게 된다.

위에서도 계속해서 언급했듯이, `render()`는 사용자가 임의로 호출할 수 없으며, 라이프 사이클의 일부로 동작된다. 만약에 진짜 진짜 강제로 호출하고 싶다면 `forceUpdate()`를 사용하면 되지만, 이는 명백한 리액트 안티패턴이다. 좋은 리액트 컴포넌트를 만들기 위해서는, state와 prop이 render 프로세스를 자연스럽게 조작해야 하며, 강제로 render를 호출할 필요성을 만들어 내서는 안된다.

라이프 사이클 내에서, render가 호출되는 시나리오는 아래와 같다.

- 리액트 컴포넌트가 처음 초기화 되었을때, `constructor()` 호출 이후에.
- components의 prop이 변경되었을 때
- `setState()`호출 된 이후애

`shouldComponentUpdate()`를 사용하면, props와 state의 변화에 무조건 렌더링을 하는 로직을 바꿀 수 있다. 특정조건의 변화에만 렌더링을 하거나 / 막고 싶다면, `shouldComponentUpdate()`를 쓰면 된다.

```javascript
  shouldComponentUpdate(nextProps: NewComponentProps, nextState: NewComponentState) {
    if (this.props.text !== nextProps.text) {
      return true;
    } else {
      return false;
    }
  }
```

명심해야할것은, `render()`는 순수 함수라는 것이다. `render()` 내부에서 컴포넌트의 state나 props를 업데이트 해서는 안된다. 당연한 얘기겠지만, `render()`함수 내에서 state나 props을 업데이트 하면 무한히 계속해서 렌더링이 돌아갈 것이다.

또하나 명심해야할 것은 JSX는 immutable 하다는 것이다. JSX는 DOM이 렌더링되는 모습을 그리는 상수를 리턴한다. 따라서 `render()`함수를 작성할 떄, 렌더링 당시에 어떤 모습으로 전체 UI가 그려져야하는 지를 상상하는 것이 도움이 된다.

https://reactjs.org/docs/rendering-elements.html#react-only-updates-whats-necessary

> In our experience, thinking about how the UI should look at any given moment, rather than how to change it over time, eliminates a whole class of bugs.

## reconciliation (재조정) 알고리즘

리액트 컴포넌트의 `render()`메소드는 이후에 리액트의 재조정 알고리즘으로 넘어간다. 이는 리액트에서 굉장히 중요한 알고리즘으로, 리액트가 내부에서 관리하고 있는 가상 DOM을 기반으로 어떻게 실제 DOM을 렌더링 해야하는지 결정한다. 매 `render()`호출마다 렌더링해야 할 사항을 지능적으로 결정하기 위해, 리액트는 가상 DOM의 현재 상태와 실제 DOM의 상태를 비교하고 UI가 업데이트되었음을 인식해야하는 물리적 DOM만 변경한다.

[Reconciliation 공식 문서](https://ko.reactjs.org/docs/reconciliation.html)

DOM트리는 위에서 아래로 파싱한다. 만약 일치 하지 않는 엘리먼트를 발견한다면, 리액트는 해당 앨리먼트에서 부터 이를 포함하고 있는 하위 엘리먼트 전체를 분해하고 다시 빌드한다. 만약 하위 트리가 복잡하다면, 이 연산비용또한 증가할 것이다. 따라서 새로운 요소가 DOM트리에 삽입된다면, 그리고 배치 위치에 대해 특별한 요구사항이 없다면 마지막 요소로 추가되는 것이 좋다.

```html
<div>
  <span key="li-1">list item 1</span>

  <span key="li-2">list item 2</span>

  <span key="li-3">list item 3</span>
</div>
```

맨 위에 컴포넌트가 추가된다면 아래와 같이 비효율적으로 변경된다.

```html
<div>
  <!-- 원래 <span key="li-1">list item 1</span> 였는데 바뀜  -->
  <NewComponent />

  <!-- 원래 <span key="li-2">list item 2</span> 였는데 바뀜 -->
  <span>list item 1</span>

  <!-- 원래 <span key="li-3">list item 3</span> 였는데 바뀜 -->
  <span>list item 2</span>

  <!-- 새로 만듬  -->
  <span>list item 3</span>
</div>
```

하지만 이를 맨 아래로 내려둔다면

```html
<div>
  <!-- 변화없음 -->
  <span>list item 1</span>

  <!-- 변화없음 -->
  <span>list item 2</span>

  <!-- 변화없음 -->
  <span>list item 3</span>

  <!-- 새로 추가만 하면 됨 -->
  <NewComponent />
</div>
```

아래로 추가했을 때가 훨씬 더 효과적인 것을 알 수 있다.

## `key` prop

`map()`으로 리스트를 렌더링 했을 때, `key` prop을 반드시 붙이는 것을 알 수 있다. `key`는 정확히 무슨일을 할까? `key`는 리액트가 DOM tree의 엘리먼트를 구별하는 역할을 하는데, 이는 재조정 알고리즘 내에서 쓰인다. 리액트가 한 엘리먼트의 모든 자식 요소를 파싱할떄, `key`를 활용하여 마지막 업데이트에서 있었던 엘리먼트와 현재 엘리먼트간의 구별을 할 수 있다. 업데이트 사이에 `key`가 똑같다면 엘리먼트는 현재 상태를 유지한다.

```html
<div>
  <span key="li-1">list item 1</span>

  <span key="li-2">list item 2</span>

  <span key="li-3">list item 3</span>
</div>
```

만약 위와 비슷한 예제에서, key를 추가해서 엘리먼트를 추가한다면 어떻게 될까?

```html
<div>
  <!-- 새로운 <span />을 초기화 한다. -->
  <span key="li-4">list item 4</span>

  <!-- li-1이 이전 렌더링에도 있었으므로, 유지한다. -->
  <span key="li-1">list item 1</span>

  <!-- li-2이 이전 렌더링에도 있었으므로, 유지한다. -->
  <span key="li-2">list item 2</span>

  <!-- li-3이 이전 렌더링에도 있었으므로, 유지한다. -->
  <span key="li-3">list item 3</span>
</div>
```

`map()` 또는 이러한 반복을 통해서 하위트리가 생성되는 경우, React는 엘리먼트와 함께 `key`속성을 요구한다. DOM 하위 트리를 수동으로 추가하는 이러한 경우에도, 조건부 렌더링과 관련된 복잡한 동작을 갖는 하위 트리에도 키를 제공하면 이점을 얻을 수 있다.
