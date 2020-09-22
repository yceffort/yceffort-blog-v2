---
title: 'Github Code Spaces 베타 당첨 및 후기'
tags:
  - javascript, react
published: true
date: 2020-09-21 18:24:44
description: '리액트 17.0 새로운 기능은 추가되지 않을 예정. 점진적 업그레이드 추가, 이벤트 위임 방식 변경이 주요 변경 내용인 것 같네용.'
category: react
template: post
---

[원문](https://reactjs.org/blog/2020/08/10/react-v17-rc.html)을 대충 요약한 글입니다.

## 새로운 기능은 없다.

리액트 17.0은 새로운 버전이 기능이 추가되는 대신에, 리액트 그 자체의 업그레이드에 초점을 두고 있다.

## 점진적 업그레이드

이전까지 리액트 버전 업그레이드는, 중간이 없었다. 이전 버전을 유지하거나, 새버전을 깔거나 둘중에 하나 였다. 이 전략이 슬슬 한계에 부딪히고 있다. 예를 들어 [legacy context api](https://reactjs.org/docs/legacy-context.html)의 경우에는 이를 자동으로 업그레이드할 방법이 존재 하지 않는다. 대부분의 애플리케이션이 이 api를 쓰고 있지 않지만, react에서는 여전히 이들을 지원해야 한다. 그래서 구 버전 앱들을 뒤로 남겨두고 지원을 하지 않을지, 아니면 계속해서 지원해야할지를 선택해야 하는데, 두 방법 모두 좋지는 않다. 따라서 새로운 방법을 염두해 두고 있다.

### 리액트 17에서는 점진적으로 업그레이드가 가능하다.

이전 버전 업그레이드는, 전체 앱을 한번에 업그레이드 해야 햇다. 이는 오래되거나 관리되지 않은 코드에서 사용하기에는 너무나 힘든 문제였다. 그래서 이후 부터는 두가지 옵션을 주려고 한다. 첫번째 옵션은 이전에 그랬던 것 처럼 한번에 전체 애플리케이션을 업데이트 하는 것이다. 그리고 다른 하나는 점진적으로 하나씩 업그레이드 하는 것이다. 예를 들어, 대부분의 앱을 리액트 18로 올릴 수 있지미나, lazy-loading 다이얼러그나 일부 라우트는 리액트 17 상태로 둘 수 있는 것이다.

그렇다고 꼭 점진적 업그레이드를 해야하는 것은 아니다. 여전히, 한번에 앱을 업그레이드 하는 것이 최선의 해결책이다. 그러나 사이즈가 큰 애플리케이션의 경우 이러한 옵션을 선택하기에 무리가 있을 수 있으며, 리액트 17부터 그것을 지원하려고 한다.

이런 점진적 업그레이드를 위해서는 리액트 이벤트 시스템을 몇가지 변경해야 하고, 이러한 변화가 breaking change가 될 수 있어서 메인 버전을 업데이트 하였다. 약 10만개 이상의 컴포넌트 들 중에, 실제로 변화가 있을 것으로 예상되는 것은 20개 정도다.

[데모버전](https://github.com/reactjs/react-gradual-upgrade-demo/) 레포를 참고해보자.

## 이벤트 위임의 변화

먼저 리액트에서 이벤트 핸들러를 붙이는 코드를 살펴보자

```jsx
<button onClick={handleClick}>
```

바닐라 DOM에서는 이렇게 작동할 것이다.

```javascript
myButton.addEventListener('click', handleClick)
```

그러나 대부분의 이벤트의 경우, 리액트는 이벤트가 실제 선언된 DOM에 붙이지 않는다. 대신, 리액트는 하나의 이벤트당 하나의 핸들러를 document node에 붙인다. 이는 이벤트 위임이라고 불린다. 큰 어플리케이션 구조에서 성능의 이점을 볼 수 있는 것 이외에도, [replaying events](https://twitter.com/dan_abramov/status/1200118229697486849)와 같은 기능을 추가할 때도 유용하게 사용할 수 있다.

리액트는 첫 릴리즈 때 부터 이벤트 위임을 자동으로 실행해 왔다. DOM이벤트가 도큐먼트에서 실행되면, 리액트는 어떤 컴포넌트에서 실행되어야 하는지 살펴보고, 리액트는 해당 컴포넌트에서 부터 위로 버블링을 시작한다. 그러나 이 뒤에는, 리액트가 이미 이벤트 핸들러를 붙인 곳에서 네이티브 이벤트가 이미 도큐먼트 레벨까지 버블링되어 있었다.

그러나, 이부분이 점진적 업그레이드 전략에서 문제가 되었다.

만약 페이지 내에서 여러 개의 리액트 버전이 존재한다면, 이벤트 핸들러가 최상단에 붙게 될 것이다. 이는 `e.stopPropagation()`을 어기게 된다. 만약 nested tree에서 이벤트에 대해 전파를 중지하더라도, 바깥 트리에서는 계속해서 이벤트를 받게 된다. 이는 리액트 내부에서 서로다른 버전의 tree를 갖는 것을 어렵게 만든다.

리액트 17부터, 이벤트 핸들러를 더이상 큐먼트의 최상 노드인 `html`에 붙이 지 않는다. 대신, 리액트 트리가 렌더링 되는 DOM Container에 이벤트를 붙이게 된다.

```javascript
const rootNode = document.getElementById('root')
ReactDOM.render(<App />, rootNode)
```

리액트 16 이하에서는, 이벤트 들이 `document.addEventListener()`로 이루어진다. 그러나 17버전 부터는 `rootNode.addEventListener()`로 변경된다.

번역이 거지 같아서 정리

- 특정 노드에 매번 이벤트 리스너를 붙이는 대신, 이벤트 리스너를 부모에게 추가하는 것이 이벤트 위임이다.
- 리액트는 이러한 이벤트 위임을 적극 활용하고 있었으며, 위임의 대상은 `document`였다.
- `document`에서 이벤트가 발생하면, 리액트 이벤트 시스템이 실제로 이벤트가 발생한 컴포넌트를 찾고, 이벤트 버블링으로 상위 컴포넌트에 전달함
- 문제는 여기에서 발생하는데, 네이티브 이벤트는 document 까지 이벤트가 버블링됨 (당연한거 아님?)
- 만약 한 페이지에 여러가지 리액트 버전이 존재한다면, 현재 구조상 모든 이벤트 들이 document에 이벤트를 위임할 것이다.
- 만약 이벤트가 발생한 컴포넌트를 찾아서, 이벤트 전파를 중단하더라도 (`stopPropagation`) 앞서 설명한 것처럼 네이티브 ㅇ이벤트는 document까지 알아서 버블링이 될 것이기 때문에, 사이드 이펙트가 발생할 수 있다.

[추가 사례 분석](https://github.com/facebook/react/pull/8117)

- atom에서는 여러개 리액트 인스턴스를 생성한뒤, 한개의 앱에서 활용하고 있었다.
- 그러나 두개의 리액트 트리가 nested되어 있는 상태에서는 `e.stopPropagation`이 잘 동작하지 않는 문제가 있었다.
- 그 이유는 두개의 이벤트 리스너가 각각의 트리에 존재했고, 따라서 이들이 각각 전파가 취소되지 않는 문제가 있었다.
  - React version A로 돌아가는 `OuterComponent`와 서로 다른 버전으로 돌아가는 `InnerComponent`가 존재한다고 가정해보자.
  - `InnerComponent`는 이벤트 리스너를 inner tree의 최상단에 이벤트를 부착하려 할 것이고, `OuterComponent`는 해당 컴포넌트의 최상단에 붙이려고 할 것이다.
  - `InnerComponent`에서 클릭이 발생하면, 이 내부버전의 리액트가 트리 더 깊숙히 있기 때문에 이벤트에 대해서 먼저 알아 차릴 것이다.
  - 그리고 이는 `InnerComponent`의 이벤트를 발생시킬 것이고, 따라서 리액트는 `e.stopPropagation()`을 발생시킨다.
  - 이벤트 전파가 중단 되었으므로, 외부버전의 리액트에서는 이를 알아챌 수 없다.
- 이러한 문제를 `html`이 아닌 컴포넌트가 최초에 렌더링되는 `dom`노드에 이벤트를 붙이면서 해결할 수 있음.
- 서로 다른 리액트 버전 (=서로 다른 리액트 인스턴스)을 가지고 있다 하더라도, 이벤트가 부착되는 element가 각각 다르기 때문에 이러한 문제에서 자유로울 수 있음.

요약

- 서로다른 리액트 버전이 페이지 내에서 존재하는 경우 이벤트가 발생했을 시, 리액트에서 일어나는 버블링과 `stopPropagation`떄문에 일부 이벤트 실행 여부를 알아채지 못할 수 있음.

![React 17 event delegation](https://reactjs.org/static/bb4b10114882a50090b8ff61b3c4d0fd/1e088/react_17_delegation.png)

이러한 변화로 인해, 한버전에 의해 관리되는 리액트 트리 내부에 서로 다른 리액트 버전을 관리하는 것이 안전해졌다. 이를 위해 두 버전이 모두 최소 리액트 17 버전 이상이어야 한다.

이는 리액트가 다른 기술과 사용하는 것을 더욱 용이하게 한다. 만약 외부에 jQuery가 존재하고, 내부에는 리액트가 존재한다면 이제 예상대로 이벤트 전파를 jQuery단까지 막을 수 있다.

이 변화로 인해, 코드에 몇 가지 변화가 필요할 수 있다. 예를 들어, DOM에 `document.addEventListener`를 활용하여 수동으로 이벤트를 붙여서 리액트의 모든 이벤트를 감지하는 코드가 존재할 수 있다. 리액트 16버전 에서는 이러한 코드가 가능했지만, 리액트 17 부터는 전파가 막히게 되므로 `document`에서도 이벤트가 발생하는지 알 수 없다.

```javascript
document.addEventListener('click', function () {
  // This custom handler will no longer receive clicks
  // from React components that called e.stopPropagation()
})
```

위와 같은 코드는 이제 , `capture: true`를 추가해야 한다.

```javascript
document.addEventListener(
  'click',
  function () {
    // Now this event handler uses the capture phase,
    // so it receives *all* click events below!
  },
  { capture: true },
)
```

결과적으로, 리액트 17의 이벤트 전파가 실제 DOM과 비슷해졌다고 볼 수 있다.

## 기타 Breaking Changes

### 브라우저 최적화

- `onScroll` 관련 이벤트 버블링을 제거하여, 네이티브 `onScroll` 이벤트가 버블링 되지 않는 것과 마찬가지로 동일하게 작동하도록 한다.
- `onFocus` `onBlur`가 네이티브 이벤트인 `focusin` `focusout`을 이용하도록 변경 (이 변경은 실제 버블링에 영향을 미치지 않는다. 리액트에서는 항상 `onFocus`이벤트는 버블링 되고, 리액트 17에서도 마찬가지 일 것이다.)
- Capture phase event (`onClickCapture`)가 실제 브라우저 리스너를 사용하도록 변경

### No Event Pooling

이벤트 풀링이 제거된다. 리액트는 오래된 브라우저의 성능 향상을 위해서 서로 다른 이벤트 사이에서 이벤트 객체를 재사용하고, 모든 이벤트 필드를 null로 설정해두었다. 리액트 16 및 이전 버전에서는 `e.persist()`를 호출하여 이벤트를 적절히 사용하거나, 필요한 속성을 미리 읽어와야 했다.

```jsx
function handleChange(e) {
  setData((data) => ({
    ...data,
    // 리애트 16버전에서는 에러가 났다.
    text: e.target.value,
  }))
}
```

이제 리액트 17에서 부터는 예상처럼 동작한다. 이벤트 풀링 최적화가 완전히 삭제 되었기 때문에, event 필드를 언제든지 원할때 마다 읽어올 수 있다.

사실 breaking changes라고는 했지만, 페이스북에서는 어떠한 문제점도 찾지 못햇다. 리액트에 `e.persist()`가 존재하긴 하지만, 실제로는 아무런 동작을 하지 않는다는 것을 알아두길 바란다.

### Effect Cleanup Timing

`useEffect` 라이프 사이클 메서드의 Cleanup 타이밍을 일관되게 동작하도록 만들고 있다.

```javascript
useEffect(() => {
  // effect
  return () => {
    // cleanup
  }
}
```

대부분의 경우 스크린 업데이트를 지연시킬 필요가 없으므로, `useEffect` cleanup은 변경사항이 스크린에 반영된 직후 비동기적으로 작동하도록 변경되었다. (스크린 엽데이트를 지연시켜야 하는 경우 `useLayoutEffect`)

기존에 `useEffect` cleanup은 16버전에서는 동기적으로 실행하기 위해 사용되었다. `componentWillUnMount`와 유사하게, 탭 변경과 같은 큰 사이즈의 페이지 전환에서 성능 저하를 유발한다.

이 변경사항 이후, 컴포넌트가 언마운트 되면 화면이 업데이트 된후 `cleanup`이 실행된다.

또한, 돔 트리에 위치한 순서와 같은 순서로 실행되도록 보장한다.

그러나 위의 변화로 인해 아래와 같은 코드에서 에러가 나는 경우가 있다.

```javascript
useEffect(() => {
  someRef.current.someSetupMethod()
  return () => {
    someRef.current.someCleanupMethod()
  }
})
```

`someRef.current` 는 mutable 하기 때문에, cleanup 함쇼ㅜ가 실행되는 순간에 null 이 될 가능성이 있다. 따라서 아래와 같이 고쳐줘야 한다.

```javascript
useEffect(() => {
  const instance = someRef.current
  instance.someSetupMethod()
  return () => {
    instance.someCleanupMethod()
  }
})
```

### undefined를 return 할 경우 일관되게 에러 발생

16버전 이하에서는, 모든 컴포넌트에서 undefined를 리턴할 경우 항상 에러를 발생했다. 하지만 코딩 실수로 인해, `forwardRef` `memo`컴포넌트 에서는 이러한 에러 처리가 누락되어 있었는데, 이제 부터 에러처리가 추가되었다.

```javascript
let Button = forwardRef(() => {
  // We forgot to write return, so this component returns undefined.
  // React 17 surfaces this as an error instead of ignoring it.
  ;<button />
})

let Button = memo(() => {
  // We forgot to write return, so this component returns undefined.
  // React 17 surfaces this as an error instead of ignoring it.
  ;<button />
})
```

렌더링을 아무것도 하지 않기 위해서는, null을 리턴하면 된다.

### Native Component Stacks

브라우저에서 에러 발생시, 자바스크립트 함수이름과 해당하는 위치를 추적하여 에러 메시지에서 보여줬지만, javascript stack이 리액트 트리 구조를 파악하고 진단하기에 충분하지 않았다. 특히 리액트는 소스코드에서 함수가 어디 선언되있는지 모르기 떄문에, 콘솔에서 에러를 클릭해서 살펴볼 수 없었다. 또한, 프로덕션 모드에서 더욱 쓸모가 없었다. 일반적인 자바스크립트 스택이 소스맵과 함께 함수명을 복구할 수 있었던 반면, 리액트 스택은 번들 사이즈(에러를 안볼 것인지) 와 프로덕션 스택 (에러를 볼 것인지) 사이에서 선택을 해야 했다.

이를 해결하기 위해, 새로운 메커니즘을 사용하여 component stacks를 생성한다. 이를 통해 프로덕션 환경에서도 리액트 컴포넌트 스택을 추적할 수 있다.

### Private Exports 삭제

- React Native for Web에서 사용하던 private exports를 삭제하였다.
- `ReactTestUtils.SimulateNative` 헬퍼 메소드를 삭제하였다.
