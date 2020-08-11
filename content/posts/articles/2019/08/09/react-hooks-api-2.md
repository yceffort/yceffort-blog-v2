---
title: React Hooks Api (2)
date: 2019-08-12 08:03:51
published: true
tags:
  - react
  - javascript
description: "### useReducer ```javascript const [state, dispatch] =
  useReducer(reducer, initialArg, init); ```  `useState`의 대체 함수다. 다수의 하윗 값을 만드는
  복잡한 로직, 혹은 다음 state가 이전 state의 의존적인 경우에 쓴다. 뭐가 뭔지 모르겠으니까 예제를 보자.  ..."
category: react
slug: /2019/08/09/react-hooks-api-2/
template: post
---
### useReducer

```javascript
const [state, dispatch] = useReducer(reducer, initialArg, init);
```

`useState`의 대체 함수다. 다수의 하윗 값을 만드는 복잡한 로직, 혹은 다음 state가 이전 state의 의존적인 경우에 쓴다. 뭐가 뭔지 모르겠으니까 예제를 보자.

useState를 쓰기전

```javascript
function Counter({ initialCount }) {
  const [count, setCount] = useState(initialCount);
  return (
    <>
      Count: {count}
      <button onClick={() => setCount(initialCount)}>Reset</button>
      <button onClick={() => setCount(prevCount => prevCount + 1)}>+</button>
      <button onClick={() => setCount(prevCount => prevCount - 1)}>-</button>
    </>
  );
}
```

```javascript
const initialState = { count: 0 };

function reducer(state, action) {
  switch (action.type) {
    case "increment":
      return { count: state.count + 1 };
    case "decrement":
      return { count: state.count - 1 };
    default:
      throw new Error();
  }
}

function Counter() {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <>
      Count: {state.count}
      <button onClick={() => dispatch({ type: "increment" })}>+</button>
      <button onClick={() => dispatch({ type: "decrement" })}>-</button>
    </>
  );
}
```

```javascript
function init(initialCount) {
  return { count: initialCount };
}

function reducer(state, action) {
  switch (action.type) {
    case "increment":
      return { count: state.count + 1 };
    case "decrement":
      return { count: state.count - 1 };
    case "reset":
      return init(action.payload);
    default:
      throw new Error();
  }
}

function Counter({ initialCount }) {
  const [state, dispatch] = useReducer(reducer, initialCount, init);
  return (
    <>
      Count: {state.count}
      <button
        onClick={() => dispatch({ type: "reset", payload: initialCount })}
      >
        Reset
      </button>
      <button onClick={() => dispatch({ type: "increment" })}>+</button>
      <button onClick={() => dispatch({ type: "decrement" })}>-</button>
    </>
  );
}
```

예제를 보니 대충 감이 온다. dispatch를 통해서 state값에 변화를 주지 않고 state값 변화에 트리거를 줄 수 있고, 이 트리거를 이용해 여러개의 state에 변화를 줄 수도 있다.

### useCallback

메모이제이션된 콜백을 반환한다.

```javascript
const memoizedCallback = useCallback(() => {
  doSomething(a, b);
}, [a, b]);
```

동일한 콜백을 바라봐야하는 자식 컴포넌트에게 사용할 때 유용하다.

### useMemo

```javascript
const memoizedValue = useMemo(() => computeExpensiveValue(a, b), [a, b]);
```

메모이제이션된 값을 반환한다. `useMemo`는 의존성이 변경되었을 때만, 메모제이션된 값을 다시 계산할 것이다.

### useRef

```javascript
const refContainer = useRef(initialValue);

function TextInputWithFocusButton() {
  const inputEl = useRef(null);
  const onButtonClick = () => {
    // `current` points to the mounted text input element
    inputEl.current.focus();
  };
  return (
    <>
      <input ref={inputEl} type="text" />
      <button onClick={onButtonClick}>Focus the input</button>
    </>
  );
}
```

`useRef`는 `.current`에 변경가능한 ref값을 담을 수 있다. 부모 클래스에서 특정 dom객체를 계속 추적해야할 때 유용하다. 다만 `.current`를 변경한다고 해서 리렌더링이 일어나지는 않는다.
