---
title: '리액트의 새로운 훅, useEvent'
tags:
  - react
  - typescript
  - javascript
published: true
date: 2022-05-12 12:03:48
description: '트위터 염탐 시리즈 제1탄'
---

## Table of Contents

## 무엇이 문제인가?

리액트 개발을 어느정도 하다보면, 리렌더링을 거치는 과정에서 함수를 고정시키기 매우 어렵다는 것을 알 수 있다. 아래 예제를 살펴보자.

```jsx
function Chat() {
  const [text, setText] = useState('')

  const onButtonClick = () => {
    console.log(text)
  }

  return (
    <>
      <input value={text} onChange={(e) => setText(e.target.value)} />
      <button onClick={onButtonClick}>버튼</button>
    </>
  )
}
```

`setState`는 리액트 컴포넌트의 리렌더링을 야기하므로, `input`의 값을 바꿀 때 마다 `onButtonClick` 함수는 새로 생성될 필요가 없는 함수임에도 불구하고 `setText`가 일어날 때 마다 새로 생성 될 것이다.

![useEvent-1](./images/useEvent-1.png)

![useEvent-2](./images/useEvent-2.png)

![useEvent-3](./images/useEvent-3.png)

위 스크린샷은 input에 두번씩 타이핑을 하면서 크롬에서 메모리 스냅샷을 촬영한 화면인데, 매번 `onButtonClick` 함수가 가리키는 메모리 주소가 달라지는 것을 볼 수 있다.

이를 해결하기 위해서 쓰는 방법 중 하나는 바로 `useCallback`이다.

```jsx
function Chat() {
  const [text, setText] = useState('')
  const [clicked, setClicked] = useState(false)

  const onButtonClick = useCallback(
    function onButtonClickCallback() {
      console.log(text)
    },
    [text],
  )

  return (
    <>
      <input value={text} onChange={(e) => setText(e.target.value)} />
      <button onClick={onButtonClick}>버튼</button>
      <button onClick={() => setClicked((prev) => !prev)}>
        {clicked ? '클릭함' : '안함'}
      </button>
    </>
  )
}
```

`useCallback`을 사용하고, deps로 `text`를 추가하는 방법을 고려해볼 수 있다. 그러나 이 경우에도 마찬가지로 `text`가 바뀔 때 마다 새로운 함수가 생성된다는 사실에는 변함이 없다.

![useEvent-4](./images/useEvent-4.png)

> 다른 state 변경으로는 함수가 재생성되지 않고 고정되지만, 여전히 deps에 의존하고 있는 값이 수정되면 다시 생성된다는 것에는 변함이 없다.

그렇다고 deps를 제거하면, 저 핸들러는 항상 최초의 `text`값만 보게 될 것이다. 이러한 문제를 해결하기 위해 나온 것이 `useEvent`다.

## useEvent

> 주의: 2022-05-12 기준으로 `useEvent`는 아직 사용할 수가 없는 상태다. 순전히 RFC를 기준으로 작성된 글이라는 걸 염두해두길 바란다.

```javascript
function Chat() {
  const [text, setText] = useState('')

  // text가 변경되도 항상 같은 함수임
  const onClick = useEvent(() => {
    sendMessage(text)
  })

  return <SendButton onClick={onClick} />
}
```

`useEvent`의 중요한 특징 두가지는 다음과 같다.

- `deps`가 없음
- state인 `text`가 변경되도 함수를 재생성하지 않고 하나의 안정된 함수만을 사용하게 됨.
- 그럼에도 불구하고 항상 최신의 `text`를 바라볼 수 있음.
- 따라서, `Memoize`된 `<SendButton />`의 리렌더링을 막을 수 있음.

## `useEvent`를 사용하면 이벤트 핸들러가 변경되도 `useEffect`는 다시 호출되지 않는다.

```jsx
function Chat({ selectedRoom }) {
  const [muted, setMuted] = useState(false)
  const theme = useContext(ThemeContext)

  useEffect(() => {
    const socket = createSocket('/chat/' + selectedRoom)
    socket.on('connected', async () => {
      await checkConnection(selectedRoom)
      showToast(theme, 'Connected to ' + selectedRoom)
    })
    socket.on('message', (message) => {
      showToast(theme, 'New message: ' + message)
      if (!muted) {
        playSound()
      }
    })
    socket.connect()
    return () => socket.dispose()
  }, [selectedRoom, theme, muted]) // 이 deps 중 하나만 변경되도 다시 실행됨.
  // ...
}
```

위 컴포넌트의 문제는, `theme`이나 `muted`가 바뀌게 되면 `useEffect`가 다시금 실행된다는 것이다. `theme`과 `muted`는 `effect`안에 있으므로 이를 의존성에 선언해 주어야 하고, 이것이 바뀌면 다시 실행되는 구조를 가지게 된다.

물론 `deps`에서 제거하는 방식도 고려할 수 있다. 그러나 이 경우 `eslint-disable-line react-hooks/exhaustive-deps`를 사용해줘야 하며 (얼마나 자주 썼던지 다 외웠다.), 이를 잘못 쓸 경우 예기치 않은 오류를 만들어 낼 수 있는 위험을 감수해야 한다. (이경우 `theme`이 다크모드 등으로 변경되어도 새로운 `toast`를 그리지 못하게 될 것이다.)

다른 방법으로 `useCallback`을 사용하는 것도 있지만, 역시나 앞서 언급했던 것 처럼 `theme` `muted`가 바뀌면 함수의 identity가 변경된다는 사실에는 변함이 없다.

```jsx
function Chat({ selectedRoom }) {
  const [muted, setMuted] = useState(false);
  const theme = useContext(ThemeContext);

  // ✅ 재생성되지 않음
  const onConnected = useEvent(connectedRoom => {
    showToast(theme, 'Connected to ' + connectedRoom);
  });

  // ✅ 재생성되지 않음
  const onMessage = useEvent(message => {
    showToast(theme, 'New message: ' + message);
    if (!muted) {
      playSound();
    }
  });

  useEffect(() => {
    const socket = createSocket('/chat/' + selectedRoom);
    socket.on('connected', async () => {
      await checkConnection(selectedRoom);
      onConnected(selectedRoom);
    });
    socket.on('message', onMessage);
    socket.connect();
    return () => socket.disconnect();
  }, [selectedRoom]); // ✅ 룸이 변경될 때만 실행됨
```

`useEvent`를 사용하여 `onConnected`와 `onMessage`를 분리했다. 이렇게 함으로써, 앞서서 예상되었던 이슈들을 모두 해결할 수 있게 되었다. `useEvent`로 값들을 내재화하여 `useEffect`의 `deps`에서 제거할 수 있게 되었고, 함수도 재생성되지 않고 안정적인 값을 가질 수 있게 되었다. 그리고 여전히, `selectedRoom`에 의존함으로서 우리가 기존에 구현하고 싶었던 기능을 안정적으로 제공할 수 있게 되었다.

```jsx
const onConnected = useEvent((connectedRoom) => {
  console.log(selectedRoom) // 이미 useState를 거쳐서 업데이트 된 값
  showToast(theme, 'Connected to ' + connectedRoom) // 이벤트를 발생시킨 값
})
```

`useEvent`의 `props`로는 이 이벤트를 발생시킨 값을 받을 수 있다.

```jsx
function Chat({ selectedRoom }) {
  const [muted, setMuted] = useState(false)
  const theme = useContext(ThemeContext)

  const onConnected = (connectedRoom) => {
    showToast(theme, 'Connected to ' + connectedRoom)
  }

  const onMessage = (message) => {
    showToast(theme, 'New message: ' + message)
    if (!muted) {
      playSound()
    }
  }

  useRoom(selectedRoom, { onConnected, onMessage })
  // ...
}

function useRoom(room, events) {
  const onConnected = useEvent(events.onConnected) // ✅ Stable identity
  const onMessage = useEvent(events.onMessage) // ✅ Stable identity

  useEffect(() => {
    const socket = createSocket(room)
    socket.on('connected', async () => {
      await checkConnection(room)
      onConnected(room)
    })
    socket.on('message', onMessage)
    socket.connect()
    return () => socket.disconnect()
  }, [room]) // ✅ Re-runs only when the room changes
}
```

또 사용하는 곳에서 `useEvent`를 사용하여 wrapping하는 전략을 취할 수도 있다.

`useEvent`가 사용될 수 있는 또다른 예시를 살펴보자. 특정 페이지에 진입했을 때 로깅하는 컴포넌트를 구현한다고 가정해보자.

```jsx
function Page({ route, currentUser }) {
  useEffect(() => {
    logAnalytics('visit_page', route.url, currentUser.name)
  }, [route.url, currentUser.name])
  // ...
}
```

이는 얼핏보면 잘 작동하는 것 처럼 보인다. 그러나 사용자가 이름을 바꾸면 어떻게 될까? 사용자는 단순히 이름만 바꿨는데, 다른 사용자로 인식되어 (=`useEffect`가 실행되어) 다시 한번 로깅을 할 것이다.

```jsx
function Page({ route, currentUser }) {
  // ✅ Stable identity
  const onVisit = useEvent((visitedUrl) => {
    logAnalytics('visit_page', visitedUrl, currentUser.name)
  })

  useEffect(() => {
    onVisit(route.url)
  }, [route.url]) // ✅ Re-runs only on route change
  // ...
}
```

`useEvent`가 이러한 문제의 해결책이 될 수 있다. `onVisit` 의 props로 주소를 받으면, `currentUser.name`이 변경되었는지 상관없이 우리가 원하던 대로 로깅을 할 수 있게 된다.

## 어떻게 구현되어 있을까?

`useEvent`의 대략적인 구현으로는 아래와 같이 설명하고 있다.

```jsx
// 대략적인 동작
function useEvent(handler) {
  const handlerRef = useRef(null)

  // 실제 구현에서는, layout effect 보다도 먼저 실행된다.
  // 하지만 정확히 언제 실행될지는 아직 개발중
  useLayoutEffect(() => {
    handlerRef.current = handler
  })

  return useCallback((...args) => {
    // 실제 구현에서는, 렌더링중 호출되면 에러를 발생 시킬 것이다.
    // 즉 렌더링 중에는 이 함수는 처리되지 않아야 한다는 것을 의미한다.
    // 렌더링 중에 함수가 호출되지 않게 함으로써 이들의 identity를 안전하게 가져갈 수 있또록 한다.
    // 렌더링 중에는 호출할 수 없으므로, 렌더링에 영향을 주지 않고, input이 변경된더라도 변경할 필요가 없다.
    const fn = handlerRef.current
    return fn(...args)
  }, [])
}
```

이와 비슷한 코드가 존재한다.

```typescript
import { useLayoutEffect, useMemo, useRef } from 'react'

type Fn<ARGS extends any[], R> = (...args: ARGS) => R

const useEventCallback = <A extends any[], R>(fn: Fn<A, R>): Fn<A, R> => {
  let ref = useRef<Fn<A, R>>(fn)
  useLayoutEffect(() => {
    ref.current = fn
  })
  return useMemo(
    () =>
      (...args: A): R => {
        const { current } = ref
        return current(...args)
      },
    [],
  )
}

export default useEventCallback
```

[https://github.com/Volune/use-event-callback/blob/master/src/index.ts](https://github.com/Volune/use-event-callback/blob/master/src/index.ts)

위 설명에서 언급했던 내용을 거의 유사하게 구현해 두었다.

## 느낀점

- 일단 RFC 이기 때문에 이런게 있을 수도 있다 정도로 알아두면 될 것 같다. 그러나 여기저기 홍보하는 걸로 봐서는 이른 시일내에 추가될 듯
- 트위터에서 이것도 염탐하다가 공감하게 된 사실인데, 확실히 리액트는 어려워 지고 있는 것 같다. 초보자들에게 친화적인 프레임워크 라고 보기 어렵지 않을까 싶다. 물론 리렌더링이고 뭐고 간에 다 생각 안하고 만든다면 상관없지만.
- 리액트를 정확하게 이해하기 위해서는 자바스크립트의 기초를 정말정말 잘 이해헤야할 것 같다.
- vue, svelte 등은 이런 문제를 어떻게 해결하고 있을까? 너무 react-way로만 길들여져 있어서 다른 프레임워크는 어떻게 처리하고 있는지 궁금하다. 리액트의 어려움을 토로하는 트위터 쓰레드를 보면, 간간히 vue 광고하시는 분들도 있다. vue는 정말 이런 문제가 없는 것인가 궁금하다.
