---
title: 'useEffect와 메모리 누수'
tags:
  - react
  - javascript  
published: true
date: 2021-02-25 22:46:17
description: 'https://overreacted.io/a-complete-guide-to-useeffect/ 도 시간나면 읽어보세용'
---

아래 코드는 일반적으로 `useEffect`를 활용해서 데이터를 가져오는 방식이다.

```javascript
import React, { useEffect } from 'react';

export default function App() {
  const [todo, setTodo] = useState(null);
  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch('https://jsonplaceholder.typicode.com/todos/1');
      const newData = await response.json();
      setTodo(newData);
    };
    fetchData();
  }, []);
  if (data) {
    return <div>{data.title}</div>;
  } else {
    return null;
  }
}
```

dependency에 아무것도 넣지 않음으로써, 딱 한번만 실행되게 끔하고 싶었지만, 이는 여전히 레이스 컨디션과 메모리 누수에 취약하다. 만약 서버에서 응답이 오는 시간이 길어졌고, 그 사이에 컴포넌트가 unmount 되었다고 생각해보자. 컴포넌트는 사라졌지만, 여전히 요청은 대기 중이다. 그리고 요청이 온다면, `setTodo`에 값을 넣을 것이며, 리액트는 이제 아래와 같은 경고문을 내뱉을 것이다.

> Can’t perform a React state update on an unmounted component. This is a no-op, but it indicates a memory leak in your application. To fix, cancel all subscriptions and asynchronous tasks in a useEffect cleanup function.

마찬가지로, `id`를 의존성 목록에 넣어서 처리하는 경우도 있을 수 있다.

```javascript
import React, { useEffect } from 'react';
export default function App( {id} ) {
  const [todo, setTodo] = useState(null);
  useEffect(() => {
    const fetchData = async () => {
      const response = await fetch(`https://jsonplaceholder.typicode.com/todos/${id}`);
      const newData = await response.json();
      setTodo(newData);
    };
    fetchData();
  }, [id]);
  if (data) {
    return <div>{data.title}</div>;
  } else {
    return null;
  }
}
```

이 경우에도 마찬가지로, ID가 변경되었지만 요청은 여전히 오지 않는 경우 위와 같은 문제가 있을 수 있다.

## 해결책

```javascript
useEffect(() => {
  let isComponentMounted = true;
  const fetchData = async () => {
    const response = await fetch('https://jsonplaceholder.typicode.com/todos/1');
    const newData = await response.json();
    if(isComponentMounted) {
      setTodo(newData);
    }
  };
  fetchData();
  return () => {
    isComponentMounted = false;
  }
}, []);
```

unmount가 될 시에 요청이 늦게 와도 `setTodo`를 방지함으로써 문제를 해결할 수 있다. 그러나 물론 백그라운드에서는 여러개의 요청이 날라가고 있기 때문에 레이스 컨디션 문제가 발생할 수는 있다. 그래도 어쨌든, 마지막 요청의 결과만 UI에 표시된다.

더욱 확실한 방법은, [http fetch를 취소하는 `AbortController`를 사용하는 것이다.](https://developer.mozilla.org/ko/docs/Web/API/AbortController)

```javascript
useEffect(() => {
  let abortController = new AbortController();
  const fetchData = async () => {
    try {
      const response = await fetch('https://jsonplaceholder.typicode.com/todos/1', {
          signal: abortController.signal,
        });
    const newData = await response.json();
      setTodo(newData);
    }
    catch(error) {
        if (error.name === 'AbortError') {
        // requset를 abort하는 과정에서 에러 발생
      }
    }
  };
  fetchData();
  return () => {
    abortController.abort();
  }
}, []);
```

unmount가 되면 cleanup을 통해서 요청을 중단시켰다. 물론, `AbortController`를 사용하기 위해서는 polyfill도 필요할 것이다.