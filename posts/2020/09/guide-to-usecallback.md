---
title: useCallback 사용 가이드
tags:
  - react
  - javascript
published: true
date: 2020-09-22 23:15:11
description: '아직도 useCallback으로 고통 받다니'
category: react
template: post
---

조금 부끄러운 이야기지만, 1년이 넘도록 리액트를 쓰면서 아직까지도 `useCallback`에 대한 정확한 개념이 서 있지 않다. 그래, `useCallback`이 함수를 재사용하는 걸 알겠는데, 그래서 어쩌라고? 라는 나를 위해서 다시한번 정리해본다.

조금 다른 미친 이야기로 넘어와서, 만약 모든 콜백 함수를 memoize 한다면, callback 함수를 사용하는 모든 자식 컴포넌트들이 불필요한 리렌더링을 막을 수 있지 않을까? 당연히 이 글을 읽는 훌륭한 개발자 분들은 말도 안되는 것임을 알 것이고, 이는 성능에 악 영향을, 그리고 컴포넌트를 느리게 한다는 것을 알 것이다.

본론으로 들어가기전에, 함수의 동일성 체크에 대해 알아보자.

```javascript
function sum() {
  return (a, b) => a + b
}

const sum1 = sum()
const sum2 = sum()

sum1(1, 2) // 3
sum2(1, 2) // 3

sum1 === sum2 // false
sum2 === sum2 // true
```

`sum1`과 `sum2`는 동일한 코드 소스를 공유하고 있지만, 리턴하는 오브젝트가 다르다. 따라서 두 개를 비고 하면 `false`를 리턴한다. 당연한 이야기이지만, object는 오로지 자기 자신만 동일하다.

이와 비슷하게, 리액트 컴포넌트에서도 동일한 소스코드로 다른 함수 인스턴스가 생성되는 경우가 종종 있다.

```javascript
import React from 'react'

function MyComponent() {
  // handleClick 은 렌더링 될 때 마다 새로 생성된다.
  const handleClick = () => {
    console.log('Clicked!')
  }

  // ...
}
```

`handleClick`은 `MyComponent`가 새롭게 렌더링 될 때마다, 다른 함수 오브젝트가 된다. 다행히도, 인라인 함수를 만드는 비용은 그다지 비싸지 않아서, 매번 새롭게 만드는 것은 큰 비용이 아니다. 다시 말해, 컴포넌트 내부에 있는 몇개의 인라인 함수정도는 괜찮다.

그러나, 이와 반대로 한가지 함수 인스턴스를 유지해야 하는 경우가 있다.

1. `React.memo()` 또는 `shouldComponentUpdate)` 내부에 있는 컴포넌트가 callback prop을 받을 때
2. 함수가 다른 hooks 함수의 dependency로 사용될 때 `useEffect(..., [callback])`

이러한 경우에 `useCallback`이 빛을 발한다. `deps`에 같은 값이 주어질 때마다, hook은 렌더링 시에 완전히 같은 함수 인스턴스를 리턴한다.

```javascript
import React, { useCallback } from 'react'

function MyComponent() {
  // handleClick 은 완전히 같은 오브젝트다.
  const handleClick = useCallback(() => {
    console.log('Clicked!')
  }, [])

  // ...
}
```

`handleClick` 변수는 `MyComponent`가 렌더링 될 때마다 같은 콜백 함수를 가진 오브젝트를 갖게 된다.

컴포넌트가 큰 사이즈의 items을 렌더링 한다고 가정해보자.

```javascript
import React from 'react'
import useSearch from './fetch-items'

function BigList({ term, handleClick }) {
  const items = useSearch(term)

  const element = (item) => <div onClick={handleClick}>{item}</div>

  return <div>{items.map(element)}</div>
}

export default React.memo(BigList)
```

`items`가 정말 많다고 가정하고, 재 렌더링을 방지하지 위하여 `React.memo`를 사용했다. 만약 `BigList`에서 아이템을 클릭할 때 실행할 핸들러 함수가 필요하다고 가정해보자.

```javascript
import React, { useCallback } from 'react'

export default function Parent({ term }) {
  const handleClick = useCallback(
    (item) => {
      console.log('You clicked ', item)
    },
    [term],
  )

  return <BigList term={term} handleClick={handleClick} />
}
```

이렇게 하게 되면, `handleClick`은 `useCallback()`에 memozied 된다. `term`의 값이 같은 이상, `useCallback()`은 항상 같은 함수 인스턴스를 반환할 것이다.

심지어 `Parent`가 리렌더링 되더라도,`BigList`의 memoziation이 깨지지 않은 이상 항상 같은 함수를 리턴할 것이다.

이제 다시 앞선 예제로 와보자.

```javascript
import React, { useCallback } from 'react'

function MyComponent() {
  const handleClick = useCallback(() => {
    // handle the click event
  }, [])

  return <MyChild onClick={handleClick} />
}
```

이 경우는 어떤가? `MyComponent`가 렌더링 될 때 마다 `useCallback()` 훅이 호출 될 것이다. 내부적으로 리액트는 같은 오브젝트 함수를 리턴하기 위해 노력할 것이다. 그렇다고 할지라도, 인라인 함수는 여전히 매번 렌더링 될 때마다 만들어진다. 왜냐하면, `MyComponent`는 특별히 memozied 되지 않았기 때문에, 그 자체 만으로도 이미 매번 렌더링을 하게 된다.

`useCallback()`이 같은 함수 인스턴스를 보장해주지만, 이는 아무런 이득이 없다. 왜냐하면 최적화의 비용이 최적화 하지 않는 비용보다 더 크기 때문이다. 또한 코드의 복잡성도 증가하게 된다. `useCallback()`의 deps에 실제로 memozied할 callback에서 사용하는 것들을 반드시 넣어두어야 한다.

모든 최적화는 복잡성을 증가시킨다. 섣불리 추가된 최적화는 최적화된 코드를 여러번 변경할 수 있기 때문에 위험하다. [성능을 최적화 하기 전에 문제를 프로파일링 해야 한다.](https://wiki.c2.com/?ProfileBeforeOptimizing) `useCallback`의 적절한 예시는 메모화된 자식 컴포넌트에 제공되는 콜백 함수를 memozie하는 것이다.
