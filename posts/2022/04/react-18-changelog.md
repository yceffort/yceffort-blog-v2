---
title: '리액트 v18 버전 톺아보기'
tags:
  - react
  - javascript
  - typescript
published: true
date: 2022-04-04 19:02:15
description: '큰거 왔다'
---

## Table of Contents

## Introduction

대규모 애플리케이션에서 버전업을 한다는 것은, 그것도 주로 사용하는 major framework의 major 버전 업을 하는 것은 꽤나 어려운 일이다. 지금도 잘 작동하고 있는 애플리케이션을 왜 업데이트 해야 하는지 개발자 부터 저 높은 어르신 까지 먼저 설득이 필요하다. 허락을 구했다면 breaking change가 있는지 살펴보고 있다면 수정해야 한다. 만약 수정 가이드가 있다면 다행이지만 없다면 코드를 하나씩 살펴보면서 고쳐야 한다. 또 고치는 데서만 끝나는 것이 아니다. regression test도 필요하고, 테스트 만으로는 못미더울 기획자나 QA 테스터 분께서 살펴보는 시간도 필요하다. 이런 저런 이유로 봤을 때 대다수의 많은 프로젝트들이 아직도 구형 버전에 머물러 있는 것은 그리 놀라운 일은 아니다. major 버전업은 누구에게나 피곤한 일이다.

그럼에도 개발자들은 항상 major 버전업에 귀기울일 필요는 있다. major 버전업은 분명 기능적으로든 성능적으로든 좋은 방향이 적용되어 있을 것이고, 이는 개발자들에게 좀 더 나은 개발 경험 내지는 고객들에게 더 좋은 애플리케이션 겅험을 제공해 줄 수 있다. 또 새로운 개발자를 유인할 수 있는 좋은 방법이기도 하다. jquery로 되어 있는 웹 애플리케이션과 최신의 자바스크립트 프레임워크와 섹시한 문법(?) 으로 작성되어 있는 웹 애플리케이션, 둘 중에 어떤 것을 개발하고 싶은지 열에 아홉은 후자를 선호할 것이다.

웹 애플리케이션 시장의 큰 파이를 차지하고 있는 react의 18 버전이 나왔다. [공식 블로그 글](https://reactjs.org/blog/2022/03/29/react-v18.html)을 통해서 어떤 것이 변경되어있는지 대략적으로 알 수 있고 또 훌륭하게 정리해놓은 블로그 글도 여기저기 많다. 하지만 조금 더 깊게 공부해보고자 [공식 CHANGELOG](https://github.com/facebook/react/blob/main/CHANGELOG.md#1800-march-29-2022)를 보고, 직접 사용해보고, 요약해 보고자한다.

## New Feature

### React

#### `useId`

`useId`는 클라이언트와 서버간의 hydration의 mismatch를 피하면서 유니크 아이디를 생성할 수 있는 새로운 훅이다. 이는 주로 고유한 `id`가 필요한 접근성 API와 사용되는 컴포넌트에 유용할 것으로 기대된다. 이렇게 하면 React 17 이하에서 이미 존재하고 있는 문제를 해결할 수 있다. 그리고 이는 리액트 18에서 더 중요한데, 그 이유는 새로운 스트리밍 렌더러가 HTML을 순서에 어긋나지 않게 전달해 줄 수 있기 떄문이다.

아이디 생성 알고리즘은 [여기](https://github.com/facebook/react/pull/22644)에서 살펴볼 수 있다. 아이디는 기본적으로 트리 내부의 노드의 위치를 나타내는 base 32 문자열이다. 트리가 여러 children으로 분기 될때 마다, 현재 레벨에서 자식 수준을 나타내는 비트를 시퀀스 왼쪽에 추가하게 된다.

```jsx
import Head from 'next/head'
import styles from '../styles/Home.module.css'
import { useId } from 'react'
import Child from '../src/components/child'
import SubChild from '../src/components/SubChild'

export default function Home() {
  const id = useId()
  return (
    <>
      <div className="field">Home: {id}</div>
      <SubChild />
      <SubChild />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
      <Child />
    </>
  )
}
```

```jsx
import { useId } from 'react'

export default function Child() {
  const id = useId()
  return <div>child: {id}</div>
}
```

```jsx
import { useId } from 'react'
import Child from './child'

export default function SubChild() {
  const id = useId()

  return (
    <div>
      Sub Child:{id}
      <Child />
    </div>
  )
}
```

```
Home: :r0:
Sub Child::r1:
child: :r2:
Sub Child::r3:
child: :r4:
child: :r5:
child: :r6:
child: :r7:
child: :r8:
child: :r9:
child: :ra:
child: :rb:
child: :rc:
child: :rd:
child: :re:
child: :rf:
child: :rg:
child: :rh:
```

자세한 알고리즘을 알고 싶다면, 앞서 언급한 PR을 참고하면 도움이 될 것 같다.

#### `startTransition` `useTransition`

이 두 메소드를 사용하면 일부 상태 업데이트를 긴급하지 않은 것 (not urgent)로 표시할 수 있다. 이것으로 표시되지 않은 상태 업데이트는 긴급한 것으로 갖누된다. 긴급한 상태 업데이트 (input text 등)가 긴급하지 않은 상태 업데이트 (검색 결과 목록 렌더링)을 중단할 수 있다.

상태 업데이트를 긴급한것과 긴급하지 않은 것으로 나누어 개발자에게 렌더링 성능을 튜닝하는데 많은 자유를 주었다고 볼 수 있다.

```javascript
function App() {
  const [resource, setResource] = useState(initialResource)
  const [startTransition, isPending] = useTransition({ timeoutMs: 3000 })
  return (
    <>
      <button
        disabled={isPending}
        onClick={() => {
          startTransition(() => {
            const nextUserId = getNextId(resource.userId)
            setResource(fetchProfileData(nextUserId))
          })
        }}
      >
        Next
      </button>
      {isPending ? 'Loading...' : null} <ProfilePage resource={resource} />
    </>
  )
}
```

- `startTransition`는 함수로, 리액트에 어떤 상태변화를 지연하고 싶은지 지정할 수 있다.
- `isPending`은 진행 여부로, 트랜지션이 진행중인지 알 수 있다.
- `timeoutMs`로 최대 3초간 이전 화면을 유지한다.

이를 활용하면, 버튼을 눌러도 바로 로딩상태로 전환되는 것이 아니고 이전화면에서 진행상태를 볼 수 있게 된다.
