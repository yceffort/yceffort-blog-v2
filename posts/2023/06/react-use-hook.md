---
title: '리액트의 신규 훅, "use"'
tags:
  - use
published: true
date: 2023-06-13 23:51:18
description: '상황에 따라 이름이 변경되거나 사라질 수도 있습니다'
---

# Table of Contents

## 서론

리액트에는 새로운 기능을 제안할 수 있는 공식적인 창구인 https://github.com/reactjs/rfcs 저장소가 존재한다. 이 저장소는 리액트에 필요한 새로운 기능 내지는 변경을 원하는 내용들을 제안하여 리액트 코어 팀의 피드백을 받을 수 있는데, 이렇게 제안된 이슈 중에는 리액트 코어 팀이 직접 제안하여 리액트 커뮤니티 개발자들의 의견을 들어보는 이슈도 존재한다.

- 서버 컴포넌트: https://github.com/reactjs/rfcs/blob/main/text/0188-server-components.md
- 서버 컴포넌트 모듈 컨벤션: https://github.com/reactjs/rfcs/blob/main/text/0227-server-module-conventions.md

이 중에 아직 머지되지는 않았지만 한가지 흥미로운 내용이 존재하는데, 바로 `use`라고 하는 새로운 훅이다. 이 훅은 이후에 설명하겠지만 이전의 훅과는 여러가지 차이점이 있는데, 그중에 하나는 조건부로 호출될 수 있다는 것이다. 이 훅도 예전부터 PR로 올라와 있어서 언제쯤 머지되는지 눈여겨 보고 있었는데 👀 도대체가 머지될 기미가 보이지 않아서 굉장히 의아한 차였다. 알고 보니 해당 proposal 을 만든 사람이 [meta에서 vercel로 이적하였고](https://github.com/reactjs/rfcs/pull/229#issuecomment-1427067863) (🤪) 이 과정에서 뭔가 이 작업이 붕뜬게 아닌가 하는 추측아닌 추측을 혼자 해봤다. 그러던 차에 리액트 카나리아 버전에서 `use`훅의 존재를 확인하게 되었다.

![react-use](./images/react-use.png)

https://www.npmjs.com/package/react/v/18.3.0-next-1308e49a6-20230330?activeTab=code

왠지 조만간 `use` 훅이 정식으로 등장할 날이 머지 않은 것 같아 이 참에 한번 다뤄보려고 한다. [react rfc에 있는 `First class support for promises and async/await`](https://github.com/reactjs/rfcs/pull/229) 을 읽어보고 `use`훅의 실체는 무엇인지 알아보자.

## 서버 컴포넌트의 등장

서버 컴포넌트의 등장으로 인해, 이제 다음과 같이 `async`한 컴포넌트를 만드는 것이 가능해졌다.

```javascript jsx
export async function Note({ id, isEditing }) {
  const note = await db.posts.get(id)
  return (
    <div>
      <h1>{note.title}</h1>
      <section>{note.body}</section>
      {isEditing ? <NoteEditor note={note} /> : null}
    </div>
  )
}
```

이와 같이 서버 자원에 직접 접근하여 서버의 데이터를 불러오는 서버 컴포넌트는 리액트 팀에서도 권장하는 방법이지만, 한가지 치명적인 사실은 대부분의 훅을 사용할 수 없다는 것이다. 물론, 서버 컴포넌트는 대부분 상태를 저장할 수 없는 기능에만 제한적으로 쓰이기 때문에 `useState`등은 필요하지 않을 것이며, `useId` 와 같은 훅은 서버에서도 여전히 사용 가능하다.

## 그렇다면 클라이언트 컴포넌트는?

이제 서버에서 쓰이는 함수형 컴포넌트가 `async`가 가능해진다... 라는 사실은 한가지 의문점을 갖게 한다. 그렇다면 클라이언트 컴포넌트가 비동기 함수가 되는 것은 불가능한 것인가? 지금까지 우리는 클라이언트 컴포넌트에서 비동기 처리를 하기 위해서는 `useEffect` 내에 비동기 함수를 선언하여 실행하는 것이 고작이었다. 그나마도, `useEffect`의 콜백 함수는 이러저러한 이유로 비동기가 되면 안되어 이상한 형태로(?) 만들어져서 사용되었다.

```javascript jsx
function ClientComponent() {
  useEffect(() => {
    async function doAsync() {
      await doSomething('...')
    }

    doAsync()
  }, [])

  return <>...</>
}
```

클라이언트 컴포넌트가 `async`하지 못한 것은 뒤이어 설명할 기술적 한계 때문이다. 그 대신, 리액트에서는 `use`라는 특별한 훅을 제공할 계획을 세운다.

## `use` 훅은 무엇인가?

`use` 훅의 정의에 대해서, rfc에서는 리액트에서만 사용되는 `await`이라고 비유했다. `await`이 `async`함수에서만 쓰일 수 있는 것 처럼, `use`는 리액트 컴포넌트와 훅 내부에서만 사용될 수 있다.

```javascript
import { use } from 'react'

function Component() {
  const data = use(promise)
}

function useHook() {
  const data = use(promise)
}
```

`use`훅은 정말 파격적이게도, 다른 훅이 할수 없는 일을 할 수 있다. 예를 들어 조건부 내에서 호출될 수도 있고, 블록 구문내에 존재할 수도 있으며, 심지어 루프 구문에서도 존재할 수 있다. 이는 `use`가 여타 다른 훅과는 다르게 관리되고 있음을 의미함과 동시에, 다른 훅과 마찬가지로 컴포넌트 내에서만 쓸 수 있다는 제한이 있다는 것을 의미한다.

```jsx
function Note({ id, shouldIncludeAuthor }) {
  const note = use(fetchNote(id))

  let byline = null
  // 조건부로 호출하기
  if (shouldIncludeAuthor) {
    const author = use(fetchNoteAuthor(note.authorId))
    byline = <h2>{author.displayName}</h2>
  }

  return (
    <div>
      <h1>{note.title}</h1>
      {byline}
      <section>{note.body}</section>
    </div>
  )
}
```

그리고 이 `use`는 `promise`뿐만 아니라 `Context`와 같은 다른 데이터 타입도 지원할 예정이다.

그렇다면 이 `use`는 왜 만들어졌는지 좀더 자세히 살펴보자.

### 왜 만들어 졌을까?

#### 자바스크립트 생태계와 원활한 통합
