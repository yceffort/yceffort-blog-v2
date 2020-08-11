---
title: NextJS 1. Page & Route
tags:
  - typescript
  - javascript
  - react
published: true
date: 2020-03-12 02:39:10
description: 요즘 리액트를 쓰는 많은 프로젝트에서, SSR을 지원하기 위해 [nextjs](https://nextjs.org/)를
  쓰고 있다. 초기 로딩 속도나, SEO 지원 이슈 등 등 때문에 아무래도 SPA는 요즘 트렌드에서 많이 밀린 기분이다. 물론
  [razzle](https://github.com/jaredpalmer/razzle) 을 쓰거나 custom ser...
category: typescript
slug: /2020/03/nextjs-01-route/
template: post
---
요즘 리액트를 쓰는 많은 프로젝트에서, SSR을 지원하기 위해 [nextjs](https://nextjs.org/)를 쓰고 있다. 초기 로딩 속도나, SEO 지원 이슈 등 등 때문에 아무래도 SPA는 요즘 트렌드에서 많이 밀린 기분이다. 물론 [razzle](https://github.com/jaredpalmer/razzle) 을 쓰거나 custom server 로 맨 바닥에 해딩하는 방법도 있지만 여기저기 컨퍼런스나 주변 사람들의 말을 들어보면 nextjs가 대세이긴 한 것 같다.

입사 이래로 nextjs를 쓰면서 별 생각 없이 썼던 것들이 많은데, 9.3 출시를 기념하여 이참에 하나씩 정리해보려고 한다.

```toc
tight: true,
from-heading: 1
to-heading: 3
```

## 1. Page

기본적으로, `pages/파일명.js|ts|tsx` 네이밍으로 파일을 만들면 `/파일명` 으로 라우팅을 할 수 있다. `pages/about.js`로 파일을 만들면 `/about`으로 접근이 가능하다.

다이나믹 라우트의 경우에도 비슷하다. `pages/디렉터리명/[id].js|ts|tsx`로 생성하게되면, `디렉토리명/id`로 접근 가능하다. 예를 들어 `pages/posts/[id].tsx`로 파일을 생성하면, `posts/1`, `posts/2` 와 같은 식으로 접근이 가능하다.

### pages/posts/[id].tsx

```typescript
import React from "react"
import { useRouter } from "next/router"

export default function Post() {
  const router = useRouter()
  const { id } = router.query
  return <div>Post id {id}</div>
}
```

임의로 선언한 id 는 위처럼 받아서 처리할 수 있다.

nested routes도 위와 마찬가지로 처리하면 된다.

## 2. Routing

Nextjs에서는 SPA와 유사한 클라이언트 사이드 라우팅을 지원한다. `Link`라고 불리는 컴포넌트를 활용하면, 클라이언트 사이드 라우팅을 할 수 있다.

```jsx
import Link from "next/link"

function Home() {
  return (
    <Link href="/">
      <a>Home</a>
    </Link>
  )
}
export default Home
```

nextjs 는 `Link`를 적절한 a 태그로 변환해 준다.

위에서 언급한 다이나믹 라우트의 경우에는, 처리하는 방식이 조금 다르다. `href`와 `as`를 전달해 주어야 한다.

- `href`: 디렉토리 명을 넘겨주면 된다. `/posts/[id]`
- `as`: 브라우저에 실제로 표시될 주소를 넘긴다. `/posts/1`

```jsx
import Link from "next/link"

function Home() {
  return (
    <ul>
      <li>
        <Link href="/posts/[id]" as="/posts/1">
          <a>To Post</a>
        </Link>
      </li>
    </ul>
  )
}

export default Home
```

## 3. Router

nextjs의 라우터 안에는 다음과 같은 정보가 포함되어 있다.

- `pathname`: (String) 현재 라우트
- `query`: (Object) object로 파싱한 query string
- `asPath`: (String) 실제로 브라우저에 표시되고 있는 path

그리고 아래와 같은 router api도 포함되어 있다.

### 3-1. Router Api

#### Router.push

클라이언트 사이드 트랜지션을 다룰 때 쓰는 api다.

```tsx
import Router from "next/router"
Router.push(url, as, options)
```

- `url`: 이동할 URL을 명시한다. 보통 `page`명을 넣는다
- `as`: 옵셔널 파라미터로, 브라우저에서 보여질 URL이다. 없으면 default로 `url`이 들어간다.
- `options`: 은 shallow만 옵션으로 가질 수 있다.
  - `shallow`: `getInitialProps`를 재실행하지 않고 현재 페이지의 라우트를 업데이트 한다. 기본값은 false다.

무슨 소리하는지 모르겠다. 예제로 알아보자.

`index.tsx`

```typescript
import React from "react"
import { useRouter } from "next/router"
import { NextPageContext } from "next"

export default function Index() {
  const { push } = useRouter()

  function pushOnlyUrl() {
    push("/posts/1")
  }

  function pushWithAs() {
    push("/posts/[id]?hello=world", "/posts/1")
  }

  function shallowPush() {
    push("/?counter=1", undefined, { shallow: true })
  }

  function notShallowPush() {
    push("/?counter=1")
  }

  function pushUrl() {
    push("/about")
  }

  function pushUrlAndAs() {
    push("/about", "/about")
  }

  return (
    <>
      <ul>
        <li>
          <button onClick={() => pushOnlyUrl()}>1번. Push only URL</button>
        </li>
        <li>
          <button onClick={() => pushWithAs()}>2번. Push with as</button>
        </li>
        <li>
          <button onClick={() => shallowPush()}>3번. shallow push</button>
        </li>
        <li>
          <button onClick={() => notShallowPush()}>
            4번. not shallow push
          </button>
        </li>
        <li>
          <button onClick={() => pushUrl()}>5번. push route</button>
        </li>
        <li>
          <button onClick={() => pushUrlAndAs()}>
            6번. push route with as
          </button>
        </li>
      </ul>
    </>
  )
}

Index.getInitialProps = function(_: NextPageContext) {
  console.log("getInitialProps of Index")

  return {}
}
```

`[id].tsx`

```typescript
import React from "react"
import { useRouter } from "next/router"
import { NextPageContext } from "next"

export default function Post() {
  const router = useRouter()

  console.log("Router", JSON.stringify(router))

  const { id } = router.query
  return <div>Post id {id}</div>
}

Post.getInitialProps = function({ req }: NextPageContext) {
  console.log("getInitialProps of Post")

  return {}
}
```

`about.tsx`

```typescript
import React from "react"
import { NextPageContext } from "next"

export default function About() {
  return <div>about page</div>
}

About.getInitialProps = function(_: NextPageContext) {
  console.log("getInitialProps of about")

  return {}
}
```

1번 버튼: getInitialProps가 서버에 찍힌다. 서버사이드에서 실행되었음을 알수가 있다. 1번 버튼 동작은 사용자가 브라우저에서 주소를 치고 들어오는 것과 동일하다.

```json
{
  "pathname": "/posts/[id]",
  "route": "/posts/[id]",
  "query": { "id": "1" },
  "asPath": "/posts/1",
  "components": {
    "/posts/[id]": { "props": { "pageProps": {} } },
    "/_app": {}
  },
  "isFallback": false,
  "events": {}
}
```

2번 버튼: getInitialProps가 클라이언트에 찍힌다. 클라이언트 사이드에서 실행되었음을 알수가 있다. 그리고 또한 url에서 보냈던 쿼리스트링이 사용자 브라우저 URL에는 감춰진 것을 알수 있다. 그러나 Post 컴포넌트에서 해당 값을 받아다가 쓸 수 있다.

```json
{
  "pathname": "/posts/[id]",
  "route": "/posts/[id]",
  "query": { "hello": "world", "id": "1" },
  "asPath": "/posts/1",
  "components": {
    "/": { "props": { "pageProps": {} } },
    "/_app": {},
    "/posts/[id]": { "props": { "pageProps": {} } }
  },
  "isFallback": false,
  "events": {}
}
```

3번 버튼: index의 getInitialProps가 실행되면서 쿼리스트링이 변했다.

4번 버튼: index의 getInitialProps가 실행되지 않고 쿼리스트링이 변했다.

5번과 6번 버튼: 다이나믹 라우트가 아니기 때문에, 동작이 동일하다. (getInitialProps가 클라이언트에 찍힘). 그러나 사용자가 주소를 직접 치고 들어간다면 서버사이드에 찍힐 것이다.

#### Router.Replace

Replace는 Push와 받는 파라미터도 동일하지만, 동작만 다르다. 이름에서 알 수 있는 것 처럼 Replace는 URL에 새로운 스택을 쌓지 않는다.

#### Router.beforePopState

몇 몇의 경우 (특히 커스텀 서버를 쓰는 경우) [popsState](https://developer.mozilla.org/en-US/docs/Web/API/Window/popstate_event) 요청을 받아서 라우트에서 액션이 일어나기 전에 무언가를 하고 싶을 수 있다.

> Window 인터페이스의 popstate 이벤트는 사용자의 세션 기록 탐색으로 인해 현재 활성화된 기록 항목이 바뀔 때 발생합니다.

`_app.tsx`

```typescript
function App({ Component, pageProps }: AppProps) {
  const router = useRouter()

  useEffect(() => {
    router.beforePopState(() => {
      console.log("beforePopState!!")
      return true
    })

    return () => {
      router.beforePopState(() => true)
    }
  }, [])
  return <Component {...pageProps} />
}
```

next의 routing이 아닌, 사용자가 히스토리를 직접 조작하는 행위 (뒤로가기, 앞으로가기 등)가 일어날 경우 해당 메소드가 호출된다. 만약 false를 리턴할 경우, Router는 `popState`를 처리하지 않는다. (주소는 바뀌지만 아무 일이 일어나지 않는다.)

#### Router.events

Router에서 일어나는 다양한 이벤트를 감지 할 수 있다.

여기서 url은 브라우저에 뜨는 url을 의미한다. 만약 as를 썼다면, 여기서 url값은 as 값이 될 것이다.

- `routerChangeStart(url)`: route가 변하기 시작할 때
- `routerChangeComplete(url)`: route의 변화가 끝났을 때
- `routerChangeError(err, url)`: route가 바뀌는 과정에서 에러가 나거나, route 로딩이 취소되었을 때
  - `err.cancelled`: 네비게이션이 취소되었는지 여부
- `beforeHistoryChange(url)`: 브라우저 히스토리가 바뀌기 전에
- `hashChangeStart(url)`: 해쉬값이 변할 때
- `hashChangeComplete(url)`: 해쉬값이 다 변하고 난 뒤 에

```typescript
useEffect(() => {
  router.events.on("routeChangeStart", as => {
    console.log("routeChangeStart", as)
  })
}, [])
```
