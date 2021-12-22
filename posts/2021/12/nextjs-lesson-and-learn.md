---
title: 'nextjs를 적용하면서 알게된 몇가지 사실들'
tags:
  - javascript
  - nextjs
published: true
date: 2021-12-20 16:55:17
description: '아 집에 가고 싶다'
---

nextjs를 본격적으로 쓴 것은 2~3년 전부터이지만, 이 정도로 대규모 프로젝트에 써본 것은 처음이었다. 이전까지는 nextjs에 대해 어느정도 알고 있다고 자부했었지만, 본격적으로 쓰고 보니 굉장히 모르는 사실들이 많았다는 것을 꺠달았다. 다시는 시행착오를 겪지 않기 위해 nextjs를 쓰면서 배운 것들을 몇가지 정리해두려고 한다.

## shallow routing은 page 리렌더링을 야기한다.

nextjs에서 routing이 일어나면 `getServerSideProps`, `getStaticProps`, `getInitialProps` 를 야기한다. https://nextjs.org/docs/routing/shallow-routing 그러나 이를 실행시키지 않고 현재 URL을 업데이트 하는 것이 shallow routing이다.

```javascript
import { useEffect } from 'react'
import { useRouter } from 'next/router'

function Page() {
  const router = useRouter()

  useEffect(() => {
    // Always do navigations after the first render
    router.push('/?counter=10', undefined, { shallow: true })
  }, [])

  useEffect(() => {
    // The counter changed!
  }, [router.query.counter])
}

export default Page
```

단순히 URL을 업데이트 하는 용도로 잘 쓰고 있었는데, 알고 보니 `router.push` 든 `router.relace`든 일어나면 해당 페이지가 리렌더링 된다는 사실을 알게 됐다.

https://github.com/vercel/next.js/discussions/18072

사실 이는 조금만 깊게 생각해보면 당연한 사실이다. `next/router`는 Context API를 내부적으로 사용하고 있고, `router.*`을 실행하는 순간 내부의 상태 값을 바꾸기 때문에 필연적으로 리액트의 리렌더링을 발생시킬 것이다. 이는 내가 생각이 짧았다.

### 해결책

해결책은 `window.history.replaceState`를 사용하는 것이다. history에 replaceState를 하는 것은 리액트의 상태를 건드는게 아니고 리액트와 별개인 페이지의 히스토리를 건드는 것이 기 때문에 리렌더링이 발생하지 않을 것이다.

```javascript
window.history.replaceState(
  window.history.state,
  '',
  window.location.pathname + '?' + `whatever=u_want`,
)
```

## getServerSideProps와 \_app.getInitialProps와의 관계

`getServerSideProps`는 무조건 서버에서 실행되는 코드로, 서버사이드 렌더링 시에 필요한 데이터를 미리 필요한 데이터를 불러올 때 쓰인다. `_app.getInitialProps`는 최초에 앱이 렌덜이되거나, 클라이언트 라우팅이 일어나는 순간에 실행된다. https://nextjs.org/docs/advanced-features/custom-app

- Persisting layout between page changes
- Keeping state when navigating pages
- Custom error handling using componentDidCatch
- Inject additional data into pages
- Add global CSS

그런데, `getServerSideProps` 가 수행되면, `_app.getInitialProps`가 실행된다는 사실을 알게되었다.

### app

```javascript
import App from 'next/app'
import '../styles/globals.css'

function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />
}

MyApp.getInitialProps = async (appContext) => {
  const appProps = await App.getInitialProps(appContext)

  console.log('getInitailProps!')

  return { ...appProps }
}

export default MyApp
```

### index

```javascript
import { useRouter } from 'next/dist/client/router'

export default function Home() {
  const router = useRouter()

  function handleClick() {
    router.replace(router.asPath)
  }

  return <button onClick={handleClick}>Replace!</button>
}

export function getServerSideProps() {
  console.log('getServerSideProps')
  return {
    props: {}, // will be passed to the page component as props
  }
}
```

버튼을 누르면

```
getInitailProps!
getServerSideProps
getInitailProps!
getServerSideProps
getInitailProps!
getServerSideProps
```

`getInitialProps`가 실행되는 것을 알 수 있다. 이는 의도한 동작인 걸까? 그냥 나는 `getServerSideProps`만 재 호출하고 싶은 건데, (새로고침 등을 이유로) `getInitialProps`까지 호출해야 할까? 사실 지금 생각해보니 이것도 어떻게 보면 당연한 것 같기도하다. 🤔 의도야 어쩄든 라우팅이 일어나는 행위고, 라우팅에는 `getServerSideProps`가 수반되어야 하니까...?

아무튼, 이 상황을 막고 싶다면 아래와 같은 조건문을 추가해주면 된다.

```javascript
import App from 'next/app'
import '../styles/globals.css'

function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />
}

MyApp.getInitialProps = async (appContext) => {
  const appProps = await App.getInitialProps(appContext)
  const {
    ctx: { req },
  } = appContext

  if (req?.url.startsWith('/_next')) {
    // serverSideProps로 호출된 경우 URL이 /_next로 시작함.
    // EX: /_next/data/development/index.json
  }

  return { ...appProps }
}

export default MyApp
```
