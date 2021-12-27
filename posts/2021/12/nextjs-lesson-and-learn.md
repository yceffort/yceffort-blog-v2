---
title: 'nextjs를 적용하면서 알게된 몇가지 사실들'
tags:
  - javascript
  - nextjs
published: true
date: 2021-12-20 16:55:17
description: '아 집에 가고 싶다'
---

## Table of Contents

## Introduction

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

사실 이는 조금만 깊게 생각해보면 당연한 사실이다. `next/router`는 Context API를 내부적으로 사용하고 있고, `router.*`을 실행하는 순간 내부의 상태 값을 바꾸기 때문에 필연적으로 리액트의 리렌더링을 발생시킬 것이다. ~~내가 생각이 짧았다.~~

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

<!-- ## getInitialProps에서 사용자의 데이터를 다뤄도 될까?

앱에 사용자가 최초 접근 시, 그러니까 `getIntialProps` 가 서버에서 실행될 때 사용자와 관련된 정보를 불러오고 그것을 애플리케이션 전체 라이프사이클에서 persistent하게 사용하고 싶었다. 그러니까, 대략 이런 코드였다.

```jsx
import App from 'next/app';
import '../styles/globals.css';

function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />;
}

MyApp.getInitialProps = async (appContext) => {
  const appProps = await App.getInitialProps(appContext);
  const {
    ctx: { req },
  } = appContext;

  // 서버사이드라면
  if (req) {
    // req에 있는 쿠키든 뭐든 활용해서 유저정보를 가져옴.
    const user = await fetchUserInfo({req})

    // 서버사이드에서는 유저정보를 내려준다.
    return { ...appProps, user}
  }

  return { ...appProps };
};

export default MyApp;
```

결론적으로, 이런 코드는 해서는 안됬었다. 그 이유를 이제 살펴보자. 

`fetchUserInfo`는 물론 빠른 비동기 함수겠지만, 어쩄거나 그 속도에는 차이가 있을 수 밖에 없다. 만약 이 API의 응답속도가 조금씩 다르다면 어떻게 될까? -->


## 환경변수 쓰기 전에 잘 점검하기

| Method              | Set at | Available in Next.js client side rendered code (browser) | Available in Next.js server side rendered code | Available in Node.js | Notes |
|---------------------|--------|----------------------------------------------------------|------------------------------------------------|----------------------|-------|
| .env                |   both     |                                                          |                      ✔️                          |        `process.env`를 구조분해할당하거나               |   `process.env`를 구조분해할당하거나 동적으로 접근할 수 없음.    |
| NEXT_PUBLIC_ .env   |   buildtime     |    ✔️                                                       |               ✔️                                 |                      |    `process.env`를 구조분해할당하거나 동적으로 접근할 수 없음.   |
| env next.config.js  |   buildtime     |    ✔️                                                      |                  ✔️                              |                      |     `process.env`를 구조분해할당하거나 동적으로 접근할 수 없음.  |
| publicRuntimeConfig |   runtime     |     ✔️                                                     |                 ✔️                               |                      |  `SSR`을 사용하는 페이지에 필요     |
| serverRuntimeConfig |   runtime     |                                                          |                   ✔️                             |                      |       |
| process.env         |   runtime     |                                                          |                                                |     ✔️                 |       |

환경 변수에 대한 설정도 잘 확인해야 한다. https://nextjs.org/docs/api-reference/next.config.js/runtime-configuration

`publicRuntimeConfig`를 사용하기 위해서는 `_app.getInitialProps`를 꼭 사용해야 한다. 초기 환경 세팅 할때, 혹은 배포 준비를 할 때 이것 때문에 헷갈리는 경우가 많으므로 주의를 요한다.

## SWC?

SWC가 러스트로 작성되어 타입스크립트나 자바스크립트를 굉장히 빠르게 컴파일한다는 사실 때문에 많은 주목을 받고 있었는데, 이번에 nextjs 12에 swc가 도입되면서 많은 관심을 끌고 있는 것 같았다. 실제로 도입을 해볼까 하고 고민을 했었는데, 결론적으로 도입하지는 않았다. 

본격적으로 적용하기에 앞서, 일단 돌아가고 있는 코드 베이스로도 안되는 문제가 많았고, 여러 다른 개발자로 부터도 이런저런 이슈가 많다는 이야기를 들어서 선뜻 적용하기 망설이고 있었다. (이 블로그는 적용되어 있다.)

- https://github.com/swc-project/swc/releases 패치가 123까지 있을 정도로 살벌하게 수정중,,
- https://github.com/vercel/next.js/issues?q=label%3A%22area%3A+SWC+transforms%22+ SWC는 아직도 이슈가 계속 나오고 있는듯,,

결론은 아직 시기상조 인 것 같다는 생각이다. 그러나 개발자 분이 워낙 능력도 출중하시고, 또 전폭적인 지원도 받고 계시니 내년 이맘 때 쯤이면 아마 babel을 걷어내고 모두가 SWC를 쓰고 있을지도 모른다.

## 잘 알고 있는 줄 알았는데..

nextjs로 블로그도 만들고, 실제 서비스 되고 있는 애플리케이션도 개발하면서 어느 정도 잘 알고 있다고 생각했었는데 대규모 애플리케이션을 만들면서, 그리고 여기에 mobx + k8s를 얹으면서 나도 몰랐던 이슈들이 터지는 것을 볼 수 있었다. (그러면서 어느정도 nextjs에 대한 신뢰도 깨지기도 했고,,)

개인적으로 서버사이드 렌더링 프레임워크를 만들어보자라는 계획이 있었는데, react 18이 나오면 해야지 하면서 차일피일 미루고 있었는데 내년에는 꼭 다시 시도해봐야겠다. (라고는 하지만 또 18 나올때까지 뭉개고 있겠지,,,)