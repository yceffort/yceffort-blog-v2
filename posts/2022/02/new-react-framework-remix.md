---
title: 'Remix nextjs와 비교하면서 살펴보기'
tags:
  - javascript
  - react
  - remix
  - nextjs
published: true
date: 2022-02-13 14:16:40
description: '늘 새로워 짜릿해 새로운게 또 나왔어'
---

## Table of Contents

## Introduction

[remix](https://remix.run/)는 새로운 리액트 기반 풀스택 웹 프레임워크다. 뭐 어떤 프레임워크고 어떻게 쓰는지는 remix 홈페이지에 잘 나와 있으므로, nextjs에서의 관점에서 remix는 어떤 웹 프레임워크고 무엇이 좋은지, 또 쓸만은 한지 한번 고민해보려고 한다.

## 클라이언트 - 서버 아키텍쳐

### nextjs

먼저 nextjs가 어떻게 애플리케이션을 구조화 하는지를 살펴보자. 일반적으로 nextjs에서는 클라리언트와 서버간의 통신을 위해서 클라이언트의 javascript에 의존하는 경우가 많다. 아래 예제를 살펴보자.

```jsx
// pages/contact.tsx

export default function ContactPage() {
  // form에 전송할 정보
  const [name, setName] = useState(null)
  const [email, setEmail] = useState(null)
  // form 제출 상태와 관련있는 상태 값
  const [submitting, setSubmitting] = useState(false)
  const [errors, setErrors] = useState(null)
  // 실제 클라이언트에 렌더링되는 form
  return (
    <form onSubmit={handleSubmit}>
      <input type="text" name="name" />
      {errors?.name && <em>Name is required</em>}
      <input type="text" name="email" />
      {errors?.email && <em>Email is required</em>}
      <button type="submit">Contact me</button>
    </form>
  )
  async function handleSubmit(e) {
    // form submit시 페이지로 넘어가지 않기 위해
    e.preventDefault()
    const formData = { name, email }
    // 클라이언트 측 validation
    const errors = validateForm()
    if (errors) {
      setErrors(errors)
    } else {
      setSubmitting(true)
      try {
        // 서버에 post 요청
        const response = await fetch('/api/contact', {
          method: 'POST',
          body: JSON.stringify({
            contact: {
              ...formData,
            },
          }),
        })
        const result = response.json()
        if (result.errors) {
          setErrors(result.errors)
        } else {
          // 홈으로 보내기
          router.push('/')
        }
      } finally {
        setSubmitting(false)
      }
    }
  }
}
```

그리고, nextjs에 `/api/` 동작을 추가할 수 있을 것이다.

```js
// pages/api/contact.js
export default function handler(req, res) {
  const { name, email } = req.body

  const errors = {}
  if (!name) errors.name = true
  if (!email) errors.email = true

  if (Object.keys(errors).length > 0) {
    res.status(400).json(errors)
  } else {
    await createContactRequest({ name, email })

    res.status(200).json({ success: true })
  }
}
```

이러한 방식은 nextjs에서 일반적인 방식으로, 위 코드에서 볼 수 있는 것 처럼 상당한 양의 자바스크립트 코드가 필요하다. (그리고 전체 페이지 번들에 hydration이 일어나지 않는다면 제대로 동작하지 않을 것이다.) 또한 위에 예제 처럼 수동으로 fetch 기반의 form을 만들어서 처리한다면, 비동기 fetch 이슈와 같은 문제도 처리해야 한다.

### remix

remix는 자바스크립트 코드를 훨씬 적게 썼던 php/rails 내부의 서버 템플릿 웹 앱 시절을 생각나게 한다. 아래 예제를 살펴보자.

```jsx
// app/routes/contact.tsx
export const action: ActionFunction = async ({ request }) => {
  const formData = await request.formData()

  const name = formData.get('name')
  const email = formData.get('email')

  const errors = {}
  if (!name) errors.name = true
  if (!email) errors.email = true

  if (Object.keys(errors).length > 0) {
    return errors
  }

  await createContactRequest({ name, email })

  return redirect('/')
}

export default function ContactPage() {
  // ActionFunction을 리턴으로 하는 action 함수가 기본으로 실행됨
  const errors = useActionData()
  return (
    <form method="post">
      <input type="text" name="name" />
      {errors?.name && <em>Name is required</em>}
      <input type="text" name="email" />
      {errors?.email && <em>Email is required</em>}
      <button type="submit">Contact me</button>
    </form>
  )
```

겉으로 보았을 때는 php 스타일로 작성된, post 핸들러를 갖춘 HTML form 일 뿐이다. 자바스크립트에 hydration이 이뤄지지 않더라도 이 코드는 작동한다. `useActionData`는 액션 핸들러에서 반환된 json 데이터를 사용할 수 있도록 한다. 이 액션 아키텍텨는 리액트를 풀스택 프레임워크로 정의하는 방법이다.

remix의 디자인은 기존의 브라우저가 지원하는 클라이언트 - 서버 모델을 본떠서 만들어졌기 때문에, 훨씬 더 적은 코드로 작성되었다.

이러한 방식에 익숙하지 않은 개발자들은, 이러한 핸들러에 일반적인 패턴이 다음과 같이 존재한다는 것을 인지하고 있어야 한다.

- validation이 실패하면, 에러를 서버로 부터 받고 동일한 페이지를 리렌더링 한다.
- validation이 성공하면 대상 페이지로 리다이렉트가 일어난다. 그리고 이 도착 페이지에 toast 메시지를 노출할 수 있다. (remix에는 빌트인 [session](https://remix.run/docs/en/v1/api/remix#using-sessions)이 있는데, 이를 활용하면 된다.)

만약 한 페이지에 여러개의 form이 있다면?

숨겨진 필드를 사용하여 각 form이 무엇을 나타내는지를 구별하거나, submit button에 추가적인 값을 넘겨주면 된다.

만약에 일반적인 REST API를 쓰고 싶다면?

nextjs의 경우에는, api 엔드포인트는 임의의 페이지와 클라이언트를 서비스할 수 있는 일반적인 REST api의 일부였다. 이에 반해 remix는 `loader` 함수를 정의하여 (뒤에서 설명) 임의의 데이터를 반환하는 라우트인 리소스 라우트를 정의할 수 있다. nextjs와 다르게, `/pages/api`에 위치할 필요는 없다.

만약 클라이언트 측에서 인터랙션이 있는 form validation을 사용하고 싶다면?

`Form`과 함께 `useTransition`을 사용하면 된다. (리액트의 `useTransition`과 다름 주의)

```jsx
// app/routes/contact.tsx

export const action: ActionFunction = async ({ request }) => {
  /*...*/
}

export default function ContactPage() {
  // 서버의 에러를 JSON 형태로 받을 수 있음. 이 경우에는 페이지를 리로드 하지 않고도 form을 리렌더링 할 수 있음
  const errors = useActionData()
  // form 성공시
  const { submission } = useTransition()
  return submission ? (
    <Confirmation contact={Object.fromEntries(submission.formData)} />
  ) : (
    <Form method="post">...</Form>
  )
}
```

서버사이드에서는 여전히 validation을 수행하지만, 이제 빠르게 클라이언트에서 피드백을 반영할 수 있다. 클라이언트 측에서 validation을 하고 싶을 경우, 추가적으로 로직을 추가하여 클라이언트에서 validation을 실행할 수 있다.

다만 이 경우 클라이언트가 JSON 데이터 fetch를 수행하거나, 자바스크립트를 아직 사용할 수 없는 경우 전체 html 응답을 요청하여 서버와 통신할 수 있다는 점이다.

## 항상 SSR

remix는 항상 서버사이드 렌더링이 일어나며, 특정 페이지가 정적으로 생성되는 것을 표시하는 개념을 지원하지 않는다. 그렇다고 정적 사이트를 만들 수 없다는 것은 아니다.

기본적인 측면에서, 서버 사이드 데이터 fetch 및 렌더링 속도가 충분하게 빠를 경우, edge server에 배치하여 정적 사이트 수준의 성능을 달성할 수 있다. (자세한 방법은 하단 예시를 참조)

그러나 이 방법은 언제나 가능한 것은 아니고, 페이지가 본질적으로 느린 백엔드 서비스에 의존할 수 있다. 이 경우 http 캐시를 활용할 수 있다. CDN을 서비스 인프라 최상단에 두고, 올바른 `Cache-Control` 헤더를 사용하면 CDN이 지연 시간을 최소화 하면서 변경되지 않은 콘텐츠를 저장하고 제공할 수 있다.

이를 통해 정적 사이트 빌드를 수행하는 대신, 변경 사항을 즉시 재구현 할 수 있다는 이점이 있다.

```jsx
// app/routes/some-page.tsx

export function headers() {
  return {
    'Cache-Control':
      'public, max-age=300, s-maxage=3600, stale-while-revalidate=300',
  }
}

export default function SomePage() {
  return <div>...</div>
}
```

이는 트레이드 오프가 있다. 직접적으로 프레임워크 수준에서 제어되는 invalidation을 포기하는 대신, CDN에 있는 전용 cache flushing 로직에 의존해야 한다. 사용자의 브라우저에서 캐시를 무효화하는 것 또 고려해야 하므로, 캐시 만료에 너무 지나치게 적극적으로 대응하지 않는 것이 좋다.

## hydration

Nextjs에서 달성하고자 하는 기능 중 하나는 부분적인 hydration이다. 특히 정적인 컨텐츠의 경우, 리액트 SSR/SSG 프레임워크는 필요한 자바스크립트 번들 보다 훨씬 더 많이 전달되고 hydration 되므로, 페이지 로드 성능에 큰 영향을 미칠 수 있다.

Remix는 개발자에게 언제 hydration이 일어날 수 있는지 결정할 수 있게 해준다. 하지만 현재는 페이지 단위로만 제공되고 있다. 이 방법은 조건부로 `Document` 내부의 `<Scripts/>`를 렌더링 하지 않는 것이다.

https://remix.run/docs/en/v1/guides/disabling-javascript

```jsx
// app/entry.server.tsx

import React from 'react'
import { Meta, Links, Scripts, Outlet, useMatches } from 'remix'

export default function App() {
  let matches = useMatches()

  // 아래 코드 참조
  let includeScripts = matches.some((match) => match.handle?.hydrate)

  // includeScripts 상태 값으로 제어 가능
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <Meta />
        <Links />
      </head>
      <body>
        <Outlet />
        {/* 스크립트 제어! */}
        {includeScripts && <Scripts />}
      </body>
    </html>
  )
}
```

```jsx
// app/routes/some-page.tsx

export let handle = { hydrate: true }

export default function SomePage() {
  return <div>...</div>
}
```

Remix 개발자가 현재 부분적인 hydration을 할 수 있는 방안에 대해서 아주 깊게 연구 하고 있다고 하니 기대해봄직하다.

## 스타일

스타일을 적용하기 위해서는, header에 링크를 추가하면 된다. 이는 HTML 에 스타일 시트를 추가하는 것과 매우 유사하다. remix에 이러한 스타일 정보를 미리 제공하면, 모든 CSS를 캐시 가능한 스타일 시트 URL과 함께 병렬로 로딩할 수 있으므로, `prefetch`와 함께 사용한다면 최적의 페이지 로드 성능을 얻을 수 있다.

> 이와 관련되서 스타일 순서가 보장되지 않는 nextjs의 오랜 버그가 있다. https://github.com/vercel/next.js/issues/16630

```jsx
// app/routes/some-page.tsx

import styles from '~/styles/global.css'
// styles is now something like /build/global-AE33KB2.css

export function links() {
  return [
    {
      rel: 'stylesheet',
      href: 'https://unpkg.com/modern-css-reset@1.4.0/dist/reset.min.css',
    },
    {
      rel: 'stylesheet',
      href: styles,
    },
  ]
}

export default function SomePage() {
  return <div>...</div>
}
```

https://remix.run/docs/en/v1/guides/styling

위 가이드를 보면 알겠지만, scss, css, css-in-js를 사용해도 `links`를 통해서 스타일을 노출 시키는 것은 필수사항이다.

## 파일 기반 라우팅

remix는 nextjs를 사용하는 사람에게 매우 친숙한 파일 기반 라우팅을 사용한다. 그러나 nextjs 와 다르게 경로를 계층으로 구성할 수 있는 React Router 스타일의 중첩을 지원한다.

예를 들어, 아래와 같은 라우팅이 있다고 가정해보자.

- `/dashboard`
- `/dashboard/settings`
- `/dashboard/reports`

```jsx
// app/routes/dashboard.tsx

export default function DashboardLayout() {
  return (
    <div>
      <Header />
      {/* 내부 중첩 라우팅에 필요한 컨텐츠가 들어간다. */}
      <Outlet />
      <Footer />
    </div>
  )
}
```

```jsx
// app/routes/dashboard/settings.tsx
export default function Settings() {
  // ...
}
```

```jsx
// app/routes/dashboard/reports.tsx
export default function Reports() {
  // ...
}
```

```jsx
// app/routes/dashboard/index.tsx
export default function DashboardMain() {
  // ...
}
```

nextjs와 마찬가지로 remix는 isomorphic routing을 지원하므로 js에 hydration이 발생한다면 클라이언트에서 전환이 빠르게 일어난다.

## 경로 위치에 있는 데이터 가져오기

nextjs에서는 페이지 내부의 `getStaticProps` 또는 `getServerSideProps`에서 데이터를 가져올 수 있다. remix에서는 nextjs와 같은 방식으로 페이지 라우팅에 필요한 `loader` 함수를 사용할 수 있다. 마찬가지로, SSR 응답에 JSON으로 이 데이터를 직렬화한다.

```jsx
// app/routes/mypage.tsx

// 데이터를 가져온다. 이는 서버에서 수행된다.
export async function loader() {
  return fetch(/*...*/)
}

export default function MyPage() {
  const data = useLoaderData()
  return <div />
}
```

그러나 remix의 `loader`가 다른 점은 페이지 레벨 뿐만 아니라 중첩 라우팅에서도 사용이 가능하다는 것이다.

이것의 이점은, 데이터를 배치하는 것이 용이 해진다는 것이다. 예를 들어 `/dashboard`라는 루트 라우팅이 있을 경우, 하위 라우팅에 공통 페이지 레이아웃을 렌더링 할 수 있다는 것이다.

```jsx
export async function loader() {
  return getCurrentUser()
}

export default function DashboardLayout() {
  const currentUser = useLoaderData()
  return (
    <div>
      <Header>
        <UserAvatar user={currentUser} />
      </Header>
      {/* 하위 렌더링 페이지가 여기에 배치된다. */}
      <Outlet />
      <Footer />
    </div>
  )
}
```

그리고 ` /dashboard/settings` 에 또다른 `loader`가 있다고 가정해보자. 루트는 이에 대해 관심이 없으며, 마찬가지로 다른 페이지에서도 이에 대해 가지고 있을 필요가 없다. nextjs에서는 `getServerSideProps`의 일부를 분산하여 이를 수동으로 구현할 수 있지만, remix는 좀더 자연스러운 지원을 제공한다.

```jsx
// parent route
import { Outlet } from 'remix-utils'

export default function Parent() {
  return <Outlet data={{ something: 'here' }} />
}
```

```jsx
// child route
import { useParentData } from 'remix-utils'

export default function Child() {
  const data = useParentData()
  return <div>{data.something}</div>
}
```

이러한 분산된 fetch 설계는 스타일 시트의 링크와 마찬가지로, remix가 전체 트리의 데이터 의존성을 미리 알고 있기 때문에 별렬로 실행할 수 있다는 장점이 있다.

## 배포

remix는 vercel, cloudflare worker, deno deploy, fly.io와 같은 다양한 배포 환경을 지원한다. 각 배포에는 프로젝트 별로 약간 다른 구성 및 패키지가 필요할 수 있겠지만, 노드 환경과 비노드 환경 (cloudflare)에서 모두 실행 가능하다.

특정 배포 대상에 따른 remix 설정은 `create-remix`를 사용할 때 자동으로 구성되지만, 기본 설정에서 다른 설정으로 옮겨갈 경우에는 약간의 수정을 추가하면 된다.

## 기타

- remix 내부에는 세션과 쿠키를 처리할 수 있는 함수가 내장되어 있으며, 이는 클라이언트 서버 아키텍쳐에서 중요하다.
- `app/routes/reports/$report.tsx` 형태의 동적 라우팅을 지원한다. nextjs는 `[param]`, remix는 `$param`이다.
  ```jsx
  export default function Report() {
    // ...
    return <div>...</div>
  }
  ```
- nextjs 에 글로벌 `App` `Document` Wrapper가 있다면, remix에는 `entry.server.tsx`가 있다. 또한 `entry.client.tsx`를 사용하여 hydration과정에서 정확히 어떤일이 일어나는지 확인할 수 있다.
- nextjs에 있지만 remix에 없는 것은
  - 이미지 최적화 컴포넌트 `next/image`
  - 구글 폰트와 Typekit를 위한 자동 폰트 CSS 인라이닝
  - 스크립트 스케쥴링 및 우선순위 지정과 같은 세부적인 제어

## 느낀점

- react-router-dom 스타일의 중첩 routing 지원, 그리고 부모 데이터를 불러올 수 있는 기능이 인상적이다. nextjs는 페이지별로 다 찢어져있어서 특정 페이지들을 위한 context 구현이 opt-out 하지 않는 이상 불가능 했는데 이점은 굉장히 맘에 든다.
- static한 페이지가 많은 애플리케이션은 여전히 nextjs가 더 좋은 방식을 제공하고 있는 것 같다. CDN과 cache를 사용하는 것은 물론 기존에 있는 접근이지만 서도, nextjs가 더 편리한 방식으로 구현했다고 본다.
- 기본적으로 ssr이라는 점은 좋은 것 같다. 프론트엔드 개발자들이 static 파일을 upload하고 서빙하는 시대는 지났다. 이제 node 서버, 더 나아가 배포와 devOps에 대해서도 고민해야할 때가 왔다. (사실 진작에 왔다)
- 위와 마찬가지로, SPA의 패러다임은 이제 조금씩 쇠퇴하고 있는 느낌이다. 기기의 성능이 갈수록 좋아지는 시대일 수록 SPA가 빛을 발한다는 이야기를 들었던 것 같은데 이제 틀린게 아닌가 싶다. 성능을 사용자의 기기에 의존해서는 안된다.
- javascript가 실행되지 않는 환경을 고민하는 것 또한 인상적이다. 성능 측면에서 우리가 항상 고민해봐야할 문제다.
- 다음 토이 프로젝트는 remix다.
