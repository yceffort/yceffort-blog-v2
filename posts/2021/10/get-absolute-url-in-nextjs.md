---
title: 'nextjs 서버사이드에서 absolute url 가져오기'
tags:
  - javascript
  - nextjs
published: true
date: 2021-10-29 17:57:02
description: '이번 달 포스팅이 더디네요... 반성합니다.'
---

nextjs에서 fetch api를 사용하다가 깨닫는 점 하나는, (당연하지만) 서버와 클라이언트에서 fetch를 하는 방식을 다르게 가져가야 한다는 것이다. 무심코 평소에 하듯이 `fetch('/api/some/info')`를 하다보면, `getServerSideProps`나 `getInitialProps`에서 에러가 날 수 있다. fetch 를 할 때 놓치지 말아야 할 이 에러를 살펴보자.

## Absolute URL

서버사이드에서 절대 경로가 아닌 상대경로로 fetch 요청을 하면 (`/api/some/info`) Absolute URL이 필요하다는 에러가 뜬다. 클라이언트에서는 origin을 추론할 수 있기 때문에 상대경로로 요청을 해도 상관없지만, 서버사이드에서는 현재 주소가 무엇인지 알리가 없기 때문에 상대경로로 요청할 수 없다.

그렇다면 아래와 같이 처리해도 되는 것인가?

```typescript
async function getUser(id: number) {
  const response = await fetch(
    typeof window === 'undefined'
      ? 'https://yceffort.kr'
      : '' + `/api/user/${id}`,
  )
  const result = await response.json()
  return result
}
```

absolute url, origin이 고정되어 있는 경우라면 괜찮겠지만 그렇지 않다면 이렇게 처리하는건 안전하지 못하다. 따라서 우리는 absolute URL을 추론해야 한다. 추론할 수 있는 가장 좋은 방법은, `ctx.req` 즉 [IncomingMessage](https://nodejs.org/api/http.html#class-httpincomingmessage)를 사용하는 것이다. 여기에는 요청과 관련한 정보가 포함되어 있는데, 이 요청이 날라온 곳이 absolute url이라고 가정하고 코딩하는 것이다.

```typescript
import { IncomingMessage } from 'http'

function getAbsoluteURL(req?: IncomingMessage) {
  // 로컬은 http, 프로덕션은 https 라는 가정
  const protocol = req ? 'https:' : 'http:'
  let host = req
    ? req.headers['x-forwarded-host'] || req.headers['host']
    : window.location.host

  // 로컬로 부터 요청이 온것이라면..
  // 물론 이것도 완전히 안전하지는 못하다. 프로덕션 주소에 local이 들어가있다면,,,
  if ((host || '').toString().indexOf('local') > -1) {
    // 개발자 머신에서 실행했을 때 로컬
    host = 'localhost:3000'
  }

  return {
    protocol: protocol,
    host: host,
    origin: protocol + '//' + host,
  }
}
```

물론 이 방법도 완전하지는 못하다. 프로젝트의 상황에 따라 조금씩 다를 수 있다. 일단 프로토콜은 서버에서 추론할 수가 없는 부분이기 때문에 하드 코딩 형태의 추론이 필요하다.

그리고 `host`가 아닌 [`x-forwarded-host`](https://developer.mozilla.org/ko/docs/Web/HTTP/Headers/X-Forwarded-Host)를 사용한 이유는 설명에도 나와있듯, 요청을 처리하는 원래 사용된 host를 확인하기 위해 사용하였다.

그리고 이제 fetch 함수는 `getAbsoluteURL`를 사용해야 한다.

```typescript
type FetchUser = {
  req?: IncomingMessage
}

async function getUser(id: number, options?: FetchUser) {
  const absoluteURL = getAbsoluteURL(options?.req).origin
  const response = await fetch(
    options?.req ? absoluteURL : '' + `/api/user/${id}`,
  )
  const result = await response.json()
  return result
}
```

```typescript
export const getServerSideProps: GetServerSideProps = async (ctx) => {
  const { req } = ctx
  const userId = ctx.query?.userId

  const user = await getUser(userId, { req })
  return {
    props: {
      user,
    },
  }
}
```

이제 서버에서 요청을 할때는 `req` 정보를 넘겨줘야 한다. 서버와 클라이언트 요청은 명확히 구별할 수 있으므로 크게 어렵지 않을 것이다.

## 또다른 방법

아무래도 하드 코딩이 들어가 있기 때문에, absolute url을 알아낼 수 있는 또다른 방법은 실행시에 환경변수로 주입하고, 이를 가져오는 것이다. 이 방법을 쓰고 있는 것이 [VERCEL](https://vercel.com/docs/concepts/projects/environment-variables)이다. `NEXT_PUBLIC_VERCEL_URL`를 사용하면 이 빌드가 실행되는 곳의 주소를 알아낼 수 있다.

물론 이는 데브옵스, 인프라에서 이러한 정보를 제공할 수 있는 환경 구축이 선행되어야 한다.
