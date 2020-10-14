---
title: NextJS 2. Data Fetching
tags:
  - typescript
  - javascript
  - react
published: true
date: 2020-03-12 02:39:10
description: '[nextjs의 공식
  문서](https://nextjs.org/docs/basic-features/data-fetching)를 보고 요약한 내용입니다.
  ```toc tight: true, from-heading: 1 to-heading: 2 ```  ## 1.
  getInitialProps  Nextjs 9.3 이전에는 `getInitialProps` 밖에...'
category: typescript
slug: /2020/03/nextjs-02-data-fetching/
template: post
---

[nextjs의 공식 문서](https://nextjs.org/docs/basic-features/data-fetching)를 보고 요약한 내용입니다.

## Table of Contents

## 1. getInitialProps

Nextjs 9.3 이전에는 `getInitialProps` 밖에 존재하지 않는다. 최신 버전인 9.3에서는 밑에서 설명할 `getStaticProps`나 `getServerSideProps`를 사용하기를 권장한다. (왠지 deprecate 될 것 같은 기분이다.)

`getInitialProps`는 페이지에서 서버사이드 렌더링을 가능하게 하며, 페이지가 호출될 때 최초로 데이터 조작을 가능하게 한다. 이 말의 뜻은, 서버에서 데이터를 불러온 다음에, 이 데이터와 함께 페이지를 내보낸다는 뜻이다. 이는 특히 SEO 등에서 유용하다.

> 주의: `getInitialProps`를 쓰는 순간 nextjs의 automatic static optimization이 불가능해진다.

예제를 살펴보자.

```typescript
import { NextPageContext } from 'next'
import React from 'react'
import fetch from 'isomorphic-fetch'

interface EmployeeInterface {
  id: number
  employee_name: string
  employee_salary: number
  employee_age: number
  profile_image: string
}

export default function Data({ data }: { data: EmployeeInterface[] }) {
  return (
    <>
      <h1>Employee list</h1>
      {data.map(
        ({ id, employee_age, employee_name, employee_salary }, index) => (
          <div key={index}>
            <span>{id}.</span>
            <span>{employee_name} </span>
            <span>${employee_salary}</span>
            <span> {employee_age} years old</span>
          </div>
        ),
      )}
    </>
  )
}

Data.getInitialProps = async (_: NextPageContext) => {
  const response = await fetch(
    'http://dummy.restapiexample.com/api/v1/employees',
  )
  const { data } = await response.json()

  return { data }
}
```

`getInitialProps` 내 에서 비동기로 데이터를 가져 온 다음에, props를 만들어 컴포넌트에 넘긴다. 한가지 명심할 것은, 여기서 컴포넌트에 넘겨주는 행위는 `JSON.stringify`와 비슷하다. 따라서 넘길 수 있는 데이터는 순수 Object여야 한다.

**중요 포인트**

1. 처음 페이지가 로딩 된다면, `getInitialProps`는 서버에서만 로딩된다. 그러나 `next/link` 또는 `next/router`를 통해서 클라이언트 사이드에서 페이지 이동이 일어난다면, 클라이언트 사이드에서 실행될 수 있다.

2. `getInitialProps` 는 자식 컴포넌트에서 사용할 수 없다. 오직 각 페이지에서만 실행 가능하다.

3. 1번의 이유에 따라서, `getInitialProps`내에서 서버사이드에서만 실행될 수 있는 모듈을 내장하고 있다면, 주의를 기울여야 한다. 만약 서버사이드에서만 작동하고 싶은 로직이 있다면, 아래처럼 하면 된다.

```typescript
Data.getInitialProps = async ({ req }: NextPageContext) => {
  console.log('fetch some data')
  const response = await fetch(
    'http://dummy.restapiexample.com/api/v1/employees',
  )
  const { data } = await response.json()

  let isServer = false
  if (req) {
    // is server side???????
    isServer = true
  }

  return { data, isServer }
}
```

## 2. getStaticProps

정적 페이지 생성을 지원하며, 데이터를 딱 빌드 타임에만! 실행된다.

```typescript
export async function getStaticProps(_: NextPageContext) {
  const response = await fetch(
    'http://dummy.restapiexample.com/api/v1/employees',
  )
  const { data } = await response.json()

  console.log('fetchData in build time!')

  return {
    props: { data },
  }
}
```

빌드를 해보면 아래와 같이 메시지가 출력된다.

```
...
Automatically optimizing pages ..fetchData in build time!
Automatically optimizing pages

Page                                                           Size     First Load
┌ λ /                                                          458 B       68.2 kB
├   /_app                                                      352 B       67.7 kB
├ λ /about                                                     301 B         68 kB
├ ● /data                                                      412 B       68.2 kB
└ λ /posts/[id]                                                303 B         68 kB
+ shared by all                                                67.7 kB
  ├ static/pages/_app.js                                       352 B
  ├ chunks/d43014630f87ab6320ffd55320a44642064161b7.111b68.js  9.77 kB
  ├ chunks/framework.9daf87.js                                 40.1 kB
  ├ runtime/main.d2cfdc.js                                     16.8 kB
  └ runtime/webpack.a34f97.js                                  744 B

λ  (Server)  server-side renders at runtime (uses getInitialProps or getServerSideProps)
○  (Static)  automatically rendered as static HTML (uses no initial props)
●  (SSG)     automatically generated as static HTML + JSON (uses getStaticProps)
...
```

data를 빌드시에 미리 땡겨와서 static하게 제공한다는 것을 알 수 있다. 그리고 next를 실행해보면 데이터 fetch를 하지 않는다는 것을 알 수 있다. 이미 빌드 시에 데이터를 땡겨 왔기 때문에, 굉장히 빠른 속도로 페이지가 로딩 된다.

`getStaticProps` 는 아래와 같은 경우에 유용할 것이다.

- 매 유저의 요청마다 fetch할 필요가 없는 데이터를 가진 페이지를 렌더링 할때
- headless CMS로 부터 데이터가 올때
- 유저에 구애받지 않고 퍼블릭하게 캐시할 수 있는 데이터
- SEO 등의 이슈로 인해 빠르게 미리 렌더링 해야만 하는 페이지. `getStaticProps`는 HTML과 JSON파일을 모두 생성해 두기 때문에, 성능을 향상시키기 위해 CDN 캐시를 하기 쉽다.

그리고 아래와 같은 사항을 유념해 두자.

- 빌드 타임에서만 실행된다.
- 서버사이드 코드다. 절대 클라이언트 사이드에서 실행되지 않는다. 심지어 브라우저 JS 번들에도 포함되지 않는다. 그냥 props결과물 자체를 JS 번들에 포함시키고 있다. 페이지에서 소스 보기를 하면, 아래 처럼 데이터를 아예 들고 있는 것을 볼 수 있다.

```html
<script id="__NEXT_DATA__" type="application/json">
  {
    "props": {
      "pageProps": {
        "data": [
          {
            "id": "1",
            "employee_name": "Tiger Nixon",
            "employee_salary": "320800",
            "employee_age": "61",
            "profile_image": ""
          }
        ]
      },
      "__N_SSG": true
    },
    "page": "/data",
    "query": {},
    "buildId": "ExAlLKs0H7K3JGmYT162x",
    "nextExport": false,
    "isFallback": false,
    "gsp": true
  }
</script>
```

- Page에서만 가능하다.
- 개발 모드에서는 매 번 요청이 간다.

## 3. getStaticPaths

위에서 언급한 `getStaticProps`와 매우 유사하다. 차이가 있다면, `getStaticPaths`는 다이나믹 라우트에서만 쓴다는 것이다. 설명보단 예시를 보는게 더 빠르다.

**/pages/post/[id].tsx**

```typescript
import React from 'react'
import fetch from 'isomorphic-fetch'
import { GetStaticProps } from 'next'

interface PostInterface {
  userId: number
  id: number
  title: string
  body: string
}

export default function Employee({ todo }: { todo: PostInterface }) {
  const { userId, id, title, body } = todo
  return (
    <>
      <h1>Todo</h1>
      <div>userId: {userId}</div>
      <div>id: {id}</div>
      <div>title: {title}</div>
      <div>body: {body}</div>
    </>
  )
}

export async function getStaticPaths() {
  const response = await fetch('https://jsonplaceholder.typicode.com/posts')
  const data = await response.json()

  const paths = data.map(({ id }: PostInterface) => ({
    params: { id: String(id) },
  }))

  return { paths, fallback: false }
}

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const response = await fetch(
    `https://jsonplaceholder.typicode.com/posts/${params?.id}`,
  )
  const data = await response.json()

  return {
    props: { todo: data },
  }
}
```

`getStaticPaths` 에서 `/pages/post/[id]`로 접근 가능한 모든 목록을 땡겨온다. 그리고 가능한 접근 목록을

```json
[{ "params": { "id": 1 } }, { "params": { "id": 2 } }]
```

와 같은 형태로 만들어 둔다. 문서와 다르게 꼭 주의 해야 할 것은 **value는 무조건 string 이어야 한다는 것이다.** 그리고 이제 빌드 타임에 가능한 모두 경우의 수를 땡겨와서 - 빌드 하게 된다.

몇 가지 더 샘플을 보도록 하자.

**pages/todo/[userId]/[id].tsx**

```typescript
export async function getStaticPaths() {
  const response = await fetch('https://jsonplaceholder.typicode.com/todos/')
  const data = await response.json()

  const paths = data.map(({ id, userId }: TodoInterface) => ({
    params: { userId: String(userId), id: String(id) },
  }))

  return { paths, fallback: false }
}
```

**pages/todo/[...slug].tsx**

```typescript
export async function getStaticPaths() {
  const response = await fetch('https://jsonplaceholder.typicode.com/posts/')
  const data = await response.json()

  const paths = data.reduce(
    (
      acc: Array<{ params: { slug: string[] } }>,
      { userId, id }: PostInterface,
    ) => {
      return acc.concat([
        { params: { slug: [String(userId), String(id)] } },
        { params: { slug: [String(id)] } },
      ])
    },
    [],
  )

  return { paths, fallback: false }
}
```

이렇게 array 형태로 넘겨주면 된다.

```json
{"slug":["10","95"]}},{"params":{"slug":["95"]}}
```

`getStaticProps`에서는 `params`로 접근하면

```json
{ "slug": ["1", "3"] }
```

여기서 꺼내 쓰면 된다.

`getStaticPaths`는 리턴 값으로 앞서 만들었던 `paths`와 `fallback`을 넘겨준다. `fallback`을 true나 false가 가능하다. false라면 nextjs의 404가 뜬다. 이는 미리 만들어 두어야 할 페이지의 수가 적을 때, 빌드 타임을 짧게 가져감으로서 이익을 볼 수 있다.

만약 `fallback`의 값이 true라면 `getStaticProps`는 아래와 같이 달라진다.

- `getStaticPaths`에서 리턴되는 `paths`는 빌드타임에 HTML이 렌더링 된다.

- 여기서 생성되지 않는 예외 Path들은 404 페이지를 리턴하지 않는다. 대신, NextJs는 fallback page를 보여주게 된다. 아래 예시를 살펴보자.

```typescript
export default function Employee({ todo }: { todo: PostInterface }) {
  const { isFallback } = useRouter()

  if (isFallback) {
    return <>Fail!</>
  }

  const { userId, id, title, body } = todo
  return (
    <>
      <h1>Todo</h1>
      <div>userId: {userId}</div>
      <div>id: {id}</div>
      <div>title: {title}</div>
      <div>body: {body}</div>
    </>
  )
}

export async function getStaticPaths() {
  const response = await fetch('https://jsonplaceholder.typicode.com/posts')
  const data = await response.json()

  const paths = data.map(({ id }: PostInterface) => ({
    params: { id: String(id) },
  }))

  return { paths, fallback: true }
}
```

Fallback 페이지의 props는 아무것도 없다. 따라서 props를 가공하는 처리를 해서는 안된다.

- 해당 path가 없는 페이지에 대해서 Nextjs는 서버단에서 정적인 HTML과 JSON을 만들어 둔다. 여기에는 `getStaticProps`을 실행하는 것도 포함된다.

- 위 작업이 끝났다면, 브라우저는 해당 path에 따라서 만든 JSON을 받게된다. 이 JSON은 페이지 렌더링에 필요한 Props를 제공하는데 사용된다. 유저 입장에서는, fallback 페이지에서 전체 페이지로 스왑되는 것으로 보일 것이다. (fallback이 잠시 보였다가 다시 받아온 props로 그리는 페이지가 나타남 (isFallback이 true에서 false로 바뀜))

- 이와 동시에, 해당 path를 미리 렌더링한 path에 추가해둔다. 같은 path로 오는 요청들은 이제 마치 빌드시에 사전에 렌더링해 둔 페이지 처럼 제공된다.

복잡하다. 예를 들어서 설명해보자.

```typescript
export async function getStaticPaths() {
  const items = Array.from(Array(10).keys())

  const paths = items.map(value => ({
    params: { id: String(value) },
  }))

  return { paths, fallback: true }
}

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const id = params?.id

  if (Number(id) > 10) {
    return {
      props: {
        todo: {
          userId: 1,
          id,
          title: `이건 에러야.`,
          body: `아 이건 에러라니깐.`,
        },
      },
    }
  } else {
    return {
      props: {
        todo: {
          userId: 1,
          id,
          title: `할일 ${id}`,
          body: `이거 하자. ${id}`,
        },
      },
    }
  }
```

개 떡 같은 코드지만 (...) `getStaticPaths`는 `/todo/0` 부터 `/todo/9`까지만 미리 빌드 타임에 만들어 둔다.

```
 ● /todo/[id]                                                 378 B       68.1 kB
    ├ /todo/0
    ├ /todo/1
    ├ /todo/2
    └ [+7 more paths]
```

그리고 만약 어떤 사용자가 처음으로 `/todo/1111`로 접근했다고 가정해보자. 그럼 사용자는 잠시 fallback 페이지를 봤다가, 다시 `getStaticProps`가 렌더링해주는 에러 페이지를 보게된다. 그리고 nextjs는 해당 path에 대해 렌더링 해둔 것을 저장해둔다. 그리고 이후에 다시 접근하는 사용자는 fallback 페이지를 보지 않고 바로 앞서 만들어 두었던 페이지를 보여주게 된다.

fallback 페이지는 언제 유용할까?

아주 큰 커머스 사이트와 같이, 데이터에 따라 만들어 두어야할 정적페이지가 많은 사이트에서 유리할 것이다. 모든 페이지를 빌드시에 만들어 두고 싶지만, 그랬다가는 빌드가 엄청나게 오래걸릴 것이다. 대신, 미리 몇개의 주요 페이지만 만들어두고, 나머지는 `fallback: true`로 처리하자. 누군가 아직 만들어지지 않은 페이지에 접근하려 한다면, 유저에게 로딩 인디케이터를 띄우자. 그러면 백그라운드에서는 `getStaticProps`를 실행해서 렌더링에 필요한 데이터를 가져올 것이다. 그리고 이 작업이 끝난다면, 다른 유저들은 이제 미리 렌더링된 정적인 페이지를 볼 수 있다.

그리고 아래와 같은 사항을 유념해 두자.

- 항상 `getStaticProps`와 짝으로 쓰자. 그리고 `getServerSideProps`와는 쓸수가 없다.
- `getStaticPaths`는 서버사이드에서 빌드 타임에만 실행된다.
- `getStaticPaths`는 페이지에서만 사용 가능하다.
- 개발 모드에서는 항상 실행된다.

## 4. getServerSideProps

`getServerSideProps`를 사용하면, 각 요청 마다 `getServerSideProps`에서 리턴한 데이터를 받아다가 서버사이드에서 미리 렌더링을 하게 된다.

```javascript
export async function getServerSideProps(context) {
  return {
    props: {},
  }
}
```

빌드를 하게 되면, 아래와 같이 나타난다.

```
Page                                                           Size     First Load
...
├ λ /server                                                    415 B       68.2 kB
...

λ  (Server)  server-side renders at runtime (uses getInitialProps or getServerSideProps)
○  (Static)  automatically rendered as static HTML (uses no initial props)
●  (SSG)     automatically generated as static HTML + JSON (uses getStaticProps)
```

context에는 다음과 같은 것들이 포함되어 있다.

- `params`: 다이나믹 라우트 페이지라면, `params`를 라우트 파라미터 정보를 가지고 있다.
- `req`: [HTTP request object](https://nodejs.org/api/http.html#http_class_http_incomingmessage)
- `res`: [HTTP response object](https://nodejs.org/api/http.html#http_class_http_serverresponse)
- `query`: 쿼리스트링
- `preview`: `preview` 모드 여부 [preview mode](https://nextjs.org/docs/advanced-features/preview-mode)
- `previewData`: `setPreviewData`로 설정된 데이터

언제 써야 할까?

`getServerSideProps`는 페이지를 렌더링하기전에 반드시 fetch해야할 데이터가 있을 때 사용한다. 매 페이지 요청시마다 호출되므로 당연히, TTFB가 `getStaticProps`보다 느리다.

그리고 아래와 같은 사항을 유념해 두자.

- `getServerSideProps`는 서버사이드에서만 실행되고, 절대로 브라우저에서 실행되지 않는다.
- `getServerSideProps`는 매 요청시 마다 실행되고, 그 결과에 따른 값을 props로 넘겨준 뒤 렌더링을 한다.
- `next/link`를 이용해서 클라이언트 사이드 페이지 트렌지션을 하더라도, `getInitialProps`와는 다르게 무조건 서버에서 실행된다.
- 당연히 page 에서만 실행할 수 있다.
