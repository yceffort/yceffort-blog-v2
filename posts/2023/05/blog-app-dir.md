---
title: '블로그 app dir 업그레이드 후기'
tags:
  - nextjs
  - typescript
published: true
date: 2023-05-23 20:52:26
description: '😬'
---

## Table of Contents

## 서론

블로그가 만들어진지도 꽤 오랜시간이 지나 새롭게 기술 스택을 수정할 필요가 있었고, 5월 초에 리액트@18 의 서버 컴포넌트를 사용할 수 있는 nextjs@13.4 가 정식으로 릴리즈 되었다. 리액트 18과 nextjs 13은 꽤나 많은 변경점을 가지고 있기 때문에 실무에 본격적으로 적용하기 전에 먼저 적용해 볼 필요가 있다고 생각하여 블로그에 우선적용하게 되었다. 약 2시간 정도를 들여 업그레이트에 성공한 기억을 바탕으로, 기존 블로그에서 업그레이드 하면서 겪었던 경험에 대해서 요약한다.

## 가이드

리액트18의 문서가 https://react.dev/ 를 기반으로 완전히 바뀐 것처럼, next@13 도 이번에 새 주버전이 올라가면서 문서가 https://nextjs.org/docs 완전히 새롭게 변경되었다. 개인적으로 한번 읽어본 바로는, 아직 공식 문서의 내용이 부족한 점이 있지만 꽤 일목 요연하게 잘 정리된 것 같은 느낌을 받았다. 업그레이드에는 이 두 문서와 [app router incremental adoption guide](https://nextjs.org/docs/app/building-your-application/upgrading/app-router-migration)를 참고했다.

## `src/pages`에서 `src/apps`로

####서버 컴포넌트

가장 큰 차이점은 `_app.tsx`와 `_document.tsx`로 대표되던 `src/pages`방식이 사라졌다는 것이다. 이 방식은 서버사이드에서 렌더링한다는 장점은 있지만, 모든 페이지가 완성되기 까지 기다려야 한다는 단점이 존재한다. 그러나 서버 컴포넌트는 이제 모든 페이지 완성을 기다릴 필요가 없이 스트림 방식으로 완성된 페이지를 조금씩 반환한다. 정확히는, 리액트 렌더링에 필요한 정보를 스트림으로 제공한다.

```bash
## https://yceffort.kr/pages/3 에 접근시

1:HL["/_next/static/css/60c057695325b064.css",{"as":"style"}]
0:[[["",{"children":["pages",{"children":[["id","3","d"],{"children":["__PAGE__?{\"id\":\"3\"}",{}]}]}]},"$undefined","$undefined",true],"$L2",[[["$","link","0",{"rel":"stylesheet","href":"/_next/static/css/60c057695325b064.css","precedence":"next"}]],["$L3",null]]]]
4:I{"id":"3238","chunks":["481:static/chunks/481-c2603ca401b0b1f5.js","222:static/chunks/222-806bbed146c8e258.js","185:static/chunks/app/layout-af351b82bfb0351c.js"],"name":"Providers","async":false}
5:I{"id":"9481","chunks":["481:static/chunks/481-c2603ca401b0b1f5.js","302:static/chunks/app/tags/[tag]/pages/[id]/page-61cc77a637db7fd9.js"],"name":"","async":false}
6:I{"id":"7","chunks":["481:static/chunks/481-c2603ca401b0b1f5.js","222:static/chunks/222-806bbed146c8e258.js","3:static/chunks/app/[year]/[...slug]/page-0546852f3fc5430b.js"],"name":"","async":false}
7:I{"id":"5008","chunks":["481:static/chunks/481-c2603ca401b0b1f5.js","222:static/chunks/222-806bbed146c8e258.js","185:static/chunks/app/layout-af351b82bfb0351c.js"],"name":"","async":false}
8:I{"id":"4567","chunks":["481:static/chunks/481-c2603ca401b0b1f5.js","222:static/chunks/222-806bbed146c8e258.js","185:static/chunks/app/layout-af351b82bfb0351c.js"],"name":"","async":false}
9:I{"id":"5690","chunks":["272:static/chunks/webpack-a5f9efca3d914538.js","618:static/chunks/81497cce-0ce4c3138c148cf8.js","905:static/chunks/905-99371aa5e5c9b1ba.js"],"name":"","async":false}
a:I{"id":"2465","chunks":["272:static/chunks/webpack-a5f9efca3d914538.js","618:static/chunks/81497cce-0ce4c3138c148cf8.js","905:static/chunks/905-99371aa5e5c9b1ba.js"],"name":"","async":false}
// ...
```

스트리밍 형태의 응답을 볼 수 있는데, `id`를 바탕으로 리액트의 어느 부분이 어떻게 렌더링이 필요한지를 서버에서 미리다 계산한 다음에 내려주게 된다. (서버 컴포넌트 기준)

### 새로운 예약어 파일

기존에는 파일명까지 라우팅을 구성하였지만, 이제는 폴더명만 라우팅 주소를 구성한다. 예를 들어 `/src/pages/hello.tsx`는 `/hello`로 접근가능하여 파일명까지 주소로 인식했지만, 이제는 폴더명까지 만 인식된다. 같은 주소로 반환되게 하려면 `/src/apps/hello/*.tsx` 로 변경해야 한다. 그리고 몇몇 파일명에 예약어가 생겼다.

#### `layout`

> https://nextjs.org/docs/app/api-reference/file-conventions/layout

과거 nextjs의 약점으로 지적받던 것 중 하나는 `react-router`와 같이 레이아웃을 구성하기 어렵다는 점이었다. 애플리케이션 전체 레이아웃은 `_app.tsx`나 `_document.tsx`에서 제한적으로 할 수 있었지만, `/hello/world` `/hello/foo`와 같이 특정 라우팅 하위에 레이아웃을 구성하는 것은 불가능하여 중복 코드를 작성해야 하는 수고가 있었다. next@13부터는 `layout.tsx` 이 생겨 이제 레이아웃을 구성할 수 있게 되었다. 그리고 이 레이아웃은 하위 라우팅에도 영향을 미친다.

```typescript jsx
import { ReactNode } from 'react'

export default function Layout({ children }: { children: ReactNode }) {
  // 여기에 레이아웃을 구성
  return <div className="body">{chlidren}</div>
}
```

이렇게 구성해두면, 하위 라우팅은 모두 `<div className="body"/>` 하단에 꽂히게 된다.

`layout`은 무조건 서버 컴포넌트이며, 따라서 `useState`등을 쓸 수는 없다. 그리고 `{children}`을 무조건 `props`으로 가지고 렌더링 해주어야 한다. 추가로 `parmas`객체를 통해 동적인 주소를 핸들링 할 수도 있다.

#### `page`

> https://nextjs.org/docs/app/api-reference/file-conventions/page

`layout`이 말그대로 레이아웃을 구성하기 위한 목적이라면, `page`는 그 레이아웃 내에 들어갈 내용을 작성하는 곳이다.

```typescript jsx
export default async function Page({
  params: { year, slug },
}: {
  params: { year: string; slug: string[] }
}) {
  // ...
  return <>...</>
}
```

`children`은 따로 필요 없으며, 동적인 주소에 대한 `params`와 `/hello?a=1`에서 `a=1`과 같은 `searchParams`을 추가로 받을 수 있다. 그리고 여기에 있는 내용이 위 `layout`의 `children`에 들어가게 된다.

#### 그 외

그 외에도 블로그에는 사용하지 않았지만, 로딩 상태를 나타낼 수 있는 [loading](https://nextjs.org/docs/app/api-reference/file-conventions/loading), api를 나타낼 수 있는 [route](https://nextjs.org/docs/app/api-reference/file-conventions/route), 에러 페이지인 [error](https://nextjs.org/docs/app/api-reference/file-conventions/error), 404 페이지인 [not-found](https://nextjs.org/docs/app/api-reference/file-conventions/not-found) 등이 있다. 블로그는 빌드 시점에 정적으로 완전히 다 빌드하기 때문에, `not-found`등만 추가하였다.

## `getStaticProps`와 `getStaticPaths`

### before

`getStaticPaths`는 미리 정해진 라우팅을 바탕으로 어떠한 주소가 가능한지를 정의하는 메서드고, `getStaticProps`는 앞서 정적으로 정한 주소에 사용자가 접근하였을 때 어떠한 `props`를 클라이언트에 반환할지 결정하는 메서드다. 먼저 구 블로그 코드를 살펴보자.

```typescript
// src/pages/[year]/[...slugs].tsx
export const getStaticPaths: GetStaticPaths = async () => {
  // 포스팅 가능한 모든 md 파일을 불러온다.
  const allPosts = await getAllPosts()

  // 불러온 정보를 Array<{ params: { year: string; slugs: string[] } } 로 반환한다.
  // ...

  // paths로 정의한 변수가 해당 페이지에서 접근가능한 페이지가 된다.
  return {
    paths,
    fallback: 'blocking',
  }
}
```

`paths`로 해당 페이지로 접근 가능한 주소를 나열한다음, `fallback: blocking`을 사용하면 빌드 시점에 모든 주소가 결정된다. 그리고 빌드 시점에 모든 페이지가 만들어지고, 사용자는 이렇게 정적으로 만들어진 페이지만 방문할 수 있게 된다. 사전에 빌드되지 않은 페이지를 방문하면 404가 반환된다.

```typescript
// src/pages/[year]/[...slugs].tsx
export const getStaticProps: GetStaticProps = async ({ params }) => {
  const { year, slugs } = params as SlugInterface

  const slug = [year, ...(slugs as string[])].join('/')
  // md 파일을 찾고 그중에 일치하는 파일을 반환한다.
  const posts = await getAllPosts()
  const post = posts.find((p) => p?.fields?.slug === slug)
  if (post) {
    const source = await parseMarkdownToMdx(post.body, post.path)

    return {
      props: {
        post,
        mdx: source,
      },
    }
  }
  return {
    notFound: true,
  }
}
```

`getStaticPaths`로 가능한 주소를 정의했다면, `getStaticProps`는 이제 해당 주소로 접근 했을 때 어떤 `props`를 반환할지 결정하게 된다. 여기에서는, `slugs`에 맞는 `markdown`파일을 찾고 이를 mdx로 직렬화하여 리액트에 반환한다.

### after

이제 `getStaticPaths`는 [generateStaticParams](https://nextjs.org/docs/app/api-reference/functions/generate-static-params)로 변경되었다.

```typescript
// src/app/[year]/[...slug]/page.tsx
export async function generateStaticParams() {
  // 마크다운 파일을 다 불러온다음
  const allPosts = await getAllPosts()
  // Array<{ year: string; slug: string[] } 로 반환한다.
  return allPosts.reduce<Array<{ year: string; slug: string[] }>>(
    (prev, { fields: { slug } }) => {
      const [year, ...slugs] = `${slug.replace('.md', '')}`.split('/')

      prev.push({ year, slug: slugs })
      return prev
    },
    [],
  )
}
```

`{params: {}}` 형태의 객체 였던 것과 다르게, 이제는 단순히 가능한 조합을 객체로 반환하면 된다. 이외에는 큰 차이가 없다.

이제 중요한 부분이 바로 마크다운을 렌더링하는 영역이다. 이제 `Page`가 `async`해지는 것이 가능해진다. 다음 예제를 보자.

```typescript jsx
export default async function Page({
  params: { year, slug },
}: {
  params: { year: string; slug: string[] }
}) {
  const post = await findPostByYearAndSlug(year, slug)

  if (!post) {
    return notFound()
  }

  return (
    <MDXRemote
      source={body}
      components={MDXComponents}
      options={{
        mdxOptions: {
          remarkPlugins: [remarkMath, remarkToc, remarkSlug, remarkGfm],
          rehypePlugins: [
            rehypeKatex,
            prism,
            parseCodeSnippet,
            rehypeAutolinkHeadings,
            imageMetadata(path),
          ],
        },
      }}
    />
  )
}
```

이 예제에서는 실 `getStaticParams`가 사라진 대신, `page`가 직접 `param`객체를 받아 렌더링한다. 그리고 이 작업은 비동기로도 가능해진다. `getStaticParmas`와 같은 예약어 함수명을 외우지 않아도 직관적으로 렌더링 할 수 있게 되어 더욱 편리해졌다.

#### 라우트 캐싱 정책

next@13 부터 [Route Segment Config](https://nextjs.org/docs/app/api-reference/file-conventions/route-segment-config)라고 하여 라우팅 별로 캐싱 정책등을 어떻게 가져갈지 선택할 수 있게 되었다. 해당 내용는 `page`에 별도 `export` 하는 변수로 선언하면 되고, 다음과 같이 동작한다.

- `dynamic`
  - `auto` (default): 컴포넌트가 가능하나 동적인 동작을 하지 못하도록 막으며 가능한 캐싱을 많이 하게 한다.
  - `force-dynamic`: 모든 캐싱을 비활성화 하고, 동적 렌더링 및 `fetch`를 수행한다. 이 옵션은 구 `getServerSideProps`와 동일하다.
  - ✅ `error`: 동적으로 가져오는 경우 에러를 발생시킨다. 다시 말하면 모든 페이지를 정적으로 렌더링하는 것을 강제한다. 이 옵션은 `getStaticProps`와 같으며 이 블로그가 이 옵션을 사용하였다.
  - `force-static`: 정적인 렌더링이 강제되고, 레이아웃이나 페이지에서 데이터 요청이 있을 경우 쿠키, 헤더, `searchParams`의 값이 모두 빈값으로 나온다.
- `dynamicParmas`: `generateStaticParams`로 생성되지 않은 파일을 방문했을 때 어떻게 동작할지 결정한다.
  - `true` (default): 해당 페이지 요청이 오면 파일을 생성한다.
  - ✅ `false`: 404를 반환한다. 위에서 만약 `force-static`나 `error`를 사용한다면 이 값이 자동으로 `false`가 된다.
- `revalidate`: 레이아웃과 페이지의 유효기간을 어떻게 가져갈지 정한다.
  - `false`: `Infinity`를 준것 과 동일하며, 무기한 캐싱된다. 단, 개별적으로 내부 페이지에서 `fetch`의 캐싱 동작을 오버라이드 하지는 않는다.
  - `0`: 동적 렌더링이 없어도 항상 페이지가 동적으로 렌더링 된다.
  - `number`: 특정 유효시간 (초) 를 정할 수 있다. 60으로 설정할 경우, 60초 마다 페이지가 렌덜이 될 것이다.

## og tag image

### before

과거 이 블로그는 ogtag 이미지 동적 생성을 위해 `generate-screenshot` 페이지와 서버리스 구글 cloud function을 사용하여 동적으로 블로그 포스트 썸네일을 생성했다. [관련 글](https://yceffort.kr/2020/12/generate-serverless-thumbnail) 이 방법은 개발자 입장에서는 재밌을지는 몰라도, 확실히 비효율적이긴했다.

### after

`opengraph-image.tsx`라는 예약어 파일이 생겼다. 이파일을 다음과 같은 형식으로 만들면, og tag image를 생성할 수 있다.

https://nextjs.org/docs/app/api-reference/file-conventions/metadata/opengraph-image

```typescript jsx
// app/opengraph-image.tsx
export const runtime = 'edge'

export const alt = SiteConfig.author.name
export const size = OpenGraphImageSize

export const contentType = 'image/png'

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <OpenGraphComponent
        title="Welcome to yceffort's blog"
        url="https://yceffort.kr"
        tags={['blog', 'frontend']}
      />
    ),
    { ...size },
  )
}
```

그러나 아직 애석하게도 `[...slug]`와 같은 동적인 데이터를 기준으로 og image를 만드는 것은 불가능해 보인다.

> https://github.com/vercel/next.js/issues/48162#issuecomment-1540040105

그러나 개발자의 말로 보아(?) 조만간 이 기능도 추가되지 않을까 싶다.

## metadata

과거 metadata는 `_document`에 일일이 추가해주어야 하는 굉장히 귀찮은 작업이었다. 그러나 이제는 `metadata`라고 하는 별도의 객체를 export 하면, 메타데이터를 필요에 따라 만들어준다.
https://nextjs.org/docs/app/building-your-application/optimizing/metadata

```tsx
// 정적인 경우
export const metadata: Metadata = {
  title: SiteConfig.title,
  description: SiteConfig.url,
  authors: [{ name: SiteConfig.author.name }],
  referrer: 'origin-when-cross-origin',
  creator: SiteConfig.author.name,
  publisher: SiteConfig.author.name,
  metadataBase: new URL('https://yceffort.kr'),
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  icons: {
    icon: '/favicon/apple-icon.png',
    shortcut: '/favicon/apple-icon.png',
    apple: '/favicon/apple-icon.png',
    other: {
      rel: '/favicon/apple-icon-precomposed',
      url: '/favicon/apple-icon-precomposed.png',
    },
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
    },
  },
  viewport: {
    width: 'device-width',
    initialScale: 1,
  },
}

// 동적인 경우
export async function generateMetadata({
  params: { year, slug },
}: {
  params: { year: string; slug: string[] }
}) {
  const post = await findPostByYearAndSlug(year, slug)

  if (!post) {
    return {}
  }

  return {
    title: post.frontMatter.title,
  }
}
```

이 `metadata`도 마찬가지로 `layout`에 따라 상속하거나 하위에서 재선언하는 등 작업이 가능하다.

## sitemap

과거 sitemap 생성을 하기 위해 빌드 이전에 별도로 모든 가능한 주소를 다 가져온 다음, 그 주소를 바탕으로 `xml`파일을 수동으로 만들어 `public`폴더에 밀어넣는 작업을 했었다.

이제는 `app/sitemap.ts`라는 예약어 파일을 만들면, 빌드 시점에 미리 sitemap도 생성해준다.

```typescript
import { MetadataRoute } from 'next'

import { getAllPosts, getAllTagsFromPosts } from '#utils/Post'

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const posts = await getAllPosts()
  const tags = await getAllTagsFromPosts()

  return [
    {
      url: 'https://yceffort.kr',
      lastModified: new Date(),
    },
    {
      url: 'https://yceffort.kr/about',
      lastModified: new Date(),
    },
    ...posts.map((post) => {
      return {
        url: `https://yceffort.kr/${post.fields.slug}`,
        lastModified: new Date(post.frontMatter.date),
      }
    }),
    ...tags.map((tag) => {
      return {
        url: `https://yceffort.kr/tags/${tag}`,
      }
    }),
  ]
}
```

## robots.txt

검색엔진에 도움이 되는 `robots.txt`도 설정이 가능하다. `app/robots.ts`를 다음과 같이 만들어 추가할 수 있다.

```typescript
import { MetadataRoute } from 'next'

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: '*',
      allow: '/',
    },
    sitemap: 'https://yceffort.kr/sitemap.xml',
  }
}
```

## 그 외 시행착오 와 소감

- 서버 컴포넌트를 본격적으로 지원하기 시작하면서, 내가 사용하는 라이브러리가 서버에서 사용가능한지, 클라이언트에서 사용가능한지 확인이 필요해졌다. 마크다운 렌더링을 위해 [next-mdx-remote](https://github.com/hashicorp/next-mdx-remote)를 사용했는데, 이 라이브러리를 서버에서 사용할 경우 내부적으로 `useState`를 사용하고 있어 렌더링 시 오류가 발생했다. 다행히 [해당 기능을 지원](https://github.com/hashicorp/next-mdx-remote#react-server-components-rsc--nextjs-app-directory-support)해줘서 큰 문제는 없었지만, 16.8 의 등장으로 훅을 지원하느냐 여부에 따라 리액트 라이브러리의 생태계가 많이 갈렸던 것 처럼 일대 혼란이 있을 것으로 보인다. 사내에서 만드는 라이브러리가 있는데, 이 라이브러리들이 어디까지가 서버컴포넌트에서 돌아갈지 고민해봐야할 필요가 있을 것 같다.
- `app`과 `pages`에 동일한 주소가 있을 경우 (당연히) 정상적으로 실행되지 않는다. 블로그의 경우 기능이 그렇게 많지 않아 과감하게 모두 날리고 다시 만들었지만, 실제 실무 프로젝트라면 당연히 그렇게 못했을 것이다. 따로 `new` prefix를 추가한 주소에서 `app`을 사용했을 것 같다.
- `next dev --turbo`를 사용해보았는데, 역시나 swc 때와 마찬가지로 베타라는 말이 무색하게 여기저기서 에러가 터졌었다. 물론 vercel 팀을 비난하려는건 아니고, 아무튼 사용에 주의가 필요해보였다. (사랑해요 vercel)
- [typescript 5.1 부터 비동기 컴포넌트를 정식으로 지원할 예정](https://devblogs.microsoft.com/typescript/announcing-typescript-5-1-rc/#decoupled-type-checking-between-jsx-elements-and-jsx-tag-types)이라서 현 버전에서는 `@ts-ignore`로 어글리하게 처리한 케이스가 몇개 있다.
- 생각보다 `pages`에서 `app`으로 전환하는데 사고가 빠르게 되지 않았다. `getServerSideProps`를 다른 프로젝트에서 마이그레이션 해보았지만, router segment 별로 caching 정책을 가져간다거나 `fetch`별로 캐싱을 하는게 익숙하지 않았다. 이 기분은 마치 next@8 인가 7을 내가 처음 써봤을 때 `getInitialProps`가 클라와 서버에서 동시에 실행될 때 느꼈던 혼란의 그것과 유사했다. 이 또한 적응 될 것이다. (늙어서 그렇지)
- 많은 사람들이 서버 컴포넌트가 가장 큰 핵심이라고 이야기 하지만, 개인적으로는 캐싱도 엄청 중요하다고 느끼게 되었다. 캐싱을 진짜 잘 만지면, 대규모 애플리케이션에서 `react-query`나 `swr`등이 없어도 데이터 호출을 효율적으로 다룰 수 있을 것 같다.
- 개인적으로 제일 기대하고 있는건 [서버액션](https://nextjs.org/docs/app/building-your-application/data-fetching/server-actions)이다. 읭 이거 완전 php 아닌가요 하며 트위터리안을 단체로 혼란에 빠트렸던 그것,, 이건 따로 기회가 된다면 다뤄볼까 한다.
