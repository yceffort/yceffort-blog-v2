---
title: 'Nextjs 10 릴리즈 및 적용 후기'
tags:
  - javascript
  - nextjs
  - react
published: true
date: 2020-10-28 19:24:01
description: 'nextjs 정말 열일하네'
---

nextjs 10.0.0이 릴리즈 되었다. 내가 좋아하는 오픈소스 중 하나 이기 때문에, 릴리즈 노트를 읽어보면서 당장 내 블로그에 적용해보았다. 그리고 적용하면서 어떤게 바뀌었는지 하나씩 확인해보려고 한다.

- 릴리즈노트: https://nextjs.org/blog/next-10

## 빌트인 이미지 컴포넌트, 그리고 자동 이미지 최적화

이미지는 마크업과 함께 웹에서 큰 트래픽을 유발하는 범인중 하나다. 이러한 이미지를 최적화 하기 위해, `next/image` 라는 전용 이미지 컴포넌트를 적용하였다. 브라우저에서 `Webp`가 사용 가능하다면, 해당 이미지를 변환해서 적은 용량으로 내려주고, 동시에 lazy loading도 해준다고 한다.

### 대상 이미지

![blog profile](./images/profile.png)

### 적용전

```typescript
const AuthorPhoto = styled.image`
  display: inline-block;
  margin-bottom: 0;
  border-radius: 50%;
  background-clip: padding-box;
  width: 75px;
  height: 75px;
  cursor: pointer;
`

return <AuthorPhoto alt={name} src={photo} />
```

```html
<img
  alt="yceffort"
  src="/profile.png"
  class="Author__AuthorPhoto-sc-1ywmx02-0 fMrwt"
/>
```

```bash
accept-ranges: bytes
access-control-allow-origin: *
age: 1331855
cache-control: public, max-age=0, must-revalidate
content-disposition: inline; filename="profile.png"
content-length: 30710
content-type: image/png
date: Wed, 28 Oct 2020 02:05:42 GMT
etag: W/"078a2ad86a1350e007d801e7f74b073ed415e5bdd60a60e5b65b9fafe972af03"
server: Vercel
status: 200
strict-transport-security: max-age=63072000w
x-content-type-options: nosniff
x-frame-options: DENY
x-vercel-cache: HIT
x-vercel-id: icn1::chhv4-1603850742953-91ba1434fc9c
x-xss-protection: 1; mode=block
```

### 적용후

그리고 이를 `next/image`로 아래와 같이 바꿨다.

```typescript
import Image from 'next/image'

const AuthorPhoto = styled(Image)`
  display: inline-block;
  margin-bottom: 0;
  border-radius: 50%;
  background-clip: padding-box;
  width: 75px;
  height: 75px;
  cursor: pointer;
`

// width와 height를 지정해줘야 한다.
return <AuthorPhoto alt={name} src={photo} width={75} height={75} />
```

```html
<img
  alt="yceffort"
  data-src="/_next/image?url=%2Fprofile.png&amp;w=320&amp;q=75"
  data-srcset="/_next/image?url=%2Fprofile.png&amp;w=320&amp;q=75 320w"
  class="Author__AuthorPhoto-sc-1ywmx02-0 fMrwt"
  style="visibility: visible; height: 100%; left: 0px; position: absolute; top: 0px; width: 100%;"
  src="/_next/image?url=%2Fprofile.png&amp;w=320&amp;q=75"
  srcset="/_next/image?url=%2Fprofile.png&amp;w=320&amp;q=75 320w"
/>
```

```bash
accept-ranges: bytes
access-control-allow-origin: *
age: 276
cache-control: public, max-age=0, must-revalidate
content-disposition: inline; filename="profile.png"
content-length: 18898
content-type: image/webp
date: Wed, 28 Oct 2020 02:06:37 GMT
server: Vercel
status: 200
strict-transport-security: max-age=63072000; includeSubDomains; preload
x-content-type-options: nosniff
x-frame-options: DENY
x-robots-tag: noindex
x-vercel-cache: HIT
x-vercel-id: icn1::q6js7-1603850797310-7e6bd3beb3f0
x-xss-protection: 1; mode=block
```

일단 이미지가 눈에 띄게 lazy loading이 되었고 (나중에 떴고) 이미지 사이즈도 webp를 사용하면서 눈에 띄게 줄어든 모습이다. 구글 크롬팀에서 이미지 성능을 향상 시킬 수 있는 리액트 컴포넌트를 만들 수 있도록 도와주었다고 하는데, 나중에 소스 코드를 보는 것도 재밌을 것 같다.

## 국제화 라우팅

당장 내가 쓸 일이 있을지는 모르겠지만, 언어별 라우팅을 지원한다. 그리고 최신 브라우저가 지원하는 `Accept-language`헤더를 기반으로 언어 감지를 할 수 있는 기능도 추가되었다고 한다.

```javascript
// next.config.js
module.exports = {
  i18n: {
    locales: ['en', 'nl'],
    domains: [
      {
        domain: 'example.com',
        defaultLocale: 'en',
      },
      {
        domain: 'example.nl',
        defaultLocale: 'nl',
      },
    ],
  },
}
```

## 성능 분석

nextjs를 사용하는 웹 어플리케이션의 성능을 분석할 수 있는 도구를 지원한다.

https://nextjs.org/analytics

당장은 vercel만 지원하는 것 같은데(?) 운좋게도(?) 블로그가 vercel로 서빙되고 있기 때문에 당장 시도해보러 갔다.

![analytics](./images/analytics1.png)

![analytics](./images/analytics2.png)

(점수가 깎이는 것은 아마도 메인페이지의 페이지 전환 버튼 때문인 것 같다. 현재 타이틀 제목 길이에 따라서 페이징 버튼이 위아래로 일관되지 못하게 움직이는 버그가 있다.)

일단 모양새와 제공데이터는 구글 라이트 하우스와 비슷한데, 차이가 있다면 page 별로 데이터도 지원한다는 것이다. 다만 서두에도 말한 것 처럼 vercel을 써야만 누릴 수 있는 기능이라 🤔 근데 라이트 하우스에 비해 크게 차별점도 없다면...

## 커머스 키트 제공

https://nextjs.org/commerce

Next.js가 본격적으로 돈을 벌 만한 비즈니스를 시작하는 것 같다. 간단하게 말헤 next.js로 커머스 사이트를 만들 수 있는 도구를 제공한다고 한다.

## 리액트 17 지원

react 17을 지원한다. breaking change가 없으므로 바로 적용 가능하며 react를 import 하지 않아도 사용할 수 있는 jsx transform 기능도 사용할 수 있게 되었다.

https://reactjs.org/blog/2020/09/22/introducing-the-new-jsx-transform.html

## getStaticProps, getServerSideProps 빠른 새로고침

이제 해당 두 함수에서 코드 변화가 일어나면, 자동으로 새로고침을 지원한다고 한다. 이게 안되었다는 걸 이제 알았는데(??????) 해보니까 잘된다.

## 써드 파티 리액트 컴포넌트에서 css 임포트 가능

```javascript
import DatePicker from 'react-datepicker'
import 'react-datepicker/dist/react-datepicker.css'
```

이를 바탕으로, 단일 컴포넌트 레벨에서 CSS 코드 스플리팅이 가능해졌다. 자세한 내용은 https://nextjs.org/docs/basic-features/built-in-css-support

## `href` 자동화

이전까지, 다이나믹 라우팅을 사용하기 위해서 `next/link`에서 `href` `as`를 넣어 주어야 했다.

```html
<link href="/categories/[slug]" as="/categories/books" />
```

여기서 `as`는 실제 브라우저 URL 바에서 보이는 주소인데, 이전까지는 `href`와 `as`를 아래와 같이 따로 넣어주어야 했다.

```javascript
const pids = ['id1', 'id2', 'id3']
{
  pids.map((pid) => (
    <Link href="/post/[pid]" as={`/post/${pid}`}>
      <a>Post {pid}</a>
    </Link>
  ))
}
```

그러나 나를 비롯해서 많은 개발자들이 `as`의 사용에 대해서 혼란이 있었던 것 같다. (`as`를 넣어야 하는데 까먹는 다든지) 그래서 `as`가 더 이상 필요없어졌다고 한다. 이제는 기존에 `as`에 넣었던 값을 `href`에 넣어주면 된다. 그게 훨씬 더 이전의 리액트나 HTML경험에서도 자연스러워 보인다.

## `@next/codemod` cli

nextjs 에서 기능이 deprecated 되는 등의 대규모 코드 베이스 변경이 필요한 경우에 사용할 수 있는 툴이다. codemode는 소스 코드 업데이트를 위해 프로젝트에서 실행할 수 있는 자동화 코드 변경 툴이다.

https://nextjs.org/docs/advanced-features/codemods

## `getStaticPaths`에 블로킹 fallback 추가

`getStaticProps` 와 `getStaticPaths`에 `fallback` 속성이 추가되어 있었다. 이 속성은 최초에는 정적 페이지 (fallback 페이지)를 제공하고, 이후 요청 시에는 완전히 렌더링된 콘텐츠를 제공할 수 있도록 도와주는 기능이었다. 그러나 몇몇 개발자들이, 사용자가 페이지를 처음 요청할때는 사전 렌더링을 차단할 수 있는 옵션을 요구했었나 보다. (나또한 fallback 페이지를 보여주는 것이 사용자들에게 안좋은 사용자 경험을 제공한다고 생각했다. ) 그래서 `blocking`옵션이 추가되었다.

```javascript
export function getStaticPaths() {
  return {
    fallback: 'blocking',
  }
}
```

이 옵션을 추가하면, fallback 페이지를 보여주는 대신에, 그냥 최초 렌더링이 서버에서 내려올 때 까지 기다린다.

## 결론

여러가지로 nextjs는 정말로 잘 관리되고 있는 리액트 SSR 프레임워크다. 단순히 nextjs 뿐만 아니라 여러가지로 nextjs를 중심으로 다양한 생태계를 만들어 가려는 것이 보인다.

모질라나 webpack 등 순수하게 기부로 운영되고 있는 대규모 오픈소스 프로젝트들이 코로나 시국이 닥치면서 생존 문제에 직면한 것을 보고 마음이 안타까웠다. 이러한 문제를 vercel도 알고 있는 듯, 여러가지로 수익화를 하려는 노력이 보였다. `vercel`, `next commerce` 등의 비즈니스를 통해서, 단순히 기부에 의존하는 것이 아니라 다각도로 생존에 대해 고민하고 있는 것이 보인다. 개인적으로 다 잘되었으면 하는 바람이다.

회사를 옮기면서 nextjs는 업무에서는 더 이상 쓰지 못하고 있지만 (ㅠㅠ) 블로그나 개인 프로젝트에서는 나름 적극적으로 사용하고 있었다. 심지어 지금 하고있는 nextjs conf 도 꼬박 꼬박 잘챙겨보고 있다. (미리 신청해서 티켓도 받아놨었는데 어디간지 모르겠네)

전 회사 사람들은 next github에 issue 도 올려서 contribute도 했는데 아직도 나는 가져다 쓰고 구글링 하기에 바쁘다 😇

같은, 그리고 연차도 더 많은 개발자로서 부끄럽지 않을 수가 없는 일이다. 조만간 오픈소스에 기여할 날도 오기를 바라며 열심히 공부를 해야겠다.
