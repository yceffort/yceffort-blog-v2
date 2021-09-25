---
title: 블로그 gatsby에서 nextjs로 옮긴 이야기
tags:
  - gatsby
  - nextjs
  - react
published: true
date: 2020-10-16 22:23:24
description: '어영부영했지만 보람은 있었다'
---

## Table of Contents

예전부터 블로그는 내가 직접 만든 적이 없고, 사람들이 만들어 놓은 템플릿에 마크다운만 얹는 식으로 운영했었다. 그러다 보니 무언가 수정이 필요할 때 마다 코드를 잘모르니 땜빵식으로 수정하게 되고, 블로그에 대한 애정도 부족하지 않았나 싶다.

## 1. 나는 왜 Next.js로 갔나

이전까지 쓰던 블로그는 static site generator로 Gatsby 기반으로 운영되고 있었는데, 나 뿐 만 아니라 많은 사람들이 여러가지로 불편함을 느끼고 있는 분위기 였다. 🤔

https://cra.mr/an-honest-review-of-gatsby/

https://jaredpalmer.com/gatsby-vs-nextjs

나의 경험과 섞어서 gatsby가 안좋았던 경험을 이야기 하자면

- 정적 사이트를 만들 뿐인데 GraphQL은 너무 무겁고 쓸모가 없다. 개인적으로 블로그 운영하면서도 써본적이 없다.
- gatsby만의 생태계, gatsby-plugin-\*\*\*에 너무 의존적이다. 버그가 있어도 수정이 어렵고, 대부분의 플러그인이 관리도 잘 안되고 있었다.
- 디버깅이 어려웠다. Gatsby 내부의 graphql 및 webpack 등등으로 인해 추상화가 있는대로 되어 있어서 디버깅 하기가 정말 쉽지 않았다.
- GraphQL, 그리고 마크다운과 혼재되어 있는 이미지 빌드 등 각종 작업으로 인해 빌드가 너무 오래 걸렸다. (그래서 대부분의 경우에 나는 빌드 결과를 확인하지 않고 머지했다.)

## 2. 스펙

기존 개발 스택은 아래와 같다.

- gatsby
- graphql
- react
- javascript
- flow
- SCSS

바뀐 개발 스택은 아래와 같다.

- nextjs
- remark
- rehype
- react
- typescript
- styled component

굳이 혼자 쓰는 프로젝트에 타입스크립트가 필요가 있겠냐만은, 이제 자바스크립트 생태계 자체가 많이 타입스크립트 쪽으로 기우는 것 같아서 함께 썼다.

마크다운을 HTML로 만들기 위해 `remark`와 `rehype`를 썼고, `SCSS`는 그냥 별로 안좋아해서 (내 jsx 에 클래스명이 덕지덕기 붙어있는게 너무 보기 힘들다) `styled component`로 넘어갔다.

## 2. 이사 과정

### 1) nextjs

nextjs에서 정적 사이트를 만들기 위해서 중요한 것은 `getStaticProps` 와 `getStaticPaths`다.

https://yceffort.kr/2020/03/nextjs-02-data-fetching#2-getstaticprops

https://yceffort.kr/2020/03/nextjs-02-data-fetching#3-getstaticpaths

`getStaticPaths`는 빌드시에 가능한 다이나믹 path를 결정하고, `getStaticProps`는 페이지 로딩 시에 서버사이드에서 내려올 props를 결정한다.

### 2) 마크다운 파일 읽어오기

내 마크다운 파일은 `/content/posts/articles`에 존재하고 있다. gatsby에서는 이것을 graphql로 처리하지만, nextjs에서는 그런 것이 없기 때문에 직접 파일시스템에 접근해서 재귀적으로 모든 `.md`파일을 찾아와서 [frontMatter](https://github.com/jxson/front-matter)로 읽어 왔다. 그리고 이렇게 읽어 온 마크다운을 HTML로 변환해주어야 한다. 이를 위해 [unified](https://github.com/unifiedjs/unified), [remark](https://github.com/remarkjs/remark) [rehype](https://github.com/rehypejs/rehype) 를 사용했다.

### 3) flow를 타입스크립트로 변환하기

이 과정이 젤 쉬웠다.

### 4) SCSS를 styled component로 변환

워낙 디자인 감각이 없는 대다가, SCSS에 postCss에 lostGrid 까지 사용되어 있어서 개인적으로 제일 고통스러운 과정이었다. 일일이 CSS를 보면서 적용했다.

## 3. 난관

### 1) Dynamic Path

내 블로그 글들을 nextjs dynamic path 문법으로 작성하면 다음과 같다.

```bash
- /[year]/[month]/[day]/[title]
- /[year]/[month]/[title]
```

그래서 아래와 같이 pages 디렉토리를 설정해주었는데

```bash
- /[year]/[month]/[day]/[title].tsx
- /[year]/[month]/[title].tsx
```

`day`와 `title`이 겹치면서 빌드가 되지 않았다. koa를 쓰면 라우팅 순서대로 타기 때문에 상관없지만, 여기까지 와서 koa를 쓰고 싶지 않아서 (...) 결국 아래와 같이 만들었다.

```bash
- /[year]/[month]/[day]/[title].tsx
- /[year]/[month]/[day]/index.tsx
```

이렇게 하면 세번째 path가 `day`로 동일해지기 때문에, 더 이상 에러가 나지 않는다. 다만 `index.tsx`에서 `day`가 아니라 `title`이라는 사실을 염두해두고 개발에 해야한다.

### 2. 마크다운 파서

내 마크다운은 다음과 같은 기능을 반드시 제공 해야만 했다.

- toc 자동생성
- latex 문법 지원
- heading link
- raw html 지원 (iframe 등). 어차피 글은 나만 쓰기 때문에 상관없다.
- 코드 하이라이팅

이를 완벽하게 지원하기 위하여 참으로 많은 삽질을 거쳤다. 특별한 노하우가 있는게 아니고 그냥 삽질의 과정이 었다. 약간의 시간을 거친 끝에, 잘 만들었다.

https://yceffort.kr/2020/07/math-for-programmer-chapter2-3-rational-irrational-real-number

### 3. 사이트맵

사이트맵도 gatsby에서는 graphql로 잘 만들어줬지만, 여기서는 내가 다 만들어야 했다.

### 4. 리다이렉트

`/categories` url이 존재하는지도 몰랐는데, 있긴 있더라. 그래서 이것들을 다 리다이렉트 시켰다. `next.config.js`에서 다 처리할 수 있다.

```javascript
module.exports = {
  async redirects() {
    return [
      {
        source: '/tag/:tag',
        destination: '/tag/:tag/page/1',
        permanent: true,
      },
      {
        source: '/category/:tag',
        destination: '/tag/:tag/page/1',
        permanent: true,
      },
      {
        source: '/categories',
        destination: '/tags',
        permanent: true,
      },
    ]
  },
}
```

## 3. 결과

일단 빌드 시간이 엄청나게 빨라졌다.

![gatsby](./images/gatsby-build.png)

7분 가까이 빌드에 소요되었었다. ㅠㅠ

```bash
16:54:32.265  	Running "npm run build"
...

16:56:13.131  	success Rewriting compilation hashes - 0.006s
16:59:23.076  	success Building static HTML for pages - 7.784s - 655/655 84.15/s
16:59:24.034  	success Generating image thumbnails - 275.708s - 2434/2434 8.83/s
```

정적파일 빌드 하는데 3분이 넘게 걸렸고, 쓸데 없이 이미지 썸네일 만드는데에 5분까지 걸렸다.

![nextjs](./images/nextjs-build.png)

빌드도 빨라지고, 빌드 결과물도 기존 232.34mb에서 29.02mb로 다이어트 할 수 있었다. (nextjs는 다이나믹 path에 대해서 모두 빌드해두는 것이 아니라, 최초 페이지 접근 요청이 올 때만 빌드하고, 이 후 접근엔 이전에 빌드해둔 페이지를 보여준다.)

또한 기존에 찾기가 너무 어려웠던 mathjax 문법 오류도 다 고칠 수 있었다.

### github issue & board

https://github.com/yceffort/yceffort-blog-v2/projects/1

https://github.com/yceffort/yceffort-blog-v2/issues?q=is%3Aissue+is%3Aclosed+label%3A%22%F0%9F%91%B7%E2%80%8D%E2%99%82%EF%B8%8F+Next.js%22

(볼 때 마다 느끼지만, 깃헙 라벨링 정말 이쁘다)

## 4. 아쉬운점

- CSS를 옮기면서 약간 생각 없이 그냥 가져온 부분들이 많다. CSS에 대해 더 분석할 필요가 있을 듯
- nextjs는 정적이미지를 `public`디렉토리에서 서빙해야되는데, 이거 때문에 이미지를 다 이 디렉토리로 옮겨 왔다. 그러다 보니 글 쓰는 과정 마크다운 프리뷰에서 이 이미지들을 제대로 볼 수 가 없었다. dev 환경에서 볼 수 는 있지만, dev 에서 .md 파일에 대해서는 HMR이 안되었다. 향 후 아이디어가 필요해 보인다.
- 일부 type이 없는 패키지가 있었는데, 맘이 급해서 `@ts-ignore` 로 무시하고 갔다. `@mapbox/rehype-prism` `remark-slug` 시간 나는대로 만들어야겠다.

## 5. 개선 예정 사항

- 글 오른쪽 하단 플로팅 버튼으로 글 최상단에 올라갈 수 있는 기능
- algolia를 활용한 검색 (디자인까지 건드려야 되서 매우 귀찮을 듯)
- 조금 더 예쁜 about, contact
