---
title: 'Nextjs 11 릴리즈 노트 살펴보고 블로그에 적용하기'
tags:
  - javascript
  - nextjs
  - react
published: true
date: 2021-06-19 22:12:48
description: 'nextjs 정말 열일하네2222'
---

작년 10월 쯤에 nextjs 10을 적용해보고 릴리즈 노트를 살펴보았는데, 어느덧 nextjs 11까지 나오게 되었다. 8개월 만에 메이저 버전을 업데이트 했는데, 정말 놀랍다. 반년 남짓한 사이에 또 발전을 만들어 냈는데 대단하다는 생각이 드는 한편으로 많이 자극도 되었다. 11 버전에서는 어떤 것들이 달라졌는지 살펴보자.

https://nextjs.org/blog/next-11

## Table of Contents

## CHANGELOG

### [Conformance](http://web.dev/conformance)

Conformance (한글로는 일치,부합,적합성이라고 하는데 딱히 뭐라고 번역해야 할지 모르겠다. 그냥 Conformance라 부르겠다.) 는 최적의 로딩 및 Core web vital을 지원하기 위해 만들어진 시스템으로, 보안 및 접근성과 같은 여러 품질 측면의 다양한 리소스를 지원하기 위한 기능이 제공된다. 이를 활용하면, 최적의 성능을 내기 위한 다양한 규칙들을 개발자들이 외우고 다닐 필요가 없이, 적합한 옵션을 선택할 수 있다.

또한 이번에 [eslint-config-next](https://github.com/vercel/next.js/tree/canary/packages/eslint-config-next)도 제공하면서, 개발 중에 생기는 프레임워크 관련 문제를 더 쉽게 파악할 수 있고, 개발 시에도 베스트 프랙티스를 보장할 수 있는 여러가지 가이드라인을 제공하게 되었다.

```bash
» npx next lint
info  - Loaded env from /Users/yceffort/private/yceffort-blog-v2/.env
info  - Using webpack 5. Reason: Enabled by default https://nextjs.org/docs/messages/webpack5
✔ No ESLint warnings or errors
```

뭐 근데, 사실 살펴보니까 대단한 룰들이 있지는 않았다.

https://github.com/vercel/next.js/blob/afa86cc5bbd488d123e2c5888205a40a5a0afe42/packages/eslint-config-next/index.js#L16-L27

```json
{
  "rules": {
    "import/no-anonymous-default-export": "warn",
    "react/react-in-jsx-scope": "off",
    "react/prop-types": "off",
    "jsx-a11y/alt-text": [
      "warn",
      {
        "elements": ["img"],
        "img": ["Image"]
      }
    ]
  }
}
```

하지만 주목할 만한 것은 `eslint-config-next`보다 `eslint-plugin-next` 였다.

https://github.com/vercel/next.js/tree/canary/packages/eslint-plugin-next/lib/rules

여기에 있는 룰들을 간단히 요약해보았다.

- `google-font-display`: 구글 폰트에 `font-display` 속성이 적절히 되어 있는지 여부
- `google-font-preconnect`: 구글 폰트에 `preconnect`가 사용되고 있는지 여부 (rel 속성)
- `link-passhref`: [next/link](https://nextjs.org/docs/tag/v9.5.2/api-reference/next/link#if-the-child-is-a-custom-component-that-wraps-an-a-tag)의 하위 컴포넌트가 커스텀으로 있다면 `passhref`를 넘겨주었는지 여부
- `no-css-tags`: html link 엘리먼트가 외부 스타일 시트를 불러오는 경우 (웹 페이지의 css 성능에 악영향)
- `no-document-import-in-page`: `next/document`가 `pages/_document.*sx`외부에서 사용되는 경우
- `no-head-import-in-document`: `pages/_document.*sx` 내부에 `next/head`를 쓰는 경우
- `no-html-link-for-pages`: `pages` 디렉토리가 없는 경우
- `no-img-element`: `<img/>` 대신 `<next/image>`를 쓰세여. `next/image`가 더 좋다. https://nextjs.org/docs/api-reference/next/image
- `no-page-custom-font`: custom font는 `pages/_document.*sx`에서 불러오세여
- `no-sync-script`: 외부 스크립트를 sync로 불러오지 말것
- `no-title-in-document-head`: `next/document`에 `<Head>`에 `<title>`에 있으면 안된다.
- `no-unwanted-polyfills`: next에서 이미 기본으로 제공하는 폴리필을 중복으로 불러오지 말것

대부분이 next와 관련된 룰이었지만, 이번에 구글과 함께 Conformance를 적용하면서 Core web vital을 많이 신경을 쓰기 시작한 것 같다. vercel에도 이와 관련한 기능이 있기도 하고.

![partner drive process](https://web-dev.imgix.net/image/0SXGYLkliuPQY3aSy3zWvdv7RqG2/QFTQX7npdBsFheXIqbuc.png?auto=format&w=845)

이전까지는 구글이 개발자들에게 제발좀 Core web vital을 지키세여!!! 라고 외치는 전략이었다면, 이번에는 영리하게도 프레임워크를 만들고 있는 파트너 (리액트, 뷰, 타입스크립트, 바벨 등)을 공략하여 이러한 프레임워크를 쓰고 있는 개발자라면 자연스럽게 Core Web Vital을 챙길 수 있게 끔 하는 전략으로 바꾼 것으로 보인다.

사실 잘 돌아가고 있는 앱에 (사실 그렇게 보이는 앱을..) 성능이 중요하다, core web vital이 중요하다, 라고 백날 이야기 해봐야 이를 챙기는 프론트엔드 개발자가 몇이나 있을까? 그래서 제발 좀 이를 지켜주세요!! 라고 하는 것보다는, 프레임워크 레벨에서 아예 이를 세트로 묶어서 공략한다면 번거롭게 추가로 무언가를 한다는 거부감을 줄일 수 있을 것 같다.

그리고 전체적인 프론트엔드 애플리케이션의 품질이 좋아지면, 구글의 웹페이지 정보 수집에도 많은 발전이 있을 것으로 보인다. 구글의 이러한 전략에 대한 내용은 https://web.dev/introducing-aurora/ 에 있는데, 나중에 나도 한번 다뤄봐야겠다.

### 성능 향상

개발자들의 개발 경험을 향상 시키기 위해, 10.1, 10.2 에서는 시작시간을 최대 24% 단축했고, React fast refresh를 활용해 개발시 업데이트 반영시간을 40%까지 단축했다. 그리고 이번 11에서는 시간을 더 줄이기 위해 babel에 추가적인 최적화가 포함되었다. 개발자들에게 느껴지는 직접적인 코드의 변화는 없지만, 개발 환경에서 더 빠른 환경을 제공할 수 있게 되었다.

### 스크립트 최적화

[next/script](https://nextjs.org/docs/basic-features/script)가 탄생했다. 써드파티 스크립트를 로딩할 때 시간과 성능을 개선할 수 있도록 도와준다.

```jsx
function Home() {
  return (
    <>
      <Script src="https://www.google-analytics.com/analytics.js" />
    </>
  )
}
```

이 `<Script>`는 `strategy` 속성있고, 다음의 값을 가질 수 있다.

- `beforeInteractive`: 페이지가 활성화되기전에 (번들된 자바스크립트가 실행되기전에) 스크립트를 가져오고 실행한다. 스크립트가 SSR된 html내부에 주입된다.
- `afterInteractive`: 페이지가 활성화 된 이후 (번들된 자바스브립트가 모두 실행되고) 스크립트를 가져오고 실행한다. 스크립트를 hydration과정에서 주입하고, 이 후 즉시 실행된다.
- `lazyOnload`: `onload` 시점에 스크립트를 실행한다. `requestIdleCallback`을 활용하여 idle 상태가 되면 바로 실행한다.

```jsx
<Script
  src="https://polyfill.io/v3/polyfill.min.js?features=Array.prototype.map"
  strategy="beforeInteractive" // lazyOnload, afterInteractive
/>
```

`onLoad` 속성도 생겼다.

```jsx
<Script
  src={url} // consent mangagement
  strategy="beforeInteractive"
  onLoad={() => {
    // 로딩이 끝나면 그 이후에 실행됨. 그 이후에 스크립트를 로딩하거나 할 수 있음.
  }}
/>
```

또한 기본 script 로딩을 `async`에서 `defer`로 변경했다. `defer`가 더 좋은 이유, 이러한 변경을 하게된 계기는 아래 링크를 참조하자.

- https://yceffort.kr/2020/10/defer-than-async
- https://github.com/vercel/next.js/discussions/24938
- https://docs.google.com/document/u/0/d/1ZEi-XXhpajrnq8oqs5SiW-CXR3jMc20jWIzN5QRy1QA/mobilebasic#

### 이미지 최적화

[Cumulative Layout Shift](https://vercel.com/blog/core-web-vitals#cumulative-layout-shift)란 이미지 로딩으로 인해 사이트의 레이아웃이 갑자기 밀리거나 변하는 현상을 의미한다. `next/image`가 이 현상을 개선했다고 한다.

- 로컬이미지에 대한 사이즈 감지: `src`에 들어가 있는 로컬 이미지에 대해 너비와 높이를 자동으로 정의 한다고 한다.

```jsx
import Image from 'next/image'
import author from '../public/me.png'

export default function Home() {
  return (
    // 로컬 이미지를 불러오면, 알아서 width height를 계산한다.
    <Image src={author} alt="Picture of the author" />
  )
}
```

- 이미지 placeholder: 이미지가 완전히 로딩 되기전에 자동으로 블러된 이미지를 보여주는 placeholder를 지원한다.

지금 내 블로그의 헤더이미지, 프로필이미지, about 이미지 등에 적용되어 있다. (본문 이미지는 mdx로 되어있어서 적용되어있지 않다.)

```javascript
<Image src={author} alt="Picture of the author" placeholder="blur" />
```

또는 `blurDataURL`을 직접 넣어서 구현가능하다.

```jsx
<Image
  src="https://nextjs.org/static/images/learn.png"
  blurDataURL="data:image/jpeg;base64,/9j/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAIAAoDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAb/xAAhEAACAQMDBQAAAAAAAAAAAAABAgMABAUGIWEREiMxUf/EABUBAQEAAAAAAAAAAAAAAAAAAAMF/8QAGhEAAgIDAAAAAAAAAAAAAAAAAAECEgMRkf/aAAwDAQACEQMRAD8AltJagyeH0AthI5xdrLcNM91BF5pX2HaH9bcfaSXWGaRmknyJckliyjqTzSlT54b6bk+h0R//2Q=="
  alt="Picture of the author"
  placeholder="blur"
/>
```

### Webpack 5

nextjs에 webpack5가 이제 기본으로 지원된다. 이로 인한 이점은 [여기](https://nextjs.org/blog/next-10-2#webpack-5)에 나와 있다. 만약 webpack 5 미만 버전으로 custom 옵션을 사용하고 있다면, [여기](https://nextjs.org/docs/messages/webpack5)에서 마이그레이션을 할 수 있다.

### create-react-app migration

[@next/codemod](https://nextjs.org/docs/advanced-features/codemods)는, nextjs에서 deprecated된 기능을 자동으로 변환해주는 도구다. 여기에는 `/pages` 디렉토리를 생성하고, css를 적절한 위치로 import 하는 등의 옵션이 포함되어 있다. 여기에 cra로 생성된 앱을 점진적으로 nextjs로 바꿔주는 기능이 추가되었다.

```bash
npx @next/codemod cra-to-next
```

아직은 실험단계라고 한다.

### Next.js Live

https://nextjs.org/live 는, next를 활용한 전체 개발 프로세스를 웹 브라우저에서 할 수 있도록 도와주는 애플리케이션이다. 빌드단계를 생략하고 URL로 즉시 개발이 가능해지고, 공동작업도 가능하다고 한다. 아직 early access 단계라서 직접 사용해보지는 못했지만,

![nextjs.live](https://nextjs.org/_next/image?url=%2F_next%2Fstatic%2Fimage%2Fpublic%2Fstatic%2Flive%2Fbrowser.f5aa736f88c19b45aa423f7b1c0ca58f.png&w=3840&q=75)

모습을 보아하니, nextjs 애플리케이션을 stackblitz 처럼 웹에서 실시간으로 개발할 수 있게 해주고, 거기에 다른 디자이너나 기획자들이 협업할 수 있도록 도와주는 도구 인 것 같다.

이를 위해 서비스워커, 웹어셈플리, ESModule, [sucrase](https://github.com/alangpierce/sucrase), Tailwind JIT 등의 최신 기술을 집약했다고 한다.

### React 버전

최소 리액트 버전이 17.0.2로 업데이트 되었다고 한다.

### Upgrade guide

https://github.com/vercel/next.js/blob/canary/docs/upgrading.md

## 블로그 적용 후기

일단 새롭게 추가된 `<next/script>`, `<next/image>`의 blur를 위주로 적용했다. 개발 속도에서의 개선은 체감상 크게 못느꼈지만 (상대적으로 무겁지 않은 애플리케이션이라 더더욱), `Script` `blur` 등의 기능은 유용하게 쓸 수 있었다. 그리고 Conformance를 읽으면서 블로그의 라이트 하우스에 소홀했었다는 점을 떠올리며 점수를 끌어올렸다.

### before

![before](./images/before-blog.png)

### after

![after](./images/after-blog.png)

> 이정도면 꽤 만족스럽게 끌어올렸다. 당분간은 라이트 하우스를 쳐다보지 않아도 될 것 같다.

nextjs가 이렇게 훌륭한 프레임워크를 제공해주어서 좋을 따름이지만, 그 뒤에 숨겨져 있는, 프론트엔드 개발에 필요한 많은 요소들을 놓치거나 혹은 이것들을 상대적으로 덜 중요한 것으로 생각하는 경우도 종종 보게 된다. 어떻게든 좋은 애플리케이션을 만들면 되는 것은 회사와 블로그에서는 중요한 것일 수도 있지만, 그 뒤에 `babel`, `webpack`, `eslint` 가 어떻게 동작하고 있는지, `blur`처리는 어떻게 자동으로 되고 있는지, `Script`의 각 값별로 어떤식으로 동작하는 지 등을 이해하는 것은 개발자로서 성장하는데 있어 정말 중요한 일이다. 무지성 버전업, 이후 documentation 감상 후 적용 또한 좋은 일이지만, 이 뒤에서 무엇이 일어나고 있는지, 오픈소스 컨트리뷰터 들이 어떠한 고민을 하고 있는지도 뒤늦게나마 함께 공부해본다면 분명 유의미한 일이 될 것이다.
