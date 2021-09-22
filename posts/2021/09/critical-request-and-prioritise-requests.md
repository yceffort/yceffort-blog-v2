---
title: 'Critical Request - request 순서는 웹사이트 속도에 어떤 영향을 미치는가'
tags:
  - javascript
  - browser
  - web
  - css
published: true
date: 2021-09-22 10:24:48
description: '서순을 정확히하는게 중요하지'
---

## Table of Contents

## Introduction

웹 사이트를 서비스하는 것은 굉장히 간단해 보일 수 있다. HTML을 내려주면, 브라우저는 다음에 어떤 리소스를 불러올지 알아낸다. 그런 다음, 브라우저가 페이지를 준비하기 까지 기다리기만 하면 된다. 그러나 사실 이 사이에는 많은 일이 일어난다. 브라우저는 과연 어떤 순서로 에셋을 요청해야 할까?

## 1. 에셋 우선순위란 무엇인가?

모던 브라우저는 streaming parser를 활용하여 HTML 구문을 붆석한다. 즉, 에셋은 다운로드 되기 전에 HTML 마크업 문서 내에서 찾을 수 있다. 브라우저가 에셋을 검색할 때, 미리 결정된 우선순위에 기반하여 네트워크 대기열에 해당 에셋을 추가시켜 둔다.

이러한 우선순위는 Lowest, Low, Medium, High, Highest라고 불리는 다섯단계로 나눠서 결정된다. 여기에서 우선순위를 할당하면, 브라우저는 페이지를 빠르게 로드하는데 필요한 가장 중요한 요청을 쉽게 구별할 수 있다.

![chrome-network-priority](./images/chrome-network-priority.png)

## 2. 크롬은 어떻게 우선순위를 결정하는가?

리소스는 어떻게 나타나는지, 그리고 어디서 발견되는지 순서에 따라서 네트워크 대기열에 추가된다. 그런 다음 브라우저는 네트워크 작업에서 가장 우선 순위가 높은 리소스를 최대한 빨리 가져오려고 시도한다.

각 리소스 유형에는 우선순위를 지정하는 규칙이 아래와 같이 존재한다.

| 리소스 타입                      | 우선순위                                                   |
| -------------------------------- | ---------------------------------------------------------- |
| HTML                             | Highest                                                    |
| Fonts                            | High                                                       |
| Stylesheets                      | Highest                                                    |
| `@import`로 불러오는 stylesheets | Highest, 스크립트를 블로킹 한 뒤에 로딩 됨                 |
| 이미지                           | 기본 값은 low, 최초 뷰포트에 존재할 경우 Medium으로 올라감 |
| 자바스크립트                     | 하단 표 참조                                               |
| Ajax, XHR, `fetch()` 등          | High                                                       |

### 자바스크립트 리소스의 우선순위

#### `<head>`내 `<script>`

- 로딩 우선순위 (네트워크, 블링크 엔진): Medium/High
- 실행 우선순위: 매우 높음, parser 블로킹
- 어디서 쓸까
  - FMP, FCP에 영향을 미치는 스크립트
  - 다른 스크립트 이전에 반드시 실행되어야 하는 스크립트
- 예제
  - 프레임워크 런타임 (스태틱 렌더링이 아닌 경우)
  - 폴리필
  - 페이지 전체의 DOM 구조에 영향을 미치는 A/B 테스트 등

#### `<link rel=preload>` + `<script async>` 또는 `<script type=module async>`

- 로딩 우선순위 (네트워크, 블링크 엔진): Medium/High
- 실행 우선순위: 높음, parser에 영향을 미침
- 어디서 쓸까
  - 중요한 컨텐츠를 만드는 스크립트 (FMP)
  - 하지만 뷰포트에 영향을 미치는 스크립트는 안됨
  - 컨텐츠의 동적인 삽입을 위한 동적인 네트워크 요청
  - 가져오는 즉시 실행해야하는 스크립트의 경우에는 `<script type=module async>`를 사용
- 예제
  - `<canvas />` 에 그려야 하는 것

#### `<script async />`

- 로딩 우선순위 (네트워크, 블링크 엔진): Lowest/Low
- 실행 우선순위: 높음, parser에 영향을 미침
- 어디서 쓸까
  - 사용할 때 주의 해야 한다. 요즘 들어 중요하지 않은 스크립트를 로딩 할 때 많이 사용하고 있지만, 로딩 우선순위만 낮을 뿐 실행 우선순위는 높다는 것을 기억해야 한다. ((https://calendar.perfplanet.com/2016/prefer-defer-over-async/) )

#### `<script defer />`

- 로딩 우선순위 (네트워크, 블링크 엔진): Lowest/Low
- 실행 우선순위: 매우 낮음, `<body />` 최하단에 있는 `<script />`가 실행된 이후에 실행
- 어디서 쓸까:
  - 중요하지 않은 컨텐츠를 만드는 스크립트
  - 페이지 방문자의 50% 이상정도가 사용하는 중요한 상호작용 기능
- 예제
  - 광고

#### `<body />` 최하단에 있는 `<script />`

- 로딩 우선순위 (네트워크, 블링크 엔진): Medium/High
- 실행 우선순위: 낮음, 파서가 끝난 뒤에 실행됨
- 어디서 쓸까
  - 이 방법은 생각만큼 낮은 우선순위로 실행되지 않는다는 것을 명심해야 한다.

#### `<body />` 최하단에 있는 `<script defer />`

- 로딩 우선순위 (네트워크, 블링크 엔진): Lowest/Low, 큐 맨 마지막
- 실행 우선순위: 매우 낮음. `<body />` 최하단에 있는 `<script />` 가 끝나면 실행됨
- 어디서 쓸까
  - 사용자들이 가끔 사용하는 상호작용 기능
- 예제
  - '연관된 기사들' 같은 컨텐츠 (중요도가 낮은)
  - '피드백을 주세요' 같은 기능들 (역시 중요도가 낮은)

#### `<link rel=prefetch />` + `<script/>`

- 로딩 우선순위 (네트워크, 블링크 엔진): Idle/Lowest
- 실행 우선순위: 스크립트가 어떻게 작동 하느냐에 따라 다름.
- 어디서 쓸까
  - 다음 라우팅을 위한 자바스크립트 번들
- 예제
  - 다음 라우팅을 위한 자바스크립트 번들

> 브라우저 별로 동작이 통일되어 있지 않으므로 사용할 때 한번 더 확인이 필요하다. (위 자료는 크롬 기준)
> https://addyosmani.com/blog/script-priorities/

## 3. Critical Request란 무엇인가?

Critical Request란 초기 뷰포트에 표시되어야 하는 리소스를 의미한다.

여기에 포함되는 리소스들은 [Core Web Vital](/2021/08/core-web-vital)의 Largest Contentful Paint, First Contentful Paint에 영향을 미친다.

여기에 포함될 수 있는 리소스들은 아래와 같다.

- HTML
- CSS
- webfont
- images
- logo

이러한 애셋들은 (자바스크립트는 없다) 최초 뷰포트를 보여주는데 필수적인 요소들이기 때문에, 가장 먼저 로딩되어야 한다. 이러한 페이지를 위해서는 아래와 같은 항목을 챙기는 것을 추천한다.

- 최초 페이지 로딩시에 보여주어야 하는 요소들에 대한 성능 측정 (above-the-fold)
- 최초 HTTP 요청은 위 요소들이어야 함
- Critical Request는 리다이렉트 되어선 안됨
- Critical Request는 최적화되고, 압축되어야 하며, 캐싱 및 올바른 HTTP 헤더와 함께 제공되어야 함.

## 4. Lighthouse, Critical Request 체이닝을 피해야 한다.

구글의 라이트하우스는 Critical Request가 체이닝 되어 여러 요청이 호출되는 것을 막도록 권장하고 있다. 

![critical-requests-chaining](./images/critical-requests-chaining.png)

가장 흔히 저지르는 critical request chaining은 바로 스타일 시트에서 폰트를 불러올 때 발생한다.

```css
@font-face {
  font-family: 'Calibre';
  font-weight: 400;
  font-display: swap;
  src: url('/Calibre-Regular.woff2') format('woff2'), url('/Calibre-Regular.woff')
      format('woff');
}

.carousel-bg {
  background-image: url('/images/main-masthead-bg.png');
}
```

이러한 요청의 체이닝의 수를 줄이면 LCP의 속도가 빨라지고, 사용자의 웹사이트 경험이 향상된다.이러한 체이닝 수를 줄이기 위해서 아래와 같은 항목을 점검하자.

- 요청의 수를 줄이기
- 리소스를 압축하거나 최소화 하여 크기를 줄이기
- 중요하지 않은 스크립트에 `async`
- HTML에 인라인 `@font-face` 선언을 고려
- css `background-image`나 `@import`를 최소화
- `preload`를 사용하여 중요한 리소스를 빨리 가져오기
- [bundlephobia](https://bundlephobia.com/)를 사용하여 더 작은 대체 라이브러리를 찾아보기

## 5. 요청 우선순위를 제어하기

요청 우선순위는 [preload](https://developer.mozilla.org/en-US/docs/Web/HTML/Link_types/preload)의 영향을 받을 수 있다. `Preloaded` 리소스는 높은 우선순위를 보여 받고, 페이지가 로딩될 때 빠르게 불러와진다.

```html
<link
            rel="preconnect"
            href="https://fonts.gstatic.com"
            crossOrigin="anonymous"
          />
          <link
            rel="stylesheet"
            href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
          />
```

![chrome-font-priority](./images/chrome-font-priority.png)

`Preload`를 사용하면, Critical Request를 최적화 할 수 있지만 너무 많이 사용해서는 안된다. 너무 많은 리소스에 붙이게 되면, 당연하게도 [페이지의 성능이 저하된다.](https://andydavies.me/blog/2019/02/12/preloading-fonts-and-the-puzzle-of-priorities/)

Preloading은 LCP와 CLS(Cumulative Layout Shift)에 영향을 미칠 수 있는데, 일부는 부정적인 영향을 미칠 수 있다. 사용하기 전에 충분히 실험해봐야 한다.

## 6. 이미지 레이지 로딩

기본적으로 브라우저는 사용자가 이미지를 실제로 보는 것과 상관없이 HTML에 있는 모든 이미지를 로드한다. 레이지 로딩을 사용하면 사용자가 이미지에 스크롤을 가까이 할 때만 이미지를 불러오도록 지정할 수 있다. 사용자가 스크롤 하지 않으면 해당 이미지를 브라우저는 불러오지 않는다.

이 접근방식을 사용하면 전반적인 렌더링 속도를 개선하고, 불필요한 데이터 전송을 피할 수 있다. 레이지로딩은 LCP를 개선하는데 매우 효과적이다.

과거 라이브러리나 스크립트를 사용하여 구현했지만, 요즘 브라우저에는 이 기능이 내장되어 있다.

> 물론 지원 가능 여부는 확인해 봐야 한다. https://caniuse.com/loading-lazy-attr

## 7. font-display

[통계에 따르면, 전체 69% 사이트가 웹 폰트를 사용하고 있다.](http://httparchive.org/interesting.php#fonts) 그리고 불행하게도, 이 경우 대부분 수준 이하의 성능을 제공하고 있다. 대부분의 유저들이 폰트가 나타나다가 사라지거나 한다거나, font-weight이 바뀌거나 하는 경험을 해본적이 있다. 이러한 변화는 이제 CLS에 부정적인 영향을 끼치게 된다.

`<link rel= “preload”/>`에서 소개했던 것 처럼, 폰트도 우선순위를 제어하면 렌더링 속도에 큰 영향을 미칠 수 있다. 따라서 대부분의 경우에는 폰트의 요청 우선순위를 결정해야 한다. 

CSS font-display를 사용하면 렌더링 속도 향상에 영향을 미칠 수 있다. 이 속성을 사용하면 폰트를 요청하고 로딩되는 동안, 폰트가 표시되는 방식을 제어할 수 있다.

> https://yceffort.kr/2021/06/ways-to-faster-web-fonts 를 참고!

## Critical Request 체크리스트

위에서 살펴본 것들을 바탕으로, 사이트에 중요한 에셋을 선택하고 그에 따른 우선순위를 지정할 수 있다. 우선 순위와 속도를 높이기 위해서 아래의 체크리스트를 한번더 확인해보자.

- 크롬 개발자 도구에서 우선순위를 확인해보기
- 가능한 경우 필요한 요청 수를 줄이기
- 사용자가 완전히 렌더링된 페이지를 보기 전에 해야하는 것들을 정리하기
- `<link rel="preload" />`를 하여 중요 요청의 우선순위를 수정하기
- 다음 페이지에서 사용될 가능성이 있는 에셋에 대해 [link prefetching](https://developer.mozilla.org/en-US/docs/Web/HTTP/Link_prefetching_FAQ)을 사용하기
- [Link Preload HTTP 헤더](https://www.w3.org/TR/preload/)를 사용하여 HTML이 완전히 전송되기전에 사전에 로드될 리소스를 선언하기
- 이미지의 사이즈가 올바른지 확인하기
- 로고나 아이콘에 인라인 svg를 사용하기
- AVIF, Webp와 같은 좋은 이미지 포맷 사용하기
- `font-display: swap`을 사용하여 초기렌더링에 텍스트를 표시하기
- `WOFF2` 와 같은 압축된 폰트 형식 사용하기
- `chrome://net-internals/#events`를 사용하여 크롬 네트워크 이벤트 살펴보기
- *가장 빠른 요청은 요청하지 않는 것*