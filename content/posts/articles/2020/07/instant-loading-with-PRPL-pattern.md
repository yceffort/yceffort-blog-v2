---
title: 빠른 로딩을 위한 PRPL 패턴
tags:
  - javascript
  - web
  - browser
published: true
date: 2020-07-06 09:38:02
description: "[Apply instant loading with the PRPL
  pattern](https://web.dev/apply-instant-loading-with-prpl/)을 번역한 글입니다. PRPL은 웹
  페이지를 로드하고 인터랙티브 할 수 있게 금 더욱 빠르게 만드는 패턴을 설명하는 약어다.  ## 요약  - 중요한 리소스를 미리 로드해라
  (Push (..."
category: javascript
slug: /2020/07/instant-loading-with-PRPL-pattern/
template: post
---
[Apply instant loading with the PRPL pattern](https://web.dev/apply-instant-loading-with-prpl/)을 번역한 글입니다.

PRPL은 웹 페이지를 로드하고 인터랙티브 할 수 있게 금 더욱 빠르게 만드는 패턴을 설명하는 약어다.

## 요약

- 중요한 리소스를 미리 로드해라 (Push (or preload) the most important resources.)
- 최초 라우팅을 가능한 빠르게 렌더링해라 (Render the initial route as soon as possible.)
- 나머지 assets을 미리 캐싱해두어라 (Pre-cache remaining assets.)
- 기타 다른 라우팅과 덜 중요한 assets을 레이지 로딩 해라. (Lazy load other routes and non-critical assets.)

## Preload critical resources

[Preload](https://developer.mozilla.org/en-US/docs/Web/HTML/Preloading_content) 속성은 브라우저에 가능한 빨리 리소스를 요청하도록 지시하는 선언적 fetch 요청이다. HTML 헤드에 있는 `<link/>` 태그에 `rel="preload"`를 붙이면 중요한 리소스를 미리 요청할 수 있다.

```html
<link rel="preload" as="style" href="css/style.css">
```

브라우저는 이 선언을 보게 되면, `window.onload` 이벤트를 지연시키지 않는 선에서 해당 리소스에 우선순위를 두고 다운로드를 시도한다.

더욱 자세한 가이드는 [여기](https://web.dev/preload-critical-assets/)를 참고 하라.

## Render the initial route as soon as possible.

First paint를 향상 시키기 위해서는, 가장 중요한 자바스크립트 코드를 인라인으로 작성하고, 나머지를 자바스크립트와 css를 [async](https://developers.google.com/web/fundamentals/performance/critical-rendering-path/adding-interactivity-with-javascript)로 작성해야 한다. 이는 렌더링을 막는 asset을 가져오기위한 서버 라운드 트립을 막음으로서 성능을 향상시킨다. 그러나 인라인 코드는 개발관점에서 유지하기가 어렵고 브라우저에 의해 별도로 캐시될 수 없으니 주의 해야 한다.

가능한 다른 방법으로는 최초 HTML 페이지를 서버사이드 렌더링에 맡기는 것이다. 이는 사용자가 스크립트를 가져오고, 파싱하고, 실행하는 시간동안에 의미있는 정보를 보여줄 수 있다. 그러나 이는 HTML 파일을 가져오는 페이로드를 증가시킬 수 있으며, 이는 [TTI](https://web.dev/interactive/) (사용자가 페이지와 상호작용하는데 걸리는 시간)에 악영향을 미칠 수도 있다. 

First Paint를 향상시킬 수 있는 절대적인 방법은 없다. 따라서 인라인 스타일이나 서버사이드 렌더링을 하는데 있어서 장단점이 무엇인지를 알아야 한다. 

- [CSS 가져오기를 최적화 하기](https://developers.google.com/speed/docs/insights/OptimizeCSSDelivery)
- [서버사이드 렌더링이란 무엇인가](https://www.youtube.com/watch?v=GQzn7XRdzxY)

## Pre-cache assets

![서비스 워커에서의 요청과 응답](https://webdev.imgix.net/apply-instant-loading-with-prpl/service-workers.png)

서비스 워커는 매 방문시 서버에서 필요한 assets을 가져오는 것이 아니라 proxy 처럼 역할하여 assets을 제공한다. 이는 사용자가 오프라인 중에도 어플리케이션을 쓸 수 있게 끔 할 뿐만 아니라, 반복적으로 페이지에 접근했을 때 페이지 로딩을 빠르게 해준다.

일반적인 라이브러리가 제공하는 것 이상으로 복잡한 요구사항이 없다면, 써드 파티 라이브러리로 서비스 워커를 만들어 이 과정을 단순화 하는 것을 고려해봄직하다. 예를 들어 [WorkBox](https://web.dev/workbox)는 asset을 캐싱할 수 있는 서비스 워커를 만들고 유지할 수 있는 다양한 툴을 제공한다. 더 많은 서비스 워커에 대한 정보와 오프라인 상태의 유지에 대해 알고 싶다면, [Service worker guide](https://web.dev/service-workers-cache-storage)를 참고하라.

## Lazy load

웹페이지에는 다양한 asset이 있지만, 특히 구문 분석및 컴파일에 걸리는 시간으로 인해 큰 자바스크립트는 많은 비용을 소모한다. 가능한 자바스크립트를 작게 만들어 최초 페이지 로딩에 필요한 chunk 들만 내보내고, 나머지는 [lazy load](https://web.dev/reduce-javascript-payloads-with-code-splitting/) chunk로 분리하는 것이 좋다.

번들을 나눈 이후에는, 이 chunk들을 `preload` 하는 것이 더욱 중요하다. `Preloading`은 중요한 리소스에 대해서 더 빠르게 다운로드 할 수 있도록 브라우저에 명시한다.

웹 페이지에서 많은 양의 이미지를 로딩한다면, 화면 밖에서 표시되는 이미지에 대해서는 페이지가 로딩 될 때 나중에 로딩되도록 하는 것이 좋다. [lazysizes](https://github.com/aFarkas/lazysizes)라이브러리의 사용을 고려해보는 것도 좋다.
