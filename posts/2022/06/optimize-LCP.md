---
title: 'Largest Contentful Paint (LCP) 최적화하기'
tags:
  - web
  - browser
published: true
date: 2022-06-09 00:21:56
description: ''
---

## Table of Contents

LCP는 가장 최적화하기 쉬운 Core Web Vital 이다. 3가지 중 유일하게 개발자의 로컬 환경과 실제 환경에서의 수치가 거의 비슷하게 나오는 요소이기도 하다. 그럼에도, 가장 최적화 되지 않는 요소 이기도 하다.

> Once more, we saw an increase in the number of origins having good Core Web Vitals (CWV) driven by improved good CLS.

> - 52.7% of origins had good LCP
> - 94.9% of origins had good FID
> - 70.6% of origins had good CLS
> - 39.0% of origins had good LCP, FID, and CLS

> https://twitter.com/ChromeUXReport/status/1501325517634490376?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1501325517634490376%7Ctwgr%5E%7Ctwcon%5Es1_c10&ref_url=https%3A%2F%2Fcsswizardry.com%2F2022%2F03%2Foptimising-largest-contentful-paint%2F

어떻게 하면 LCP를 쉽게 최적화 할 수 있는 지 살펴보자.

## 정의

LCP를 최적화 하기전에, 먼저 그 정의를 살펴보자.

> 최대 콘텐츠풀 페인트(LCP) 메트릭은 페이지가 처음으로 로드를 시작한 시점을 기준으로 뷰포트 내에 있는 가장 큰 이미지 또는 텍스트 블록의 렌더링 시간을 보고합니다.

> 최대 콘텐츠풀 페인트(LCP)는 페이지의 메인 콘텐츠가 로드되었을 가능성이 있을 때 페이지 로드 타임라인에 해당 시점을 표시하므로 사용자가 감지하는 로드 속도를 측정할 수 있는 중요한 사용자 중심 메트릭입니다. LCP가 빠르면 사용자가 해당 페이지를 사용할 수 있다고 인지하는 데 도움이 됩니다.

> https://web.dev/lcp/

여기에서 알아둬야 할 점 중 하나는, 구글 (라이트 하우스)는 LCP가 빨리 도달하기만 하면, 그 방법이야 어찌되었던 크게 신경쓰지 않는 다는 것이다. 페이지 로딩 라이프사이클과 LCP 사이에서는 다음과 같은 많은 일 들이 일어난다.

- DNS, TCP, TLS
- 리다이렉트
- TTFB
- First Paint
- FCP

만약 이것들 중 어떤 것이라도 느리다면, LCP에는 안좋은 영향을 미친다. 이러한 것들을 가능한한 낮게 얻을 수 있다면 LCP에 도움이 될 것이다.

## LCP 최적화 기법

이미지 기반의 LCP가 있다면, 적절한 파일 포맷, 적절한 크기, 그리고 또 압축이 잘되어 있는지 확인해봐야 한다. 그리고 LCP 요소들은 3MB TIFF 를 넘어서는 안된다.

물론 가장 좋은 방법은 LCP를 텍스트 기반으로 만드는 것이다. 당연하게도 텍스트는 이미지보다 훨씬 크기가 작고, LCP 수치를 높이는데 많은 도움을 준다.

물론, 이미지를 빼는 선택을 하기에는 어려운 상황들이 많을 것이다. 당장에 이미지를 뺴고 글자를 채우자고 한다면 그 누구도 좋아하지 않을 것이다. 그렇다면 이 것을 어떻게 최적화 하면 좋을지 살펴보자.

LCP 내의 이미지를 선언하는 방법들은 다음과 같은 것이 있다.

- `<img />`
- `<svg />` 내부의 `<img />`
- `<video />`의 poster
- HTMLElement에 css로 background image `url()` 를 사용하여 이미지를 깔아 놓은 경우
- 텍스트 노드 또는 인라인 레벨의 텍스트를 포함하는 [블록 레벨](https://developer.mozilla.org/ko/docs/Web/HTML/Block-level_elements) HTMLElement

```html
<img src="lcp.jpg" ... />
```

```html
<svg xmlns="http://www.w3.org/1000/svg">
  <image href="lcp.jpg" ... />
</svg>
```

```html
<video poster="lcp.jpg" ...></video>
```

```html
<div style="background-image: url(lcp.jpg)">...</div>
```
