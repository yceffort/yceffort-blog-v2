---
title: 'Largest Contentful Paint (LCP) 최적화하기'
tags:
  - web
  - browser
published: true
date: 2022-06-09 00:21:56
description: '브라우저 동작 방식을 이해하기'
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

물론, 이미지를 빼는 선택을 하기에는 어려운 상황들이 많을 것이다. 당장에 이미지를 빼고 글자를 채우자고 한다면 그 누구도 좋아하지 않을 것이다. 그렇다면 이 것을 어떻게 최적화 하면 좋을지 살펴보자.

LCP 내의 이미지를 선언하는 방법들은 다음과 같은 것이 있다.

- `<img />`
- `<svg />` 내부의 `<img />`
- `<video />`의 poster
- HTMLElement에 css로 background image `url()` 를 사용하여 이미지를 깔아 놓은 경우
- 텍스트 노드 또는 인라인 레벨의 텍스트를 포함하는 [블록 레벨](https://developer.mozilla.org/ko/docs/Web/HTML/Block-level_elements) HTMLElement

### 테스트

```html
<img src="lcp.jpg" ... />
```

https://yceffort.kr/LCP/img.html

```html
<svg xmlns="http://www.w3.org/1000/svg">
  <image href="lcp.jpg" ... />
</svg>
```

https://yceffort.kr/LCP/svg.html

```html
<video poster="lcp.jpg" ...></video>

https://yceffort.kr/LCP/video.html
```

```html
<div style="background-image: url(lcp.jpg)">...</div>
```

https://yceffort.kr/LCP/background-image.html

각 페이지를 https://www.webpagetest.org/ 에서 테스트 해보자.

### 테스트 결과

![LCP_result](./images/LCP_result.png)

![LCP_progress](./images/LCP_progress.png)

https://www.webpagetest.org/video/compare.php?tests=220610_AiDc96_7AZ%2C220610_AiDc3F_7B4%2C220610_BiDc9B_7JJ%2C220610_AiDc1M_7B6&thumbSize=150&ival=500&end=visual

#### `<img />`

![LCP_img](./images/LCP_img.png)

이미지가 LCP에서 차지하는 비중이 높다면, 가장 먼저 선택해야할 방법이다. `<img />`는 특별하게 개발자가 망치지 않는 한, [프리로드 스캐너](https://andydavies.me/blog/2013/10/22/how-the-browser-pre-loader-makes-pages-load-faster/)에 의해 빠르게 요청되기 때문에, 병렬적으로 요청하여 미리 그려질 수 있다. (블로킹 리소스와 함께도 가능하다)

#### `<picture />`, `<source />`

`<picture />`가 `<img />`와 같은 방식으로 동작한 다는 점을 알고 있어야 한다. 따라서 `srcset` `size` 속성에 상세하게 작성해야 한다. 이를 정확히 제공하면, 앞서 언급한 프리로드 스캐너를 통해 이미지에 대한 충분한 정보를 브라우저에 제공하게 되면, 레이아웃을 기다릴 필요가 없어진다. (물론 기술적으로 연산에 필요한 오버헤드는 존재할 수 있다.)

https://developer.mozilla.org/en-US/docs/Web/HTML/Element/picture

#### `<svg/>`

![LCP_svg](./images/LCP_svg.png)

svg 내부의 이미지는 두가지 흥미로운 동작이 있다. 먼저 한가지는 크롬이 svg 내부의 이미지가 미처 불러와지지 않았음에도 LCP가 완료된 것 처럼 판단한다는 것이다. 상황에 따라 점수만 먹고 튀고 싶다면(?) 이 방법을 응용할 수도 있다.

![LCP_chrome_bug](./images/LCP_chrome_bug.png)

> 물론 이는 어디까지나 현재 블로그 글을 작성 중인 102 버전 이하의 동작으로, 이후에는 수정될 수도 있다.

어쨌든 이는 LCP 상의 버그 일뿐, 실제로 브라우저가 리소스를 처리하는 방법에는 영향을 주지는 않는다. 위 스크린샷에서도 볼 수 있는 것처럼, 워터폴 방식으로 느리게 불러오는 것을 볼 수 있다.

svg 내부에 있는 img는 프리로드 스캐너에서 숨겨진 것 처럼 보인다. 즉, 내부의 `href`는 브라우저의 기본 파서가 이를 발견할 때까지 구분 분석되지 않는다. 이를 미루어 보아 프리로드 스캐너는 SVG가 아닌 HTML 를 스캔하기 위해 만들어졌다는 것을 알 수 있다.

#### `<video />`의 `poster`

![LCP_video](./images/LCP_video.png)

사실 `video` 의 `poster` 가 이렇게 선방할 줄은 몰랐다. 이는 `img`와 동일하게 동작하는 것으로 보이며, 프리로드 스캐너에 의해 조기에 발견된다. 이는 본질적으로 poster가 굉장히 빠르다는 것을 의미한다.

그리고 또다른 소식 중 하나는, `poster`가 없는 `video`는 첫번쨰 프레임을 LCP로 가져가려는 의도가 있다는 것이다. https://bugs.chromium.org/p/chromium/issues/detail?id=1289664 즉 동영상을 실제로 로딩해서 LCP로 가져가려 한다는 뜻인데, 아무래도 동영상은 이미지보다 용량이 크므로, LCP로 동영상을 가져가기 위해서는 `poster`가 필수적으로 보인다.

#### `background-image: url()`

![LCP_css](./images/LCP_css.png)

CSS에 정의된 리소스 (url 을 통해 요청된 모든 리소스) 는 기본적으로 느리다. 여기서 말하는 리소스는 background-image과 web font 등이다.

이러한 리소스가 느린 이유는, 브라우저가 해당 리소스를 필요로 하는 DOM노드를 그릴 준비가 될 때 까지 리소스를 요청하지 않기 때문이다.

이는 background-image 가 LCP의 마지막 순간에 요청된다는 것을 의미인데, 이는 사실 너무 늦은 타이밍이긴하다. 따라서 background-image가 있는 LCP는 적절하지 못하다.

현재 LCP에 background-image가 있는 사이트는 지금 리팩토링할 수 있는 여지가 존재한다. 그리고 이를 가장 빠르게 해결하는 방법이 있다.

```html
<div style="background-image: url(lcp.jpg)">
  <img
    src="lcp.jpg"
    alt=""
    width="0"
    height="0"
    style="display: none !important;"
  />
</div>
```

이러한 방법을 활용하면 프리로드 스캐너로 하여금 브라우저가 `<div />`를 렌더링할 때 까지 않고 이미지를 가져오게 만들 수 있다. 이것은 단순 구현보다 1.058초 더 빠르게 앞당길 수 있다. 이는 `<img/>` 옵션과 거의 똑같다는 것을 알 수 있게 될 것이다.

## 요약

- 텍스트 기반의 LCP는 언제나 옳다. 가장 빠르다.
- `<img />` 와 `poster`는 프리로드 스캐너에 발견될 수 있기 때문에 빠르게 로딩할 수 있다.
- `poster`가 없는 `<video/>`는 동영상의 첫번째 프레임이 LCP에 들어갈 수 있다.
- `<svg />` 에 있는 `<img />` 는 매우 느리고, 이는 href가 프리로드 스캐너에 포함되지 않기 때문이다.
- `background-image` 는 기본적으로 CSS의 동작 방식 때문에 느리다.
  - 이 안에 보이지 않는 `<img/>`를 집어 넣으면 개선할 수 있다.

## 조심해야할 것들

이와 반대로, LCP를 개선하기 위한 작업 외에 하지 말아야 할 것들이 있다. 개발자들이 무심코 LCP 점수를 깎아먹는 일들이 무엇이 있는지 알아보자.

### LCP 를 lazy load 하지 말 것

`loading=lazy`는 프리로드 스캐너에서 이미지를 숨긴다는 문제가 있다. 즉 이미지가 뷰포트에 있더라도 브라우저는 이미지를 늦게 요청한다. 따라서 모든 이미지에 `loading=lazy`를 하는 것은 안전하지 못한 선택이고, 만약 다 추가했다면 브라우저가 잘 로딩해주기를 기도하는 수밖에 없다.

### Fade in 하지 말기

예상했던 것처럼, 이미지를 만약 500ms fade in 한다면 LCP도 그만큼 늦춰진다.

### LCP 에셋은 직접 호스팅할 것

가능하다면, LCP 에셋은 셀프 호스팅하는 것이 좋다. 사이트 관리자들이 Cloudinary와 같은 이미지 최적화 서비스를 사용하여 크기 조정, 포맷 변환, 압축 등 자동으로 이미지를 최적화 하는 솔루션을 사용하곤 한다. 그러 나 이러한 서비스 성능 향상을 고려한다면, 이러한 다른 origin에서 이미지를 가져오는 것은 이렇게 최적화 한 비용을 상쇄해버리고도 남는다. 중요하지 않은 이미지에서는 기존 처럼 자동화된 이미지를 사용하되, 가능하면 최적화된 이미지는 직접 호스팅하는 것이 좋다.

### LCP를 클라이언트에서 빌드 하지 말것

사이트에서 정보를 요청했을 때, 처음으로 오는 응답은 HTML이다. 그리고 이 안에서 `<img />`를 발견하고, 프리로드 스캐너가 이를 읽어드리길 바랄 것이다. 그러나 만약 이 작업 이전에 `framework.js` 와 같은 요청이 온다면 이미지를 발견하기전에 이 자바스크립트를 다운로드 하고 파싱하는 작업이 수행될 것이다. 이렇게 되면 LCP는 js를 해석하여 삽입하는 시간 만큼 뒤로 밀리게 될 것이다.

즉 LCP 영역은 가능하다면 서버사이드에서 미리 빌드된 채로 오는 것이 좋다.

### LCP를 오염하지 말 것

LCP 내부에 있는 컨텐츠를 뒤늦게 로드 하면 안된다. 여기에서 말하는 것은 콘텐츠를 덮고 매우 늦게 로딩되는 쿠키 배너나 각종 광고, 모달 등이 포함된다. 이런 쿠키 배너 등의 요소가 필요하다면

- 가능한 즉시 렌더링한다
- 컨텐츠 크기를 너무 크게 하지 않는다.

## 요약

텍스트 기반의 LCP가 빠르다는 것은 누구나 알 것이다. 그러나 대부분의 경우 이는 불가능할 것이다. LCP에는 `<img/>`나 `<video/>`의 poster를 활용하는 것이 가장 빠르다. 그리고 LCP 내부의 컨텐츠를 lazy 로딩 하지 말고, LCP에 js 빌드 작업을 포함시켜서는 안된다.

## 참고

- [브라우저의 프리로드 스캐너(pre-load scanner)와 파싱 동작의 이해](/2022/06/preload-scanner)
