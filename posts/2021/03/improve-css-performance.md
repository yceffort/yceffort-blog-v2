---
title: 'CSS 성능 향상 시키기'
tags:
  - css
  - browser
published: true
date: 2021-03-26 20:43:10
description: 'CSS의 황제가 출간한 CSS 완벽가이드를 장식용으로 구매...'
---

모던 웹사이트의 복잡성과 더불어 브라우저가 CSS를 처리하는 방식 까지 얹혀진다면, 일부 구식장치, 네트워크 지연, 제한된 데이터를 경험하는 사람들에게는 그리 많지 않은 CSS도 병목현상을 겪을 수 있다. 성능은 사용자 경험에서 필수적인 부분이기 때문에, 다양한 디바이스에서 일관된 고품질의 환경을 제공해야하며 이를 위해선 CSS 최적화가 필수다.

이 포스트에서는 CSS의 성능 이슈와, CSS의 성능을 향상 시키기 위해서는 어떤 작업들이 필요한지 다뤄보려고 한다.

## Table of Contents

## CSS는 어떻게 동작하는가

### CSS는 렌더링을 막는다

CSS의 존재 자체 만으로도, CSS가 파싱되기 전까지 브라우저는 렌더링이 지연된다. 대부분의 모던 웹사이트에서 CSS가 존재하지 않는다면 정상적으로 페이지를 이용할 수 없을 것이다. 만약 브라우저가 CSS가 없는 페이지를 그대로 노출된다면, 잠깐 동안 CSS가 파싱되면서 스타일이 적용되는 페이지가 나타나기 전까지 의 시간이 생기고 말 것이다. 이러한 것을 [FOUC(Flash of Unstyled Content)](https://en.wikipedia.org/wiki/Flash_of_unstyled_content)라고 한다.

### CSS는 HTML 파싱도 막을 수 있다.

브라우저가 CSS가 파싱되기 전까지 콘텐츠를 보여주지 않더라도, HTML의 로딩된 부분만을 일단 보여줄 수도 있다. 그러나 스크립트의 경우 `async` `defer` 이 없다면 파싱을 막게 된다. 스크립트는 잠재적으로 페이지를 조작할 여지가 있으므로, 브라우저는 스크립트 실행에 매우 주의를 기울일 필요가 있다.

![js-blocking-html-parsing](https://web-now-rbviiass9-calibreapp.vercel.app/_next/image?url=%2Fimages%2Fblog%2Fcss-performance%2Fparser-blocking-script.png&w=1920&q=75)

스크릡트가 페이지의 스타일에 영향을 줄 수 있기 때문에, 만약 브라우저가 CSS 관련 작업을 진행중이라면, 이 작업이 완료될 때 까지 기다렸다가 스크립트를 실행할 것이다. 스크립트가 실행되기 전까지 문서 파싱을 할 수 없기 때문에, CSS는 더이상 렌더링을 차단하는 요소로 작용하지 않는다. (하단 그림 참조) 문서의 외부 스타일시트 및 스크립트 순서에 따라서 때로는 HTML 파싱도 중지할 수 있다.

![css-can-block-html-parsing](https://web-now-rbviiass9-calibreapp.vercel.app/_next/image?url=%2Fimages%2Fblog%2Fcss-performance%2Fparser-blocking-css.png&w=1920&q=75)

파싱을 차단하는 상황을 피하기 위해서는, CSS를 최대한 빨리 불러와야 하고, 리소스를 최적의 순서로 불러와야 한다.

## CSS 사이즈 지켜보기

### CSS 압축하고 최소화 하기

외부 스타일 시트를 다운로드 하는 작업은 필연적으로 네트워크 지연이 발생지만, 네트워크에 전송되는 바이트의 양을 줄임으로써 이 과정을 최소화 할 수 있다.

파일을 압축하는 것은 속도 향상에 지대한 영향을 미치며, 많은 호스팅 플랫폼과 CDN에서는 기본적으로 애셋을 압축해준다. 가장 널리알려져 있는 압축 솔루션은 GZip이고, Brotil 또한 존재하지만, [Brotli는 일부 브라우저에서 지원을 하지 않는다.](https://caniuse.com/brotli)

Minification (최소화) 과정은 코드에서 필요없는 공백을 지우는 과정이다. 결과물은 이전 코드에 비해서 작아지지만, 브라우저는 충분히 코드를 파싱할 수 있으며 이를 통해 몇 바이트라도 더 절약할 수 있다. 가장 유명한 자바스크립트 압축 툴로 [terser](https://github.com/terser/terser)가 있고, [웹팩 v4 버전 이상 부터는 빌드 파일을 작게하는 도구가 내장되어 있다.](https://webpack.js.org/plugins/css-minimizer-webpack-plugin/)

### 사용하지 않는 CSS 제거

CSS 프레임워크를 사용하게 될 경우, 필요한 컴포넌트만 번들링 하지 않는 이상 사용되지 않는 CSS가 포함되는 것은 일반적인 문제다. 이와 비슷하게, 오랜시간에 걸쳐서 쌓이는 큰 코드 베이스에도 안쓰는 CSS가 남는 경우가 더러 있다.

사용하지 않는 CSS를 제거하는 것은 수동 작업이다. 따라서 코드가 얼마나 복잡하느냐에 따라서 난이도가 증가하게 된다. 이 작업은 웹사이트 전체에서, 가능한 모든 디바이스에서, 가능한 모든 상황에서, 가능한 모든 자바스크립트 실행 결과에 따라서 결정해야 한다. [UnusedCSS](https://unused-css.com/)나 [PurifyCSS](https://purifycss.online/) 와 같은 유명한 툴이 있지만, 항상 visual regression 테스트도 병행해서 이뤄져야 한다.

**바로 이것이 CSS-in-JS를 쓸 때 얻을 수 있는 가장 큰 이점이다. 각 컴포넌트가 렌더링에 필요한 CSS가 js내에 포함되어 있다. (따라서 컴포넌트 레벨로 관리하기 때문에 편하다는 것)** CSS-in-JS는 페이지 내부에 CSS를 인라인 처리하거나, 외부 CSS파일로 따로 번들링 해버린다. CSS를 자바스크립트 내부에 포함시켜 버리면 CSS 파싱과 평가가 느려진다.

## CSS의 우선순위 정하기

Critical CSS란 화면에 보이는 컨텐츠 (above-the-fold content)의 CSS 에 대해서만 inline 처리하는 것을 의미한다. HTML 문서의 `<head/>`에 있는 스타일을 따로 추출해서 인라이닝 하면 스타일을 가여오는 추가 요청을 할 필요가 없어져 렌더링이 빨라진다.

첫 렌더링 시의 라운드트립을 최소화 하기 위해서는, above-the-fold content의 크기를 14kb내로 유지해야 한다. (압축시)

Critical CSS를 정확히 정의하는 것은 어렵다. 디바이스의 크기에 따라서 사용자가 보이는 영역이 달라지기 때문이다. 이는 특히 매우 유동적인 사이트의 인 경우에는 더욱 어려워 진다. 그러나 이는 여전히 성능 향상에 중요한 부분 이므로, [Critical](https://github.com/addyosmani/critical) [CriticalCSS](https://github.com/filamentgroup/criticalCSS) [Penthouse](https://github.com/pocketjoso/penthouse) 등의 도구를 활용해서 자동화 할 필요가 있다.

### CSS 비동기로 불러오기

위 above-the-fold content를 최대한 빠르게 불러오는데 집중했다면, 나머지 영역은 비동기로 로딩하는 것이 최선이다.

```html
<link
  rel="stylesheet"
  href="non-critical.css"
  media="print"
  onload="this.media='all'"
/>
```

`"Print"` 미디어 타입이란, 사용자가 페이지를 프린트를 하려고 하는 경우에만, 브라우저가 해당 스타일 시트를 불러오는 것으로 렌더링에는 영향을 미치지 않는다. 그리고 여기에 `onload` 이벤트로 `this.media='all'`를 추가한다면, 스타일 시트가 로드가 완료되면 미디어 속성을 다시 `all`로 바꾸면서 스타일 시트가 적용된다.

또 다른 방법은 `<link rel="preload">`를 사용하는 것이다. 그러나 이 방법은 아래와 같은 단점이 있다.

- [브라우저 지원이 여전히 시원치 않으므로](https://caniuse.com/?search=preload), [loadCSS](https://github.com/filamentgroup/loadCSS/)와 같은 폴리필이 필요하다.
- `preload`는 생각보다 매우 이른 타이밍에, 높은 우선순위로 다운로드 되므로 다른 중요한 애셋의 다운로드 우선순위를 밀어 버릴 수 있다.
  - https://developer.mozilla.org/ko/docs/Web/HTML/Preloading_content

```html
<link rel="preload" href="/path/to/my.css" as="style" />
<link
  rel="stylesheet"
  href="/path/to/my.css"
  media="print"
  onload="this.media='all'"
/>
```

> https://www.filamentgroup.com/lab/load-css-simpler/

### @import 사용 자제하기

`@import`는 CSS파일의 렌더링 속도를 느리게 한다. 특히 `@import url(imported.css)`와 같은 코드는 네트워크 흐름을 아래와 같이 바꿔버린다.

![@import](https://web-now-rbviiass9-calibreapp.vercel.app/_next/image?url=%2Fimages%2Fblog%2Fcss-performance%2Fcss-import-blocking.png&w=1920&q=75)

그러나 두 파일을 별개로 분리하면 이런식으로 동시에 다운로드 하게 된다.

![parallel](https://web-now-rbviiass9-calibreapp.vercel.app/_next/image?url=%2Fimages%2Fblog%2Fcss-performance%2Fcss-parallel-download.png&w=1920&q=75)

## 효과적인 CSS 애니메이션 사용하기

페이지에 애니메이션이 있는 요소가 있는 경우, [브라우저는 종종 문서 내 요소의 위치와 크기를 재 계산한다.](https://calibreapp.com/blog/investigate-animation-performance-with-devtools#animation-performance) (레이아웃 발생) 예를 들어, 어떤 요소의 너비를 바꾸게 되면, 그 자식 요소들 까지 영향을 미치면서 페이지 내부에서 큰 레이아웃이 발생할 수 있다. 그리고 이 레이아웃의 크기가 커질 수록, 성능에 안좋은 영향을 미칠 것이다.

요소에 애니베이션을 넣을 때는, 레이아웃과 리페인트가 최소한으로 이뤄지도록 해야 한다. 모든 CSS 애니메이션 기술이 동일하지 않으며, 모던 브라우저에서는 위치, 크기, 회전, 불투명도 등을 가진 고성능 애니메이션을 만들 수 있다.

- `height` `width` 대신 `transform: scale()`을 쓰자
- 요소를 움직이게 하기 위해서는, `top` `right` `bottom` `left` 대신 `transform: translate()`를 쓰자.
- 배경에 blur를 먹이고 싶다면, opacity를 바꾸는 것보다 그냥 blur 된 이미지를 불러오는게 낫다.

### `contain` 속성

[contain](https://developer.mozilla.org/en-US/docs/Web/CSS/contain)은 브라우저에 요소와 하위 요소가 그 문서 트리와 무관한 것으로 간주된다는 것을 알려주는 속성이다. 페이지의 하위 트리와 나머지 페이를 분리한다. 그런 다음, 브라우저는 페이지에서 독립된 부분의 렌더링을 최적화하여 성능을 향상 시킬 수 있다.

`contain`는 페이지 내부에서 독립적으로 작동하는 위젯 등에서 매우 효과적이다. 이 속성을 활용하여 위젯의 내부에서의 변경사항이 바깥으로 전파되는 것을 막을 수 있다.

## CSS로 폰트 로딩 최적화 하기

### 폰트 로딩 중에는 보이지 않는 텍스트를 두지 않기

폰트는 로딩하는데 시간이 걸리는 큰 파일인 경우가 많다. 일부 브라우저는 폰트가 로딩되기 전까지 텍스트를 숨긴다. (FOIT, flash of invisible text) 속도를 최적화 하기 위해서는 보이지 않는 텍스트가 갑자기 나타나는 현상(FOIT)을 피하고, 시스템 폰트를 기본으로 사용하여 즉시 사용자에게 콘텐츠를 보여주는 것이 좋다. 폰트가 로딩 되면, 시스템 기본 글꼴로 로딩된 폰트를 대체 하는 FOUT(flash of unstyled text) 현상이 일어날 것이다.

이를 위한 방법 중 하나가 [font-display api](https://developer.mozilla.org/ko/docs/Web/CSS/@font-face/font-display)를 사용하는 것이다. `swap`을 사용하면, 브라우저가 폰트가 다운로드 되기 전에는 즉시 시스템 글꼴로 보여줘야 한다는 것을 알려줄 수 있다.

### variable font를 사용하여 파일 크기 줄이기

[Variable Font](https://variablefonts.io/)는 모든 너비, weight,style에 대해서 별도의 폰트 파일이 아니라 하나의 파일에 여러가지 다양한 폰트를 통합해줄 수 있도록 한다. CSS와 단일 `@font-face` 참조로 지정된 글꼴의 다양한 변형에 액세스 할 수 있다.

이는 글꼴의 여러 변형이 필요한 파일 크기를 획기적으로 줄일 수 있다. 일반, 볼드, 기울임꼴 폰트 버전을 각각 로딩 하는 대신 모든 정보가 포함된 하나의 단일 파일을 로딩할 수 있다.

[Monotype](https://www.monotype.com/resources/expertise/truetype-gx-variable-fonts) 에서는 12개의 폰트를 결합하여 이탤릭과 로만 모두에 3개의 다른 너비, 8개의 다른 weight를 생성하는 실험을 한 바 있다. 하나의 variable font에 48개의 개별글꼴을 모두 저장함으로써 파일의 크기가 88% 감소했다.

## CSS 선택자의 속도에 대해서 걱정할 필요는 없다.

CSS 셀렉터를 구조화하는 방법에 따라서 브라우저가 CSS를 매칭하는데 필요한 속도가 달라진다. 브라우저는 셀렉터를 오른쪽에서 왼쪽으로 읽기 때문에 자식에서 부모로 거쳐서 올라가게 된다.. 예를 들어, `nav a {}`가 있다면, 모든 `<a/>`를 찾고, 그 다음에 `nav` 하위에 있는 `<a/>`를 찾는다. 따라서 만약 선택자를 `<nav/>` 내부에 있는 `<a/>`에 대해서 `.nav-link`로 지정하 찾는다면, 전체 페이지에서 `<a/>`를 찾는 수고로움을 덜할 것이다. 따라서 `.container ul li a { }`와 같은 선택자는 따라서 비용이 많이 발생하게 된다.

선택자를 매칭 시키는 속도는 굉장히 빠르므로 굳이 걱정할 필요는 없다. CSS 선언은 압축 알고리즘에 매우 유연하게 동작하기 때문에, CSS 선택자를 최적화하는데 필요한 노력은 투자대비 더 큰 수익으로 다가올 것이다.

## 결론

CSS는 페이지 로딩과 유익한 사용자 경험을 주기 위한 필수적인 요소다. js나 image와 같은 요소에 최적화 하다보면, css에도 관심이 필요하다는 것을 잊을 수 있다. 위의 전략들을 활용하다보면, 사용자에게 더 빠르고 최적화된 웹사이트를 제공할 수 있을 것이다.
