---
title: '웹 폰트 로딩을 더 빠르게 하는 방법'
tags:
  - web
  - browser
  - css
published: true
date: 2021-06-27 17:34:52
description: '개발할 때 간지나는 이쁜 폰트 추천받습니다'
---

## Table of Contents

## 들어가기에 앞서

web font에 대해서 이야기 할 때, 자주나오는 두 용어의 정의와 차이점에 대해서 먼저 알아본다.

- typeface: 한글로는 `서체`라고 하며 공통 디자인을 공유하는 글꼴 전체를 의미한다. 이 서체에는 굵기나 스타일 등이 포함될 수 있다. 예를 들어 Helvetica는 서체의 한 종류다. 서체는 일종의 폰트 패밀리로 보면 된다.
- font: 한글로는 `글꼴` 이라고 한다. 서체의 단일 굵기와 스타일이다. 글꼴은 특정 크기, 굵기 및 스타일을 포함하여 제공된다. (예 10 포인트 Helvetica 볼드 이태릭체) 벡터 기반의 최신 디지털 글꼴 디자인은 단일 글꼴을 무한히 확장하거나 축소할 수 있지만, 각 굵기와 스타일에 대해 별도의 파일이 필요하다.

## 모던 파일 포맷을 사용하자

[Web Open Font Format 2.0](https://www.w3.org/TR/WOFF2/)은, 현재 기준 가장 작고 효율적인 웹 폰트 파일 형태다. CSS에서 `@font-face`룰을 사용할때, woff2 글꼴이 ttf와 같은 오래된 구식의 덜 효율적인 폰트보다 더 앞서서 선언되있게 끔 해야 한다. 브라우저는 더 큰 파일이라 할지라도, 먼저 선언되어 있는 글꼴을 인식하여 사용하게 된다.

```css
@font-face {
  font-family: 'Typefesse';
  src:
    url('typefesse.woff2') format('woff2'),
    url('typefesse.woff') format('woff');
}
```

IE8 지원을 할 것이 아니라면, WOFF2나 WOF 보다 더 오래된 폰트 형식을 사용할 필요가 없다. IE 11 지원을 배제한다면, WOFF2만 사용 하면 된다.

- https://caniuse.com/woff
- https://caniuse.com/woff2

만약 TTF 파일 만 가지고 있다면, https://onlinefontconverter.com/ 와 같은 사이트를 방문해서 변환하는 것이 좋다. 물론 그전에 폰트에 대한 라이센스를 확인해봐야 한다.

## `font-display` 지시자를 사용하자

1. Flash of Invisible Text (FOIT): 브라우저가 폰트를 다운로드 하기전에 폰트가 보이지 않는 현상이다.
2. Flash of Unstyled Text (FOUT): 브라우저가 폰트를 다운로드하기전에 폰트가 적용되지 않은 글자가 보이는 현상이다.

![FOIT vs FOUT](https://d2.naver.com/content/images/2018/12/helloworld-201812-webfont_14.gif)

물론 두 상황 모두 이상적이지는 않지만, 만약 웹 폰트를 사용하게 되면 사용자가 처음 웹사이트를 방문할때 둘 중 하나의 현상이 발생하게 될 것이다. (두 번째 방문 시 부터는 브라우저가 캐시에서 폰트를 제공하겠지.....?) 만약 [font-display](https://developer.mozilla.org/en-US/docs/Web/CSS/@font-face/font-display)지시자 이전에 `font-face`와 같은 규칙을 추가한다면, 브라우저에 위에서 언급한 두개중 어떤 것을 선택할지 알려줄 수 있다.

```css
@font-face {
  font-family: 'Typefesse';
  src:
    url('typefesse.woff2') format('woff2'),
    url('typefesse.woff') format('woff');
  font-display: swap;
}
```

여기에서 `font-display`에 적용할 수 있는 다섯가지 값이 있다. 일단 첫번째 값은 `auto`로, 브라우저의 기본값에 의존하는 것이다. 일단 브라우저의 기본값은 대부분 FOIT 이다.

https://developer.mozilla.org/ko/docs/Web/CSS/@font-face/font-display

### swap

![swap](/images/font-swap.svg)

swap이란 이름에서 느껴지는 것처럼, 웹폰트가 로딩되기 전까지 fallback 폰트로 글자를 보여주는 것이다. (FOUT) 폰트 다운로드가 끝나자마자 폰트 스왑이 일어나게 된다. 폰트가 로딩되지 않더라도 사용자들이 글자를 읽을 수 있기 때문에 좋다고 볼 수 있다. 그러나 fallback font를 웹폰트와 비슷한 것으로 설정 하지 않는다면, 화면전환이 크게 일어날 수 있으므로 조심해야 한다.

### block

![block](/images/font-block.svg)

웹 폰트가 로딩되기전가지 브라우저에 텍스트를 숨기기 위해서 (FOIT) 사용되는 방식이다. 그러나 웹 폰트가 다운로드될 때까지 하염없이 기다리는 것은 아니다. 글꼴이 특정 시간 (보통 3초)내에 로딩되지 않으면 브라우저는 fallback font를 사용하여 로딩 한 후에 웹 폰트로 교체한다. FOUT가 별로라고 생각한다면 아마도 이게 최선의 방법일 수도 있다. 그러나 다시한번 말하지만, 폰트가 다운로드 되기 전까지 글자가 보이지 안않는다.

### fallback

![fallback](/images/font-fallback.svg)

swap이랑 비슷하긴 한데, 두가지 차이점이 있다.

1. 0.1초 정도 텍스트가 보이지 않는 블록이 발생하며, 이후에는 fallback font가 보여진다.
2. 3초 이내로 다운로드 되지 않는다면, 웹 폰트 다운로드와 상관없이 앞으로 계속 fallback font가 보여진다.

사용자가 처음 사이트르 방문했을때, 웹 폰트로 제공되지 않더라도 별로 상관이 없다면 이 옵션도 괜찮을 수 있다.

### optional

![optional](/images/font-optional.svg)

`fallback`과 비슷한데, `fallback`의 2번 기능이 제거된 버전이라 볼 수 있다. 여기에 추가로 폰트가 다운로드되는데 너무 오래걸린다면, 브라우저가 연결을 취소 시켜버릴 수 있는 기능까지 추가되어 있다.

> 페이지에 있는 각 폰트는 고유한 FOIT, FOUT 기간을 가진다. 따라서 폰트가 개별적으로 따로따로 swap 되어 버린다. 이와 관련된 [웃긴 사건](https://www.zachleat.com/web/mitt-romney-webfont-problem/이 있다. 따라서 이를 완벽하게 제어하기 위해서는 , 자바스크립트를 써야 한다.

## 폰트 파일을 미리 로딩하기

FOIT/FOUT 기간을 최소화 하기 위해, 웹 폰트를 가능한 빠르게 로딩을 시작할 필요가 있다. `<head/>`에 `<link rel="preload">`를 사용해서, 브라우저에 가능한 빠르게 폰트를 다운로드 하게 할 수 있다. 가능한 `<head>`의 가장 앞자리에 설정하는 것이 좋다.

```html
<link
  rel="preload"
  href="/typefesse.woff2"
  as="font"
  type="font/woff2"
  crossorigin
/>
```

이 태그를 추가하면, 브라우저에 즉시 폰트 파일을 로드하도록 지시하게 된다. 일반적으로 css에서 특정 글꼴에 대한 참조가 발견하고 이를 참조하는 DOM요소를 찾을 때까지는 시작되지 않는다.

브라우저는 현재 페이지에 필요한 폰트만 다운로드 할 수 있을 정도로 충분히 똑똑하다. `preload`를 사용하면, 이 동작이 리셋되어 브라우저가 사용하지 않더라도 폰트를 다운로드 해야 한다. 따라서 각 폰트의 단일 포맷에 대해서만 이 옵션을 사용해야 한다.

그러므로, 더 많은 폰트를 사전 로드할 수록 이 기법의 이점이 줄어든다. 따라서, 화면에 최초에 표시되는 (above the fold, 100vh)로 표시되는 폰트에 이 속성을 지정하는 것이 좋다.

> https://www.smashingmagazine.com/2016/02/preload-what-is-it-good-for/

## 폰트 파일의 부분 집합 만들기

폰트의 하위 집합을 설정하면, 딱 필요한 glyph(개별 문자 또는 기호)만 포함하는 더 작은 폰트 파일을 생성할 수 있다. https://everythingfonts.com/subsetter 와 같은 도구를 사용한다면, 폰트에서 필요한 글꼴만 지정하여 포함시킬 수 있다.

이 도구는 강력한 도구지만, 몇가지 약점이 있다. 예를 들어 사용자가 생성한 컨텐츠, 이름, 장소를 표시하는 웹 사이트를 구축하려면 일반적인 26개의 알파벳과, 10개의 숫자 및 기호 이외에 사용자가 추가할 수 있는 다양한 문자에 대한 고려를 해야 한다. 프랑스어, 베트남어, 스페인어, 그리스어, 히브리어와 같은 언어에서 나오는 분음 부호들도 고민해봐야 한다.

물론 이를 다 수동으로 해야 하는 것은 아니다. [Glyphhanger](https://www.zachleat.com/web/glyphhanger/)를 사용하면, 두가지 도움을 얻을 수 있다. 먼저 웹페이지를 보고 사용되는 유니코드 문자의 범위를 결정한다. 그리고, 지정된 범위의 문자만 포함하는 새로운 버전의 폰트 파일을 출력한다.

> https://www.sarasoueidan.com/blog/glyphhanger/

## 폰트를 셀프 호스팅하기

앞선 네가지와는 다르게 이는 보편적으로 적용해도 좋은 규칙은 아니다.

- https://fonts.google.com/
- https://fonts.adobe.com/

와 같은 폰트 호스팅 서비스를 사용하는게 좋은 이유도 있다.

1. 웹에서 특정 폰트를 가장 저렴하게 사용할 수 있으면서, 법적으로도 유효한 방법이다. 이러한 서비스 중 하나를 사용하는 경우, 앞서 언급한 하위집합 또는 font-display 지시자를 지원하는지 확인해야 한다.
2. 편리하다. html 라인을 그냥 무지성으로 복사해서 head에 붙여넣는 것이 모든 방법보다 빠르다.

만약 순전히 편리함 때문에 구글 폰트를 사용하고 있다면, https://google-webfonts-helper.herokuapp.com/ 를 한번 방문해보는 것도 좋다. 이 도구를 사용하면 전체 구글 폰트 집합에서 커스텀 웹 폰트 번들을 만들고, 필요한 두께나 문자 집합을 정의한다음, 모든 css 및 폰트 파일을 포함하는 다운로드를 만들 수 있다.

> 동일한 글꼴을 동일한 소스에서 로드하는 사이트를 사용자가 이전에 방문한적이 있다면, 브라우저 캐시로 인해 다시 다운로드할 필요가 없다는 이야기도 있다. https://developers.google.com/fonts/faq#what_does_using_the_google_fonts_api_mean_for_the_privacy_of_my_users 이는 한 때 사실일 수도 있지만, 아닐 수도 있다. 구글 크롬이나 사파리 모두 추적에 대한 염려 때문에 서로 다른 도메인간에 캐시된 타사 리소스를 공유하는 것을 명시적으로 금지하고 있다. https://www.stefanjudis.com/notes/say-goodbye-to-resource-caching-across-sites-and-domains/

반면 폰트를 셀프 호스팅해서 사용하면 아래와 같은 장점이 있다.

### 성능

도메인 룩업을 하는데에는 시간이 소요된다. 물론 [preconnect 리소스 힌트](https://web.dev/uses-rel-preconnect/)를 사용하면 문제를 해결할 수 있지만, 어쨌건간에 새로운 도메인 연결에 TCP를 열면 항상 성능저하가 발생한다.

![font-of-web-dev](/images/font-of-web.dev.png)

> web.dev 도 보면 폰트를 셀프 호스팅 하고 있다.

### 프라이버시

Adobe fonts와 같은 유료 웹 폰트 서비스는 빌링으로 인한 이유 때문에 페이지 뷰를 감지하지만, 딱 필요한 것보다 더 많은 데이터를 수집하고 있을 수도 있다. 만약 가능하다면, 자바스크립트 `<script/>` 대신, `<link rel="stylesheet>` 를 사용하여 데이터 수집을 막아야 한다.

구글 폰트의 경우 ip 주소, user-agent외에 웹 사이트 방문자들에 대한 정보를 수집하지 않는 것처럼 보이지만, 구글이 무료로 서비스를 제공함으로써 [많은 양의 데이터를 수집한 것으로 보인다](https://fonts.google.com/analytics).

### 제어권

셀프 호스팅 폰트를 사용하면, 폰트 로딩 방법을 완벽하게 제어할 수 있으므로, 커스텀 폰트 하위 세트를 제공하거나 `font-display`를 정의하거나, 브라우저가 폰트를 캐시할 기간을 지정할 수 있다.

### 신뢰도

써드파티 서비스는 느려지거나, 중단되거나 혹은 [서비스가 종료될 수 있는](https://web.archive.org/web/20180617081657/http://blog.fontdeck.com/post/133794978966/why-fontdeck-is-retiring) 위험이 있다. 셀프 호스팅 폰트를 사용하면 웹사이트가 살아있는 한 폰트는 사용가능할 것이다.

## 결론

각 단계는 자체적으로도 장점이 있지만, 함께 다같이 적용하면 큰 개선으로 이어질 수도 있다. 여기에서 언급한 몇가지를 구현하기로 결정했다면, 적용 전후로 [Light House](https://developers.google.com/web/tools/lighthouse)나 [Web Page Test](https://www.webpagetest.org/)로 각 개별 변경에 따른 성능 개선을 확인해보자.

## 참고

- https://d2.naver.com/helloworld/4969726
- https://iainbean.com/posts/2021/5-steps-to-faster-web-fonts/
