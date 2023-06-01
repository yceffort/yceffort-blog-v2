---
title: '새로 바뀐 네이버 메인 훔쳐보기'
tags:
  - javascript
  - web
  - html
published: true
date: 2023-05-30 14:51:26
description: '사실 나도 잘 몰라요'
---

## Table Of Contents

## 들어가며

[이번에 새롭게 네이버 PC 메인이 바뀌면서](https://campaign.naver.com/naverdetails/newpc/?pcode=naver_pcanniversary) 많은 것이 바뀌었다고 한다. 내부에서 무슨 일이 일어나는지는 잘 모르지만🤔이번에 리액트 17로 새롭게 개편하면서 내부적으로도 변경된 것이 많을 것 같은데, 새롭게 개편된 네이버는 어떤 스펙으로 어떻게 만들어졌는지 직접 크롬 소스보기로 까보면서 알아보고자 한다.

> 네이버 메인 개발자가 아니기 때문에 진짜 그냥 웹사이트만 보고 배운 것들만 적어둔다. 실제 개발 내용과는 다를 수 있다는 점을 미리 경고한다.

## 기술 스택

### react@17.0.2

![naver-01](./images/naver-01.png)

가장 먼저 눈에 띄는 점은 react@17.0.2 를 사용하고 있다는 것이다. 최신 버전인 react@18을 사용하지 못하는 이유는 아마도 추측 컨데 [18버전에서 internet explorer 11 지원을 중단](https://github.com/facebook/react/blob/main/CHANGELOG.md#react-1) 했기 때문으로 보인다. `Promise` `Symbol` `Object.assgin` 폴리필을 넣어주면 될 것 같은 뉘앙스로 쓰여있긴 하지만, 아무래도 리스크가 있을 것으로 판단하여 17 만 사용하는 것으로 보인다. 대한민국에서 internet explorer 11 이 관짝으로 가지 않는 이상, 이 버전이 올라갈일은 없어보인다.

### nextjs?

한 가지 흥미로운 사실은 크롬 성능 도구로 분석한 결과, `nextjs`의 hydration 과정을 확인할 수 있다는 것이다.

![naver-02](./images/naver-02.png)

오우 매우 신기하다라고 생각하던 차에, 조금 더 알아보니, naver.com이 nextjs로 만들어진 건 아니었다. (그럼 그렇지)

![naver-03](./images/naver-03.png)

nextjs로 만들어진 것으로 추정되는 영역은 쇼핑블록 https://shopsquare.naver.com/ 으로, `iframe` 형태로 이 쇼핑 블록을 naver.com 페이지에 삽입하는 것으로 확인되었다.

이 페이지를 몇번 사용해보니, 내가 어떤 페이지에 방문했건간에 새로고침하면 무조건 1페이지로 돌아가는 것을 확인할 수 있었는데, 그에 반해 nextjs의 `pageProps`는 18페이지의 모든 정보를 다 들고 있는 것으로 확인되었다. 사용자의 방문이 예상되는 1, 2 그리고 마지막 페이지 (18)의 정보만 `getServerSideProps`로 들고 왔으면 조금 더 다운로드 해야 하는 페이지의 크기를 줄일 수 있을 것으로 보인다.

### jquery

네이버 어딘가에는 아직 jquery를 사용하는 코드가 존재하는 것으로 보인다.

![naver-04](./images/naver-04.png)

### corejs

corejs 사용도 확인할 수 있었다.

![naver-05](./images/naver-05.png)

### 스타일

소스보기로 naver.com 을 살펴보면, 단 두개의 css 파일만 존재하는 것을 볼 수 있다.

- https://ssl.pstatic.net/sstatic/search/pc/css/sp_autocomplete_220526.css
- https://pm.pstatic.net/resources/css/main.f6c8441d.css

이 두 파일의 존재로 추정해 본 게 하나 있는데, 홈페이지가 최 상단의 네이버 검색바 부터 렌더링 되지 않을 까 하는 것이다. 이 를 크롬 개발자 도구로 확인해보자.

![naver-06](./images/naver-06.png)

First Contentful Painting 시점을 확인해 본 결과, 네이버 검색바 부터 스타일이 입혀진 채로 렌더링 된 이후에 이후에 모든 내용이 그려지는 것을 볼 수 있었다.

`sp_autocomplete`는 내용이나 파일명 등으로 미루어 보아 소스 코드 내부가 아닌, 외부에서 정적으로 관리되는 파일로 보인다. 그렇게 추정하는 이유는

- 파일명에 날짜로 추정되는 버전이 존재하고
- 내부 클래스명이 module.scss 등을 사용했을 때와 다르게 난독화되어 있지 않고 정제되어 있기

때문이다.

그리고 이 두 리소스는 네트워크 차단 요청으로 선언되어 있기 때문에 최대한 다이어트를 하는 것이 좋아 보인다. `main.css`의 경우 특히 그 크기가 큰데 비해 above the fold 영역에서 사용하지 않는 스타일도 함께 존재하는 것으로 보인다.

![naver-07-1](./images/naver-07-1.png)

![naver-07-2](./images/naver-07-2.png)

`main.css`에 존재하는 스타일 중 하나는, 최근 검색어가 존재하며 사용자가 클릭을 했을 때만 필요한 스타일인 것으로 확인된다. 이런 스타일은 별도 스타일 파일로 분리한다음 `aysnc`나 `defer`를 사용하여 lazy 하게 불러오는 것이 좋다.

이러한 미사용 자바스크립트나 css 를 확인할 수 있는 좋은 방법 중 하나는 크롬 개발자 도구의 범위 (coverage)를 사용하는 것이다.

![naver-08](./images/naver-08.png)

이렇게 빨간색으로 표신된 부분이 로딩은 되었으나 실제로 사용되지 않은 영역이다. 물론 크롬 시크릿 탭으로 들어간 경우에만 해당되므로, 로그인된 사용자 등 다양한 시나리오에 따라서 미사용 스타일과 자바스크립트의 크기는 달라질 수 있으므로 이는 어디까지나 참고용으로만 사용하는 것이 좋다.

## 줄 단위로 살펴보기

트위터인가 어디선가 면접 문제로 트위터의 html을 보고 한 줄 씩 해석하라고 하는 낸다고 들었다.🤔 ~~네이버에 합격하기 위해 그런 것은 아니다.~~

> 정말로 그냥 소스보기에서 라인바인 라인으로 복사해왔기 때문에, close tag 등이 안맞을 수 있다.

### 1. `<html/>`

```html
<!doctype html>
<html lang="ko" data-dark="false" class="fzoom">
<head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=1190">
  <title>NAVER</title>
  <meta name="apple-mobile-web-app-title" content="NAVER" />
  <meta name="robots" content="index,nofollow" />
  <meta name="description" content="네이버 메인에서 다양한 정보와 유용한 컨텐츠를 만나 보세요" />
  <meta property="og:title" content="네이버">
  <meta property="og:url" content="https://www.naver.com/">
  <meta property="og:image" content="https://s.pstatic.net/static/www/mobile/edit/2016/0705/mobile_212852414260.png">
  <meta property="og:description" content="네이버 메인에서 다양한 정보와 유용한 컨텐츠를 만나 보세요" />
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="">
  <meta name="twitter:url" content="https://www.naver.com/">
  <meta name="twitter:image" content="https://s.pstatic.net/static/www/mobile/edit/2016/0705/mobile_212852414260.png">
  <meta name="twitter:description" content="네이버 메인에서 다양한 정보와 유용한 컨텐츠를 만나 보세요" />
  <link rel="shortcut icon" type="image/x-icon" href="/favicon.ico?1">
  <link rel="apple-touch-icon-precomposed" href="https://s.pstatic.net/static/www/nFavicon96.png" />
  <link rel="apple-touch-icon" sizes="114x114" href="https://s.pstatic.net/static/www/u/2014/0328/mma_204243574.png" />
  <link rel="apple-touch-icon" href="https://s.pstatic.net/static/www/u/2014/0328/mma_20432863.png" />
  <link rel="stylesheet" href="https://ssl.pstatic.net/sstatic/search/pc/css/sp_autocomplete_220526.css">
  <script>
    window.gladsdk = window.gladsdk || {}, window.gladsdk.cmd = window.gladsdk.cmd || [], window.ndpsdk = window.ndpsdk || {}, window.ndpsdk.cmd = window.ndpsdk.cmd || [], window.ndpsdk.polyfill = window.ndpsdk.polyfill || {
      cmd: []
    };
    var g_ssc = "navertop.v5";
    window.nsc = g_ssc
  </script>
  <script async src="https://ssl.pstatic.net/tveta/libs/ndpsdk/prod/ndp-loader.js"></script>
  <script async src="https://ssl.pstatic.net/tveta/libs/glad/prod/gfp-core.js"></script>
  <script>
```

- `!doctype html`: 현재 선언된 웹페이지의 html 버전을 선언하는 일종의 지시자 역할을 한다. 이 내용은 대소문자를 구분하지 않으며, naver는 html5로 작성되어 있음을 알 수 있다. 아주 구닥다리 사이트의 경우 html 4.01의 표준인 `<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">`와 같은 방식으로 작성되어 있는데, MZ한 개발자 이므로 이런 건 딱히 다루지 않는다. 한가지 중요한 것은, 페이지 최상단에 있어야 한다는 것이다.
- `<html/>`: html 태그의 시작이다. [lang](https://html.spec.whatwg.org/multipage/dom.html#attr-lang)은 내부 요소의 콘텐츠 및 텍스트의 기본 언어를 지정하는 속성이다. `data-dark`나 `class`는 내부적으로 사용하는 용도로 보인다. 참고로 다크 모드는 쿠키의 `NSCS` 값을 제어하여 컨트롤하는 것으로 보인다.
- `<meta charset="utf-8">` HTML의 문서의 문자 인코딩 방식이 utf-8 이라는 뜻이다.
- `<meta http-equiv="X-UA-Compatible" content="IE=edge">`: 오직 인터넷 익스플로러에서만 유요한 태그로, 인터넷 익스플로러에 IE Edge 모드를 사용하여 렌더링하라고 지시하는 것이다. MZ하지 않은 개발자들은 IE 브라우저의 호환성보기 모드를 알 텐데, 이 호환성 보기 모드에 가장 최신의 브라우저 렌더링 방식인 Edge를 사용하도록 지시하는 것이다.
- `<meta name="viewport" content="width=1190">`: 뷰포트에 최소너비를 지정해주는 내용의 태그다. 이로 미루어보아 네이버 메인 페이지는 최소 1190px에서 잘 렌더링 되도록 설계한 것을 알 수 있다.
- `<meta name="apple-mobile-web-app-title" content="NAVER" />`: 이 내용은 [애플 개발자 문서](https://developer.apple.com/library/archive/documentation/AppleApplications/Reference/SafariWebContent/ConfiguringWebApplications/ConfiguringWebApplications.html)에서 확인할 수 있는데 iOS에서 런치아이콘을 생성했을 때 해당 앱의 타이틀이 된다.
- `<meta name="robots" content="index,nofollow" />`: 검색 수집 로봇의 동작을 정의할 수 있는 메타 태그다. 이 경우, `index`페이지는 크롤링하나, `notfollow`옵션으로 인해 내부에 있는 링크는 따라가지 않는다.
- `<link rel="apple-touch-icon-precomposed" href="https://s.pstatic.net/static/www/nFavicon96.png" />`: iOS 7에서 동작하는 터치 아이콘 이미지다. iOS 7 이하 버전 지원을 위해 남겨 둔 것으로 보인다.
- 이하 스크립트: 무언가 제3자 스크립트의 초기화를 위한 작업을 추가해준 것으로 보인다.

### 2. `window["EAGER-DATA"]`

네이버의 동작을 깊게 살펴보면서 처음 알게된 동작이다. `EAGER-DATA`가 무엇을 하는지는 개발자가 아닌 이상 정확히 알수 없지만, `EAGER-DATA`라는 이름으로 보아 웹사이트 렌더링에 즉시 필요한 데이터로 보인다.

```html
<script>
  window['EAGER-DATA'] = window['EAGER-DATA'] || {}
</script>
```

이는 아마도 nextjs의 pageProps 처럼 서버사이드에서 가져온 데이터를 html에 바로 삽입하여 즉시 자바스크립트 단에서 사용하기 위한 목적으로 추정된다. 이렇게 페이지 내부에 `window['EAGER-DATA']`로 삽입 해두면, 자바스크립트에서는 `window['EAGER-DATA']`를 사용해 손쉽게 데이터를 가져올 수 있다. 실제로 소스코드에서도 이렇게 접근해서 가져오는 것을 확인할 수 있었다.

![naver-09](./images/naver-09.png)

정확한 분석은 아니지만, 아마도 사용자에게 정적 페이지를 제공하기 위해 다음과 같은 과정을 거칠 것으로 예상해볼 수 있다.

1. 미리 빌드해둔 head 영역 html을 가져온다.
2. 1번의 파일엔 `<html> ... <script>` 까지 코드가 존재한다.
3. 서버에서 렌더링에 필요한 동적인 데이터를 불러온다.
4. 2번의 파일에 3번의 서버에서 불러온 데이터를 `window["EAGER-DATA"] = window["EAGER-DATA"] || {};`을 시작으로 내용을 붙여 넣는다.
5. 4번 내용이 끝나면 `</script>`로 닫는다.
6. 광고나 지표 수집같이 우선순위가 떨어지는 내용을 마지막 스크립트로 붙인다.

```html
<!DOCTYPE html>
<html>
  <script>
    <!-- 미리 빌드해둔 html 영역 여기까지가 1번째 줄 -->
    <!-- 서버에서 불러온 데이터를 window["EAGER-DATA"] 형태로 때려 넣어 파일형태로 삽입한다.-->
  </script>
</html>
```

이렇게 추측한 이유는 첫번째 줄에서 `<script>`태그가 닫히지 않고 마치 다음 페이지에 무언가 삽입되는 내용을 기대하는 것 처럼 끝났기 때문이다. 그리고 줄 단위로 데이터를 하나씩 넣는 걸로 보아, 이 줄 단위로 요청을 하고 삽입을 하는게 아닐까 싶다.

참고로 nextjs는 다음과 같은 형태로 한줄로 깔끔하게 나온다.

```html
<script id="__NEXT_DATA__" type="application/json">
  {"props":{"pageProps": {}}},"isFallback":false,"gssp":true,"customServer":true,"appGip":true,"scriptLoader":[]}
</script>
```

이렇게 처리한 이유는 아마도 내부에서 스크립트로 인한 fetch 요청을 최소화 하기 위함일 것이다. 서버에서 클라이언트 렌더링에 필요한 모든 데이터를 다 fetch 해온 다음에, 이를 클라이언트에 제공하여 마치 서버사이드 렌더링을 하는 것과 같은 효과를 누릴 수 있게 된다. 이러한 노력 덕분에 네이버 메인의 네트워크 요청을 보면 xhr 요청이 없는 것을 확인할 수 있다.

그러나 서버사이드 렌더링과는 다른 점은, 미리 만들어진 html을 내려주는 방식이 아니라는 것이다. 이는 소스보기로 부터 만들어진 html을 보면 알 수 있다.

```html
<div id="wrap">
  <div id="header" role="banner">
    <div id="topSearchWrap" class="header_inner">
      <!--  상단 검색영역    -->
    </div>
  </div>
  <div id="container" role="main">
    <div id="root"></div>
  </div>
  <div id="footer" role="contentinfo"></div>
</div>
```

html은 검색영역, 그리고 리액트의 루트 영역으로 추정되는 `div#container` 를 확인할 수 있었다. 요약하자면 서버의 데이터를 hydration 해서 내려주는 것은 nextjs와 비슷하지만, 이를 클라이언트에서만 가지고 렌더링 한다는 점은 서버사이드 렌더링 애플리케이션과 SPA 어딘가쯤에 있는 것 같은 느낌을 준다.

> 정확히는 네이버의 스크립트로 인한 xhr 요청이 없다는 뜻이다. 제3자 스크립트로 인한 요청은 확인할 수 있었다.

### 3. 각종 스크립트

이제 부터 본격적으로 스크립트 영역이다. 주요한 내용만 몇가지 살펴본다.

- `<script defer="defer" src="https://pm.pstatic.net/resources/js/polyfill.f47ccc9a.js?o=www" crossorigin="anonymous">`: 안정적인 최신 기능 사용을 위한 폴리필이다. 여기서 확인한 주요 폴리필은 다음과 같다.
  - `Map`
  - `WeakMap`
  - `MutationObserver`
  - `DOMRectReadOnly`
- `<script defer="defer" src="https://pm.pstatic.net/resources/js/preload.06cdf21d.js?o=www" crossorigin="anonymous">`: jquery와 sizzle.js가 초기화되는 파일로 보여진다. `preload`라는 이름이 달려있는 것으로 보아, 미리 `jquery`를 전역에 안정적으로 추가하기 위한 목적으로 보여진다.

> 참고로 `defer`의 경우 우선순위가 제일 낮으며, 이들이 실행되는 시점은 모든 `<script/>`가 실행된 이후다. `defer`간의 실행되는 순서는 태그 순서이므로 `preload`나 `polyfill`을 먼저 선언해주는 것이 정상적인 동작을 보장해줄 수 있다.

- `<script defer="defer" src="https://pm.pstatic.net/resources/js/search.c0419952.js?o=www" crossorigin="anonymous">`: 검색바와 관련된 스타일이 별도로 있었던 것 처럼, 검색과 관련된 스크립트 역시 따로 빼둔 것을 확인할 수 있다. 여기서 유추해볼 수 있는 내용은 두가지다.
  - 네이버의 비즈니스로직 중 검색을 최우선 순위로 삼고 있다?
  - 모종의 이유로 검색 영역은 리액트로 못넘어가지 못했다?
  - 네이버 SE (구글 비스무리한 심플버전, 이걸 알면 당신도 MZ 하지 못하다) 의 잔재다?
- `<script defer="defer" src="https://pm.pstatic.net/resources/js/main.acc2da58.js?o=www" crossorigin="anonymous">`: 이제 진짜 메인 chunk다. 한가지 특이한점은 다른 여러 사이트와 다르게 chunk를 여러 단위로 분리하지 않고 하나의 뭉탱이로 관리하고 있다는 것이다. 이 뭉탱이의 크기는 약 164.8kb 이다. 필요데 따라서 여러개의 `chunk`로 나눠서 제공할 수도 있지만,

### 4. `<div>`

이제 본격적으로 div를 포함한 body 영역이 렌더링 된다. 이 영역은 앞서 언급했듯 딱 검색 영역과 리액트 루트만 존재한다.

````html
```html
<div id="wrap">
  <div id="header" role="banner">
    <div id="topSearchWrap" class="header_inner">
      <!--  상단 검색영역    -->
    </div>
  </div>
  <div id="container" role="main">
    <div id="root"></div>
  </div>
  <div id="footer" role="contentinfo"></div>
</div>
````

## 기타

### 꼼꼼한 에러 바운더리

리액트로 만들어진 애플리케이션 들은 대게 에러 바운더리를 활용하여 에러를 처리하는데, 네이버의 경우 컴포넌트 단위로 아주 빡세게 에러 바운더리 관리를 하고 있는 것을 볼 수 있었다.

![naver-10](./images/naver-10.png)

컴포넌트 디버깅 도구로 추측컨데 `withErrorBoundary`라는 HOC를 만들고 이를 특정 컴포넌트 단위로 묶어서 에러 처리를 하는게 아닐까 싶다. 이렇게 하면 컴포넌트의 에러를 애플리케이션 전체에 증폭시키지 않아도 되므로 유용하다.

실제로 이 영역을 `false`로 바꿔서 에러를 어떻게 보여주는지 살펴보았더니 🤔

![naver-12-1](./images/naver-12-1.png)

![naver-12-2](./images/naver-12-2.png)

그냥 해당영역을 `null`로 반환하는 것을 확인할 수 있었다. 별도의 오류 컴포넌트를 선언해두지 않은 것은 장애가 난걸 동네방네 알리고 싶지 않아서가 아닐까?

### `iframe`

네이버에는 각종 iframe이 존재한다.

- 앞서 언급했던 쇼핑 https://shopsquare.naver.com/
- 최상단 배너, 로그인 하단 등 광고 영역

네이버의 경우 `X-XSS-Protection` 헤더로 cross site scripting을 막는 반면, `iframe`으로 제공되는 페이지들에는 딱히 그런 처리가 없는 것으로 확인되었다. (광고라서 누가 띄워주면 개이득?)

광고를 `iframe`으로 하는 것은 일반적인 기술이다. 이는 광고를 표시하는 컨텐츠 제공자에게 컨텐츠 제어권을 넘기지 않아도 되며, 구조적으로 분리할 수도 있으며, 성능을 별도 측정할 수 있다는 장점이 있다. 아마도 어딘가 구글 adsense 처럼 광고 sdk 가 따로 동작해서 `iframe`으로 광고를 띄우는 것이 아닐까 싶다.

### MIME 타입 요약

![naver-11](./images/naver-11.png)

대부분 요청이 이미지를 차지하고 있다. 이러한 이미지들을

- 최대한 화질 손상 없이 압축
- progressive jpeg으로 변경 (IE에서는 소용 없지만)
- 중요하지 않은 이미지 들에 대한 [`loading=lazy` 속성 추가](https://developer.mozilla.org/en-US/docs/Web/Performance/Lazy_loading#images_and_iframes)

## 요약

- naver 는 클라이언트에서 fetch 요청을 최소화 하기 위해 서버에서 미리 필요한 데이터를 요청 한다음, `window['EAGER-DATA']` 에 넣어둔다.
- 클라이언트는 클라이언트 사이드 렌더링을 채택하여 `window['EAGER-DATA']` 를 기준으로 렌더링한다
- 네이버의 핵심 기능인 검색은 별도의 html, js, css 로 구성되어 있다.
- 네이버는 광고 노출을 위해 여러 `iframe`을 사용하고 있으며, 네이버 본체를 제외하고 별도 X-XSS-Protection 처리는 되어 있지 않다.
