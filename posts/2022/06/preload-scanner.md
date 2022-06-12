---
title: '브라우저의 프리로드 스캐너(pre-load scanner)와 파싱 동작의 이해'
tags:
  - web
  - browser
published: true
date: 2022-06-12 18:10:40
description: '브라우저랑 싸우지마'
---

## Table of Contents

## Introduction

웹 개발자가 웹 페이지 속도를 개선하기 위해서 가장 먼저 알아야할 것은, 웹 페이지 로딩 속도 개선을 위한 테크닉이나 팁이 아닌 바로 브라우저의 동작 원리다. 우리가 최적화 하려고 노력하는 것 만큼, 브라우저도 웹페이지를 로딩할 때 최적화 하려고 더 노력한다. 하지만 우리의 이런 최적화 노력이 의도치 않게 브라우저의 최적화 노력을 방해할 수도 있다.

페이지 속도 개선에 있어 가장 먼저 이해해야할 것 중 하나는 바로 브라우저의 프리로드 스캐너다. 프리로드 스캐너는 무엇인지, 그리고 우리가 이 작업을 방해하지 ㅇ낳기 위해서는 무엇을 해야 하는지 알아보자.

## Pre-load Scanner란 무엇인가

모든 브라우저는 raw markup 상태의 파일을 [토큰화](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization) 하고, [객체 모델](https://developer.mozilla.org/ko/docs/Web/API/Document_Object_Model)로 처리하는 HTML 파서를 기본적으로 가지고 있다. 이 모든 작업은 이 파서가 `<link />` 또는 `async` `defer`가 없는 `<script />`와 같은 [블로킹 리소스](https://web.dev/render-blocking-resources/)를 만나기 전 까지 계속 된다.

![html-parser](https://web-dev.imgix.net/image/jL3OLOhcWUQDnR4XjewLBx4e3PC3/mXRoJneD6CbMAqaTqNZW.svg)

CSS 파일의 경우, [스타일링이 적용되지 않는 콘텐츠가 잠깐 뜨는 현상(Flash of unstyled content, AKA FOUC)](https://en.wikipedia.org/wiki/Flash_of_unstyled_content)을 방지하기 위해 파싱과 렌더링이 차단된다.

그리고 자바스크립트의 경우에는, 앞서 언급했듯 `async` `defer`가 없는 `<script/>`를 만나게 되면 파싱과 렌더링 작업이 중단된다.

> `<script />`에 `type=module`이 있다면 기본적으로 `defer`로 동작한다.

그 이유는 HTML 파서가 동작하는 동안, 이 스크립트가 DOM을 수정할 것인지 브라우저 입장에서는 알 수 없기 떄문이다. 따라서 일반적으로는 이러한 자바스크립트를 문서의 끝에 두어 렌더링 및 파싱에 미치는 영향을 제한하는 것이 일반적이다.

어쩄든, 이러한 중요한 파싱 단계를 차단하는 것은 바람직하지 않다. 왜냐하면 다른 중요한 리소스를 찾는 과정을 지연시킴으로써 퍼포먼스를 저하시킬 수 있기 때문이다. 이러한 문제를 완화 시키기 위한 것이 바로 프리로드 스캐너라고 하는 보조 HTML 파서다.

![pre-load scanner](https://web-dev.imgix.net/image/jL3OLOhcWUQDnR4XjewLBx4e3PC3/6lccoVh4f6IJXA8UBKxH.svg)

> 기본 HTML 파서는 CSS를 로딩하고 처리할 떄 블로킹 되지만, 프리로드 스캐너는 마크업에서 이미지 리소스를 찾고 기본 HTML 파서가 차단 해제되기전에 로드를 시작할 수 있다.

이 프리로드 스캐너의 역할은 speculative, 즉 추측에 근거하며, 이 말의 뜻은 기본 HTML 파서가 리소스를 발견하기 전에 먼저 리소스를 찾기위해 마크업 문서를 훑는 다는 것을 의미한다.

## 프리로드 스캐너가 언제 동작하는지 확인하는 방법

앞서 이야기 하였듯이, 렌더링과 블로킹을 차단하는 리소스가 있기 때문에 프리로드 스캐너가 존재한다. 이러한 두 가지 성능 문제가 존재하지 않는다면 딱히 프리로드 스캐너가 필요하지 않을 것이다. 따라서 웹 페이지가 프리로드 스캐너의 이점을 얻을 수 있는지 여부를 파악하는 열쇠는 이러한 블로킹 현상의 존재 여부에 따라 다르기 떄문에, 프리로드 스캐너가 동작하는지 알기 위해서는 인위적인 딜레이를 집어넣을 필요가 있다.

[프리로드 스캐너의 동작 방식을 확인해보기 위한 예제 페이지](https://preload-scanner-fights.glitch.me/artifically-delayed-requests.html)

> 결과: https://www.webpagetest.org/result/220612_BiDcZ0_2E9/

먼저 CSS 파일은 렌더링과 파싱을 모두 차단하기 때문에, 스타일 시트를 집어 넣음으로서 아래와 같은 인위적인 딜레이를 집어넣을 수 있다. 이러한 딜레이 덕분에, 프리로드 스캐너가 작동하는 것을 볼 수 있다.

![waterfall-chart](https://www.webpagetest.org/waterfall.php?test=220612_BiDcZ0_2E9&run=1&cached=&step=1)

> css가 블로킹 리소스라서 잠시간 딜레이가 있었지만, 이미지는 프리로드 스캐너가 먼저 발견해서 찾은 모습이다.

이 차트에서 볼 수 있는 것 처럼, 프리로드 스캐너는 렌더링 및 파싱이 차단되는 와중에서 `<img />`를 검색한다. 이 최적화 없이는 차단 되는 동안 리소스를 가져올 수 없으므로, 리소스를 가져오는 과정은 동시적이 아닌 연속적으로 이루어질 것이다.

## async script 삽입

`<head/>`에 아래와 같은 자바스크립트를 삽입해보자.

```html
<script>
  const scriptEl = document.createElement('script')
  scriptEl.src = '/yall.min.js'

  document.head.appendChild(scriptEl)
</script>
```

삽입되는 스크립트는 `async`가 기본값이기 때문에 비동기적으로 동작하는 것 처럼 보일 것이다. 즉 가능한 빨리 실행되고, 렌더링을 차단하지 안흔다. 물론, 이는 최적화가 적용되어 있는 것 처럼 보이지만.. 이 인라인 스크립트가 외부 CSS 파일을 로드하는 `<link/>`뒤에 온다고 가정하면 다음과 같은 결과가 나타난다.

[테스트 페이지](https://preload-scanner-fights.glitch.me/injected-async-script.html)

[결과](https://www.webpagetest.org/result/220612_BiDcPM_2FE/)

![waterfall-chart](https://www.webpagetest.org/waterfall.php?test=220612_AiDcHH_2FY&run=1&cached=&step=1)

무슨일이 일어났는지 하나씩 살펴보자.

1. 0초에 문서를 요청했다.
2. 1.3초 쯤에 요청에 대한 첫번째 바이트 응답이 왔다.
3. 2.4초 쯤에 이미지와 CSS 요청이 이루어졌다.
4. 파서가 스타일 시트를 로딩하느라 차단되고, 비동기 스크립트를 주입하는 인라인 자바스크립트가 2.8초 쯤에 나타나기 때문에 스크립트가 제공하는 `async` 기능을 바로 사용할 수 없게 됨

스타일 시트 다운로드가 완료된 이후에만 스크립트에 대한 요청이 발생한다. 따라서 스크립트가 최대한 빨리 실행되지 않는다. 이는 페이지의 TTI(Time To Interactive)에 영향을 미칠 수 있다. 이와 반대로, `<img/>`는 서버에서 제공하는 마크업에서 감소하기 떄문에 프리로드 스캐너에 의해 검색될 것이다.

그렇다면 스크립트를 DOM에 주입하는 대신, `async` 값과 함께 일반 `<script/>`를 사용하면 어떻게 될까?

```html
<script src="/yall.min.js" async></script>
```

[웹페이지](https://preload-scanner-fights.glitch.me/inline-async-script.html)

[결과](https://www.webpagetest.org/result/220612_AiDcSY_2HM/)

![waterfall-chart](https://www.webpagetest.org/waterfall.php?test=220612_AiDcSY_2HM&run=1&cached=&step=1)

> 이 페이지는 하나의 스타일 시트와 `async` `<script/>`한개가 포함되어 있다. 프리로드 스캐너는 렌더 블로킹 단계에서 스크립트를 발견하고, CSS와 동시에 로딩한다.

[`rel=preload`](https://developer.mozilla.org/en-US/docs/Web/HTML/Link_types/preload)를 사용하면 문제를 해결할 수도 있지 않을까? 이는 효과적일 수도 있지만, 약간의 부작용이 있을 수 있다. 그렇다면 왜 `<script/>`를 DOM에 주입하지 않음으로써 피할 수 있는 문제를 해결하기 위해 `rel=preload`를 사용하는 이유는 무엇일까?

[웹페이지](https://preload-scanner-fights.glitch.me/preloaded-injected-async-script.html)

[결과](https://www.webpagetest.org/result/220612_AiDcVX_2JA/)

![waterfall-chart](https://www.webpagetest.org/waterfall.php?test=220612_AiDcVX_2JA&run=1&cached=&step=1)

> 이 페이지는 하나의 스타일 시트와 `async` 스크립트를 삽입했는데, `async` 스크립트는 발견되는 즉시 프리로딩 된 것을 알 수 있다.

프리로딩은 여기서 문제를 해결한 것 처럼 보이지만, 처음 두 데모의 `async` 스크립트는 `<head />`에 삽입되었음에도 불구하고 낮은 우선순위로 로드 되는반면, 스타일 시트는 높은 우선순위로 로딩된다. 비도기 스크립트가 preloading된 마지막 데모페이지에서는, 스타일 시트와 스크립트의 우선순위가 모두 높음으로 승격되었다.

> 우선순위는 크롬의 dev tool 네트워크 탭에서 볼 수 있다.

리소스의 우선순위가 올라가면, 브라우저는 더 많은 대역폭을 리소스에 할당한다. 즉, 스타일 시트 우선순위가 가장 높더라도 스크립트의 우선순위가 높아지면 이러한 대역폭에 경합이 발생할 수 있다. 연결 속도가 느리거나, 리소스가 상당히 크다면 이러한 문제가 더 두드러질 수 있다.

답은 간단하다. 스크립트가 웹 페이지 시작중에 실행되어야 한다면 DOM에 삽입하여 프리로드 스캐너를 방해하지 말자. `<script/>` 요소의 위치 뿐만 아니라, `defer` `async` 속성에 대한 실험도 해보면서 확인해봐야 한다.

## 자바스크립트 Lazy Loading

레이지 로딩은 데이터를 불러오기 위한 효과적인 방법으로 일반적으로 이미지에 적용된다. [그러나 때때로 이 레이지 로딩이 above the fold 영역에 있는 이미지에 잘못 적용될 수 있다.](https://yceffort.kr/2022/06/optimize-LCP#lcp-%EB%A5%BC-lazy-load-%ED%95%98%EC%A7%80-%EB%A7%90-%EA%B2%83)
