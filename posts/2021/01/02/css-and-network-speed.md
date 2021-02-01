---
title: 'CSS와 네트워크 속도'
tags:
  - browser
  - CSS
published: true
date: 2021-02-01 16:19:06
description: '내 일이 아니라고 생각하면 관심이 안가더라고'
---

## Table of Contents

CSS는 웹 페이지의 성능에 영향을 미치는 중요한 요소중 하나다. 그 이유는

1. 브라우저는 렌더 트리를 만들기 전가지 페이지를 렌더링 할 수 없음
2. 렌더트리는 DOM과 CSSOM을 합쳐서 만드는 것
3. DOM은 HTML과 함께 자바스크립트의 실행을 차단함
4. CSSOM 이란 DOM에 적용해야 하는 모든 CSS 규칙을 의미함
5. 자바스크립트는 `async` 와 `defer`로 차단하지 않도록 만드는 것은 쉽지만
6. CSS를 비동기로 처리하는 것은 매우 어렵
7. 따라서 페이지가 가장 느린 스타일 시트가 렌더링 되는 순간에 페이지가 렌더링 된다는 것을 명심해야 한다.

이러한 점을 염두해두고, DOM과 CSSOM을 최대한 빠르게 구성해야 한다. 일반적으로 DOM은 HTML 응답에 따라서 만들어지므로 비교적 빠르다. 그러나 CSS는 거의 대부분 HTML의 하위리소스이기 때문에 CSSOM을 구성하는 것은 일반적으로 시간이 더 걸린다.

## Critical CSS

렌더링 시작 시간을 줄이는 가장 효과적인 방법은 Critical CSS Pattern을 사용하는 것이다. 이는 렌더링 시작에 필요한 모든 스타일 (일반적으로 스크롤 하지 않아도 맨 처음에 필요한 모든 항목에 필요한 스타일)을 의미한다. 문서의 `<head/>`에 `<style/>` 태그로 인라인처리하고, 나머지 스타일 시트는 비동기적으로 로드 하는 방식이다.

물론 이방식은 효과적이지만 간단하지는 않다. 사이트가 엄청나게 동적일 경우, 스타일을 추출하는 것이 어렵다. 또한 프로세스를 자동화 해야 하며, 안보이는 부분에 대한 정의를 내려야 하고, 예외 처리를 하기가 어렵다. 이는 코드가 커질 수록 어렵다.

## 미디어 쿼리로 나누기

현재 컨텍스트 (medium, 스크린크기, 해상도, 방향 등)에 맞는 css를 가장 최우선 순위로 다운로드 하고, 그 외의 것은 나중에 다운로드 하는 방식이다. 기본적으로 현재 뷰를 렌더링하는데 필요하지 않은 CSS는 브라우저에 의해 지연되어 로딩 된다.

```html
<link rel="stylesheet" href="all.css" />
```

이처럼 되어 있는 것을 미디어 쿼리로 분할 할 수 있다면 네트워크가 분할해서 다르게 취급할 것이다.

```html
<link rel="stylesheet" href="all.css" media="all" />
<link rel="stylesheet" href="small.css" media="(min-width: 20em)" />
<link rel="stylesheet" href="medium.css" media="(min-width: 64em)" />
<link rel="stylesheet" href="large.css" media="(min-width: 90em)" />
<link rel="stylesheet" href="extra-large.css" media="(min-width: 120em)" />
<link rel="stylesheet" href="print.css" media="print" />
```

https://caniuse.com/css-mediaqueries 대부분의 브라우저에서 사용할 수 있다.

물론 여전히 브라우저가 모든 css 파일을 다운로드 하긴 하지만, 현재 컨텍스트를 충족하는데 필요한 파일의 렌더링만 차단한다.

## `@import` 사용하지 않기

`@import`는 아무튼 느리다. 그래서 렌더링 성능에 안좋다. `@import`의 작동 과정을 살펴보자.

1. HTML 다운로드
2. HTML이 CSS를 요청
3. CSS가 또 다른 `@import`에 있는 CSS를 요청
4. 이게 다 끝나면 렌더 트리 생성

```html
<link rel="stylesheet" href="all.css" />
```

안에

```css
@import url(imported.css);
```

와 같은 코드가 있다면, 폭포수 형태로 다운로드를 시작할 것이다.

이는 그냥 단순히

```html
<link rel="stylesheet" href="all.css" />
<link rel="stylesheet" href="imported.css" />
```

로 처리한다면 두개를 동시에 병렬화 하여 다운로드 할 것이다. 그런데 현재 파일에서 `@import` 구문을 지울 수 없는 상황이라 할지라도, 저렇게 별개로 따로 선언해주는 것이 좋다. 그렇다고 해서 브라우저가 중복으로 파일을 다운로드 받지 않을 것이다.

## `<link rel="stylesheet" />`를 비동기 코드 전에 두지 않기

**브라우저는 현재 실행중인 CSS가 있을 경우 `<script/>`를 실행하지 않는다.**

```html
<link rel="stylesheet" href="slow-loading-stylesheet.css" />
<script>
  console.log('I will not run until slow-loading-stylesheet.css is downloaded.')
</script>
```

이는 의도된 다분히 의도적인 동작이다. CSS가 다운로드 중이라면, HTML은 어떤 동기 `<script/>`도 실행하지 않는다. 스크립트 태그 내에서 CSS 가 도착하여 파싱되기 전까지 이에 대한 정보를 찾는 경우, Javascript가 응답하는 내용이 잘못될 수도 있다. 이를 방지하기 위해 브라우저는 CSSOM이 구성될 떄 까지 `<script/>`를 실행하지 않는다.

따라서 CSS의 다운로드 시간이 비동기 코드에 영향을 미친다는 것이다. 만약 `<link rel="stylesheet" />` 앞에 비동기 코드를 둔다면, 이는 CSS파일이 다운로드 되어 파싱되기 전까지 실행되지 않을 것이다. 이 말인 즉슨 CSS가 모든 작업을 뒤로 미룬다는 뜻이 된다.

```html
<link rel="stylesheet" href="app.css" />

<script>
  var script = document.createElement('script')
  script.src = 'analytics.js'
  document.getElementsByTagName('head')[0].appendChild(script)
</script>
```

이렇게 순서가 주어지면, CSSOM이 생성되기 전까지 자바스크립트 파일이 다운로드 되지 않는 다는 것을 알 수 있다. 이는 구글 애널리틱스와 같이 제 3자 스크립트를 안전하게 로드 하기 위해 비동기 코드를 제공하는 방식이다. 개발자가 이러한 제3자 코드를 코드 맨 뒷부분에 배치 하고 싶은 것은 일반적이다. 그러나 이는 실수가 될 수 있다.

따라서 `<script/>` 내에 css에 의존하는 코드가 없다면, 이를 상단에 배치하는 것이 좋다.

```html
<script>
  var script = document.createElement('script')
  script.src = 'analytics.js'
  document.getElementsByTagName('head')[0].appendChild(script)
</script>

<link rel="stylesheet" href="app.css" />
```

따라서

- CSSOM에 의존하지 않는 자바스크립트는 CSS이전에
- CSSOM에 의존하는 자바스크립트는 이후에

둔다.

파일이 서로 의존적이지 않은 경우, 차단 스크립트를 차단 스타일 위해 배치 해야 한다. 자바스크립트가 실제로 의존하지 않는 CSS때문에 자바스크립트 실행을 지연시킬 필요는 없다. 만약 자바스크립트 중 일부는 CSS에 의존하고, 일부는 그렇지 않는 경우에는 두개로 분할하는 것이 가장 빠른 방식이다.

```html
<!-- This JavaScript executes as soon as it has arrived. -->
<script src="i-need-to-block-dom-but-DONT-need-to-query-cssom.js"></script>

<link rel="stylesheet" href="app.css" />

<!-- This JavaScript executes as soon as the CSSOM is built. -->
<script src="i-need-to-block-dom-but-DO-need-to-query-cssom.js"></script>
```

이러한 로딩 순서를 이용하면, 다운로드와 실행이 모두 최적의 순서로 발생된다.

## `<link rel="stylesheet" />`는 `<body/>`에 위치 시키기

기존 HTTP 1.1 에서는 모든 스타일을 하의 기본 번들로 연결하는 것이 일반적이다.

```html
<!DOCTYPE html>
<html>
  <head>
    <link rel="stylesheet" href="app.css" />
  </head>
  <body>
    <header class="site-header">
      <nav class="site-nav">...</nav>
    </header>

    <main class="content">
      <section class="content-primary">
        <h1>...</h1>

        <div class="date-picker">...</div>
      </section>

      <aside class="content-secondary">
        <div class="ads">...</div>
      </aside>
    </main>

    <footer class="site-footer"></footer>
  </body>
</html>
```

그러나 위 코드는 아래와 같은 비효율성이 있다.

1. 실제 페이지에서 필요한 css는 일부분 이지만, 필요한 것보다 더 많은 `app.css`를 다운로드 하고 있다.
2. 캐시전략이 비효율적이다. 예를 들어서 특정 섹션의 배경색만 변경하기 위해서는, `app.css` 전체의 캐시를 버려야 한다.
3. 현재 페이지에서 `app.css`의 어느정도를 필요로 하든지간에, 전체 내용이 도착하기 전까지 렌더링이 블로킹 된다.

이제 http/2 를 사용하면 1, 2번 문제를 해결할 수 있다.

```html
<!DOCTYPE html>
<html>
  <head>
    <link rel="stylesheet" href="core.css" />
    <link rel="stylesheet" href="site-header.css" />
    <link rel="stylesheet" href="site-nav.css" />
    <link rel="stylesheet" href="content.css" />
    <link rel="stylesheet" href="content-primary.css" />
    <link rel="stylesheet" href="date-picker.css" />
    <link rel="stylesheet" href="content-secondary.css" />
    <link rel="stylesheet" href="ads.css" />
    <link rel="stylesheet" href="site-footer.css" />
  </head>
  <body>
    <header class="site-header">
      <nav class="site-nav">...</nav>
    </header>

    <main class="content">
      <section class="content-primary">
        <h1>...</h1>

        <div class="date-picker">...</div>
      </section>

      <aside class="content-secondary">
        <div class="ads">...</div>
      </aside>
    </main>

    <footer class="site-footer"></footer>
  </body>
</html>
```

이제 모든 것을 다 다운로드 하는 대신에, 페이지에 필요한 css만 다운로드 할 수 있다. 그러면 주요 렌더링 과정에서 차단 CSS의 크기가 줄어든다. 또한 캐시 전략을 보다 신중하게 선택할 수 있다. 캐시가 필요한 파일만 날리고, 나머지는 유지한다.

하지만 여전히 해결하지 못한 점은 이 모든 것이 렌더링을 가로막는 다는 것이다. 여전히 우리 사이트는 가장 느린 스타일시트의 속도에 좌우되고 있다. 그러나 이제 크롬 69버전 이후, 파이어폭스, IE/Edge 등에서 아래와 같은 코드가 가능해진다.

```html
<!DOCTYPE html>
<html>
  <head>
    <link rel="stylesheet" href="core.css" />
  </head>
  <body>
    <link rel="stylesheet" href="site-header.css" />
    <header class="site-header">
      <link rel="stylesheet" href="site-nav.css" />
      <nav class="site-nav">...</nav>
    </header>

    <link rel="stylesheet" href="content.css" />
    <main class="content">
      <link rel="stylesheet" href="content-primary.css" />
      <section class="content-primary">
        <h1>...</h1>

        <link rel="stylesheet" href="date-picker.css" />
        <div class="date-picker">...</div>
      </section>

      <link rel="stylesheet" href="content-secondary.css" />
      <aside class="content-secondary">
        <link rel="stylesheet" href="ads.css" />
        <div class="ads">...</div>
      </aside>
    </main>

    <link rel="stylesheet" href="site-footer.css" />
    <footer class="site-footer"></footer>
  </body>
</html>
```

이를 활용하면 페이지를 점진적으로 렌더링할 수 있다. 자세한 내용은 [여기](https://jakearchibald.com/2016/link-in-body/)를 참조.

## 요약

- 렌더링 시작에 필요하지 않은 CSS 를 지연시킨다.
  - Critical CSS로 분리하거나
  - CSS를 미디어 쿼리로 분리한다.
- `@import`의 사용을 줄인다.
- 동기 CSS와 자바스크립트의 순서에 주의한다.
  - CSS 이후에 있는 자바스크립트는 CSSOM이 구성되기 전까지 실행되지 않는다.
- DOM이 필요로 하는 CSS만 불러온다.
  - 이는 점진적 렌더링을 가능하게 하고, 초기 렌더링을 블록하지 않는다.
