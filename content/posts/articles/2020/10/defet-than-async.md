---
title: '왜 Async 보다는 Defer를 써야할까'
tags:
  - chrome
  - javascript
  - browser
published: true
date: 2020-10-20 23:32:39
description: '스크립트 실행 최적화를 위해 잘 고민해봐야 한다.'
---

웹사이트의 렌더링 성능을 최적화하면서, 가장 중점적으로 살펴봐야 할 것은 렌더링을 막는 자바스크립트 실행이다. 그런데 종종, 블로킹 자바스크립트의 원인이 `async` 태그라는 것을 발견하게 되다. 많은 사람들이 async가 렌더링을 막지 않는다고 생각한다. 그러나 슬프게도, 그렇지 않다. async 태그는 렌더링을 블로킹할 뿐만아니라, 스크립트를 동기로 실행하는 것을 막을 수도 있다. 추정컨데, 동기식으로 만든 스크립트는 페이지에 중요한 컨텐츠 이므로, 이를 지연시키는 것은 사용자 경험에 안좋은 영향을 미친다.

이번 글에서는 `defer`가 `async` 보다 더 기본적으로 선택해야할 것인지를 알아본다.

## 병렬화는 성능에 중요하다.

아마도 실제 웹 성능 향상에 가장 중요하게 영향을 미친 것 중 하나는, 스크립트 다운로드를 병렬로 다운로드 받는 것이다. 2006년 이전에는, 브라우저는 모든 외부 스크립트를 순서대로 받았다.

```html
<script src="aphid.js"></script>
<script src="bmovie.js"></script>
<script src="seaserpent.js"></script>
<img src="deejay.gif" />
<img src="elope.gif" />
```

이런 페이지가 있다고 하면, 아래와 같은 순서로 실행된다.

```bash
aphid.js        ====xxx
bmovie.js              =====xx
seaserpent.js                 =====xx
deejay.gif                           =====
elope.gif                            =====
DOM Interactive                      *
image render                              *
```

여기서 `==`는 다운로드하는 작업을 의미하고, `xx`는 스크립트 파싱과 실행을 의미한다. DOM interactive는 브라우저가 HTML 구문 분석을 완효한 후에 실행된다.

이러한 접근 방식에는 많은 비효율이 존재한다. 스크립트는 뭐 순차적으로 실행될 순 있지만, 다운로드는 병렬로 진행할 수 있다. 또한 스크립트가 이미지와 같은 비 스크립트 리소스를 차단할 필요도 없다.

IE8을 필두로 많은 브라우저가 스크립트를 병렬로 다운로드 할 수 있는 `preloader`개념을 도입하기 시작했다. 이로 인해 페이지 속도가 엄청나게 빨라졌다.

```bash
aphid.js        ====xxx
bmovie.js       =====  xx
seaserpent.js   =====    xx
deejay.gif      =====
elope.gif       =====
DOM Interactive            *
image render               *
```

`preloader`를 활용하여 모든 리소스가 병렬로 다운로드 되는 것을 볼 수 있다. 여전히 DOM interactive는 세 리소스를 기다려야 하지만, 이전에 비해서 훨씬더 빠르게 동작하는 것을 볼 수 있다.

## 동기 스크립트는 HTML 구문 분석을 차단한다.

preloader는 동기식 스크립트가 다른 리소스의 다운로드를 차단하지 않도록 변경하여 웹 성능을 향상시켰다. 그러나 동기 스크립트는 여전히 HTML 구문 분석을 차단한다. HTML 구문 분석이 `<script/>` 태그에 도달하면 해당 스크립트가 다운로드되고, 구문 분석이 실행 될 때까지 중지된다. HTML 파서가 차단되면 사용가자 페이지의 컨텐츠를 보기 위해 기다려야 한다. 이러한 이유로 인해 앞선 예제에서, 이미지가 다 다운로드 되었음에도 나중에 뵤여지게 된다.

동기스크립트가 HTML 분석을 차단하는 것을 막기 위해, 개발자들은 스크립트를 비동기적으로 로드하는 방법을 찾기 시작했다. 모든 스크립트가 비동기로 동작할 필요는 없다. 페이지 렌더링에 있어서 중요한 스크립트는 동기적으로 로딩되어야 한다. 그러나 중요한 렌더링 이외의 요소는 비동기적으로도 할 수 있다. 2009년 이전에는 이런 것을 처리하기 위한 [꼼수](https://www.stevesouders.com/blog/2009/04/27/loading-scripts-without-blocking/)가 존재했다. 그리고, HTML에 async와 defer가 추가되었다.

## async와 defer

async와 defer는 HTML 파서를 차단하지 않고 스크립트를 로드할 수 있다는 점에서 유사하다. 즉, 둘다 사용자는 페이지를 더 빨리 볼 수 있따. 그러나 여기에 차이점이 존재한다.

- `async`로 로드된 스크립트는 다운로드가 완료되면 즉시 구문 분석을 하고 실행된다. 그에 반해 `defer`는 HTML 문서가 파싱되기 전까지 실행되지 않는다.
- `async`는 순서없이 로드가 가능하지만 `defer`는 마크업 순서대로 로딩된다.

### async

```html
<script ASYNC src="aphid.js"></script>
<script src="bmovie.js></script>
<script src="seaserpent.js"></script>
<img src="deejay.gif">
<img src="elope.gif">
```

```bash
aphid.js        ====xxx
bmovie.js       =====  xx
seaserpent.js   =====    xx
deejay.gif      =====
elope.gif       =====
DOM Interactive            *
image render               *
```

`aphid`가 async였지만, 다운로드가 먼저 되었기 때문에 다른 스크립트 실행을 차단하고 먼저 실행되었다. 다시말해, `async`는 다운로드 된 뒤에 모든 동기 스크립트 실행을 차단해 버린다.

### defer

```html
<script DEFER src="aphid.js"></script>
<script src="bmovie.js></script>
<script src="seaserpent.js"></script>
<img src="deejay.gif">
<img src="elope.gif">
```

```
aphid.js        ====     xxx
bmovie.js       =====xx
seaserpent.js   =====  xx
deejay.gif      =====
elope.gif       =====
DOM Interactive          *
image render             *
```

defer는 DOM이 상호작용이 가능해지는 시점에 실행된다. defer는 다운로드가 먼저 되었음에도 다른 동기 스크립트가 실행된 이후에 실행된 것을 볼 수 있다.

## defer를 더 선호해야하는 이유

`defer`는 항상 `async`와 동시에, 또는 그 이후에 스크립트 실행을 발생시킨다. 아마도 스크립트 중에서도 덜 중요한 것들을 `defer`나 `async`로 만들 것이다. 따라서 기본 렌더링 시간 외에 실행 될 수 있도록 `defer`로 하는 것이 좋다. `defer`는 동기 스크립트를 차단할 수 없지만, `async`는 스크립트 다운로드에 따라 차단할 수도 있따. 동기 스크립트는 일반적으로 페이지에서 중요한 내용을 담고 있으므로, 다른 작업이 방해하지 않도록 `defer`를 쓰는 것이 좋다.

이러한 잘못된 최적화의 예는 종종 찾아볼 수 있다. [예전 인스타그램 웹페이지 테스트 결과](https://www.webpagetest.org/result/161204_ZY_9a82d23e52565194cb985a10cf8d5465/2/details/)를 한번 살펴보자. 4번째 스크립트가 `async`로 로딩되는 것을 볼 수 있다. 그리고 [Timeline](https://www.webpagetest.org/chrome/timeline.php?test=161204_ZY_9a82d23e52565194cb985a10cf8d5465&run=2)을 살펴보면, 이 스크립트가 다른 스크립트 보다 0.6초정도 먼저 (c0456c81549b.js) 실행되서 다른 스크립트를 블록했다.

반대로 yelp의 [웹 페이지 테스트결과](https://www.webpagetest.org/result/161206_RP_DBM/3/details/)를 보자. 두번째 스크립트가 defer로 로딩되어있는데, 다운로드가 된 이후에 실행은 `DOM Content Loaded` 이후에 이루어졌다. 따라서 다른 동기스크립트가 먼저 로딩 되었고, 렌더링이 더 빨리 이루어졌다.

## 결론

결론은 DOM Interactive 까지 defer 스크립트 실행을 지연하는 것과는 다르게, async는 다운로드가 빨리 되서 실행이 먼저되버리고, 다른 스크립트 분석을 멈춰버리게 하는 위험성을 가지고 있다는 것이다. 따라서, DOM Interative 시간을 아는 것이 굉장히 중요하다. [Alexa Top 100](https://www.alexa.com/topsites)에 따르면 2016년 11월 기준 DOM interactive의 중앙값은 2.1초이지만, 하위 95%의 경우에는 11.2초이다. 이정도 값이면, async 스크립트가 DOM Interative 이전에 다운로드를 마치고, 페이지 렌더링을 방해하고 있는지 합리적으로 의심해 볼법하다.

스크립트에서 async를 사용하면, defer로 바꿔서 렌더링이 더 빨리 되는지 확인해보자.

출처: https://calendar.perfplanet.com/2016/prefer-defer-over-async/
