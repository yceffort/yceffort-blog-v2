---
title: '브라우저와 Nodejs의 이벤트 루프는 무엇이 다를까'
tags:
  - web
  - javascript
  - browser
published: true
date: 2021-08-10 22:22:37
description: '인생은 돌고 도는 이벤트 루프'
---

## 이벤트 루프는 정확히 무엇인가?

`이벤트 루프` 사실 일반적인 프로그래밍 패턴을 지칭하는 용어다. 프로그래밍의 이벤트나 메시지를 대기하나가 처리하는 일종의 프로그래밍 구조체라고 볼 수 있다. (https://ko.wikipedia.org/wiki/%EC%9D%B4%EB%B2%A4%ED%8A%B8_%EB%A3%A8%ED%94%84) 자바스크립트와 Nodejs의 이벤트 루프도 별반 다르지 않다. 자바스크립트는 애플리케이션이 실행되면 다양한 이벤트를 발생시키고, 이러한 이벤트는 처리를 위해 이벤트 핸들러 형태로 대기열에 존재한다. 이벤트 루프는 대기중인 이벤트 핸들러를 지속적으로 지켜보다가, 이벤트 핸들러가 존재하면 이를 실행한다.

### HTML5 스펙으로 살펴보는 이벤트 루프

[HTML5의 스펙](https://html.spec.whatwg.org/)은 여러 벤더가 브라우저나 자바스크립트 런타임, 또는 기타 관련한 라이브러리를 개발하는데 사용할 수 있는 표준 가이드라인을 제시한다.

대부분의 브라우저와 자바스크립트 런타임은 이러한 가이드라인을 그대로 따르기 때문에 전세계 웹서비스에 더 나은 호환성을 제공한다. 그러나 사실은 이 단일 소스에서 약간씩 벗어나서 흥미로운 (혹은 짜증나는) 결과를 유발하기도 한다.

여기에서는 이러한 흥미로운 결과, 특히 Nodejs와 브라우저와의 차이에 대해서 알아보려고 한다. 개별 브라우저 구현은 언제든 조금씩 변할 수 있으므로, 자세히 알아보지는 않는다.

### 클라이언트 사이드와 서버사이드 자바스크립트

지난 수년간, 자바스크립트는 브라우저에서 실행되는 웹 애플리케이션에서만 사용되어져 왔다. 그리고 이 후 자바스크립트는 nodejs를 사용하여 서버 사이드 애플리케이션을 만드는데에도 사용할 수 있다. 두 곳 모두 자바스크립트를 사용하지만, 클라이언트와 서버사이드에서의 요구사항은 조금씩 다를 수 있다.

브라우저는 일종의 샌득박스 환경이며, 파일 시스템 작업, 네트워크 작업 등 자바스크립트가 수행할 수 있는 작업에 권한 제한이 있다. 그러나 서버사이드 자바스크립트(Nodejs)는 이벤트루프에서 이러한 것들을 모두 실행할 수 있다.

브라우저와 Nodejs 모두 자바스크립트를 사용하여 비동기 이벤트 기반 패턴을 구현한다. 그러나 브라우저의 맥락에서 봤을 때에 "이벤트"란 웹 페이 지 내에서의 상호작용 (클릭, 마우스 이동, 키보드 이벤트 등..)이지만, Nodejs에서의 맥락에서 이벤트란 파일 I/O, 네트워크 I/O 등이다. 이러한 요구 사항의 차이로 인해 크롬과 Node는 자바스크립트 실행을 위해 모두 V8 엔진을 사용하지만, 이벤트 루프 구현에는 차이가 있다.

'이벤트루프'란 결국 프로그래밍 패턴에 불과하기 때문에, V8은 자바스크립트 런타임과 함께 외부 이벤트 루프 구현을 플러그인 해줄 수 있록 해준다. 이러한 유연성을 바탕으로, 크롬 브라우저는 [libevent](https://libevent.org/)를, nodejs는 [libuv](https://blog.insiderattack.net/javascript-event-loop-vs-node-js-event-loop-aea2b1b85f5c#:~:text=and%20NodeJS%20uses-,libuv,-to%20implement%20the)를 각각 이벤트 루프 구현을 위해 사용한다. 그러므로, 자바스크립트와 Nodejs의 이벤트루프는 기본적으로 다른 라이브러리를 사용하여 약간의 차이가 있을 수 있지만, '이벤트루프'라고 하는 일반적인 프로그래밍 패턴을 구현하고 있다는 것에서 비슷하다.

## 브라우저 vs Nodejs 무엇이 다른가?

### 마이크로, 그리고 매크로 태스크

> 간단히말해, 마이크로 태스크와 매크로 태스크는 서로 다른 비동기 태스크 처리기다. 매크로 태스크에 비해 마이크로 태스크의 우선순위가 더 높다. 마이크로 태스크의 예로는 `Promise`가 있다. `setTimeout은 대표적인 매크로 태스크다.

브라우저와 Nodejs에 눈에 띄는 차이점은 **마이크로 태스크와 매크로 태스크의 우선순위를 어떻게 정하느냐** 이다. Nodejs 11 이상에서는 브라우저의 동작과 일치하지만, 이전 버전은 상당히 다르다. 자, 아래 면접 질문으로 나올 것 만 같은 아래 코드르 보자.

> nodejs 11이전 버전에서 무슨일이 있는지 살펴보려면 https://blog.insiderattack.net/new-changes-to-timers-and-microtasks-from-node-v11-0-0-and-above-68d112743eb3

```javascript
Promise.resolve().then(() => console.log('promise1 resolved'))
Promise.resolve().then(() => console.log('promise2 resolved'))
setTimeout(() => {
  console.log('set timeout3')
  Promise.resolve().then(() => console.log('inner promise3 resolved'))
}, 0)
setTimeout(() => console.log('set timeout1'), 0)
setTimeout(() => console.log('set timeout2'), 0)
Promise.resolve().then(() => console.log('promise4 resolved'))
Promise.resolve().then(() => {
  console.log('promise5 resolved')
  Promise.resolve().then(() => console.log('inner promise6 resolved'))
})
Promise.resolve().then(() => console.log('promise7 resolved'))
```

> `queueMicrotask`를 사용하여 마이크로 태스크를 스케쥴링 할 수도 있다.

브라우저 (크롬, 파이어폭스, 사파리. IE는 브라우저가 아니므로 제외) + Nodejs 11 이상

```bash
promise1 resolved
promise2 resolved
promise4 resolved
promise5 resolved
promise7 resolved
inner promise6 resolved
set timeout3
inner promise3 resolved
set timeout1
set timeout2
```

nodejs 11 미만

```bash
promise1 resolved
promise2 resolved
promise4 resolved
promise5 resolved
promise7 resolved
inner promise6 resolved
set timeout3
set timeout1
set timeout2
inner promise3 resolved
```

[HTML5 스펙에 정의된 이벤트 루프 가이드라인](https://html.spec.whatwg.org/multipage/webappapis.html#event-loop-processing-model)에 따르면, 이벤트 루프는 매크로 태스큐에서 하나의 매크로 태스크를 처리하기전에 마이크로 태크스에 있는 모든 것을 처리해야 된다. 이 예제에서는, `set timeout3` 콜백이 실행되면, promise 콜백을 예약한다. HTML5의 스펙에 따라서, 타이머 콜백 큐의 다른 콜백을 처리하기전에, 이벤트 루프가 마이크로태스크 큐가 비어있는지 확인해야 한다. 따라서 새로 추가된 promise callback을 실행하고 처리하여야 한다. 이 작업을 처리하면, 비로소 마이크로 태스크 큐가 비어 이벤트 루프가 남은 `setTimeout1` `setTimeout2`을 실행할 수 있게 된다.

그러나 11 버전 이전의 nodejs에서는, 이벤트 루프의 두 사이 단계에서만 마이크로 태스크열을 비우게 된다. 따라서 `inner promise3`은 모든 `setTimeout3`이 실행되기 전까지 실행될 수가 없게 된다.

### 내부 타이머 동작의 차이

타이머 동작은 nodejs, 브라우저 간 뿐만아니라 브라우저 벤더간, 버전마다 다르다. 여기서 가장 주목할만한 두가지는 timeout이 0일때와, timeout이 중첩되어 있을 때다. 이 러한 두가지 동작의 차이를 알기 위해 nodejs v10.19.0, v11.0.0, chrome, firefox, safari에서 아래의 코드를 실행해보자. 이 코드는 timeout이 0 인 중첩타이머 8개를 스케쥴링하고, 각 콜백이 스케쥴링 된이후 실행되기까지의 걸린 시간을 계산한다.
