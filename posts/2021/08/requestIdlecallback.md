---
title: 'requestIdleCallback으로 최적화하기'
tags:
  - javascript
  - nodejs
  - browser
published: true
date: 2021-08-15 17:24:10
description: '내 인생은 언제 idle 할 것인가'
---

사이트와 애플리케이션에는 실행해야할 스크립트가 잔뜩 쌓여있다. 이러한 자바스크립트가 최대한 빨리 실행되야 하는 것이 좋지만, 그와 동시에 사용자의 방해가 되지 않도록 해야 한다. 사용자가 페이지를 스크롤 할 때 데이터를 보내거나, DOM에 element를 추가해야 하는 경우 웹 애플리케이션이 응답하지 않아 사용자 경험이 저하될 수 있다.

이를 해결하기 위해 [requestIdleCallback](https://developer.mozilla.org/ko/docs/Web/API/Window/requestIdleCallback)이라는 API가 있다. `requestAnimationFrame`을 사용하면 애니메이션을 적절하게 스케쥴링하고, 60fps를 달성하는데 도움을 줄 수 있는 것 처럼, `requestIdleCallback`은 프레임이 끝나는 지점에 있거나, 사용자가 비활성화 상태일 때 작업을 예약할 수 있다.

- https://developer.mozilla.org/ko/docs/Web/API/Window/requestIdleCallback
- https://w3c.github.io/requestidlecallback/
- https://github.com/pladaria/requestidlecallback-polyfill

## 왜 `requestIdleCallback`인가

필수적이지 않은 작업을 스케쥴링해서 처리하는 것은 매우 어렵다. `requestAnimationFrame` 콜백을 실행한 후 스타일 연산, 레이아웃, 페인팅 및 기타 브라우저 내부에서 실행해야하는 작업을 수행하기 떄문에, 현재 남은 프레임 시간을 정확히 파악하는 것은 어렵다. 개발자가 여기에서 해볼 수 있는 시도는 많지 않다. 사용자가 어떤 방식으로든 인터랙션을 하지 못하게 하려면, 사용자가 할 수 있는 모든 종류의 인터랙션 (스크롤, 터치, 클릭 등)에 listener를 달아야 한다. 반면 브라우저는 프레임 작업이 끝난 이후에 얼마나 여유가 있는지, 그리고 사용자가 인터랙션 중인지 알 고 있기 때문에 `requestIdleCallback`을 사용해 가능한 효율적으로 이 빈 시간을 활용할 수 있는 api를 쓸 수 잇다.

## `requestIdleCallback`

- https://caniuse.com/requestidlecallback

IE에서는 사용이 불가능하고, safari에서는 (여전히) 실험적 기능으로 제공되고 있다.

polyfill

- https://github.com/pladaria/requestidlecallback-polyfill/blob/master/index.js
  - timeout을 이용해서 적용
- https://github.com/aFarkas/requestIdleCallback
  - 말그대로 사용자가 할 수 있는 모든 이벤트에 리스너를 달아둬서 해결

## `requestIdleCallback` 사용해보기

`requestIdleCallback`은 [requestAnimationFrame](https://developer.mozilla.org/ko/docs/Web/API/Window/requestAnimationFrame)과 매우 비슷하다.

```javascript
requestIdleCallback(myNonEssentialWork)
```

`myNonEssentialWork`가 호출되면, 이 작업의 남은시간을 나타내는 함수가 포함된 [deadline](https://developer.mozilla.org/ko/docs/Web/API/IdleDeadline)객체를 넘겨받는다.

```javascript
function myNonEssentialWork(deadline) {
  while (deadline.timeRemaining() > 0) doWorkIfNeeded()
}
```

`timeRemaining`함수를 호출하여 현재 최신 값을 가져올 수도 있다. `timeRemaining`의 값이 0 이면서, 다음 작업이 또 있는 경우에는 `requestIdleCallback`으로 다음 작업을 또 예약할 수도 있다.

```javascript
function myNonEssentialWork(deadline) {
  while (deadline.timeRemaining() > 0 && tasks.length > 0) doWorkIfNeeded()

  if (tasks.length > 0) requestIdleCallback(myNonEssentialWork)
}
```

## 함수 호출을 보장받는 방법

만약 작업이 정말 정말 바쁘면 어떻게 될까? 콜백이 실행되지 않을지 걱정될 수도 있다. `requestIdleCallback`은 `requestAnimationFrame`와 다르게 두번째 인수가 존재한다. 이 인수에서는, timeout을 넘길 수 있는데, 이 설정된 시간이 초과된 경우 idle 상태와 상관없이 그냥 실행해버린다.

```javascript
// 2초는 내가 기다려본다...
requestIdleCallback(processPendingAnalyticsEvents, { timeout: 2000 })
```

이렇게 시간 초과로 인해 콜백이 실행되는 경우 아래 두가지를 확인할 수 있다.

- `timeRemaining()`은 0을 반환
- `didTimeout`이 true가 됨

```javascript
function myNonEssentialWork(deadline) {
  while (
    (deadline.timeRemaining() > 0 || deadline.didTimeout) &&
    tasks.length > 0
  )
    doWorkIfNeeded()

  if (tasks.length > 0) requestIdleCallback(myNonEssentialWork)
}
```

이 timeout으로 인해 사용자 작업이 중단될 수도 있으므로 (작업으로 인해 애플리케이션이 응답하지 않거나 오류가 나거나), 이 인수를 사용할 때는 주의해야 한다.

## 데이터 분석을 위해 `requestIdleCallback`사용하기

`requestIdleCallback`를 사용하는 예제를 살펴보자. 이 경우 메뉴를 클릭하는 것과 같은 이벤트를 추적할 수 있다 그러나 일반적으로 메뉴를 클릭하면 화면에 애니메이션이 함께 표시되므로, google analytics에 이 이벤트를 즉시 보내지 않도록 설정해보자.

```javascript
var eventsToSend = []

function onNavOpenClick() {
  // 메뉴를 여는 이벤트
  menu.classList.add('open')

  // 보낼 이벤트를 저장해둔다.
  eventsToSend.push({
    category: 'button',
    action: 'click',
    label: 'nav',
    value: 'open',
  })

  schedulePendingEvents()
}
```

`requestIdleCallback`를 활용하여 이 이벤트를 실행해보자.

```javascript
function schedulePendingEvents() {
  // isRequestIdleCallbackScheduled 가 있으면 예약하지 않는다.
  if (isRequestIdleCallbackScheduled) return

  // 없으면 작업시작 준비
  isRequestIdleCallbackScheduled = true

  if ('requestIdleCallback' in window) {
    // 최대 2초 대기
    requestIdleCallback(processPendingAnalyticsEvents, { timeout: 2000 })
  } else {
    processPendingAnalyticsEvents()
  }
}
```

이 예제에서는 2초로 설정했지만, 애플리케이션에 따라 이 값이 달라질 수 있다.데이터 분석의 경우, 데이터를 미래의 특정 시점에 리포트 하는 것이 아니라 적절한 시간에 리포팅 해야 한다.

```javascript
function processPendingAnalyticsEvents(deadline) {
  // false 상태로 만들어 다음 작업도 받게함
  isRequestIdleCallbackScheduled = false

  // deadline이 없다면, 바로 실행
  if (typeof deadline === 'undefined')
    deadline = {
      timeRemaining: function () {
        return Number.MAX_VALUE
      },
    }

  // 작업이 남아있고, 여유가 있는 경우 실행
  while (deadline.timeRemaining() > 0 && eventsToSend.length > 0) {
    var evt = eventsToSend.pop()

    ga('send', 'event', evt.category, evt.action, evt.label, evt.value)
  }

  // 해야할 작업이 있따면 다시 예약
  if (eventsToSend.length > 0) schedulePendingEvents()
}
```

이 예제에서는, `requestIdleCallback`가 없으면 바로 전송하도록 해두었다. 그러나 프로덕션 애플리케이션에서는 사용자의 상호작용과 충돌하지 않고 에러가 발송하지 않도록 timeout으로 지연해서 전송하는 것이 좋다.

## `requestIdleCallback`으로 DOM 조작하기

`requestIdleCallback`이 성능에 도움이 될 수 있는 또다른 상황은, 필수적이지 않은 DOM을 변경해야 하는 경우가 있다. 예를 들어, 지속적으로 children 하단에 붙어서 로딩되는 DOM과 같은 것을 들 수 있다.

![](https://developers.google.com/web/updates/images/2015-08-27-using-requestidlecallback/frame.jpg)

먼저, 브라우저가 지속적으로 사용중이어서, 작업을 할 수 있는 여유시간이 없는 경우도 가정해야 한다. 이 경우, 프레임별로 `setImmediate`를 실행해야 한다.

프레임이 끝나는 지점에서 콜백이 실행되면, 현재 프레임이 커밋된 이후에 실행할 수 있도록 스케쥴링 될 것이다. 즉, 스타일 변경사항이 적용되고, 레이아웃이 다시 계산될 것이다. idle callback내에서 DOM을 조작하려면, 레이아웃 계산이 취소될 수 있다. 다음 프레임에서 `getBoundingClientRect`이나 `clientWidth`와 같이 현재 레이아웃을 읽어오는 메소드가 있는 경우, [강제 동기식 레이아웃](https://developers.google.com/web/fundamentals/performance/rendering/avoid-large-complex-layouts-and-layout-thrashing#avoid-forced-synchronous-layouts)을 수행해야 하는데 이 경우 브라우저에서 성능 저하가 일어날 수 있다.

idle callback에서 DOM 조작을 트리거하지 않는 또다른 이유는, DOM 에 걸리는 시간을 예측할 수 없기 때문에, 브라우저에서 제공한 deadline을 쉽게 넘길 수 있기 때문이다.

따라서 가장 좋은 방법은 브라우저가 스스로 스케쥴링할 수 있는 `requestAnimationFrame` 콜백 내부에서 DOM 조작을 하는 것이다. 하나 주의해야할 것은, 만약 가상돔 라이브러리를 사용한다면 `requestIdleCallback`를 사용할 경우 실제 DOM 조작은 idle callback이 아닌 다음 `requestAnimationFrame` 에서 일어날 수도 있다는 것이다.

```javascript
function processPendingElements(deadline) {
  // deadline이 없으면, 바로 실행
  if (typeof deadline === 'undefined')
    deadline = {
      timeRemaining: function () {
        return Number.MAX_VALUE
      },
    }

  if (!documentFragment) documentFragment = document.createDocumentFragment()

  // 작업에 여유가 있고, 작업이 있으면 바로 실행
  while (deadline.timeRemaining() > 0 && elementsToAdd.length > 0) {
    var elToAdd = elementsToAdd.pop()
    var el = document.createElement(elToAdd.tag)
    el.textContent = elToAdd.content

    documentFragment.appendChild(el)

    // 바로 실행하는 것이 아니고, 다음 requestAnimationFrame 까지 대기
    scheduleVisualUpdateIfNeeded()
  }

  if (elementsToAdd.length > 0) scheduleElementCreation()
}
```

```javascript
function scheduleVisualUpdateIfNeeded() {
  if (isVisualUpdateScheduled) return

  isVisualUpdateScheduled = true

  requestAnimationFrame(appendDocumentFragment)
}

function appendDocumentFragment() {
  // Append the fragment and reset.
  document.body.appendChild(documentFragment)
  documentFragment = null
}
```
