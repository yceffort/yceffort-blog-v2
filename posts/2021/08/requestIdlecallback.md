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

사이트와 애플리케이션에는 실행해야할 스크립트가 잔뜩 쌓여있다. 이러한 자바스크립트가 최대한 빨리 실행되야 하는 것이 좋지만, 그와 동시에 사용자의 방해가 되지 않도록 해야 한다.사요앚가 페이지를 스크롤 할 때 데이터를 보내거나, DOM에 element를 추가해야 하는 경우 웹 애플리케이션이 응답하지 않아 사용자 경험이 저하될 수 있다.

이를 해결하기 위해 [requestIdleCallback](https://developer.mozilla.org/ko/docs/Web/API/Window/requestIdleCallback)이라는 API가 있다. `requestAnimationFrame`을 사용하면 애니메이션을 적절하게 스케쥴링하고, 60fps를 달성하는데 도움을 줄 수 있는 것 처럼, `requestIdleCallback`은 프레임이 끝나는 지점에 있거나, 사용자가 비활성화 상태일 때 작업을 예약할 수 있다.

- https://developer.mozilla.org/ko/docs/Web/API/Window/requestIdleCallback
- https://caniuse.com/requestidlecallback
- https://w3c.github.io/requestidlecallback/
- https://github.com/pladaria/requestidlecallback-polyfill
- https://github.com/aFarkas/requestIdleCallback

## 왜 `requestIdleCallback`인가
