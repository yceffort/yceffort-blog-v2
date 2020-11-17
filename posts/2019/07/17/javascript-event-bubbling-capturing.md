---
title: javascript event bubbling & capturing
date: 2019-07-17 07:22:22
published: true
tags:
  - javascript
description: "![image](https://miro.medium.com/max/1200/1*Et5UjVPGLfF1L43T7Errx\
  Q.png) ## Javascript Event
  Capturing  https://codepen.io/yceffort/pen/GbVaaY  Event Capturing은 특정 요소에서
  이벤트가 발생했을 때, 최상위 요소에서 부터 이벤트를 탐..."
category: javascript
slug: /2019/07/17/javascript-event-bubbling-capturing/
template: post
---
![image](https://miro.medium.com/max/1200/1*Et5UjVPGLfF1L43T7ErrxQ.png)

## Javascript Event Capturing

https://codepen.io/yceffort/pen/GbVaaY

Event Capturing은 특정 요소에서 이벤트가 발생했을 때, 최상위 요소에서 부터 이벤트를 탐색하여 특정요소까지 찾아오는 이벤트 전파 방식을 의미한다. 위 예시에서 가장 내부의 element를 클릭했을 때, 최상위 요소 부터 해당 `click`이벤트를 전파시켜 이벤트가 실행되는 것을 볼 수 있다.

## Javascript Event Bubbling

https://codepen.io/yceffort/pen/Prrgob

반대로 Event Bubbling은 특정요소에서 이벤트가 발생했을 때, 해당 요소에서 부터 이벤트를 전파시키는 것을 의미한다. 위 예시에서 가장 내부의 `three`를 클릭했을때, `three`, `two`, `one`으로 이벤트가 전파되는 것을 볼 수 있다.

## stopPropagation

`stopPropagation`는 이벤트의 전파 (Event Bubbling 및 Capturing)을 막는 것을 의미한다. 즉, 현재까지의 이벤트만 실행하고 이후의 이벤트를 막게된다.

## preventDefault

`preventDefault`는 해당 DOM에서 내가 원하는 이벤트만 실행하고, 기본적인 (취소할 수 있는) 이벤트를 취소하는 것을 의미한다.

https://codepen.io/yceffort/pen/xovoRp

해다아 예제에서는 `preventDefault`를 이용해서 a tag의 기본 이벤트인 `href`를 막는 것을 볼 수 있다.
