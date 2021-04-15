---
title: Intersection Observer
date: 2019-06-24 06:01:35
published: true
tags:
  - javascript
description:
  '## Intersection Observer Intersection Observer는 엘리먼트가 viewport에
  노출되고 있는지 여부를 확인해주는 API다. 간단히 말해 브라우저의 어떤 요소가 화면에 노출되고 있는지 안되고 있는지를 확인해주는
  라이브러리라고 생각하면 될 것 같다. 이 라이브러리가 없이 엘리먼트가 노출중인지 확인하려면 어떻게 해야할까? 이...'
category: javascript
slug: /2019/06/24/intersection-observer/
template: post
---

## Intersection Observer

Intersection Observer는 엘리먼트가 viewport에 노출되고 있는지 여부를 확인해주는 API다. 간단히 말해 브라우저의 어떤 요소가 화면에 노출되고 있는지 안되고 있는지를 확인해주는 라이브러리라고 생각하면 될 것 같다. 이 라이브러리가 없이 엘리먼트가 노출중인지 확인하려면 어떻게 해야할까? 이전까지 주로 사용되던 API는 [getBoundingClientRect](https://developer.mozilla.org/ko/docs/Web/API/Element/getBoundingClientRect)다. 이 메서드는 해당 엘리먼트의 크기와 viewport에서의 상대적인 위치를 알려준다.

```javascript
function isInViewport(element) {
  // viewport의 height, width
  const viewportHeight = document.documentElement.clientHeight
  const viewportWidth = document.documentElement.clientWidth
  // 엘리먼트의 rect
  const rect = element.getBoundingClientRect()

  if (!rect.width || !rect.height) {
    return false
  }

  var top = rect.top >= 0 && rect.top < viewportHeight
  var bottom = rect.bottom >= 0 && rect.bottom < viewportHeight
  var left = rect.left >= 0 && rect.left < viewportWidth
  var right = rect.right >= 0 && rect.right < viewportWidth

  return (top || bottom) && (left || right)
}
```

물론 이 함수로도 충분히 확인이 가능하다. 그러나 문제는 이 함수를 시도 때도 없이 불러야 할것이다. 문서가 처음로딩 되었을때, 스크롤 이벤트가 발생했을때, 브라우저 크키가 변경되었을때, (모바일의 경우) 화면이 회전 됐을 때 등 고려해야할 사황이 너무나도 많다. 그리고 결정적으로 `getBoudingClientRect()`는 호출이 될 때 마다 레이아웃을 다시 그리기 때문에 성능에 매우 부담이 간다. [참고](https://gist.github.com/paulirish/5d52fb081b3570c81e3a)

## 사용법

```javascript
var options = {
  root: document.querySelector('#scrollArea'),
  rootMargin: '0px',
  threshold: 1.0,
}

var observer = new IntersectionObserver(callback, options)
```

`root`: 엘리먼트 노출을 감시할 viewport 영역이다. null이 기본값이고, null이라면 브라우저 전체를 감시하게 된다.

`rootMargin`: css의 margin property와 사용법이 같다. root에 margin 값을 주는 것이다.

`threshold`: number, 또는 array of number가 가능하다. 여기서 숫자는 viewport의 n%가 되었을 때 이벤트를 호출할 것인지 결정하는 것이다. array라면 해당 숫자만큼 노출될때 마다 이벤트가 발생하게 된다. 기본값은 0 으로, 단 1픽셀이라도 노출될 경우 이벤트가 발생된다.

`callback`: 타겟이 노출될때 실행되는 콜백함수다. 여기서는 두개의 값을 반환한다.

`entries`: [IntersectionObserverEntry](https://developer.mozilla.org/en-US/docs/Web/API/IntersectionObserverEntry)의 array이며, 각각 얼마나 노출되었는지 값이 나온다.

`observer`: callback을 호출한 `IntersectionObserver`

```javascript
var target = document.querySelector('#listItem')
observer.observe(target)
```

https://codepen.io/yceffort/pen/RzVyML

콘솔창을 보면, 50% 이상 나온 엘리먼트가 찍히는 것을 볼 수 있다.

## 활용해보기

### 무한스크롤

https://codepen.io/yceffort/pen/VJbxNM

문서의 맨 마지막에 height가 10px인 `#watch_end_of_document`를 넣어서, 이 엘리먼트가 브라우저에 노출되게 되면 스크롤의 마지막에 온것으로 간주하고 그 때마다 새로운 아이템들을 로딩하도록 명령을 내렸다.

### 이미지 레이지 로딩

https://codepen.io/yceffort/pen/YoVvya

콘솔 - 네트워크 창으로 가보면 이미지가 동적으로 로딩되는 것을 알 수 있다. 각각의 엘리먼트들에 observer를 걸어두고, viewport에 걸칠때 마다 backgroundImage를 줘서 이미지가 로딩되도록 하였다.
