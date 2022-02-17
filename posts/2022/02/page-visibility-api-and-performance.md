---
title: 'Page Visibility API와 성능과의 관계'
tags:
  - javascript
  - html
published: false
date: 2022-02-17 21:06:23
description: ''
---

## Table of Contents

## Introduction

웹 페이지가 사용자의 백그라운드에서 로드 되는 경우는 얼마나 있을까? 백그라운드에서 페이지가 로드된다면, 어차피 사용자가 보지 않는 상왕이기 때문에 페이지 로딩이 느려도 괜찮지 않을까? 그러나 웹 사이트를 관리하고, 사용자의 환경을 제대로 측정하기 위해서는 현재 사용자의 visibility 상태가 분석중인 데이터에 어떤 영향을 미치는지 이해하는 것이 중요하다.

현재 제공되고 있는 [Page Visibility API](https://w3c.github.io/page-visibility) 와 [akmai](https://www.akamai.com/products/mpulse-real-user-monitoring)에서 제공하는 데이터를 바탕으로 페이지의 visibility와 이 것이 웹 페이이지의 성능에 미치는 영향을 알아보자.

## Page Visibility

Page Visibility API는 현재 콘텐츠의 가시성 뿐만 아니라 시간에 따른 가시성 변화를 측정하는 방법을 정의한다. 개발자는 이 정보를 활용하여 사용자에게 페이지가 표시되고 있는지 여부를 확인할 수 있다. 또한 페이지 로드 시 수행될 작업을 조금더 추려낼 수 있다.

https://caniuse.com/pagevisibility

이 api를 사용하는 방법은 간단하다. `document.visibilityState`를 사용하면 현재 페이지가 보여지고 있는지 아닌지를 알 수 있다. 만약 브라우저의 탭이 전환되면서 이 값을 알고 싶다면, 이 값을 모니터링 하는 것 또한 가능하다.

```javascript
console.log(document.visibilityState + ': ' + Date())
document.onvisibilitychange = () =>
  console.log(document.visibilityState + ': ' + Date())
```

w3의 스펙 문서에서는, 현재 페이지의 가시성에 따라서 페이지 로드시 비디오를 자동으로 시작될 수 있도록 이 상태 변화를 볼 수 있는 listener를 제공한다.

```javascript
const videoElement = document.getElementById('videoElement')

// Autoplay the video if application is visible
if (document.visibilityState === 'visible') {
  videoElement.play()
}

// Handle page visibility change events
function visibilityListener() {
  switch (document.visibilityState) {
    case 'hidden':
      videoElement.pause()
      break
    case 'visible':
      videoElement.play()
      break
  }
}

document.addEventListener('visibilitychange', visibilityListener)
```

https://www.w3.org/TR/page-visibility/#examples-of-usage

## 가시성 상태

사용자가 링크를 마우스 오른쪽 단추로 클릭해서 백그라운 탭에서 페이지를 로드하는 빈도가 얼마나 될까? 아니면 모바일 디바이스에서 링크를 클릭하고, 페이지가 채 보이기도 전에 애플리케이션을 전환해버리는 경우는 얼마나 있을까? 또는 모바일 페이지가 보이기도 전에 화면을 잠그는 경우는?

아래 그래프는 장치 유형별로 visibility 상태를 분류해서 보여준다. 전체 테스크톱 페이지 뷰의 11.18%가 보이지 않는 상태로 로딩되어 있다. 마찬가지로, 모바일의 경우에는 9.59%도 보이지 않는 상태로 로드 되었다.

![distribution of visibility states](https://calendar.perfplanet.com/images/2021/paul/image8.jpg)

## 이 가시성이 중요한 이유

대부분의 모던 브라우저는 현재 메인으로 보이고 있는 (foreground)에 대한 작업에 우선순위를 두기 때문에, 백그라운드 탭에서 로드되는 페이지의 경우에는 느릴 수가 있다.

아래 데이터를 확인하면 이러한 visibility의 차이에 따라 성능차이가 크게 나타나는 것을 알 수 있다.

![Median Load time by visibility state](https://calendar.perfplanet.com/images/2021/paul/image1.jpg)

위 데이터를 보면, 확실히 현재 visibility가 없는 페이지의 로딩시간이 느린 것을 알 수 있다.

그런데, 이게 정말로 중요한 데이터인 것일까?

![Desktop Load Time Percentiles by Visibility State](https://calendar.perfplanet.com/images/2021/paul/image2.jpg)

사용자 경험에 비추어보았을때, 페이지가 표시
