---
title: '웹 개발자가 본 사파리 15의 변화와 대응'
tags:
  - javascript
published: true
date: 2021-09-19 17:46:41
description: '죽인다 사파리'
---

## Introduction

사파리 15가 나왔다. 애플을 굉장히 좋아하고, 또 다수의 애플 제품을 보유하고 있는 나로서는 매우 즐거운 일이지만, 이번 safari 15는 나에게 몇가지 이슈를 안겨줬다. 무엇이 달라졌고, 어떻게 대응해야 하는지 살펴보자. 

## 주소 창 위치의 변화

주소 창 위치가 변화하였다. 종전에 위에 있었지만, 이제는 밑에 주소창이 나타난다. 이로 인해 발생하는 문제에 대해서는 후술한다.

## 버튼 기본 색상 및 radius 변경

먼저 버튼의 기본 색상과 raidus가 변경되었다.

![safari14-button](./images/safari14-button.png)

![safari15-button](./images/safari15-button.jpeg)

애플에서 자주 보던 그 파란색이다. 이제 스타일 리셋을 할때 버튼의 색깔까지 클리어 해주어야 한다. 

## 사파리 100vh 문제 해결?

모바일 사파리에서는 기존 버전까지 `100vh`가 의도대로 동작하지 않는 문제가 있었다. 요약하자면 모바일 사파리에서는 스크롤시 주소창이 사라지는데, 이 경우 `100vh`가 뷰포트의 100% 높이가 변경되어 버리는 문제가 있다. 즉, `100vh`라는 값이 정적이지 않다는 뜻이다. 문제를 자세히 살펴보자.

먼저 우리가 아는 `vh` 란 viewport 너비의 1%를 말한다.

그리고 모바일 사파리에서 동작하는 `100vh`는 아마도 아래와 같을 것이라고 추측하고 있다.

> 가장 큰 문제는 모바일 브라우저 (크롬, 사파리)가 주소창이 보여지거나 숨겨져서 view port의 크기가 변경될 수 있다는 것이다. 이러한 브라우저는 view port 높이가 변경될때 현재 가시적인 부분으로 100vh를 수정하는 것이 아니라, 브라우저 주소 표시줄이 숨겨진 상태에서 100vh를 설정해둔다는 것이다. 그 결과, 주소표시줄이 다시 보이게 될 때 화면 하단 부분이 잘려나가서, 100vh의 목적을 위반하게 된다.

> https://chanind.github.io/javascript/2019/09/28/avoid-100vh-on-mobile-web.html

![100vh](https://chanind.github.io/assets/100vh_problem.png) 

### 테스트

아래 테스트 페이지를 살펴보자. 하단에는 버튼이 있고, 이 모든 요소들은 `100vh`로 감싸져 있다.

![safari14-100vh](./images/safari14-100vh.png)

![safari15-100vh](./images/safari15-100vh.jpeg)

오오 해결된 것 같지만...

![safari15-100vh-floating-address-bar](./images/safari15-100vh-floating-address.jpeg)

> 짜잔 사실 해결되지 않았습니다. 

Safari15에서도 `100vh`에는 변화가 없다. 이 쯤 되면 사실상 해결할 생각이 없거나, 혹은 이를 문제라고 보고 있는 것 같지 않다.

이를 해결하기 위해서는 어떻게 해야할까? 

시간을 과거로 돌려, 아이폰 X가 처음나왔을때, 노치에 컨텐츠가 가려지는 문제를 해결하기 위하여 애플이 [`env`와 `safe-area-inset`을 소개했던 것](https://webkit.org/blog/7929/designing-websites-for-iphone-x/)을 떠올려보자.

사파리 14에서는, `safe-area-inset-bottom`의 값이 주소 창에 상관없이 0으로 고정되어 있었다. 그러나 사파리 15에서는 주소창이 활성화 되지 않은 상태에서의 `safe-area-inset-bottom`값은 0 이지만, 주소창이 펼쳐졌을 때는 그 값만큼 제공이 된다.

```css
footer {
    padding-bottom: calc(1em + env(safe-area-inset-bottom));
}
```

## Theme color

탭 모음 배경색은 더이상 흰색 또는 회색으로 고정되어 있지 않고, 현재 페이지의 색 구성표에 맞게 조정된다. 이렇게 하면 화면에 좀더 몰입도를 가져올 수 있다. 기본적으로는 헤더나 바디의 배경색을 사용하여 사파리에서 자동으로 선택되지만, 문서헤더에 메타 태그를 사용하여 설정할 수도 있다. 

![safari14-themecolor](./images/safari14-themecolor.png)

![safari15-themecolor](./images/safari15-themecolor.jpeg)

사파리 14, 15에서 내 블로그의 색이 다르게 나오는 것을 볼 수 있는데, 이는 아래 코드처럼 내가 강제로 색을 설정해두었기 떄문이다.

```html
<meta name="theme-color" content="#00b7ff" />
```

이 말인 즉슨, 다른 DOM 노드 처럼 자바스크립트를 활용하여 사용자가 특정 작업을 사용하거나 특정 페이지를 방문할 때, `theme-color`를 동적으로 업데이트하여 사용자에게 더 큰 몰입감을 줄 수도 있다.

또한 이는 다크테마도 지원한다. 그래서 나는 아래 처럼 변경해보았다.

```html
<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)" />
<meta name="theme-color" content="#121826" media="(prefers-color-scheme: dark)" />
```

![safari15-theme-color-light](./images/safari15-theme-color-light.jpeg)

![safari15-theme-color-light](./images/safari15-theme-color-dark.jpeg)

> https://github.com/whatwg/html/issues/6495

