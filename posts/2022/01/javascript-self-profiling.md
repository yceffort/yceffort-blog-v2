---
title: '웹 애플리케이션에서 자바스크립트 프로파일링 해보기'
tags:
  - javascript
  - chrome
  - browser
published: true
date: 2022-01-20 21:54:46
description: '해봤지만 해보지 않았습니다'
---

자바스크립트를 프로파일링 할 수 있는 api가 있다. https://wicg.github.io/js-self-profiling/ 이 api를 활용하면 실제 고객의 디바이스에서 자바스크립트 웹 애플리케이션의 성능 프로파일을 가져올 수 있다. 즉, 브라우저 개발자 도구에서 로컬 머신 (컴퓨터)로 애플리케이션을 프로파일링 하는 수준 이상을 해볼 수 있다. 애플리케이션을 프로파일링하는 것을 성능을 파악할 수 있는 좋은 방법이다. 프로파일을 활용해서 시간이 지남에 따라 실행되는 항목 (스택)을 확인하고 코드에서 성능에 문제가 되는 핫스팟을 식별할 수 있도록 도와준다.

브라우저에서 개발자 도구를 사용해 봤따면, 자바스크립트 프로파일리에 익숙할 수 있다. 예를 들어, 크롬 브라우저의 개발자도구에서 성능탭을 보면 프로파일을 기록할 수 있다. 이 프로파일은 시간이 지남에 따라 애플리케이션에서 실행 중인 내용을 보여준다.

![performance-example](./images/performance-example.png)

> 내가 만든거 아님

이 api는 크롬에서 여전히 사용할 수 있는 자바스크립트 프로파일러 탭을 상기시켜준다.

![javascript-profiler-example](./images/javascript-profiler-example.png)

이 js self profiling api는 새로운 api로, 크롬 94+ 버전에서만 사용 가능하다. 자바스크립트에서 방문자를 위해 사용할 수 있는 샘플링 프로파일러를 제공한다.

## Sample Profiling이란 무엇인가