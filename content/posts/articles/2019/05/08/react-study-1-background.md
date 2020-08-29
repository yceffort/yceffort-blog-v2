---
title: React 공부하기 1 - background
date: 2019-05-08 07:46:42
published: true
tags:
  - react
  - javascript
description: "## 리액트 요약 기존에 많은 자바스크립트 기반 플게임워크들이 있었는데, 대부분의 프레임워크들은 MVC
  (Model-View-Controller), MVVM(Model-View-View Model), MVW(Model-View-Whatever)
  아키텍쳐를 사용하여 개발되었다.  ![MVC](https://mdn.mozillademos.org/files/1..."
category: react
slug: /2019/05/08/react-study-1-background/
template: post
---
## 리액트 요약

기존에 많은 자바스크립트 기반 플게임워크들이 있었는데, 대부분의 프레임워크들은 MVC (Model-View-Controller), MVVM(Model-View-View Model), MVW(Model-View-Whatever) 아키텍쳐를 사용하여 개발되었다.

![MVC](https://mdn.mozillademos.org/files/16042/model-view-controller-light-blue.png)

그러나 이 작업은 애플리케이션 규모가 커지면 복잡해지고, 제대로 관리하지 않으면 성능이 떨어진다는 문제점을 지니고 있다.

이렇게 부분적으로 찾아서 업데이트 하는 대신에, 데이터가 변할 때마다 기존 뷰를 날려 버리고 처음부터 새로 렌더링을 하면 어떨까? 이렇게 하면 애플리케이션 구조가 간단해지고, 작성할 코드의 양도 적어지며, 변화가 있으면 기존의 뷰를 날려버리고 그냥 다시 만들어 버리면 된다. 그러가 이 방식대로 하면 웹브라우저에 CPU 점유율도 늘어날 것이고, 메모리도 많이 사용 될 것이다. 그리고 사용자도 잠깐이지만 뷰가 날라가는 모습을 볼 수 있을지도 모른다.

리액트는 오직 V, 즉 View만 신경 쓰는 라이브러리다. 리액트에서 중요한 두가지는 `초기렌더링`과 `리렌더링`이다. 즁요한 것은 react자체는 view만을 담당하는 라이브러리 라는 것이다.

### 초기 렌더링

```js
render () {...}
```

이 함수는 컴포넌트가 어떻게 생겼는지 정의 한다. 여기에는 html 구문이 아니라, 뷰가 어떻게 생겼고 무슨 정보를 지니고 있는지에 대한 내용을 가지고 있다.  이 함수를 실행하면, 내부 컴포넌트들도 재귀적으로 실행해서,html 마크업을 만들고, 실제 페이지의 dom 요소에 주입하게 된다.

### Reconcilation

앞서 말한 뷰를 새로 갈아끼는 과정을 Reconcilation이라고 한다. 이 작업도 render함수에서 진행된다. 이 작업은 render가 반환한 결과를 바로 적용하는 것이 아니라, 이전에 만들었던 컴포넌트 정보와 비교를 먼저한다. 이렇게 비교한 후, 둘의 차이를 알아내 최소한의 연산으로 dom 트리를 바꿔 치는 것이다.

### Virtual Dom

DOM (Document Object Model)은 동적 UI에 최적화되어 있지 못하다. HTML은 정적인 문서이지만, Javascript를 통해 동적으로 만드는 것이다. 흔히들 DOM은 느리다고 하지만, 이는 정확한 표현이 아니다. 정확히는 `DOM에서 변화가 일어났을 때, 브라우저가 CSS를 연산하고 리페인트 하는 과정이 느린 것`이다. 여기애서 React는 Virtual Dom 방식을 사용하여, DOM 업데이트를 추상화 하여 처리횟수를 최소화 하고 효율적으로 처리한다.

![Virtual Dom](https://i1.wp.com/programmingwithmosh.com/wp-content/uploads/2018/11/lnrn_0201.png?ssl=1)

Virtual Dom을 사용한다고 해서 무조건 빠른 것은 아니다. 작업이 매우 간단한 경우에는 리액트가 없는 편이 나을 수 있다. 어쨌거나, React Virtual Dom이 주는 것은 업데이트 처리의 간결성이다.