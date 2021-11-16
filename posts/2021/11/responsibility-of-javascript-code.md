---
title: '자바스크립트 코드가 가져야할 책임감'
tags:
  - javascript
  - web
published: true
date: 2021-11-16 19:21:28
description: '책임감 있는 코드를 작성하고 있는지 항상 뒤돌아보기'
---

## Table of Contents

## Introduction

가끔씩 심심할 때 마다 들어가는 사이트가 있는데, 그 중 하나가 [State of Javascript](https://httparchive.org/reports/state-of-javascript?start=2021_01_01&end=latest&view=list)다. 이 레포트가 어떻게 생성되는지는 모르겠지만, 아무튼 전세계 사이트의 자바스크립트 관련 통계를 보여준다. 보여주는 내용은 크게 3가지로 볼 수 있다.

- javascript bytes: 사이트 접근시 내려받는 자바스크립트 크기의 평균
- javascript requests: 사이트 접근시 수행되는 평균 요청 갯수
- javascript boot-up time: 페이지당 스크립트가 소비하는 cpu 시간

2021년이 되면서 이제 평균 자바스립트 크기는 450kb, 500kb를 넘어섰다. 하지만 우리가 알아둬야 할 것은 단지 전송된 크기라는 점, 대부분의 전송에는 압축이 수반되기 때문에 실제 압축해제한 크기는 더 크다는 것이다. 물론, 리소스를 전송하는데 시간을 단축하는 것도 유의미한 일이지만, 클라이언트는 실제 다운로드한 450kb 정도의 리소스를 압축해제 후 처리해야 하기 때문에, 실제 처리해야하는 자바스크립트 코드의 크기는 아마도 1mb를 넘을 것이다. 1mb가 크다면 크고, 작다면 작은 크기 이겠지만, 이를 잘 처리하는지는 처리하는 기기에 달려 있을 것이다. 이를 연구하기 위한 많은 노력이 이뤄지고 있지만, 아무튼 간에 처리하는데 소요되는 시간은 장치마다 크게 다를 것이다.

긍정적으로 생각하자면, 고객이 웹을 탐색하는 기기와 네트워크 환경은 점차 개선되고 있다. 그러나 한편으로는 우리는 그런 이득을 상쇄할 만큼 복잡한 사이트를 만들고 있다. 따라서 우리는 책임감 있게 자바스크립트 코드를 작성해야 한다. 그리고 이 책임감은, 우리가 어떻게 코드를 작성하고 있는지 부터 이해해야 한다.

## 웹 앱, 그리고 웹 사이트

우리는 일반적으로 '웹 사이트'와 '웹 앱' 이라는 용어를 혼용해서 쓰고 있다. 그러나 이를 혼동하는 것은 안좋은 결과를 가져올 수 있다. 비즈니스용 웹 사이트를 만드는 경우, 강력한 프레임워크에 의존하여 DOM 변경사항을 관리하거나, 클라이언트 사이드 라우팅 등을 만들 가능성이 적다. 작업에 적합하지 않은 도구를 사용하는 것은 만드는 사람들에는 생산성을 떨어뜨리고, 사이트를 사용하는 사람들에게도 앞선 관점에서 피해를 끼칠 수 있다.

하지만 '웹 앱'을 만들 때를 생각해보자. 수백에서 수천개의 dependencies를 가지는 패키지를 아주 자연스럽게 설치하는데, 사실 우리는 이 패키지가 안전한지 100% 확신하지 않고 설치한다. 모듈 번들을 위한 구성도 매우 복잡하다. 어떻게 보면 이러한 복잡성이 흔하게 몰아치는 개발 환경 속에서, 빠르게 프로그램을 만들고 접근이 쉽도록 하기 위해서는 프론트엔드 전반에 걸쳐 넓은 지식이 필요하고, 경계심 또한 늦추지 말아야 한다. 만약 의존성이 의심스럽다면 `npm ls --prod`를 실행하여 모든 의존성에 대해 내가 인지하고 있는지 확인해야 한다. 물론 이렇게 한다고 해서 모든 써드파티 스크립트를 다 고려할 수 있는 것은 아니다.

아무튼, 우리가 잊어버리는 것은 웹 사이트와 이 웹 앱이 모두 하나의 환경, '웹'이라고 하는 생태계에서 건설되고 있다는 것이다. 둘 모두 네트워크의, 기기의 영향을 받고 있다. 그러나 많은 사람들이 '웹 앱'을 만든다고 결심하기 시작하면서 부터 이러한 환경을 잊어도 되는 것처럼 생각한다. '앱'이라고 부르기로 결정했다고 해서 이러한 제약이 사라지는 것도 아니고, 갑자기 사용자의 기기가 마법과도 같은 힘을 발휘하는 것도 아니다.

우리가 만든 것을 누가 사용하는지 이해하고, 이들이 인터넷을 접속하는 조건이 우리와 다르다는 것을 받아드리는 것 부터 이러한 책임이 시작되는 것이다. 우리가 달성하고자 하는 목적을 알아야 하며, 그리고 이러한 목적을 달성하라 수 있는 무언가를 만들어야 한다.

결론적으로, 자바스크립트의 의존성과 자바스크립트의 활용 (HTML, CSS를 제외하더라도) 이 성능과 접근성을 해치는 패턴을 사용하는 것을 항상 경계해야 한다.

## 프레임워크로 인해 지속 불가능한 패턴을 만드는 것을 경계할 것

아래 리액트 코드를 살펴보자.

```javascript
import React, { Component } from "react";
import { validateEmail } from "helpers/validation";

class SignUpForm extends Component {
  constructor (props) {
    super(props);

    this.handleSubmit = this.handleSubmit.bind(this);
    this.updateEmail = this.updateEmail.bind(this);
    this.state.email = "";
  }

  updateEmail (event) {
    this.setState({
      email: event.target.value
    });
  }

  handleSubmit () {
    if (validateEmail(this.state.email)) {
      // ... sign up...
    }
  }

  render () {
    return (
      <div>
        <span>Enter your email:</span> <input type="text">

        <button onClick={handleSubmit}>Sign Up</button>
      </div>
    );
  }
}
```

위 코드엔 몇가지 문제가 있다.

1. `<form>`을 사용하지 않는 다면 그것은 form이라고 부를 수 없다. `<div role="form">`이라는 기법도 있지만, form을 작성하기 위해서는 적절한 동작과 메서드를 가진 `<form/>`을 쓰는 것이 좋다. `<form/>`의 `action`은 해당 컴포넌트가 서버사이드에서 렌더링된 경우라도, 자바스크립트 없이도 해당 작업을 수행할 수 있도록 보장할 수 있기 때문이다.
2. `<label>`이 없다면 접근성 이점을 누릴 수 없다.
3. form을 제출하기전에 클라이언트에서 무언가를 하기 원한다면, `<button/>` 의 `onClick`이 아닌 `<form/>`의 `onSubmit`을 활용해야 한다.
4. 이메일 유효성 검사에 특별한 기능이 필요한게 아니라면, IE 10 부터 광범위하게 지원하는 HTML5의 폼 validation의 활용하는 것이 여러모로 좋다. `<input type="email">`은 [많은 브라우저에서 지원하고 있으므로](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/email), `required` 속성과 함꼐 활용한다면 된다. 다만 스크린리더를 고려한다면 [몇가지 주의사항](https://www.tpgi.com/required-attribute-requirements/)이 있다.
5. 이 컴포넌트는 라이프 사이클 메소드에 의존적이지 않다. 따라서 stateless한 컴포넌트로 리팩토링할 수 있다. 이는 일반적인 리액트 컴포넌트보다 훨씬더 적은 자바스크립트를 사용한다.

따라서 이를 리팩토링한다면,

```javascript
import React from 'react'

const SignupForm = (props) => {
  const handleSubmit = (event) => {
    // 비동기로 폼 이벤트를 발생시키기 위해서 필요하다.
    // 자바스크립트가 disable된 서버사이드 렌더링 환경에서도 유효 할 것이다.
    event.preventDefault()

    // Do Something...
  }

  return (
    <form method="POST" action="/signup" onSubmit={handleSubmit}>
      <label for="email" class="email-label">
        Enter your email:
      </label>
      <input type="email" id="email" required />
      <button>Sign Up</button>
    </form>
  )
}
```

이제 이 컴포넌트는 접근성도 향상되었고, 자바스크립트도 덜 사용하게 되었다. 자바스크립트 범벅인 웹 세상에서, 자바스크립트를 줄이는 것은 거의 대부분 옳다. 브라우저는 우리에게 많은 기능을 공짜로 제공하고 있으므로, 항상 이를 최대한 활용할 수 있어야 한다.

이와 같은 예제는, 프레임워크에 의존적일 때 접근성이 떨어지는 패턴이 생긴다는 것을 의미하는 것 뿐만 아니라 HTML과 CSS를 제대로 이해하지 못하는 이해의 격차가 있다는 것을 방증한다. 자바스크립트, 그리고 HTML과 CSS 사이의 지식의 격차는 우리가 인지하지 못하는 실수를 초래하곤한다. 프레임워크는 생산성을 높이는 도구가 될 수도 있지만, 그것보다 더 중요한 것은 핵심적인 웹 기술에 대해서 이해하고, 나아가 무슨 툴을 사용하던지 사용자에게 좋은 사용자 경험을 안겨주는 것이다.

## 웹 플랫폼에 의존하기

angular, vue, react와 같은 웹 프레임워크에 많은 시간을 쏟고 있지만, 웹 플랫폼 또한 그자체 만으로도 어마어마한 프레임워크라 할 수 있다. 이전 섹션에서 알아 본 것처럼, 이미 확립된 마크업 패턴과 브라우저의 기능에 의존하는 것이 더욱 효과적이다. 

### 싱글 페이지 애플리케이션

개발자들이 가장 많이 하는 실수 중 하나는 별다른 고민없이 Single Page Application을 채택하는 것이다. SPA가 물론, 클라이언트 라우팅을 통해 성능적 이점을 누릴수도 있다. 하지만 반대로 잃는 것은 무엇인가? 브라우저의 내비게이션은 비록 동기적으로 작동하지만 많은 이점을 제공한다. 이러한 방문이력은 [복잡한 스펙](https://alistapart.com/article/responsible-javascript-part-1/#:~:text=a%20complex%20specification)에 따라 관리된다. 자바스크립트를 활용할 수 없는 환경에서도 SPA를 사용할 수 있게 하려면, 서버사이드 렌더링을 고려해야 한다.

> https://kryogenix.org/code/browser/everyonehasjs.html

![CSR vs SSR](https://i2.wp.com/alistapart.com/wp-content/uploads/2019/04/fig2.png?resize=960%2C324&ssl=1)

클라이언트의 라우터가 페이지의 어떤 컨텐츠가 변경되었는지 알리지 않는다면 접근성 측면 에서도 좋지 못하다. 

일부 클라이언트 라이브러리는 매우 작지만, 이를 [리액트와 함께 사용하는 것을 고려한다면](https://bundlephobia.com/package/react-router@6.0.2),그리고 여기에 [상태 관리 라이브러리를 얹는다면](https://bundlephobia.com/package/redux@4.1.2) 더 커진다.  따라서 개발하기전에 무엇을 구축하고 있는지, 그리고 클라이언트 사이드 라우팅이 손익계산을 해봤을 때 충분히 필요한 것인지 신중하게 고려해봐야 한다. 일반적으로는, 없는게 낫다.

만약 네비게이션 성능이 우려된다면, [rel=prefetch](https://www.w3.org/TR/resource-hints/#prefetch-link-relation-type)을 사용하여 동일한 오리진의 문서를 미리 가져올 수도 있다. 이는 document를 캐시에서 즉시 사용할 수 있으므로, 페이지의 로딩 성능을 높이는데 큰 영향을 미친다. prefetch는 낮은 우선순위로 진행되므로, 다른 중요한 리소스와 경쟁할 가능성도 낮다.

이 기법의 단점 중 하나는 일단 쓰고 보는 식으로 낭비가 될 수 있다는 것이다. 이 경우 구글의 [Quicklink](https://github.com/GoogleChromeLabs/quicklink)를 사용하는 것도 좋은 방법이 될 수 있다. 클라이언트 연결이 느리다면, 데이터 보호 모드가 사용되고 있는지를 확인하여 이 문제를 완화시킬 수 있다. 기본적으로, 교차 오리진에서는 링크를 미리 가져오지 않는다.

또한 서비스 워커를 활용한다면 클라이언트 라우팅을 사용하는지와 관계 없이 사용자 성능에 큰 도움을 줄 수 있다. 서비스 워커가 미리 경로를 파악해두면 앞서 언급한 기법과 마찬가지로 이점을 얻을 수 있지만, 요청과 응답에 대해 더 큰 제어권을 얻을 수 있다.

- https://developers.google.com/web/ilt/pwa/caching-files-with-service-worker

오늘날 서비스워커를 활용하는 것은 아마도 자바스크립트의 책임감을 높이는데 있어 가장 좋은 방법중 하나일 것이다.

## 자바스크립트는 레이아웃 문제를 해결하는데 별로 도움이 되지 않는다

레이아웃 문제로 인해 패키지를 설치하기전에, 반드시 내가 해결하고자 하는 문제가 무엇인지 명확히 할 필요가 있다. 이 레이아웃 문제를 해결하기 위한 가장 좋은 방법은 CSS다. [박스 위치 조정, 정렬, 사이즈](https://www.npmjs.com/package/flexibility) [문자열 오버플로우](https://www.npmjs.com/package/shave) 혹은 [레이아웃 시스템 전반](https://www.npmjs.com/package/lost)을 해결하기 위한 대부분의 자바스크립트 패키지들은 대부분 CSS로도 마찬가지로 해결할 수 있다. Flexbox, Grid와 같은 최신 레이아웃 엔진은 특별히 프레임워크가 필요 없을 정도로 잘 지원 된다. CSS가 곧 프레임워크다. CSS 내부에서 [feature queries](https://hacks.mozilla.org/2016/08/using-feature-queries-in-css/)를 사용한다면, 점진적으로 레이아웃 문제를 해결하기에 유용하다.

```css
/* Your mobile-first, non-CSS grid styles goes here */

/* The @supports rule below is ignored by browsers that don't
   support CSS grid, _or_ don't support @supports. */
@supports (display: grid) {
  /* Larger screen layout */
  @media (min-width: 40em) {
    /* Your progressively enhanced grid layout styles go here */
  }
}
```

우리는 모든 브라우저에서 사이트가 동일하게 페이지가 보이도록 개발해야 한다. 2009년으로 시간을 되돌려보자. IE 보다 더 좋은 브라우저에서도, 그리고 IE6 에서도 똑같이 보이게 하는 일을 해야만 하곤 했다. 그리고 2021년 현재, 만약 우리가 모든 브라우저에서 동일하게 페이지를 보이는 것을 목표로 하고 있다면 이 목표를 수정할 필요가 있다.에버그린 브라우저가 할 수있는 일을 하지 못하는 일부 브라우저를 지원하는 일을 계속해서 지원해 나가야 한다. 모든 플랫폼에서 동일하게 보이길 바라는 것을 바라는 것은 헛된 노력이며, 점진적 향상을 이룩하는데 있어 걸림돌이 될 것이다.

## 자바스크립트를 그만 쓰자는 이야기는 아니다

자바스크립트에 악의가 있는 것은 아니다. 자바스크립트로 인해 많은 것을 배울 수 있었고, 해마다 할 수 있는 것도 많아지고 기능도 성숙해지고 있다.

그러나 자바스크립트와 의견이 다를 때가 종종있다. 자바스크립트에 대해 항상 비판적인 자세를 취해야 한다. 좀더 정확히 말하자면, 우리가 웹을 구축하기 위한 첫번째 수단으로 자바스크립트를 생각해서는 안된다. 뒤에서 엉킨 전선, 코드 줄을 뜯어보다보면 웹이 자바스크립트에 취해 (drunken) 있다는 것을 볼 때가 한두번이 아니다. 우리는 거의 모든 문제에 자바스크립트를 가져다 댄다. 우리는 이러한 자바스크립트의 과도한 '숙취'를 막기 위해 실용적으로 다가갈 필요가 있다.

