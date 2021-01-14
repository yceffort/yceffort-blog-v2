---
title: 'overflow: auto vs overflow: scroll 왜 윈도우에서만 쓸모없는 스크롤바가 노출될까'
tags:
  - css
  - browser
published: true
date: 2021-01-14 22:40:55
description: '맨날 맥만 봐서 이런 줄도 몰랐다 반성합니다'
---

맥에서 웹을 개발하다보면 단점 아닌 단점이 하나 있는데, 바로 웹 화면이 다른 플랫폼에서 스크롤로 도배되어 있다는 것이다. 맥의 경우에는 커서가 플랫폼 화면에 올라오는 순간에 스크롤 막대가 올라온다.

![naver-mac](./naver-mac.png)

> 맥은 스크롤바가 보이지 않는다.

![naver-window](./naver-window.png)

> 그러나 윈도우는 기본적으로 스크롤바를 깔고 간다.

같은 사이트라 할지라도, 맥은 기본적으로 애플리케이션 영역에 커서가 올라오지 않는 이상 스크롤바를 보여주지 않는다. 이는 다음과 같은 속성 때문이다.

![mac](./mac-scroll-preference.png)

> 스크롤 막대보기: 마우스 또는 트랙패드에 따라 자동으로 설정이 기본값으로 되어 있다.

종종 이러한 특징 때문에 맥에 대한 비난 내지는 혼선이 빚어지는데, 사실 범인은 개발자다.

**결론부터 이야기 하자면 `overflow:scroll`는 항상 스크롤 막대를 표시한다.** 그러나 대부분의 개발자들은 맥의 기본동작 처럼, 필요한 경우에만 스크롤 바를 표시하고 싶을 것이다. 이를 위해서는 `overflow: auto`를 활용하여 브라우저 스스로가 스크롤 바가 필요한지 여부를 자동으로 결정하도록 해야 한다.

이러한 문제가 발생하고 있는 곳이 링크드인 이다.

![linkedin-window-scroll](./linkedin-window-scrollbar.png)

링크드인에서 포스트를 올릴 때, 스크롤이 필요하지 않은 상황임에도 윈도우에서 비활성화된 스크롤바가 보이는 것을 볼 수 있다. 사진까지 올리면 이제 스크롤바가 불필요하게 중첩까지 된다.

![linkedin-window-duplicated-scrollbar.png](./linkedin-window-duplicated-scrollbar.png)

문제는 `overflow-y: scroll`이다. 해당 옵션을 수정하면 컨텐츠가 넘쳐날 때만 스크롤 바가 뜬다.

![linkedin-post-window-scroll-auto](./linkedin-post-window-scroll-auto.png)

![linkedin-post-window-scroll](./linkedin-post-window-scroll.png)

> 스크롤바를 의도한 걸 수도 있다. 하지만 대부분은 버그입니다.

https://developer.mozilla.org/ko/docs/Web/CSS/overflow

> `scroll`:콘텐츠를 안쪽 여백 상자에 맞추기 위해 잘라냅니다. 브라우저는 콘텐츠를 실제로 잘라냈는지 여부를 따지지 않고 항상 스크롤바를 노출하므로 내용의 변화에 따라 스크롤바가 생기거나 사라지지 않습니다. 프린터는 여전히 넘친 콘텐츠를 출력할 수도 있습니다.

> `auto`: 사용자 에이전트가 결정합니다. 콘텐츠가 안쪽 여백 상자에 들어간다면 visible과 동일하게 보이나, 새로운 블록 서식 문맥을 생성합니다. 데스크톱 브라우저의 경우 콘텐츠가 넘칠 때 스크롤바를 노출합니다.

이와 별개로 가로 스크롤바가 생기는 문제가 간혹 있는데, 이는 `body`나 `html`에 `100vw`를 설정했을 때다. 이 또한 맥에서는 정상적으로 작동하지만, 다른 플랫폼에서는 그렇지 못하다.

![window-100vw](./window-100vw.png)

> 윈도우 크롬에서 `body`에 `100vw`를 적용

![mac-100vw](./mac-100vw.png)

> 맥 크롬에서 `body`에 `100vw`를 적용

그 이유는 아마도 다른플랫폼에서는 `100vw`에 스크롤바의 너비까지 포함하기 때문일 것이다. 이 경우 `100%`를 적용하면 해결이 된다.
