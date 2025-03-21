---
title: '리액트 18에서 변경될 새로운 SSR 아키텍쳐'
tags:
  - javascript
  - react
  - web
published: true
date: 2021-09-25 17:24:06
description: '따라가는 것만 해도 바쁜 인생'
---

## Table of Contents

## Overview

React 18의 다가올 변경사항 중에는 서버사이드 렌더링 (Server-Side Rendering, 이하 SSR) 성능을 향상 시키기 위한 아키텍처 개선이 있다. 이러한 개선은 몇년간의 노력으로 이루어져 있으며 뛰어난 향상을 만들어 낼 것이다. 개선사항의 대부분은 미공개가 될 예정이지만 (behind-the-scenes) 프레임워크를 사용하지 않는 경우 (nextjs와 같은) 주의해야할 몇가지 옵트인 메커니즘이 존재한다.

새롭개 공개되는 API는 `pipeToNodeWritable`로, [여기](https://github.com/reactwg/react-18/discussions/22)에서 찾아볼 수 있다. 아직 완성된 것이 아니기 때문에 차후에 자세히 글을 쓸 예정이다.

현재는 `<Suspense>` API가 기본으로 자리잡고 있다.

그리고 이것이 React 18에서 어떻게 변화하는지, 설계와 어떤 문제를 해결하려 하는지 살펴보자.

## 요약

SSR을 사용하여 React 컴포넌트를 서버에서 HTML로 생성하고, 해당 HTML을 사용자에게 전송할 수 있다. SSR을 사용하면 자바스크립트 번들이 로드되어 실행되기 전에 페이지의 내용을 보여줄 수 있다는 장점이 있다.

리액트에서 SSR은 아래와 같은 순서로 일어난다.

1. 서버에서 애플리케이션 전체를 위한 데이터를 불러온다.
2. 서버에서 애플리케이션 전체를 HTML로 렌더링 한다음 이를 응답으로 돌려 보내준다.
3. 클라이언트에서 애플리케이션 전체 자바스크립트 코드를 실행한다.
4. 클라이언트에서 서버에서 만들어진 HTML과 자바스크립트 로직을 결합한다. (이를 `hydration`이라 부른다.)

여기서 핵심은 다음 단계를 시작하기 전에 각 단계가 전체 애플리케이션에 걸쳐서 완료되어야 한다는 것이다. 거의 모든 앱에서 그렇듯이, 여기서 일부분이라도 다른 부분에 비해 느릴 경우 비효율적으로 애플리케이션이 동작하게 된다.

React 18을 사용하면, `<Suspense>`를 사용하여 이 단계를 서로 독립적으로 실행하고, 나머지 부분을 서로 차단하지 않는 더 작은 독립 장치로 프로그램을 나눌 수 있다. 결과적으로, 애플리케이션의 사용자는 콘텐츠를 더 빨리보고 훨씬 더 빠르게 인터랙션을 할 수 있다. 애플리케이션에서 가장 느린부분이 더 이상 전체 프로세스의 짐이 되지 않는다. 이러한 개선사항은 자동으로 수행되고, 작동하기 위해 특별히 코드를 조정할 필요도 없다.

이는 `React.lazy`가 SSR과 함께 작동한다는 의미이기도 하다. [데모](https://codesandbox.io/s/festive-star-9hfqt?file=/src/App.js)를 살펴보자.

> SSR을 위한 프레임워크를 사용하지 않는다면, [HTML을 생성하는 방식을 변경해야 한다.](https://codesandbox.io/s/festive-star-9hfqt?file=/server/render.js:1043-1575)

```javascript
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

import * as React from 'react'
// import {renderToString} from 'react-dom/server';
import { pipeToNodeWritable } from 'react-dom/server'
import App from '../src/App'
import { DataProvider } from '../src/data'
import { API_DELAY, ABORT_DELAY } from './delays'

// 실제 애플리케이션에서는, webpack이 처리하는 일
let assets = {
  'main.js': '/main.js',
  'main.css': '/main.css',
}

module.exports = function render(url, res) {
  // 기존에 우리가 했던 방식
  //
  // res.send(
  //   '<!DOCTYPE html>' +
  //   renderToString(
  //     <DataProvider data={data}>
  //       <App assets={assets} />
  //     </DataProvider>,
  //   )
  // );

  res.socket.on('error', (error) => {
    console.error('Fatal', error)
  })
  let didError = false
  const data = createServerData()
  // 여기가 바로 리액트 18에서 변경된 부분이다.
  const { startWriting, abort } = pipeToNodeWritable(
    <DataProvider data={data}>
      <App assets={assets} />
    </DataProvider>,
    res,
    {
      onReadyToStream() {
        // 스트리밍 시작전에 에러가 발생할 경우, 에러코드를 내려준다.
        res.statusCode = didError ? 500 : 200
        res.setHeader('Content-type', 'text/html')
        res.write('<!DOCTYPE html>')
        startWriting()
      },
      onError(x) {
        didError = true
        console.error(x)
      },
    },
  )
  // 렌더링을 준비할 만큼 시간을 줬는데, 이 시간이 지나가버리면 그냥 클라이언트에서 렌더링 하도록 하게 한다.
  // ABORT_DELAY를 낮춰서 클라이언트가 렌더링하는 것을 살펴볼 수도 있다.
  setTimeout(abort, ABORT_DELAY)
}

// 데이터 fetch로 인한 지연을 시뮬레이션
// 스트리밍 HTML 렌더러가 아직 실제 데이터 가져오는 것과 일치 하지 않도록 하기 위해 고의로 타임아웃을 줘서 시뮬레이션
function createServerData() {
  let done = false
  let promise = null
  return {
    read() {
      if (done) {
        return
      }
      if (promise) {
        throw promise
      }
      promise = new Promise((resolve) => {
        setTimeout(() => {
          done = true
          promise = null
          resolve()
        }, API_DELAY)
      })
      throw promise
    },
  }
}
```

## SSR은 무엇인가?

> - 검정 네모: 컴포넌트
> - 초록색 빗금: 사용자가 사용할 수 있는 준비가 되었다.
> - 흰색 빗금: HTML만 로딩 되었다. (아직 자바스크립트 코드가 로딩되지 않아 대부분의 기능을 사용할 수는 없는 상태)

유저가 웹사이트를 처음 로딩한다면, 개발자들은 최대한 빨리 사용자에게 사용가능한 페이지를 제공하고 싶을 것이다.

![fully-loaded-website](https://camo.githubusercontent.com/8b2ae54c1de6c1b24d9080d2a50a68141f7f57252803543c30cc69cdd4b82fa1/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f784d50644159634b76496c7a59615f3351586a5561413f613d354748716b387a7939566d523255565a315a38746454627373304a7553335951327758516f3939666b586361)

이 그림에서는, 녹색 영역이 '사용자가 사용가능한 페이지'를 의미한다. 즉 모든 자바스크립트의 이벤트 핸들러가 연결되어 있고, 버튼을 클릭하면 상태를 업데이트 하는 등의 여러가지 작업을 수행할 수 있다.

만약 페이지에서 자바스크립트 코드가 완전히 로딩 되지 않았다면 페이지 내에서 사용자가 작업을 수행할 수 없을 것이다. 이 자바스크립트 코드에는 리액트 그 자체와 애플리케이션 내 코드가 모두 포함된다. 크기가 작은 웹 사이트의 경우, 로드 시간의 대부분이 애플리케이션 코드를 다운로드 하는데 보내버린다.

SSR을 사용하지 않는다면, 자바스크립트를 로딩하는 동안 사용자가 볼 수 있는 페이지는 빈 페이지일 뿐이다.

![blank-website](https://camo.githubusercontent.com/7fac45f105cd741a94db77234465c4c85843b1e6f902b21bbdb1fe5b52d25a05/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f39656b30786570614f5a653842764679503244652d773f613d6131796c464577695264317a79476353464a4451676856726161375839334c6c726134303732794c49724d61)

이러한 상태는 좋지 못하다. 그래서 우리는 SSR을 사용하게 된다. SSR을 사용하면 서버에서 리액트 컴포넌트를 HTML로 렌더링 하여 사용자에게 전송할 수 있다. HTML은 링크나 form 입력 같은 아주 기초적인 웹 인터랙션 말고는 할 수 있는게 별로 없긴하다. 그러나 사용자는 자바스크립트 코드가 완전히 로딩되는 동안 최소한 아래와 같은 화면은 볼 수 있다.

![SSR-website](https://camo.githubusercontent.com/e44ee4be56e56e74da3b9f7f5519ca6197b24e9c34488df933140950f1b31c38/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f534f76496e4f2d73625973566d5166334159372d52413f613d675a6461346957316f5061434668644e36414f48695a396255644e78715373547a7a42326c32686b744a3061)

위 그림에서 회색 영역은 화면에서 이러한 인터랙션이 가능하지 않음을 의미한다. 아직 애플리케이션의 자바스크립트 코드가 로딩되지 않았기 때문에 버튼을 클릭해도 아무런 소용이 없을 것이다. 그러나 콘텐츠가 많은 웹 사이트의 경우, SSR은 속도가 느린 환경의 유저에세 자바스크립트를 로딩하는 동안 최소한 콘텐츠를 볼 수 있게는 해주므로 유용하다.

리액트와 프로그램의 코드가 모두 로딩 되면, 이 HTML을 다시 인터랙션이 가능한 상태로 만드려고 한다. 여기에서 우리는 리액트에게 이렇게 명령을 전달한다. "서버사이드에서 생성된 페이지가 있다. 여기에 이벤트 핸들러를 붙여" SSR이 아닌 리액트는 컴포넌트 트리를 메모리에 렌더링하지만, DOM 노드를 생성하는 대신에 이미 생성되어 있는 HTML에 이 로직을 붙여 나가게 된다.

**컴포넌트를 렌더링하고 이벤트 핸들러를 연결하는 이러한 프로세스를 "hydration"이라고 부른다. 이는 "dry"한 HTML에 인터랙션이 가능한 "water"를 주는 것이다.**

"hydration" 한 뒤에는 리액트는 드디어 비로소 적절하게 동작한다. 컴포넌트가 상태를 설정하고, 클릭에 반응하는 등의 작업을 수행할 수 있게 된다.

![fully-loaded-website](https://camo.githubusercontent.com/8b2ae54c1de6c1b24d9080d2a50a68141f7f57252803543c30cc69cdd4b82fa1/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f784d50644159634b76496c7a59615f3351586a5561413f613d354748716b387a7939566d523255565a315a38746454627373304a7553335951327758516f3939666b586361)

SSR은 일종의 매직 트릭과도 같다. 그렇다고 해서 애플리케이션이 인터랙션하는 속도가 빨라지는 것은 아니다. 사용자가 JS가 로딩되는 것을 기다리는 동안 정적 콘텐츠라도 볼 수 있도록 애플리케이션을 조금더 빠르게 보여주는 것이다. 이 트릭은 네트워크 연결이 좋지 않은 사람들에게 큰 차이를 만들어 주고, 전반적으로 성능을 향상시켜 줄 수 있다. 또한 인덱싱이 쉽고, 속도가 향상되어 검색엔진 우선순위 지정에도 도움이 된다.

> SSR과 [Server Components](https://reactjs.org/blog/2020/12/21/data-fetching-with-react-server-components.html)는 다른 것이다. Server Components는 React 18 릴리즈 대상이 아닐 수도 있는 좀더 실험적인 기능이다.

## 오늘날 SSR의 문제점은 무엇인가?

위 작업은 동작하지만, 몇가지 최적화가 필요한 부분이 있다.

### 모든 fetch가 끝나야 뭐라도 보여줄 수 있다.

오늘날 SSR의 문제점은 컴포넌트가 "데이터 대기" 상태를 허용하지 않는 다는 것이다. 현재 API를 사용하면, HTML으로 렌더링할 때 쯤이면 서버에서 컴포넌트에 대한 모든 데이터가 준비가 되어 있어야 한다. 즉, HTML을 클라이언트에 전송하기 전에 서버에서 모든 데이터가 수집되어 있어야 한다. 이는 상당히 비효율적이다.

예를 들어, 댓글이 있는 게시물을 렌더링 한다고 가정해보자. 댓글은 일찍 표시하는게 좋기 때문에 서버 HTML 결과물에 포함시킬 수도 있다. 그러나 데이터 베이스 또는 API 수준에서 속도는 우리가 제어할 수 없다. 이제 우리는 선택을 해야 한다. 서버에서 내보내지 않는다면, 자바스크립트가 로딩되기 전까지 사용자가 댓글을 볼 수 없다. 혹은 서버에 포함시키는 경우 댓글이 로딩되고 전체 트리를 렌더링 할수 있을 때 까지 기다려야 한다.

### hydration 하기 전까지 모든 자바스크립트를 로딩해야 한다.

자바스크립트 코드가 로딩되면, 리액트에 HTML을 "hydrating" 하여 상호작용할 수 있게 끔 만들어 달라고 해야 한다. 리액트는 컴포넌트를 렌더링 하는 동안 서버에서 생성한 HTML을 순회하면서 이벤트 핸들러를 HTML에 연결해야 한다. 이 작업을 수행하려며 브라우저의 컴포넌트에서 생성된 트리가 서버에서 생성된 트리와 일치 되어야 한다. 그렇지 않으면 React가 이를 맞추지 못하게 된다. 결과적으로 클라이언트에 있는 모든 컴포넌트의 자바스크립트를 로딩해야 hydrate 가 가능해 진다는 것이다.

예를 들어, 댓글 위젯에 복잡한 기능이 포함되어 있고 이를 위한 자바스크립트 로딩을 하는데 시간이 걸린다고 가정해보자. 이제 우리는 어려운 선택을 해야 한다. 서버에서 댓글을 불러와 HTML을 조기 렌더링 하는 것 까지는 오케이. 그러나 hydration은 한번에 일어나야 하기 때문에, 댓글 위젯에 있는 코드를 로딩하기 전까지는 사이드바나 게시글에 대해서 hydration을 할 수가 없다. 물론 코드 분할을 통하여 이를 달성할 수도 있지만, 서버 HTML에서 댓글을 제거해야 한다. 그렇지 않으면 리액트는 이 HTML을 어떻게 해야할지 모르고 hydration 중에 삭제해 버릴 것이다.

### 상호작용이 가능해지기전까지 모든 것을 hydration 해야 한다.

"hydration" 자체에도 비슷한 문제가 있다. 현재 리액트는 한번에 모든 트리에 hydration을 진행한다. 즉 일단 hydration을 시작하면 (컴포넌트 함수를 호출하면) 트리 전체에 이 작업을 완료 할 때 까지 멈출 수 없다. 따라서 모든 컴포넌트가 hydration을 할 때 까지 기다려야만 컴포넌트와 상호작용할 수 있다.

예를 들어, 댓글 위젯에 많은 렌더링 로직이 포함되어 있다고 가정해보자. 일반적인 컴퓨터에서는 빠르게 동작할 수 있지만, 보급형 모바일 기기에서는 이러한 로직을 실행하는 것이 결코 저렴한 작업이 아니며, 화면을 몇 초 동안 얼어붙어있게 할 수도 있다. 물론, 이상적인 상황에서는 이러한 로직이 클라이언트에 담겨 있어서는 안된다. (Server Components가 도움이 될 수도 있다.) 그러나 일부 로직에서는 이벤트 핸들러가 무엇을 해야 하는지 결정해야 하며, 상호작용이 일어나는 것이 필수적이기 때문에 이러한 상황을 피하는 것은 어렵다. 따라서 일단 hydration이 일어나면 전체 트리에 이 작업이 끝나기전까지는 다른 콘텐츠를 사용할 수 없다. 사용자가 이 페이지에서 완전히 벗어나고 싶어하는 경우 (다른 페이지로 가고 싶어 하는 경우)에도, 불행하게도 hydration 작업 때문에 바빠서 사용자가 원하지 않는 콘텐츠를 현재 페이지에서 계속해서 가지고 있어야 한다. (로딩이 끝나기도 전에 다른 페이지로 가고 싶지만 hydration 작업 중이라 현재 로딩을 멈추지 못한다)

## 이 문제를 해결하는 방법

위에서 언급한 문제들 사이에는 공통점이 있다. 작업을 그냥 시작해서 되도록 빨리 끝나게 하거나 (하지만 다른 작업을 블로킹하기 때문에 UX에 좋지 않음), 나중에 수행하도록 작업을 미루는 (사용자가 시간을 낭비하게 됨) 이지선다밖에 없다는 것이다.

그 이유는 바로 이 작업이 폭포수처럼 실행되기 때문이다.

1. 데이터 가져오기 (서버)
2. HTML 렌더링 (서버)
3. 자바스크립트 로드 (클라이언트)
4. hydration (클라이언트)

이 단계들은 이전 단계가 완료되기 전까지는 시작할 수가 없다. 이것이 비효율적인 이유이다. 리액트의 해결책은 애플리케이션 전체에 걸쳐 이 작업이 일너아는 것이 아니라, 화면의 각 부분이 이 작업을 단계별로 수행할 수 있도록 작업을 분리하는 것이다.

이는 완전히 새로운 생각은 아니다. [Marko](https://markojs.com/)는 이 패턴을 구현하는 자바스크립트 웹 프레임워크 중 하나다. 문제는 리액트가 어떻게 이 패턴을 적용하는가 이다. 이러한 문제를 해결하기 위해 `<Suspense>`를 2018년에 소개했다. 처음 도입했을 때는 물론 클라이언트에 코드를 레이지 로드 하는 용도였다. 그러나 최종 목표는 이를 SSR과 통합하여 위에서 언급한 문제를 해결하는 것이다.

## React 18: HTML 스트리밍과 선택적 hydration

리액트 18 에서는 `Suspense`를 활용한 두가지 주요 SSR 기능을 제공한다.

- 서버에서 HTML을 스트리밍. 이를 위해 `renderToString` 대신 `pipeToNodeWritable`을 사용해야 한다.
- 클라이언트에서 선택적 hydration: 이를 위해 `createRoot`를 사용하고 `<Suspense>`로 감싼다.

이 두 기능을 어떻게 사용하고, 문제를 어떻게 해결하는지 예제를 통해 살펴보자.

### 모든 데이터를 불러오기전에, HTML을 스트리밍한다.

오늘날 SSR은, HTML을 렌더링하고 hydration 하는 것은 모 아니면 도 다. 다 되거나, 안되거나 둘중 하나다. 아래 HTML을 보자.

```html
<main>
  <nav>
    <!--NavBar -->
    <a href="/">Home</a>
  </nav>
  <aside>
    <!-- Sidebar -->
    <a href="/profile">Profile</a>
  </aside>
  <article>
    <!-- Post -->
    <p>Hello world</p>
  </article>
  <section>
    <!-- Comments -->
    <p>First comment</p>
    <p>Second comment</p>
  </section>
</main>
```

위 HTML을 받으면 클라이언트는 아래를 그릴 것이다.

![SSR-website](https://camo.githubusercontent.com/e44ee4be56e56e74da3b9f7f5519ca6197b24e9c34488df933140950f1b31c38/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f534f76496e4f2d73625973566d5166334159372d52413f613d675a6461346957316f5061434668644e36414f48695a396255644e78715373547a7a42326c32686b744a3061)

그리고 코드를 로딩하고 hydration이 끝나면 아래와 같은 완전한 애플리케이션이 완성된다.

![fully-loaded-website](https://camo.githubusercontent.com/8b2ae54c1de6c1b24d9080d2a50a68141f7f57252803543c30cc69cdd4b82fa1/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f784d50644159634b76496c7a59615f3351586a5561413f613d354748716b387a7939566d523255565a315a38746454627373304a7553335951327758516f3939666b586361)

하지만 리액트 18에서는 다르다. 페이지의 일부를 `<Suspense>`로 감쌀 수 있다.

예를 들어, 댓글을 `<Suspense>`로 감싸고, 로딩되기전까지는 `<Spinner>`를 보여지게 할 수 있다.

```html
<Layout>
  <NavBar />
  <Sidebar />
  <RightPane>
    <Post />
    <Suspense fallback={<Spinner />}>
      <Comments />
    </Suspense>
  </RightPane>
</Layout>
```

`<Comments>`를 `<Suspense>`로 감싼 효과로, 리액트는 댓글 컴포넌트를 기다리지 않고 HTML을 스트리밍할 수 있다. 아직 로딩 되지 않는 댓글 컴포넌트 대신, 아래와 같이 화면이 나타날 것이다.

![comments](https://camo.githubusercontent.com/484be91b06f3f998b3bda9ba3efbdb514394ab70484a8db2cf5774e32f85a2b8/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f704e6550316c4253546261616162726c4c71707178413f613d716d636f563745617955486e6e69433643586771456961564a52637145416f56726b39666e4e564646766361)

그리고 클라이언트가 받은 최초 HTML은 아래와 같을 것이다.

```html
<main>
  <nav>
    <!--NavBar -->
    <a href="/">Home</a>
  </nav>
  <aside>
    <!-- Sidebar -->
    <a href="/profile">Profile</a>
  </aside>
  <article>
    <!-- Post -->
    <p>Hello world</p>
  </article>
  <section id="comments-spinner">
    <!-- Spinner -->
    <img width="400" src="spinner.gif" alt="Loading..." />
  </section>
</main>
```

그리고 댓글 데이터가 준비된다면, 리액트는 같은 스트림으로 추가적인 HTML을 보낼 텐데, 여기에는 HTML을 올바른 위치에 삽입하기 위한 최소한의 인라인 script 태그가 포함되어 있다.

```html
<div hidden id="comments">
  <!-- Comments -->
  <p>First comment</p>
  <p>Second comment</p>
</div>
<script>
  // This implementation is slightly simplified
  document
    .getElementById('sections-spinner')
    .replaceChildren(document.getElementById('comments'))
</script>
```

그 결과, 리액트 자체가 클라이언트에 로드되기 이전에, 뒤늦게 댓글용 HTML 코드가 삽입될 것이다.

![complete](https://camo.githubusercontent.com/e44ee4be56e56e74da3b9f7f5519ca6197b24e9c34488df933140950f1b31c38/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f534f76496e4f2d73625973566d5166334159372d52413f613d675a6461346957316f5061434668644e36414f48695a396255644e78715373547a7a42326c32686b744a3061)

우리는 이로써 첫번째 문제를 해결할 수 있게 되었다. 더 이상 모든 데이터를 준비해둘 필요가 없다. 화면 일부분이 초기 HTML을 지연시킨다면, 모든 HTML을 지연시키거나, HTML에서 일부를 제외시킬 필요가 없다. HTML 스트림에서 나중에 해당 부분을 별도로 삽입할 수 있다.

전통적인 HTML 스트리밍 기술과는 다르게, 꼭 하향식 순서로 일어날 필요가 없다. 예를 들어 사이드바에 적용하고 싶다면, 사이드바를 `<Suspense>`로 묶을 수 있다. 그런다음 사이드바 HTML이 준비되면 React는 HTML 전송이 이미 한차례 끝났지만 다시한번 script 태그와 함께 나머지 HTML을 스트리밍 할 수 있다. 데이터가 특정 순서대로 로딩될 필요는 없다. 스피너가 나타날 위치를 지정하면, 리액트가 나머지를 알아서 계산한다.

> 이 작업을 수행하기 위해서는, 데이터를 가져오는 것을 Suspense와 인테그레이션 해야 한다. Server Components는 Suspense와 함께 쉽게 통합되지만, 그 외에도 다른 fetch 라이브러리와도 통합할 수 있는 방법을 제공할 예정이다.

### 모든 코드가 로드되기 전에 페이지를 hydration 하기

초기 HTML은 일찍 보낼 수 있지만 아직 모든 문제가 해결 된 것은 아니다. 댓글 위젯의 자바스크립트 코드가 모두 로딩되기 전까지 앱에서 hydration을 진행할 수 없다. 이는 코드 크기에 따라서 더 오래 걸릴 수도 있다.

큰 번들을 피하기 위해, 보통은 "코드 스플릿" 을 사용한다. 코드의 일부를 동기적으로 로딩하지 않아도 되고, 혹은 번들러가 이를 별도의 script 태그로 분할하는 방법도 있다.

`React.lazy`로 코드를 분할하여 메인 번들에서 댓글 코드를 아래처럼 분리할 수 있다.

```jsx
import { lazy } from 'react'

const Comments = lazy(() => import('./Comments.js'))

// ...

;<Suspense fallback={<Spinner />}>
  <Comments />
</Suspense>
```

과거 이 방법은 서버사이드 렌더링에서는 동작하지 않았다. 우리가 아는한 SSR에서는, SSR에서 코드 스플릿 컴포넌트를 제외하거나, 코드를 모두 로딩한 후 hydration하거나 둘 중 하나일 뿐이다. 두 방법 모두 어쨌거나 코드 스플릿의 목적을 다소간 손상시킨다.

그러나 리액트 18 부터는 `<Suspense>`를 통해 댓글 위젯이 로드되기전에 애플리케이션에 hydration을 진행할 수 있다.

사용자 관점에서, 처음에 HTML로 스트리밍되는 상호작용이 불가능한 콘텐츠를 살펴보자.

![load-comment](https://camo.githubusercontent.com/484be91b06f3f998b3bda9ba3efbdb514394ab70484a8db2cf5774e32f85a2b8/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f704e6550316c4253546261616162726c4c71707178413f613d716d636f563745617955486e6e69433643586771456961564a52637145416f56726b39666e4e564646766361)

![load-complete](https://camo.githubusercontent.com/e44ee4be56e56e74da3b9f7f5519ca6197b24e9c34488df933140950f1b31c38/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f534f76496e4f2d73625973566d5166334159372d52413f613d675a6461346957316f5061434668644e36414f48695a396255644e78715373547a7a42326c32686b744a3061)

리액트는 이제 hydrate할 준비가 끝났다. 댓글 코드가 아직 오지 않았지만, 뭐 괜찮다. 나머지를 hydration 하면 된다.

![selective-hydration](https://camo.githubusercontent.com/4892961ac26f8b8dacbd53189a8d3fd1b076aa16fe451f8e2723528f51b80f66/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f304e6c6c3853617732454247793038657149635f59413f613d6a396751444e57613061306c725061516467356f5a56775077774a357a416f39684c31733349523131636f61)

이것이 선택적 hydration의 예시다. `Comments`를 `<Suspense>`로 감쌈으로써, 리액트에 페이지의 나머지 부분을 스트리밍하는 것을 차단해서는 안되고, hydration 과정에서도 차단하면 안된다고 말할 수 있게 됐다. 이제 두번째 문제도 해결되었다. hydrating을 하기 위해 더 이상 모든 코드가 로딩될 때 까지 기다릴 필요가 없다. 리액트는 이제 각 부분 별로 코드가 준비되면 hydration을 할 수 있게 되었다.

댓글 부분이 hydration 까지 끝나면 이제 전체애플리케이션을 사용할 수 있게 되었다.

그리고이 선택적 hydration 덕분에, 무거운 자바스크립트 코드가 로딩되지 않더라도 페이지 나머지 부분이 사용가능해 졌다.

### HTML 스트리밍이 모두 끝나기 전에 hydrating

리액트는 이 모든 작업을 알아서 처리하므로, 예기치 않은 순서로 인해 발생하는 일에 대해 걱정할 필요가 없다. 예를 들어 HTML을 스트리밍하는 동안에도 로드하는데 오랜 시간이 걸릴 수도 있다.

![loading](https://camo.githubusercontent.com/484be91b06f3f998b3bda9ba3efbdb514394ab70484a8db2cf5774e32f85a2b8/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f704e6550316c4253546261616162726c4c71707178413f613d716d636f563745617955486e6e69433643586771456961564a52637145416f56726b39666e4e564646766361)

자바스크립트 코드 로딩이 HTML 보다 빨리 끝난다면, 리액트는 HTML을 기다릴 이유가 없다. 그냥 나머지 페이지를 hydration 하면 된다.

![loading2](https://camo.githubusercontent.com/ee5fecf223cbbcd6ca8c80beb99dbea40ccbacf1b281f4cf8ac6970c554eefa3/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f384c787970797a66786a4f4a753475344e44787570413f613d507a6a534e50564c61394a574a467a5377355776796e56354d715249616e6c614a4d77757633497373666761)

댓글의 HTML이 이제서야 로딩되었다면, 자바스크립트 코드가 로딩되지 않아 아래처럼 나타날 것이다.

![comment loading](https://camo.githubusercontent.com/4892961ac26f8b8dacbd53189a8d3fd1b076aa16fe451f8e2723528f51b80f66/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f304e6c6c3853617732454247793038657149635f59413f613d6a396751444e57613061306c725061516467356f5a56775077774a357a416f39684c31733349523131636f61)

그리고 모든 작업이 끝나면, 페이지가 이제 완전히 작동할 것이다.

### 모든 컴포넌트가 hydrate 되기전에 페이지와의 상호작용

앞서 언급한 것 외에도 한가지더 개선점이 존재한다. 이제 더 이상 `hydration`작업이 브라우저가 다른 작업을 하는 것을 막지 않는다.

예를 들어, 댓글 컴포넌트가 hydrate 하는 동안 사용자가 사이드바를 클릭한다고 가정해보자.

![comment-hydrate-interaction](https://camo.githubusercontent.com/6cc4eeef439feb3c17d0ac09c701c0deffe170c60a039afa8c0b85d7d4b9c9ef/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f5358524b357573725862717143534a3258396a4769673f613d77504c72596361505246624765344f4e305874504b356b4c566839384747434d774d724e5036374163786b61)

리액트 18에서는, Suspense 바운더리 내에 있는 hydration 콘텐츠는 브라우저가 이벤트를 처리할 수 있는 한에서 만 수행된다. 덕분에 클릭이 즉시 처리되고, 보급형 모바일 디바이스에서도 hydration이 오래걸리지만 브라우저가 계속 작동하는 것 처럼 보일 수 있다. 예를 들어, hydration 작업 중에도 사용자는 더 이상 관심없는 페이지를 나갈 수도 있다.

이 예제에서는, 댓글만 `Suspense`로 감싸져있기 때문에 페이지 나머지 부분에 hydration을 하는 것은 한번에 이뤄진다. 하지만 많은 다른 부분들도 `Suspense`로 감싼다면 이를 고칠 수 있다. 예를 들어 사이드바도 똑같이 적용해보자.

```html
<Layout>
  <NavBar />
  <Suspense fallback={<Spinner />}>
    <Sidebar />
  </Suspense>
  <RightPane>
    <Post />
    <Suspense fallback={<Spinner />}>
      <Comments />
    </Suspense>
  </RightPane>
</Layout>
```

이제 두 컴포넌트 모두 navbar와 post를 포함하는 초기 HTML 렌더링 작업이후에 서버에서 스트리밍 될 수 있다. 하지만 이 작업은 hydration에도 영향을 미칠 수 있다. 두 컴포넌트 모두 HTML은 로딩되었지만, 코드는 로딩되지 않았다고 가정해보자.

![step1](https://camo.githubusercontent.com/9eab3bed0a55170fde2aa2f8ac197bc06bbe157b6ee9446c7e0749409b8ed978/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f78744c50785f754a55596c6c6746474f616e504763413f613d4e617972396c63744f6b4b46565753344e374e6d625335776a39524473344f63714f674b7336765a43737361)

그 다음, 사이드바와 댓글을 모두 포함하는 번들이 로딩된다. 리액트는 Suspense 경계 부분에서 시작하여 이제 두 컴포넌트에 모두 hydration을 시도할 것이다.

![step2](https://camo.githubusercontent.com/6542ff54670ab46abfeb816c60c870ad6194ab15c09977f727110e270517b243/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f424333455a4b72445f72334b7a4e47684b33637a4c773f613d4778644b5450686a6a7037744b6838326f6533747974554b51634c616949317674526e385745713661447361)

그런데 사용자가 댓글 위젯에 접근했다고 가정해보자.

![step3](https://camo.githubusercontent.com/af5a0db884da33ba385cf5f2a2b7ed167c4eaf7b1e28f61dac533a621c31414b/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f443932634358744a61514f4157536f4e2d42523074413f613d3069613648595470325a6e4d6a6b774f75615533725248596f57754e3659534c4b7a49504454384d714d4561)

리액트는 이 클릭 이벤트를 기억해두었다가, 사이드바보다 댓글을 hydration 하는게 중요하다고 판단하고 더 먼저 hydrate를 하게 된다.

![step4](https://camo.githubusercontent.com/f76a33458a3e698125063884035e7f126104bc2c27c30c02fe8e9ebdf3048c7b/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f5a647263796a4c49446a4a304261385a53524d546a513f613d67397875616d6c427756714d77465a3567715a564549497833524c6e7161485963464b55664f554a4d707761)

hydration이 끝나면, 클릭이벤트를 다시 dispatch하여 컴포넌트가 응답을 낼 수 있도록 할 것이다. 그리고 리액트는 더 이상 급하게 처리해야할 작업이 없으므로, 다시 사이드바 컴포넌트를 hydration 할 것이다.

![step5](https://camo.githubusercontent.com/64ea29524fa1ea2248ee0e721d1816387127507fd3d73a013f89266162b20fba/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f525a636a704d72424c6f7a694635625a792d396c6b773f613d4d5455563334356842386e5a6e6a4a4c3875675351476c7a4542745052373963525a354449483471644b4d61)

이는 3번째 문제도 해결하였다. 선택적 hydration 덕분에, _페이지가 하나라도 작동하기 위해 모두 hydration을 기다릴 필요_가 없어졌다. 리액트는 가능한 빨리 모든 것을 hydration 시도하고, 그리고 유저의 동작에 기반하여 급한 부분을 먼저 우선순위를 두고 hydration 작업을 수행한다. 선택적 hydration의 이점은 애플리케이션 전체에 `Suspense`를 사용할 수로 그 경계가 더욱 세분화 된다는 점을 고려한다면 더욱 분명해진다.

![step6](https://camo.githubusercontent.com/dbbedbfe934b41a8b4e4ed663d66e94c3e748170df599c20e259680037bc506c/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f6c5559557157304a38525634354a39505364315f4a513f613d39535352654f4a733057513275614468356f6932376e61324265574d447a775261393739576e566e52684561)

위 예제에서는, 사용자가 댓글을 시작하자마자 댓글 컴포넌트를 먼저 hydration 한다. React는 모든 부모 Suspense 바운더리의 컨텐츠에 hydration 하는 것을 우선시하지만, 관계 없는 자식 컨텐츠에 대해서는 이를 건너뛴다. 이는 상호작용 경로에 있는 컴포넌트가 먼저 hydration 되기 때문에 hydration이 즉각적으로 일어나는 것과 같은 착각이 일어난다. 그 뒤, 리액트는 나머지 hydration 작업을 수행할 것이다.

실제 예시에서는, `Suspense`를 가능한 애플리케이션의 루트와 가깝게 추가해둘 것이다.

```html
<Layout>
  <NavBar />
  <Suspense fallback={<BigSpinner />}>
    <Suspense fallback={<SidebarGlimmer />}>
      <Sidebar />
    </Suspense>
    <RightPane>
      <Post />
      <Suspense fallback={<CommentsGlimmer />}>
        <Comments />
      </Suspense>
    </RightPane>
  </Suspense>
</Layout>
```

위 예시처럼 한다면, 초기 HTML은 `<Navbar>` 만을 포함하지만, 나머지는 사용자가 상호작용한 부분을 우선시하여 관련 코드가 로딩되는 직시 스트리밍하여 컴포넌트에서 hydration 할 것이다.

> 어떻게 애플리케이션이 전체적으로 hydration 되지 않았는데 동작할 수 있는걸까? 리액트는 개별 컴포넌트에 개별적으로 hydration 하는 것이 아닌 `<Suspense>` 바운더리에 대해 hydration을 발생시킨다. `<Suspense>`는 당장 나타나지 않는 컨텐츠에 사용되므로, 코드는 이 자식 컨텐츠가 즉시 이용할 수 없는 상태에 대해 탄력적으로 대처할 수 있다. 리액트는 항상 부모 컴포넌트를 우선순위로 hydration 하므로, 컴포넌트는 항상 props set을 가지고 있을 수 있다. 리액트는 이벤트가 발생될 때 이벤트 지점에서 전체 상위 트리에 hydration이 진행될 때 까지 이를 보류 시켜 둔다. 마지막으로 상위 항목이 그럼에도 hydration이 되지 않는다면, 리액트는 이를 숨기고 코드가 로드 될 때 까지 `fallback`으로 화면을 바꿔 둔다. 이렇게 하면 트리가 일관되게 유지된다.

## 데모

https://codesandbox.io/s/festive-star-9hfqt?file=/src/App.js

위 데모코드는 `server/delays.js`에서 인위적으로 지연시켜서 확인할 수 있다.

- `API_DELAY`: 댓글을 가져오는 시간을 오래 걸리게 하여 HTML의 나머지 부분을 초기에 전송하는 것을 보여준다.
- `JS_BUNDLE_DELAY`: script 태그가 로딩되는 것을 지연하여 댓글 HTML이 나중에 삽입되는 것을 볼 수 있다.
- `ABORT_DELAY`: 서버에서 가져오는 시간이 너무 길어질 경우, 서버가 렌더링을 포기하고 클라이언트에서 렌더링이 되는 것을 볼 수 있다.

## 결론

리액트 18은 SSR을 위한 두가지 주요 기능을 제공한다.

- **HTML 스트리밍**: 개발자가 원하는 만큼 HTML을 조기에 스트리밍 할 수 있게 해주며, 나중에 로딩된 HTML을 올바른 위치에 놓아주는 `<script>`태그와 함께 추가적으로 스트리밍할 수 있다.
- **선택적 hydration**: HTML과 자바스크립트 코드의 나머지 부분이 완전히 다운로드 되기전에 가능한 빨리 애플리케이션이 hydration 할 수 있도록 한다. 또한 사용자가 상호작용하는 컴포넌트에 hydration 하는 것을 우선시하여, 즉각적으로 hydration 되는 것과 같은 착각을 불러일으킨다.

이 두가지 기능은 SSR과 관련된 아래 세가지 문제를 해결해준다.

- **HTML을 내보내기전에 서버에서 모든 데이터가 로딩될 때 까지 기다릴 필요가 없다.** 대신, HTML을 보낼 수 있는 상황이라면 바로 HTML을 보내고, 나머지부분은 준비되는 대로 스트리밍 할 수 있다.
- **hydration을 하기 위해 모든 자바스크립트 코드가 로드 될 때 까지 기다릴 필요가 없다.** 대신 SSR과 함께 코드 스플릿을 사용할 수 있다. 이렇게 하면 서버 HTML은 그대로 보존할 수 있고, 리액트는 관련 코드가 로드될 때 추가로 hydration 한다.
- **페이지와 상호작용하기 위해 모든 컴포넌트가 hydration 되는 것을 기다릴 필요가 없다.** 대신 선택석 hydration을 사용하여 사용자가 상호작용하는 컴포넌트에 우선순위를 지정하고 조기에 hydration을 수행할 수 있다.

`<Suspense>`는 이러한 모든 기능에 대한 옵트인 역할을 한다. 이 개선사항은 리액트 내부에서 자동으로 수행되며, 기존 리액트 코드의 대부분과 함께 작동 될 것으로 보인다. 이는 로딩중 상태를 선언적으로 표현하는 역할을 한다. `if (isLoading)`과 크게 달라보이지 않을 수 있지만, `<Suspense>`는 이러한 모든 개선사항을 실현해낸다.

> 위 글은 https://github.com/reactwg/react-18/discussions/37 을 번역하고, 본문의 내용 이해를 돕기 위해 몇가지 각색 및 별도 설명을 추가하였습니다.
