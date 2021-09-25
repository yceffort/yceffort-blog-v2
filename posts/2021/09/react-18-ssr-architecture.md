---
title: '리액트 18에서 변경될 새로운 SSR 아키텍쳐'
tags:
  - browser
  - javascript
  - react
  - web
published: true
date: 2021-09-25 17:24:06
description: '따라가는 것만 해도 바쁜 인생'
---

## Table of Contents

## Overview

React 18의 다가올 변경사항 중에는 서버사이드 렌더링 (Server-Side Rendering, 이하 SSR) 성능을 향상 시키기 위한 아키텍처 개선이 있따. 이러한 개선은 몇년간의 노력으로 이루어져 있으며 상당한 향상을 가져올 것이다. 개선사항의 대부분은 미공개가 될 예정이지만 (behind-the-scenes) 프레임워크를 사용하지 않는 경우 (nextjs와 같은) 주의해야할 몇가지 옵트인 메커니즘이 존재한다.

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

React 18을 사용하면, `<Suspense>`를 사용하여 이 단계를 서로 독립적으로 실행하고, 나머지 부분을 서로 차단하지 않는 더 작은 독립 장치로 프로그램을 나눌 수 있다. 결과적으로, 애플리케이션의 사용자는 콘텐츠를 더 빨리보고 훨씬 더 빠르게 인터랙션을 할 수 있다. 애플리케이션에서 가장 느린부분이 더이상 전체 프로세스의 짐이 되지 않는다. 이러한 개선사항은 자동으로 수행되고, 작동하기 위해 특별히 코드를 조정할 필요도 없다.

이는 `React.lazy`가 SSR과 함꼐 작동한다는 의미이기도 하다. [데모](https://codesandbox.io/s/festive-star-9hfqt?file=/src/App.js)를 살펴보자.

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

유저가 웹사이트를 처음 로딩한다면, 개발자들은 최대한 빨리 사용자에게 사용가능한 페이지를 제공하고 싶을 것이다.

![fully-loaded-website](https://camo.githubusercontent.com/8b2ae54c1de6c1b24d9080d2a50a68141f7f57252803543c30cc69cdd4b82fa1/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f784d50644159634b76496c7a59615f3351586a5561413f613d354748716b387a7939566d523255565a315a38746454627373304a7553335951327758516f3939666b586361)

이 그림에서는, 녹색 영역이 '사용자가 사용가능한 페이지'를 의미한다. 즉 모든 자바스크립트의 이벤트 핸들러가 연결되어 있고, 버튼을 클릭하면 상태를 업데이트 하는 등의 여러가지 작업을 수행할 수 있다.

만약 페이지에서 자바스크립트 코드가 완전히 로딩 되지 않았다면 페이지 내에서 사용자가 작업을 수행할 수 없을 것이다. 이 자바스크립트 코드에는 리액트 그 자체와 애플리케이션 내 코드가 모두 포함된다. 크기가 작은 웹 사이트의 경우, 로드 시간의 대부분이 애플리케이션 코드를 다운로드 하는데 보내버린다.

SSR을 사용하지 않는다면, 자바스크립트를 로딩하는 동안 사용자가 볼 수 있는 페이지는 빈 페이지일 뿐이다.

![blank-website](https://camo.githubusercontent.com/7fac45f105cd741a94db77234465c4c85843b1e6f902b21bbdb1fe5b52d25a05/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f39656b30786570614f5a653842764679503244652d773f613d6131796c464577695264317a79476353464a4451676856726161375839334c6c726134303732794c49724d61)

이러한 상태는 좋지 못하다. 그래서 우리는 SSR을 사용하게 된다. SSR을 사용하면 서버에서 리액트 컴포넌트를 HTML로 렌더링 하여 사용자에게 전송할 수 있다. HTML은 링크나 form 입력 같은 아주 기초적인 웹 인터랙션 말고는 할 수 있는게 별로 없긴하다. 그러나 사용자는 자바스크립트 코드가 완전히 로딩되는 동안 최소한 아래와 같은 화면은 볼 수 있다.

![SSR-website](https://camo.githubusercontent.com/e44ee4be56e56e74da3b9f7f5519ca6197b24e9c34488df933140950f1b31c38/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f534f76496e4f2d73625973566d5166334159372d52413f613d675a6461346957316f5061434668644e36414f48695a396255644e78715373547a7a42326c32686b744a3061)

위 그림에서 회색 영역은 화면에서 이러한 인터랙션이 가능하지 않음을 의미한다. 아직 애플리케이션의 자바스크립트 코드가 로딩되지 않았기 때문에 버튼을 클릭해도 아무런 소용이 없을 것이다. 그러나 콘텐츠가 많은 웹 사이트의 경우, SSR은 속도가 느린 환경의 유저에세 자바스크립트를 로딩하는 동안 최소한 콘텐츠를 볼 수 있게는 해주므로 유용하다.

리액트와 프로그램의 코드가 모두 로딩 되면, 이 HTML을 다시 인터랙션이 가능한 상태로 만드려고 한다. 여기에서 우리는 리액트에게 이렇게 명령을 전달한다. "서버사이드에서 생성된 페이지가 있다. 여기에 이벤트 핸들러를 붙여" 리액트는 컴포넌트 트리를 메모리에 렌더링하지만, DOM 노드를 생성하는 대신에 이미 생성되어 있는 HTML에 이 로직을 붙여 나가게 된다.

**컴포넌트를 렌더링하고 이벤트 핸들러를 연결하는 이러한 프로세스를 "hydration"이라고 부른다. 이는 "dry"한 HTML에 인터랙션이 가능한 "water"를 주는 것이다.**

"hydration" 한 뒤에는 리액트는 드디어 비로소 적절하게 동작한다. 컴포넌트가 상태를 설정하고, 클릭에 반응하는 등의 작업을 수행할 수 있게 된다.

![fully-loaded-website](https://camo.githubusercontent.com/8b2ae54c1de6c1b24d9080d2a50a68141f7f57252803543c30cc69cdd4b82fa1/68747470733a2f2f717569702e636f6d2f626c6f622f5963474141416b314234322f784d50644159634b76496c7a59615f3351586a5561413f613d354748716b387a7939566d523255565a315a38746454627373304a7553335951327758516f3939666b586361)

SSR은 일종의 매직 트릭과도 같다. 그렇다고 해서 애플리케이션이 인터랙션하는 속도가 빨라지는 것은 아니다. 사용자가 JS가 로딩되는 것을 기다리는 동안 정적 콘텐츠라도 볼 수 있도록 애플리케이션을 조금더 빠르게 보여주는 것이다. 이 트릭은 네트워크 연결이 좋지 않은 사람들에게 큰 차이를 만들어 주고, 전반적으로 성능을 향상시켜 줄 수 있따. 또한 인덱싱이 쉽고, 속도가 향상되어 검색엔진 우선순위 지정에도 도움이 된다.

> SSR과 [Server Components](https://reactjs.org/blog/2020/12/21/data-fetching-with-react-server-components.html)는 다른 것이다. Server Components는 React 18 릴리즈 대상이 아닐 수도 있는 좀더 실험적인 기능이다.

## 오늘날 SSR의 문제점은 무엇인가?

위 작업은 동작하지만, 몇가지 최적화가 필요한 부분이 있다.
