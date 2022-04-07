---
title: '리액트 서버 컴포넌트의 동작 방식'
tags:
  - javascript
  - react
  - html
published: true
date: 2022-01-29 18:42:50
description: '리액트 18 존버 하는 중'
---

## Table of Contents

## 리액트 서버 컴포넌트는 무엇인가

React Server Component(이하 RSC)를 사용하면, 서버와 클라이언트 (브라우저)가 리액트 애플리케이션을 서로 협력하여 렌더링 할 수 있다. 이야기 하기에 앞서, 페이지를 렌더링하는 일반적인 리액트 컴포넌트 트리를 생각해보자. 이 트리에는 리액트 컴포넌트가 있고, 이 컴포넌트는 또다른 리액트 컴포넌트를 렌더링한다. RSC를 사용하면 이 트리의 일부 컴포넌트는 서버에서 렌더링하거나, 일부 컴포넌트는 브라우저에서 렌더링 하는 등 처리를 할 수 있다.

![RSC-tree](https://blog.plasmic.app/static/images/react-server-components.png)

### 서버사이드 렌더링이 아니다?

RSC는 정확히 말해서 서버사이드 렌더링이 아니다. 물론 둘다 명칭에서 '서버'가 포함되어 있어서 혼란의 여지가 있다. RSC를 사용하면 SSR을 사용할 필요가 없고, 반대의 경우도 마찬가지다. SSR은 응답 받은 트리를 raw html로 렌더링하기 위한 환경을 시뮬레이션 한다. 즉, 서버와 클라이언트 컴포넌트를 별도로 구별하지 않고 동일한 방식으로 렌더링한다.

물론 SSR와 RSC를 함께 사용하여 서버 컴포넌트를 서버 컴포넌트를 서버쪽에서 렌더링을 하고, 브라우저에서는 적절하게 하이드레이션을 거치게 할 수 있다.

그러나 일단은, SSR을 무시하고 RSC에만 집중하자.

### 왜 필요할까?

RSC 이전에는, 모든 리액트 컴포넌트는 '클라이언트' 컴포넌트 이며, 모두 브라우저에서 실행된다. 브라우저가 리액트 페이지를 방문하면, 필요한 모든 리액트 컴포넌트 코드를 다운로드 하고, 리액트 컴포넌트 트리를 만든 후 DOM에 렌더링한다. (SSR을 사용하면, DOM에 하이드레이션만 진행한다.) 브라우저는 이벤트 핸들러를 부착하고, 상태를 추적하고, 이벤트에 따른 응답 트리 변경 및 DOM의 효율적인 업데이트 등 리액트 애플리케이션이 인터랙션 할 수 있도록 처리할 수 있는 좋은 곳이다. 그런데, 우리가 왜 서버에서 무언가를 렌더링 하려고 하는 걸까?

브라우저에 대신, 서버에서 렌더링을 한다면 다음과 같은 장점을 얻을 수 있다.

- 서버는 데이터 베이스, GraphQL, 파일시스템 등 데이터 원본에 직접 접근 할 수 있다. 서버는 공용 api 엔드 포인트를 거치지 않고 데이터를 직접 가져올 수 있고, 일반적으로 데이터 소스와 더 가깝게 배치되어 있으므로 브라우저보다 더 빠르게 데이터를 가져올 수 있다.
- 브라우저는 자바스크립트 번들링된 모든 코드를 다운로드 해야하는 것 과 달리, 서버는 모든 의존성을 다운로드 할 필요가 없기 때문에 (미리 다운로드 해놓고 수시로 재사용이 가능하므로) 무거운 코드 모듈을 저렴하게 사용할 수 있다.

**즉, RSC를 활영하면 서버와 브라우저가 각자 잘 수행하는 작업을 처리할 수 있다.** 서버 컴포넌트는 데이터를 가져오고 콘텐츠를 렌더링하는데 초점을 맞출 수 있으며 페이지 로딩 속도가 빨라지고 자바스크립트 번들 크기가 작아져서 사용자의 환경이 향상될 수 있다.

## 개괄

이제 어떻게 작동하는지 살펴보자.

RSC는 작업 분담, 즉 서버가 할 수 있는 일을 먼저 처리하게 두고 나머지는 브라우저에게 넘겨준다.

일부 컴포넌트는 서버에서 렌더링되고, 일부 컴포넌트는 클라이언트에서 렌더링 되는 상황을 고려해보자. 서버는 일반적인 리액트 컴포넌트를 html의 `<p>` `<div>` 태그로 렌더링한다. 그러나 브라우저에서 렌더링해야하는 클라이언트 컴포넌트가 나타나면, 클라이언트에서 여기를 렌더링하라는 의미에서 `placeholder` 같은 것을 둔다. 브라우저가 이 결과물을 받으면, 앞서 빈 곳으로 나왔던 부분을 채워 넣는다.

물론, 실제로 이렇게 동작하지는 않지만, 대략적인 그림으로 살펴보았다.

## 클라이언트와 서버 컴포넌트로 나누기

먼저, 컴포넌트를 어떻게 서버 컴포넌트인지, 클라이언트 컴포넌트인지 나눌 수 있을까?

리액트 팀에서는 이를 확장자로 구별하는 방식을 택했다. `.server.jsx` 면 서버 컴포넌트이고, `client.jsx`면 클라이언트 컴포넌트다. 만약 둘다 아니라면, 이 컴포넌트는 서버나 클라이언트 컴포넌트 양쪽 모두에서 가능한 것이다.

이러한 정의는 굉장히 실용적인 방식으로 보인다. 개발자와 번들러 입장에서 모두 구별하기 쉽다. 특히 번들러의 경우, 파일 이름을 검사하여 클라이언트 컴포넌트를 별도로 처리할 수 있다. 곧 알게 되겠지만, 번들러는 RSC를 작동하는데 중요한 역할을 한다.

서버 컴포넌트는 서버에서, 클라이언트 컴포넌트는 클라이언트에서 실행되므로 각 컴포넌트에서 수행할 수 있는 작업에는 제한이 있다. 여기에서 특히 기억해야 할 것은, 클라이언트 컴포넌트가 서버 컴포넌트를 `import` 할 수 없다는 점이다. 이는 서버 컴포넌트를 브라우저에서 실행할 수 없기 때문이다.

```jsx
// ClientComponent.client.jsx
// 안됨.
import ServerComponent from './ServerComponent.server'
export default function ClientComponent() {
  return (
    <div>
      <ServerComponent />
    </div>
  )
}
```

클라이언트 컴포넌트가 서버 컴포넌트를 import 할 수 없고, 그래서 서버 컴포넌트를 초기화 할 수 없다면 리액트 컴포넌트 트리를 어떻게 만들 수 있을까? 아래와 같은 그림의 구조는 가능한 것일까?

![rsc](https://blog.plasmic.app/static/images/react-server-components.png)

클라이언트 컴포넌트에서 서버 컴포넌트를 import 해서 렌더링 할 순 없지만, 그래도 여전히 합성은 가능하다. 즉, 클라이언트 컴포넌트는 opaque `ReactNode`를 props로 받을 수 있고, 그리고 이 `ReactNode`는 여전히 서버컴포넌트에 의해 렌더링 가능하다.

```jsx
// ClientComponent.client.jsx
export default function ClientComponent({ children }) {
  return (
    <div>
      <h1>Hello from client land</h1>
      {children}
    </div>
  )
}

// ServerComponent.server.jsx
export default function ServerComponent() {
  return <span>Hello from server land</span>
}

// OuterServerComponent.server.jsx
// OuterServerComponent 는 클라와 서버 모두에서 초기화 가능하다
// 따라서 서버 컴포넌트를 클라이언트 컴포넌트의 children으로 보낼 수 있다.
import ClientComponent from './ClientComponent.client'
import ServerComponent from './ServerComponent.server'
export default function OuterServerComponent() {
  return (
    <ClientComponent>
      <ServerComponent />
    </ClientComponent>
  )
}
```

이러한 제한은 RSC를 효과적으로 활용하기 위해 컴포넌트를 구성하는 방법에 큰 영향을 미친다.

## RSC 렌더링의 라이프 사이클

RSC 서버 컴포넌트를 렌더링하기 위해서 실제로 어떤 일이 일어나는지 알아 보기 위해 핵심적인 세부 서항을 살펴보자. 서버 컴포넌트를 사용하기 위해 모든 것을 이해할 필요는 없지만, 작동방식에 대해 직관적으로 이해할 필요가 있다.

### 1. 서버가 렌더링 요청을 받는다.

서버가 렌더링 과정의일부를 수행해야 하므로, 페이지의 라이프 사이클은 항상 서버에서 시작된다. 이 중 'root' 컴포넌트는 항상 서버 컴포넌트고, 다른 서버 또는 클라이언트를 렌더링할 수 있다. 서버는 요청에 전달된 정보에 따라 서버 컴포넌트와 어떤 props를 사용할지 결정한다. 이러한 요청은 일반적으로 특정 URL에서 페이지를 요청하는 형태로 나온다.

- https://shopify.dev/custom-storefronts/hydrogen/framework/server-state
- https://github.com/reactjs/server-components-demo/blob/main/server/api.server.js

### 2. 서버가 루트 컴포넌트 엘리먼트를 JSON으로 직렬화

여기에서 최종 목표는 최초 root 서버 컴포넌트를 기본 html 태그와 클라이언트 컴포넌트 placeholder 트리로 렌더링하는 것이다. 그리고 이 트리를 직렬화하여 (json으로) 브라우저로 보내면, 브라우저가 이를 다시 역 직렬화 하여 클라이언트 placeholder에 실제 클라이언트 컴포넌트를 채우고 최종 결과를 렌더링하는 작업을 수행할 수 있다.

그럼 위 예제에서, `OuterServerComponent`를 렌더링하고 싶다면, 단순히 `JSON.stringify(<OuterServerComponent />)`를 한다면 직렬화된 렌더링 트리를 얻을 수 있을까?

거의 그렇다고 볼 수 있지만, 그것만으로 충분하지는 않다. 리액트 엘리먼트의 구조를 다시한번 생각해보자.

```jsx
// React element for <div>oh my</div>
> React.createElement("div", { title: "oh my" })
// 객체 형태로 표현
{
  $$typeof: Symbol(react.element),
  type: "div",
  props: { title: "oh my" },
  ...
}

// React element for <MyComponent>oh my</MyComponent>
> function MyComponent({children}) {
    return <div>{children}</div>;
  }
> React.createElement(MyComponent, { children: "oh my" });
// 객체 형태로 표현
{
  $$typeof: Symbol(react.element),
  type: MyComponent  // reference to the MyComponent function
  props: { children: "oh my" },
  ...
}
```

기본 html 태그 엘리먼트가 아닌 컴포넌트 엘리먼트의 경우, 컴포넌트를 참조하려고 하기 때문에 직렬화 할 수 없다.

따라서 모든 것을 적절히 JSON-stringify를 하기 위해서는, 리액트는 [replacer function](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/stringify#the_replacer_parameter) 을 `JSON.stringify()`에 넘겨 준다. 관련 코드는 [여기](https://github.com/facebook/react/blob/42c30e8b122841d7fe72e28e36848a6de1363b0c/packages/react-server/src/ReactFlightServer.js#L368)에서 찾을 수 있다.

- 기본 HTML 태그의 경우에는 JSON으로 처리가 가능하므로 특별히 처리할 것이 없다.
- 만약 서버 컴포넌트라면, 서버 컴포넌트 함수를 props와 함께 호출하고, 그 결과를 JSON으로 만들어서 내려보낸다. 이렇게 하면 서버 컴포넌트가 효과적으로 렌더링된다. 여기서 목적은 모든 서버 컴포넌트를 html 태그로 바꾸는 것이다.
- 만약 클라이언트 컴포넌트라면, 사실 JSON으로 직렬화가 가능하다. 이미 필드가 컴포넌트 함수가 아닌 모듈 참조 객체 (module reference object)를 가리키고 있다.

### `module reference` 객체란?

RSC는 `module reference` 라고 불리우는, 리액트 엘리먼트의 `type` 필드에 새로운 값을 넣을 수 있도록 제공한다. 이 값으로 컴포넌트 함수대신, 이 `참조`를 직렬화 한다.

예를 들어, `ClientComponent`는 아래와 같은 형태를 띈다.

```javascript
{
  $$typeof: Symbol(react.element),
  // 실제 컴포넌트 함수 대신에, 참조 객체를 갖게됨
  type: {
    $$typeof: Symbol(react.module.reference),
    // ClientComponent is the default export...
    name: "default",
    // from this file!
    filename: "./src/ClientComponent.client.js"
  },
  props: { children: "oh my" },
}
```

그렇다면 클라이언크 컴포넌트 함수에 대한 참조를 직렬화 할 수 있는 `module reference` 객체로 변환하는 것은 어디에서 이루어지는 것일까?

이러한 작업은 번들러에서 이루어진다. 리액트 팀은 RSC를 웹팩에서 사용할 수 있는 `react-server-dom-webpack`을 [webpack loader](https://github.com/facebook/react/blob/main/packages/react-server-dom-webpack/src/ReactFlightWebpackNodeLoader.js)나 [node-register](https://github.com/facebook/react/blob/main/packages/react-server-dom-webpack/src/ReactFlightWebpackNodeRegister.js) 에서 제공하고 있다.서버 컴포넌트가 `*.client.jsx` 파일에서 가져올 때, 실제로 import 하는 것이 아닌 파일 이름과 그 것을 참조하는 모듈 참조 객체만을 가져온다. 즉, 클라이언트 컴포넌트 함수는 서버에서 구성되는 리액트 트리의 구성요소가 아니었다.

위 예제를 `JSON` 트리로 직렬화 한다면 아래와 같을 것이다.

```javascript
{
  // ClientComponent 엘리먼트를 `module reference` 와 함께 placeholder로 배치
  $$typeof: Symbol(react.element),
  type: {
    $$typeof: Symbol(react.module.reference),
    name: "default",
    filename: "./src/ClientComponent.client.js"
  },
  props: {
    // 자식으로 ServerComponent가 넘어간다.
    children: {
      // ServerComponent는 바로 html tag로 렌더링됨
      $$typeof: Symbol(react.element),
      type: "span",
      props: {
        children: "Hello from server land"
      }
    }
  }
}
```

### 직렬화된 리액트 트리

![RSC placeholder](https://blog.plasmic.app/static/images/react-server-components-placeholders.png)

#### 모든 props는 직렬화되야 한다.

전체 리액트 트리를 JSON 직렬화하고 있기 때문에, 클라이언트 컴포넌트가 기본 html 태그에 전달하는 props도 직렬화 할 수 있어야 한다. 그러나 (당연하게도) 서버 컴포넌트에서는 이벤트 핸들러를 props 전달할 수 없다.

```jsx
// 서버 컴포넌트는 함수를 prop으로 넘겨줄 수 없다.
// 함수는 직렬화 할 수 없기 때문이다.
function SomeServerComponent() {
  return <button onClick={() => alert('OHHAI')}>Click me!</button>
}
```

그러나 여기서 유의할 점은, RSC 프로세스 중에 클라이언트 컴포넌트를 마주하게 된다면, 클라이언트 컴포넌트 함수를호출하거나 클라이언트 컴포넌트를 내림차순으로 정렬하지 않는다는 것이다. 그러므로, 다른 클라이언트 컴포넌트를 인스턴스화 하는 클라이언트 컴포넌트가 있는 경우,

```jsx
function SomeServerComponent() {
  return <ClientComponent1>Hello world!</ClientComponent1>;
}

function ClientComponent1({children}) {
  // 클라이언트에서는 가능
  return <ClientComponent2 onChange={...}>{children}</ClientComponent2>;
}
```

`ClientComponent2` 는 RSC 트리에서 나타나지 않는다. 대신, module reference 가 있는 엘리먼트와 `ClientComponent1`의 props만 볼 수 있다. 그러므로, `ClientComponent1`에 이벤트 핸들러가 있는 `ClientComponent2`를 자식으로 보내는 것은 안전하다.

### 3. 브라우저가 리액트 트리를 재구조화

브라우저는 서버로 부터 JSON 결과물을 받고, 이제 브라우저에서 렌더링될 리액트 트리를 재구성하기 시작한다. type이 `module reference`인 엘리먼트를 만날 때마다, 실제 클라이언트 컴포넌트 함수에 대한 참조로 대체를 시도할 것이다.

이 작업은 다시 번들러의 도움이 필요하다. 클라이언트 컴포넌트 함수의 기능을 서버 module reference로 대체해 주었던 것도 번들러였고, 이 module reference를 브라우저가 실제 클라이언트 컴포넌 함수로 대체하는 것을 아는 것도 번들러다.

이를 그림으로 구성하면 다음과 같다.

![RSC-client](https://blog.plasmic.app/static/images/react-server-components-client.png)

이제 이 트리를 렌더링하고, DOM에 커밋한다.

## Suspense에서도 같은 원리 일까?

`Suspense`에 대해서 간략하게 이야기 하자면, 아직 준비되지 않는 요소가 필요할 때 (데이터를 빠르게 가져오거나, 컴포넌트를 느리게 가져오는 등) 리액트 컴포넌트로에서 promises를 던질 수 있다. 이 promise는 `Suspense boundary`에서 잡을 수 있다. Suspense에서 하위 트리를 렌더링 할 때, promise가 던져질 때마다 리액트는 이 promise가 resolve 될 때 까지 리액트 하위 트리 렌더링을 일시 중지한 다음 다시 시도한다.

우리가 RSC 결과물을 만들기 위해 서버에서 서버 컴포넌트 함수를 호출 할 때, 이 함수들은 각자 필요한 데이터를 가져올 때 promise를 던질 수 있다. 그리고 [이 promise를 만나면](https://github.com/facebook/react/blob/42c30e8b122841d7fe72e28e36848a6de1363b0c/packages/react-server/src/ReactFlightServer.js#L416), 앞서와 마찬가지로 placeholder를 위치 시킨다. 그리고 이 promise가 resolve되면 서버 컴포넌트 함수를 다시 호출하고, 성공하면 이 완료된 청크를 내보낸다. 실제로 RSC 출력 스트림을 생성하고, promise가 나타나면 일시 중지하고, 이것이 resolve되면 추가적인 chunck를 스트리밍한다.

마찬가지로, 브라우저에서 `fetch` 함수 호출로 RSC JSON 결과물을 스트리밍하고 있다. 이 프로세스 역시 결과물에서 placeholder를 마주하거나 (서버에서 던진 promise를 맞닥뜨린 경우), 스트림에서 placeholder를 아직 보지 못한 경우 (https://github.com/facebook/react/blob/main/packages/react-client/src/ReactFlightClientStream.js) promise를 던지는 것으로 끝날 수도 있다. 또는 클라이언트 컴포넌트 module reference를 마주치지만, 아직 브라우저에 로드된 클라이언트 컴포넌트 함수를 가지고 있지 않은 경우에도 promise를 던질 수 있다.

Suspense를 활용하면 서버 컴포넌트가 데이터를 가져올 때, 서버 스트리밍 RSC출력을 사용할 수 있으며 브라우저가 데이터를 점진적응로 렌더링하고 필요에 따라 클라이언트 컴포넌트 번들을 동적으로 가져올 수 있다.

## RSC Wire format

그런데 정확히 서버가 어떤 형태의 데이터를 보내는 것일까? 정확시 어떤 데이터가 서버에서 브라우저로 스트리밍 되는 것일까?

꽤 간단한 형태로 구성되어 있다. 한줄에 JSON blob 데이터가 있고, 여기에 ID로 태그되어 있는 간단한 형식이다.

```
M1:{"id":"./src/ClientComponent.client.js","chunks":["client1"],"name":""}
J0:["$","@1",null,{"children":["$","span",null,{"children":"Hello from server land"}]}]
```

`M`으로 시작하는 라인은, 클라이언트 번들에서 컴포넌트 함수를 조회하는데 필요한 정보와 클라이언트 컴포넌트 module reference를 정의 한다.
`J`로 시작하는 줄은 앞서 `M`라인에서 정의된 클라이언트 컴포넌트를 참조하는 것으로, 실제 리액트 컴포넌트 element 트리를 정의한다.

이 형식의 포맷은 스트리밍으로 전송이 가능하다. 클라이언트가 전체 행을 읽는 즉시 JSON의 일부 구문을 분석하여 작업을 진행할 수 잇다. 서버가 렌더링하는 동안 suspense 바운더리에 도달한 경우, resolve시 각 청크에 해당하는 여러 `J`라인을 볼 수 있다.

아래 예제를 살펴보자.

```jsx
// Tweets.server.js
import { fetch } from 'react-fetch' // React's Suspense-aware fetch()
import Tweet from './Tweet.client'
export default function Tweets() {
  const tweets = fetch(`/tweets`).json()
  return (
    <ul>
      {tweets.slice(0, 2).map((tweet) => (
        <li>
          <Tweet tweet={tweet} />
        </li>
      ))}
    </ul>
  )
}

// Tweet.client.js
export default function Tweet({ tweet }) {
  return <div onClick={() => alert(`Written by ${tweet.username}`)}>{tweet.body}</div>
}

// OuterServerComponent.server.js
export default function OuterServerComponent() {
  return (
    <ClientComponent>
      <ServerComponent />
      <Suspense fallback={'Loading tweets...'}>
        <Tweets />
      </Suspense>
    </ClientComponent>
  )
}
```

위 예제에서, RSC 스트림은 아래와 같이 나타난다.

```
M1:{"id":"./src/ClientComponent.client.js","chunks":["client1"],"name":""}
S2:"react.suspense"
J0:["$","@1",null,{"children":[["$","span",null,{"children":"Hello from server land"}],["$","$2",null,{"fallback":"Loading tweets...","children":"@3"}]]}]
M4:{"id":"./src/Tweet.client.js","chunks":["client8"],"name":""}
J3:["$","ul",null,{"children":[["$","li",null,{"children":["$","@4",null,{"tweet":{...}}}]}],["$","li",null,{"children":["$","@4",null,{"tweet":{...}}}]}]]}]
```

`J0` 은 추가 자식 컴포넌트를 갖게 되었다. `Suspense` 바운더리의 하위 항목으로, `@3`을 가리킨다. 여기서 흥미로운 점은 `@3`은 아직 정의 되지 않았다는 것이다. 서버가 `tweets`를 완전히 로드 하면, `Tweet.client.js` 컴포넌트에 대한 module reference를 참조하는 `M4`행과 `@3`이 있는 위치로 스왑되어야 하는 다른 리액트 트리를 정의하는 `J3`행을 출력하게 된다. 그리고 `J3`의 자식들이 `M4`에 정의된 트윗 컴포넌트를 참조하고 있음)

한가지 또 주의 할점은, 번들러가 자동으로 `ClientComponent`와 `Tweet` 을 두개의 개별 번들로 나누어, 브라우저가 `Tweet`번들 다운로드를 나중으로 미룰 수 있다는 점이다.

### RSC Format 사용하기

이 `RSC` 스트림을 브라우저에서 실제 리액트 엘리먼트로 어떻게 전환할까? `react-server-dom-webpack`는 [진입점 (`entrypoints`)을](https://github.com/facebook/react/blob/main/packages/react-server-dom-webpack/src/ReactFlightDOMClient.js) 가지고 있는데 여기에서 RSC 응답을 받아 리액트 엘리먼트 트리를 다시 만든다.

```jsx
import { createFromFetch } from 'react-server-dom-webpack'
function ClientRootComponent() {
  // fetch() from our RSC API endpoint.  react-server-dom-webpack
  // can then take the fetch result and reconstruct the React
  // element tree
  const response = createFromFetch(fetch('/rsc?...'))
  return (
    <Suspense fallback={null}>
      {response.readRoot() /* Returns a React element! */}
    </Suspense>
  )
}
```

API 엔드 포인트에서 RSC 응답을 읽도록 `react-server-dom-webpack`에 요청한다. 그런다음, `response.readRoot()`는 응답 스트림이 처리될 때 업데이트 되는 react element를 반환한다. 스트림을 읽기에 앞서, 아직 클라이언트가 준비되지 않았으므로 promise가 반환될것이다. 리액트는 렌더링을 재개하지만, 아직 준비되지 않은 `@3` 참조가 발견되면 또다른 promise가 던져질 것이다. 그리고 `J3`을 읽게되면 그 promise가 resolve되고, 리액트는 다시 렌더링을 재개하여 이번에는 완료할 것이다. 따라서 RSC 응답을 스트리밍할때, Suspense 바운더리에 의해 정의된 청크로, 현재 가지고 있는 element 트리를 계속 업데이트하고 렌더링할 것이다.

### 왜 그냥 html을 내보내지 않는 것일까

왜 이렇게 새로운 포맷을 만들어서 번거롭게 처리하는 것일까? 클라이언트의 목표는 리액트 element 트리를 재구성하는 것이다. HTML 을 파싱하여 react element를 만드는 것보다 이 형식을 사용하는 것이 훨씬 쉽다. 이를 통해 DOM에 최소한의 커밋으로 리액트 트리에 대한 후속 업데이트를 병합할 수 있으므로, 리액트 트리를 효과적으로 재구성할 수 있는 방향으로 검토했을 것이다.

### 단순히 클라이언트 컴포넌트에서 데이터를 가져오는 것보다 더 나은걸까?

어차피 콘텐츠를 가져오기 위해 서버에 API 요청을 해야 한다면, 이것이 현재 자주 사용하고 있는 방식, 즉 데이터를 가져오고 클라이언트에서 렌더링하는 것보다 더 나은 방법인 걸까?

결론부터 이야기 하자면 화면에 렌더링하는 내용에 따라 다르다. RSC를 사용하면 denormalized된, 즉 이미 '처리된' 데이터를 사용자에게 직접 매핑할 수 있으므로, 가져오는 데이터의 작은 부분만 렌더링하거나 렌더링 자체가 브라우저로 다운로드 되는 것을 피하고 싶은 자바스크립트를 필요로 할 때 (렌더링 하는데 많은 자바스크립트 코드가 다운로드 되어야 하는 경우) 도움이 될 수 있다. 또한 렌더링시 서로 의존성이 얽혀 있는 여러 데이터를 가져와야 하는 경우, 브라우저 보다는 지연시간이 짧은 서버에서 처리해서 가져오는 것이 훨씬 나을수 있다.

### 서버사이드 렌더링?

React 18을 사용하면, SSR과 RSC를 모두 활용하여 서버에서 html을 생성하고, 브라우저에서 html을 RSC를 hydrate할 수도 있다. 이 주제에 대해서는 다음에 다뤄보자.

## 서버 컴포넌트의 렌더링 업데이트

예를 들어, 한 제품의 페이지를 보다가 다른 제품으로 넘어가는 경우와 같이, 새로운 내용을 렌더링 하기 위해 서버 컴포넌트가 필요한 경우 어떻게 해야할까?

렌더링 자체가 서버에서 수행되므로, RSC 형식의 새 콘텐츠를 가져오기 위해 서버에 다른 API 호출이 필요하다. 브라우저가 새 컨텐츠를 받으면, 새로운 리액트 element 트리를 구성하고, 이전 리액트 트리와 reconciliation 을 수행하여 DOM에 필요한 최소한의 업데이트를 파악할 수 있으며, 모든 상태와 이벤트 핸들러는 클라이언트 컴포넌트에 유지된다.

지금은 루트 서버 컴포넌트에서 전체 응답 트리를 다시 렌더링 해야 하지만, 앞으로는 하위 트리에 대해서만 이 작업을 수행하도록 할 수 있다.

## RSC에서 프레임워크를 사용해야 하는 이유는 무엇일까?

리액트 팀은 RSC가 처음에는 플레인 리액트 프로젝트 대신에 nextjs, shopify hydrogen과 같은 프레임워크를 사용해야 한다고 언급했다. 왜그랬을까?

그 이유는 개발자들의 편의성 때문이다. 프레임워크는 보다 쉽게 래퍼와 추상화를 제공하므로, 앞서 언급했던 것처럼 서버에서 RSC 스트림을 생성하거나, 브라우저에서 이를 사용할 준비같은 것을 할 필요가 없다. 이 프레임워크들은 또 SSR을 제공하며, 서버 컴포넌트를 사용할 경우 서버에서 생성된 html 에 hydrate도 제공해준다.

앞서 언급했던 것처럼, 브라우저 클라이언트 컴포넌트에 적절하게 데이터를 내려주고 사용하기 위해서는 번들러의 도움이 필수적이다. 이미 webpack에 내장되어 있으며, shopify에서는 [vite로 사용할 수 있는 준비](https://github.com/facebook/react/pull/22952)를 하고 있는 것 같다. RSC에 필요한 많은 부분이 npm에 공개 패키지로 올라와있지 않기 때문에, 이러한 플러그인 들은 리액트 저장소의 일부가 되어야 한다. 그러나 일단 개발되면, 프레임워크 없이도 사용가능해야 할 것이다.

## RSC는 지금 사용할 수 있을까요

- https://nextjs.org/docs/advanced-features/react-18
- https://hydrogen.shopify.dev/

앞서 언급한 프레임워크에서 사용은 가능하지만, 아직 프로덕션에서 사용하기에는 이른감이 있다.

그러나 RSC가 미래에 리액트에서 큰 부분을 차지할 것이라는 사실에는 의심의 여지가 없다. 더 빠른 페이지 로딩, 더 작은 자바스크립트 번들, 그리고 더 빠른 인터랙션에 대한 리액트의 결과로, 리액트를 활용하여 여러 페이지를 만드는 애플리케이션을 만드는 방법에 대한 보다 효과적인 방법론이 될 것이다.
