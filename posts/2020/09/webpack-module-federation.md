---
title: Webpack Module Federation에 대해 알아보자
tags:
  - javascript
  - webpack
published: true
date: 2020-09-16 17:42:06
description: '자바스크립트 아키텍쳐의 게임체인저라고 하는데, 과연 그렇게 될 수 있을까?'
category: javascript
template: post
---

[이 글](https://indepth.dev/webpack-5-module-federation-a-game-changer-in-javascript-architecture/)을 위주로 번역한 글이며, 추가적으로 micro frontend에 대한 개념도 넣어두었습니다.

## Table of Contents

> Module Federation은 서버, 클라이언트 모두에서 자바스크립트 애플리케이션을 다른 번들/빌드로 부터 코드를 다이나믹하게 실행해준다.

이는 자바스크립트 번들러를 아폴로가 GraphQL에서 하는 것과 동일한 기능을 할 수 있게 해준다. (근데 내가 안써봐서 모름)

서로 독립된 애플리케이션 간에 코드를 공유할 수 있는 확장 가능한 솔루션은 여지껏 편리하지 않았으며, 확장은 불가능에 가까웠다. 코드를 공유하는 것도 번거롭고, 실제로 애플리케이션은 독립적이지도 않았으며, 종속성 또한 제한되어 있었다. 또한 실제 코드 형상이나 컴포넌트를 별도로 묶은 애플리케이션 간 공유는 불가능에 가까웠고, 비생산적이었으며, 수익성이 없었다.

Module Federation은 자바스크립트 애플리케이션이 다른 애플리케이션에서 동적으로 코드를 불러오고, 그 과정에서 종속성을 가질 수 있도록 허용한다. 만약 모듈을 사용하는 애플리케이션이 Module Federation에서 필요로 하는 종속성을 가지고 있지 않다면, 웹팩은 Module Entry Point에서 누락된 종속성을 다운로드 할 것이다.

코드는 가능하면 공유되지만, fallback은 각각의 케이스에 별도로 존재한다. Federated된 코드는 항상 종속성을 불러올수 있지만, 더 많은 페이로드를 다운로드 하기전에 사용하는 측의 종속성을 사용하려고 시도한다. 이말은, 획일적인 웹팩의 빌드 처럼 코드 중복과 의존성 공유가 적다는 것을 의미한다.

### 용어

- Module Federation: Apollo GraphQL federation 과 같은 개념을, 자바스크립트 모듈에 녹였다고 보면 된다. 브라우저와 node에서 사용가능하다.
- Host: 첫 페이지 로딩을 담당하는 웹팩 빌드 (`onLoadEvent`를 트리거하는)
- Remote: host에 의해 사용되는 다른 웹팩 빌드, host의 일부다.
- Bi-directional hosts: 번들 또는 웹팩 빌드가 호스트 또는 리모트로 작동할 수 있는 때. 런타임시에 각각 서로를 실행할 수 있다.

애플리케이션은 bi-directional host가 될 수 있다. 즉, 어떤 애플리케이션이든 최초에 로딩된다면, host가 될 수 있다. 예를 들어, 애플리케이션에서 라우팅이 바뀌거나 다른 페이지로 이동할 경우, federated된 모듈을 불러오게 되는데, 이는 dynamic import와 마찬가지로 동작하게 된다. 만약에 이동한 페이지에서 새로고침한다면, 그 지점이 host가 된다.

정리해서 예를 들어보자.

웹 애플리케이션에서 home 페이지에 들어갔다면, 이는 host가 된다. 만약 about 페이지로 간다면, host는 별도로 독립된 애플리케이션(about)을 동적으로 임포트 하게 된다. 이는 메인 entry point나 전체 애플리케이션을 로딩하는 것이 아니다. 단지 몇 kb 코드만 불러올 뿐이다. 만약 현재 상태에서 새로고침을 하면 about 페이지는 host가 되고, 뒤로가기를 시도할 경우 앞서 host였던 home을 remote로 불러오게 된다. 모든 애플리케이션은 리모트나 호스트가 될 수 있으며, 다른 federated된 모듈에 의해 실행 될 수 있다.

## federated application 만들어보기

`./src/App`을 `app_one_remote`라고 선언하였다. 이는 다른 애플리케이션에서 실행될 수 있다.

```javascript
const HtmlWebpackPlugin = require('html-webpack-plugin')
const ModuleFederationPlugin = require('webpack/lib/container/ModuleFederationPlugin')

module.exports = {
  // other webpack configs...
  plugins: [
    new ModuleFederationPlugin({
      name: 'app_one_remote',
      remotes: {
        app_two: 'app_two_remote',
        app_three: 'app_three_remote',
      },
      exposes: {
        AppContainer: './src/App',
      },
      shared: ['react', 'react-dom', 'react-router-dom'],
    }),
    new HtmlWebpackPlugin({
      template: './public/index.html',
      chunks: ['main'],
    }),
  ],
}
```

애플리케이션 헤드에, `app_one_remote.js`를 불러오도록 했다. 이렇게 하면 다른 웹팩 런타임에 연결되고, 런타임에 오케이스트레이션 계층을 프로비저닝 할 수 있다. 이는 특별히 설계된 웹팩 런타임과 진입점이다. 이는 일반적인 애플리케이션 진입점과 다르게, 몇 kb에 불과하다.

```html
<head>
  <script src="http://localhost:3002/app_one_remote.js"></script>
  <script src="http://localhost:3003/app_two_remote.js"></script>
</head>
<body>
  <div id="root"></div>
</body>
```

`App One`에서 `App Two`에 있는 코드를 사용하고 싶다면,

```javascript
const Dialog = React.lazy(() => import('app_two_remote/Dialog'))

const Page1 = () => {
  return (
    <div>
      <h1>Page 1</h1>
      <React.Suspense fallback="Loading Material UI Dialog...">
        <Dialog />
      </React.Suspense>
    </div>
  )
}

export default Page1
```

라우터는 일반적인 표준과 비슷하다.

```javascript
import { Route, Switch } from 'react-router-dom'

import Page1 from './pages/page1'
import Page2 from './pages/page2'
import React from 'react'

const Routes = () => (
  <Switch>
    <Route path="/page1">
      <Page1 />
    </Route>
    <Route path="/page2">
      <Page2 />
    </Route>
  </Switch>
)

export default Routes
```

App Two에서는 Dialog를 내보낼 것이며, 이는 위에서 봤던 것 처럼 App One에서 사용한다.

```javascript
const HtmlWebpackPlugin = require('html-webpack-plugin')
const ModuleFederationPlugin = require('webpack/lib/container/ModuleFederationPlugin')
module.exports = {
  plugins: [
    new ModuleFederationPlugin({
      name: 'app_two_remote',
      filename: 'remoteEntry.js',
      exposes: {
        Dialog: './src/Dialog',
      },
      remotes: {
        app_one: 'app_one_remote',
      },
      shared: ['react', 'react-dom', 'react-router-dom'],
    }),
    new HtmlWebpackPlugin({
      template: './public/index.html',
      chunks: ['main'],
    }),
  ],
}
```

루트 앱은 이런 모양이다.

```javascript
import React from 'react'
import Routes from './Routes'
const AppContainer = React.lazy(() => import('app_one_remote/AppContainer'))

const App = () => {
  return (
    <div>
      <React.Suspense fallback="Loading App Container from Host">
        <AppContainer routes={Routes} />
      </React.Suspense>
    </div>
  )
}

export default App
```

```javascript
import React from 'react'
import { ThemeProvider } from '@material-ui/core'
import { theme } from './theme'
import Dialog from './Dialog'

function MainPage() {
  return (
    <ThemeProvider theme={theme}>
      <div>
        <h1>Material UI App</h1>
        <Dialog />
      </div>
    </ThemeProvider>
  )
}

export default MainPage
```

`App Three`의 경우, `<App>`에서 실행되는 것이 없이 독립되어 있으므로, 아래와 같이 처리하면 된다.

```javascript
new ModuleFederationPlugin({
  name: "app_three_remote",
  library: { type: "var", name: "app_three_remote" },
  filename: "remoteEntry.js",
  exposes: {
    Button: "./src/Button"
  },
  shared: ["react", "react-dom"]
}),
```

트위터에 제작자가 공유해준 실제 코드를 살펴보자.

네트워크 탭을 살펴보면, 세 코드가 모두 다른 번들에 존재하고 있음을 알 수 있다.

의존성 중복이 존재하지 않는다. `shared` 옵션에 나와있듯, `remote`는 `host`의 의존성에 의존하게 된다. 만약 호스트에 해당 의존성이 존재하지 않는다면, 리모트는 알아서 다운로드 할 것이다.

`vendor`나 다른 모듈을 `shared`에 추가하는 것은 확장성에 그다지 좋지 못하다. `AutomaticModuleFederationPlugin`를 제공하여, 웹팩 코어 외부에 있는 코드들을 관리할 수 있도록 할 것이다.

## Server Side Rendering

Module Federation은 브라우저 node 모든 환경에서 동작한다. 단지 서버 빌드가 commonjs 라이브러리 타겟을 사용하기만 하면 된다.

```javascript
module.exports = {
  plugins: [
    new ModuleFederationPlugin({
      name: 'container',
      library: { type: 'commonjs-module' },
      filename: 'container.js',
      remotes: {
        containerB: '../1-container-full/container.js',
      },
      shared: ['react'],
    }),
  ],
}
```

Module Federation에 대한 다양한 예제를 아래에서 살펴볼 수 있다.

- https://github.com/module-federation/module-federation-examples
- https://github.com/module-federation/next-webpack-5
- https://github.com/ScriptedAlchemy/mfe-webpack-demo
- https://github.com/ScriptedAlchemy/webpack-external-import

> 결과적으로 하나의 큰 애플리케이션을 여러개의 독립된 애플리케이션으로 만든 다음, 다이나믹 로딩을 하듯이 필요한 순간에 필요한 컴포넌트 (소스)를 불러오게 한다는 개념인 것 같다. webpack5 에 포함될 예정이라고 하니, 정식 출시 될 때 실제 동작하는 예제를 만들어보고 고민해봐야겠다.
