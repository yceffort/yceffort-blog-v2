---
title: Typescript, React, NextJs, Koa, Styled Component 로 프론트엔드 환경 만들기
date: 2019-06-21 04:07:40
published: true
tags:
  - react
  - javascript
  - typescript
description: '이 문서는 더 이상 업데이트 하지 않을 생각이다. 대신
  https://github.com/yceffort/koa-nextjs-react-typescript-boilerplate 여기에서 계속 해서
  만들어 가고 있다. ## 사용한 오픈소스  ### React  자세한 설명은 생략 한다  ###
  Nextjs  [NextJs](https://nextjs.org...'
category: react
slug: /2019/06/20/react-nextjs-koa/
template: post
---

이 문서는 더 이상 업데이트 하지 않을 생각이다. 대신 https://github.com/yceffort/koa-nextjs-react-typescript-boilerplate 여기에서 계속 해서 만들어 가고 있다.

## 사용한 오픈소스

### React

자세한 설명은 생략 한다

### Nextjs

[NextJs](https://nextjs.org/) 리액트에서 서버사이드 렌더링을 할 수 있도록 해주는 프레임워크다. angular나 react 등은 SPA라서 불편한 점이 더러 있는데, React에서 NextJS를 활용하면 react를 ssr(server side rendering)이 되도록 바꿔줄 수 있다. 그리고 자동으로 code splitting이 되고, 파일 시스템을 기준으로 라우팅이 되며, .. 뭐 이런저런 장점이 있다.

### koa

express를 만든 개발자들이 따로 떨어져 나와서 만든 web framework가 바로 koa다. express와 비교했을 때는 koa가 비교적 가볍고, node.js v7의 async/await 를 자유자재로 쓸 수 있다는 데 있다. 그리고 es6를 도입해서 `generator`도 사용할 수 있다. IBM이 express를 인수해버린 관계로, 많은 개발자들이? koa로 넘어가는 추세라고 하는데, 아직은 잘 모르겠다.

### Styled Component

[Styled Component](https://www.styled-components.com/)

## 시작

### package.json

```json
{
  "name": "hello-world",
  "version": "0.0.1",
  "description": "hello-world",
  "main": "main.js",
  "scripts": {
    "build": "tsc --outDir dist server/index.ts && next build",
    "start": "NODE_ENV=production node dist",
    "dev": "concurrently 'tsc -w --outDir dist server/index.ts' 'npm run watch-server -- --delay 2'",
    "watch-server": "nodemon --exec 'node dist' --watch dist -e '*'"
  },
  "author": "",
  "license": "UNLICENSED",
  "dependencies": {
    "@zeit/next-typescript": "^1.1.1",
    "@zeit/next-css": "^1.0.1",
    "@zeit/next-stylus": "^1.0.1",
    "formik": "^1.5.7",
    "isomorphic-fetch": "^2.2.1",
    "koa": "^2.7.0",
    "koa-body": "^4.1.0",
    "koa-bodyparser": "^4.2.1",
    "koa-morgan": "^1.0.1",
    "koa-mount": "^4.0.0",
    "koa-proxies": "^0.8.1",
    "koa-router": "^7.4.0",
    "next": "^8.1.0",
    "react": "^16.8.6",
    "react-dom": "^16.8.6",
    "styled-components": "^3.4.10"
  },
  "devDependencies": {
    "@types/isomorphic-fetch": "0.0.35",
    "@types/koa": "^2.0.48",
    "@types/koa-bodyparser": "^4.3.0",
    "@types/koa-morgan": "^1.0.4",
    "@types/koa-mount": "^3.0.1",
    "@types/koa-router": "^7.0.40",
    "@types/next": "^8.0.5",
    "@types/node": "^12.0.4",
    "@types/react": "^16.8.22",
    "babel-eslint": "^10.0.1",
    "babel-plugin-styled-components": "^1.10.0",
    "concurrently": "^4.1.0",
    "nodemon": "^1.19.1",
    "npm": "^6.9.0",
    "typescript": "^3.5.1"
  }
}
```

### ./typings/koa-proxies/index.d.ts

애석하게도 `koa-proxies`의 typing이 존재하지 않는다. `./typings/koa-proxies`에 아래와 같이 추가하자.

```typescript
declare module 'koa-proxies' {
  import {Middleware} from 'koa'
  namespace koaProxies {}
  function koaProxies(name: string, options?: any): Middleware
  export = koaProxies
}
```

타입스크립트로 `nextjs`를 사용하기 위하여 `@zeit/next-typescript`를 사용하였다.

### ./next.config.js

별도의 설정은 넣지 않았다.

```javascript
const withCSS = require('@zeit/next-css')
const withStylus = require('@zeit/next-stylus')
const withTypescript = require('@zeit/next-typescript')

module.exports = withTypescript(
  withStylus(
    withCSS({
      webpack: (config) => ({
        ...config,
        plugins: [...(config.plugins || [])],
        node: {
          fs: 'empty',
        },
      }),
    }),
  ),
)
```

### ./.babelrc

```json
{
  "presets": ["next/babel", "@zeit/next-typescript/babel"]
}
```

### ./pages/index.tsx

nextjs의 유일한 제약은 pages 폴더다. pages에 렌더링 할 페이지를 만들어 둬야 한다.

```typescript
import * as React from 'react'
import styled from 'styled-components'

const MainHeading = styled.div`
  font-size: 50px;
  color: red;
`

export default class IndexPage extends React.PureComponent {
  render() {
    return <MainHeading>hello?</MainHeading>
  }
}
```

### ./server/index.ts

가장 중요한 서버 부분이다. koa를 사용한 이유는 `*/api/*`로 요청이 오는 호출에 대해서는 외부에 있을지도 모르는 api서버를 활용하기 위함이다. 이를 별도로 처리 하지 않는다면 CORS이슈가 있을수 있기 때문이다. 그래서 `koa`를 통해서 `nextjs`를 호출하는 방식으로 바꾸었다.

```typescript
import * as next from 'next'
import * as Koa from 'koa'
import * as morgan from 'koa-morgan'
import * as Router from 'koa-router'
import * as proxy from 'koa-proxies'
import * as bodyparser from 'koa-bodyparser'
import * as mount from 'koa-mount'

const isDev = process.env.NODE_ENV !== 'production'

function renderNext(nextApp: next.Server, route: string) {
  return (ctx: Koa.Context) => {
    ctx.res.statusCode = 200
    ctx.respond = false

    nextApp.render(ctx.req, ctx.res, route, {
      ...((ctx.request && ctx.request.body) || {}),
      ...ctx.params,
      ...ctx.query,
    })
  }
}

async function main() {
  const nextApp = next({isDev})
  const app = new Koa()
  const router = new Router()

  await nextApp.prepare()
  const handle = nextApp.getRequestHandler()

  router.get('/', renderNext(nextApp, '/index'))

  app
    .use(morgan('combined'))
    .use(bodyparser())
    .use(
      proxy('/api', {
        target: 'https://jayg-api-request.test.com',
        rewrite: (path: string) => path.replace(/^\/api/, ''),
        changeOrigin: true,
      }),
    )
    .use(
      mount('/health', (ctx: Koa.Context) => {
        handle(ctx.req, ctx.res)
        ctx.status = 200
      }),
    )
    .use(router.routes())
    .use(
      mount('/', (ctx: Koa.Context) => {
        handle(ctx.req, ctx.res)
        ctx.respond = false
      }),
    )
    .listen(3000)
}

main()
```
