---
title: 타입스크립트로 koa 미들웨어 만들기
tags:
  - typescript
  - javascript
published: true
date: 2020-04-15 05:46:43
description: 'koa 미들웨어 만들기'
category: typescript
slug: /2020/04/koa-middleware-with-typescript/
template: post
---

```typescript
export async function MyMiddleware(
  ctx: Koa.Context,
  next: (ctx: Koa.Context) => Promise<any>,
) {
  console.log('first middleware started..')

  // ctx를 조작하여 인증등의 옵션을 처리할 수 있다.
  const {
    header: { auth },
  } = ctx

  if (auth === 'foo') {
    ctx.state.user = user
  } else {
    // 401
    ctx.status = 401
    // 다음 미들웨어로 넘어가지 못하고 끝나게 된다.
    return
  }

  // 다음 미들웨어로 넘어간다.
  await next(ctx)

  console.log('first middleware finished..')
}
```

이런 미들웨어를 활용해서 logger를 만들 수도 있다.

expressjs의 경우 https://github.com/expressjs/morgan 가 있고,

koa를 활용할 경우 https://github.com/koa-modules/morgan 를 활용하면 된다.
