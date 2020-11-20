---
title: '서버 사이드 이벤트 (Server Side Events, SSE)'
tags:
  - javascript
  - nodejs
  - browser
  - web
published: true
date: 2020-11-17 23:41:26
description: '이거 꼭 한번 해보고 싶었는데 😭'
---

## Server Side Events

일반적이고 전통적인 웹페이지의 경우, 새로운 데이터를 받기 위해서는 서버에 데이터 요청을 해야만 한다. 이른바 폴링이라는 기술로, 웹페이지가 서버에 요청을 해야만, 서버가 그 요청에 따른 데이터를 적절하게 리턴해주는 방식이라고 볼 수 있다. 하지만 Server Side Events, 이하 (SSE)를 활용하면, 웹페이지가 별도로 요청하지 않아도 서버가 데이터를 보내는 것이 가능하다. 즉, 서버에서 클라이언트로 업데이트 되는 내용을 스트리밍을 하는 것이 가능해진다. SSE를 활용하면, 서버와 클라이언트 사이에 단방향 채널을 여는 것과 같은 이점을 얻르 수 있다.

## vs Web Socket?

그렇다면 우리가 일반적으로 알고 있는 web socket 과의 차이는 무엇일까? 웹 소켓은 양반향 통신을 위한 프로토콜은 제공하지만 (채팅과 같은), 일부 시나리오에서는 그러한 양방향 통신이 필요하지 않을 때가 있다. 클라이언트에서 굳이 데이터를 전송하지 않고, 서버의 데이터만 클라이언트에 보내서 업데이트를 해야하는 경우가 있다. (긴 시간이 걸리는 요청에 대해서 요청을 일부분씩 나눠서 보내는 등) 이러한 경우에는 Web Socket보다는 SSE가 훨씬 더 좋은 대안이 될 수 있다. 또한 웹 소켓과는 다르게, 전통적인 HTTP로도 전송이 가능하다. 즉, 특별한 프로토콜이나 서버구현이 필요하지 않다.

## How to use

### Support

대다수의 모던 브라우저가 지원하는 반면, 아쉽게도 역시나 우리의 IE는 SSE를 지원하지 않는다.

https://caniuse.com/eventsource

폴리필을 사용하면 될 것 같다. (써보진 않았지만) https://github.com/Yaffle/EventSource

### Javascript API

이벤트 스트림을 구독하기 위해서는, EventSource를 만들고 URL을 넘겨야 한다.

```javascript
if (!!window.EventSource) {
  var source = new EventSource('stream.php')
} else {
  // SSE를 사용할 수 없는 환경
}
```

만약 URL이 절대 주소로 되어 있다면, 호출 페이지와 scheme, domain, port 등이 일치해야 한다.

이제 소스에 이벤트 핸들러를 달아서 실제로 구독을 해보자.

```javascript
source.addEventListener(
  'message',
  function (e) {
    console.log(e.data)
  },
  false,
)

source.addEventListener(
  'open',
  function (e) {
    // 연결성공
  },
  false,
)

source.addEventListener(
  'error',
  function (e) {
    if (source.readyState == EventSource.CLOSED) {
      // 연결이 닫히는 경우
    }
  },
  false,
)
```

서버에서 데이터를 푸쉬하면, `message`가 실행되고, `e.data`에서 데이터를 가져올 수 있다.

소스의 이벤트 스트림은 SSE 형식인 `Content-type` `text/event-stream`을 사용하여 텍스트 응답을 작성해야 한다. 기본적인 응답형식은 아래와 같다.

```bash
data: response \n\n
```

`data:`행 다음에 메시지가 오고, 스트림 맨 마지막에는 `\n` 문자가 두개 있다면 스트림이 끝난 것으로 간주한다.

메시지가 길어서 여러줄을 보내야 한다면, `data:`행을 사용하여 메시지를 분할하면 된다.

```bash
data: first response\n
data: second response\n\n
```

`\n`으로 하나만 줄바꿈이 되어 있다면, `message`이벤트는 하나만 발생한다.

JSON 데이터를 보내야 한다면 어떻게 할까?

```bash
data: {\n
data: "msg": "hello world",\n
data: "id": 12345\n
data: }\n\n
```

```javascript
source.addEventListener(
  'message',
  function (e) {
    var data = JSON.parse(e.data)
    console.log(data.id, data.msg)
  },
  false,
)
```

물론 json 데이터를 압축해서 한줄로 보내도 가능할 것이다.

이벤트에 ID를 달아서 고유한 ID도 함께 보낼 수 있다.

```bash
id: 123\n
data: hello\n
data: world\n
```

ID를 설정하게 되면, 브라우저는 마지막에 발생한 이벤트를 추적할 수 있게 된다. 이는 만약 연결이 끊겼을 때, `Last-Event-ID`라고 불리는 특별한 HTTP 헤더가 새 요청으로 설정된다. 이는 브라우저가 어떤 이벤트를 발생하기에 적합한지 판단할 수 있게 해준다. 이 메시지 이벤트네는 `e.lastEventId` 속성이 포함되어 있다.

브라우저는 각 연결이 종료 된 후에 3초후에 다시 연결을 시도하려고 한다. 여기에 `retry:`를 시간과 함께 설정하여, 이 시간제한을 변경할 수 있다.

```bash
retry: 10000\n
data: hello world\n\n
```

이렇게 하게 되면 10초 후에 다시 연결을 시도하게 된다.

하나의 이벤트 소스에 이벤트 이름을 넣어두면, 여러가지 이벤트를 생성할 수 있다. `event:`로 시작하는 행에 이벤트 명을 명시하는 경우, 그 이벤트를 해당 이름에 바인딩 시킬 수 있다. 클라이언트에서는 이벤트 리스너를 설정하여 해당 이벤트를 구독할 수 있다.

```bash
data: {"msg": "First message"}\n\n
event: userlogon\n
data: {"username": "John123"}\n\n
event: update\n
data: {"username": "John123", "emotion": "happy"}\n\n
```

```javascript
source.addEventListener(
  'message',
  function (e) {
    var data = JSON.parse(e.data)
    console.log(data.msg)
  },
  false,
)

source.addEventListener(
  'userlogon',
  function (e) {
    var data = JSON.parse(e.data)
    console.log('User login:' + data.username)
  },
  false,
)

source.addEventListener(
  'update',
  function (e) {
    var data = JSON.parse(e.data)
    console.log(data.username + ' is now ' + data.emotion)
  },
  false,
)
```

## 예제

```javascript
const koa = require('koa')
const Router = require('koa-router')

const router = new Router()

router.get('/event', async (ctx) => {
  ctx.res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-store',
    'Access-Control-Allow-Origin': '*',
  })

  const lastEventId =
    Number(ctx.request.headers['last-event-id']) || Number(ctx.query.id) || 100
  let timeoutId = 0
  let i = lastEventId
  let c = i + 100

  let f = function () {
    if (++i < c) {
      ctx.res.write(`id: ${i} \n`)
      ctx.res.write(`data: ${i} \n\n`)
      timeoutId = setTimeout(f, 1000)
    } else {
      ctx.res.end()
    }
  }

  f()

  ctx.res.on('close', function () {
    clearTimeout(timeoutId)
  })
})

router.get('/', async (ctx) => {
  ctx.res.write(`<!DOCTYPE html>
  <html>
  <head>
      <meta charset="utf-8" />
      <title>EventSource example</title>
      <meta http-equiv="X-UA-Compatible" content="IE=edge">
      <script>
        var es = new EventSource("/event?id=50");        
        
        es.addEventListener('message', function(e) {
          var div = document.createElement("div");     
          div.appendChild(document.createTextNode('>>' + e.data));
          document.body.appendChild(div);
        }, false);
        
        es.addEventListener('open', function(e) {          
          var div = document.createElement("div");     
          div.appendChild(document.createTextNode("SSE connected!"));
          document.body.appendChild(div);
        }, false);
        
        es.addEventListener('error', function(e) {
          console.log('failed')
        }, false);
      </script>
  </head>
  <body>
  </body>
  </html>`)
})

async function main() {
  const app = new koa()

  app.use(router.routes()).use(router.allowedMethods())

  app.listen(3001)
}

try {
  main()
} catch (err) {
  console.error(err)
}
```

결과

```bash
SSE connected!
>>51
SSE connected!
>>52
SSE connected!
>>53
SSE connected!
>>54
SSE connected!
>>55
SSE connected!
>>56
SSE connected!
>>57
SSE connected!
>>58
SSE connected!
>>59
SSE connected!
>>60
SSE connected!
>>61
SSE connected!
>>62
SSE connected!
>>63
SSE connected!
>>64
SSE connected!
>>65
SSE connected!
>>66
SSE connected!
>>67
SSE connected!
>>68
SSE connected!
...
```
