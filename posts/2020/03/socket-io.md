---
title: Socket.IO 공부하기 (1)
tags:
  - javascript
  - typescript
  - react
published: true
date: 2020-03-22 03:17:25
description:
  '## WebSocket 웹은 전형적으로 HTTP 요청에 대한 HTTP 응답을 받고, 이에 따라 브라우저 화면을 새로
  만드는 방식이다. 따라서 데이터 통신은 요청과 응답이 한 쌍으로 묶여왔다. 그러나 웹 페이지가 보다 쉽게 상호작용을 하려면, 브라우저와 웹
  사이에 이러한 요청 - 응답 방식이 아닌 더 자유로운 양방향 메시지 송수신 기술이 필요하다. 이러한 ...'
category: javascript
slug: /2020/03/socket-io/
template: post
---

## WebSocket

웹은 전형적으로 HTTP 요청에 대한 HTTP 응답을 받고, 이에 따라 브라우저 화면을 새로 만드는 방식이다. 따라서 데이터 통신은 요청과 응답이 한 쌍으로 묶여왔다. 그러나 웹 페이지가 보다 쉽게 상호작용을 하려면, 브라우저와 웹 사이에 이러한 요청 - 응답 방식이 아닌 더 자유로운 양방향 메시지 송수신 기술이 필요하다. 이러한 요구를 충족하기 위해 HTML5에서 표준안의 일부로 WebSocket이 등장하였다.

- 문서보기: https://html.spec.whatwg.org/multipage/web-sockets.html
- Caniuse: https://caniuse.com/#feat=websockets

## Socket.io

https://github.com/socketio/socket.io

Socket.io는 WebSocket이 나올 당시 (2011년 쯤?) 모든 브라우저가 지원하지는 않았으므로, 대다수의 브라우저에 WebSocket 기능을 사용할 수 있도록 구현한 라이브러리다. Github의 used by 로 봐서는 요즘에도 많이 쓰고 있는 것 같다.

### Express를 활용한 기본적인 예제

#### 1. 기본설정

먼저 npm에서 socket.io를 설치한다.

```javascript
var app = require('express')()
var http = require('http').createServer(app)
var io = require('socket.io')(http)

// index.html을 서빙한다
app.get('/', function (req, res) {
  res.sendFile(__dirname + '/index.html')
})

// 'connection' 이라는 이벤트를 감지한다.
io.on('connection', function (socket) {
  console.log('a user connected')
})

// http를 3000포트에서 실행한다.
http.listen(3000, function () {
  console.log('listening on *:3000')
})
```

이제 localhost:3000 으로 접속하면 아래와 같은 로그를 확인할 수 있다.

```
listening on *:3000
a user connected
```

연결 외에 연결 종료를 감지하면 아래와 같이 코드를 추가한다.

```javascript
io.on('connection', function (socket) {
  console.log('a user connected')
  socket.on('disconnect', function () {
    console.log('user disconnected')
  })
})
```

다시 실행하고, 페이지를 닫으면 이제 아래와 같이 로그가 찍힌다.

```
listening on *:3000
a user connected
user disconnected
a user connected
user disconnected
```

#### 2. 이벤트 보내기

이제 클라이언트에서 이벤트를 보내보자. 기본적으로 보낼 수 있는 객체는 JSON형태이며, binary data도 가능하다.

```html
<script src="https://code.jquery.com/jquery-1.11.1.js"></script>
<script>
  $(function () {
    var socket = io()
    $('form').submit(function (e) {
      e.preventDefault() // prevents page reloading
      socket.emit('chat message', $('#m').val())
      $('#m').val('')
      return false
    })
  })
</script>
```

```javascript
io.on('connection', function (socket) {
  socket.on('chat message', function (msg) {
    console.log('message: ' + msg)
  })

  socket.on('disconnect', function () {
    console.log('user disconnected')
  })
})
```

```
> node index.js

listening on *:3000
message: 와
message: 이렇게 메시지가 가는구나
message: 신기하네
```

#### 3. 브로드캐스팅

브로드 캐스팅은 서버에서 현재 connection으로 접속한 모든 유저에게 이벤트를 보내는 것이다. `io.emit`을 활용하면 된다.

```javascript
io.on('connection', function (socket) {
  socket.on('chat message', function (msg) {
    // chat message를 보낸 사용자를 제외한 모든 사용자에게 emit
    // socket.broadcast.emit(msg)
    // 그냥 모든 사용자에게 emit
    io.emit('chat message', msg)
  })

  socket.on('disconnect', function () {
    console.log('user disconnected')
  })
})
```

이제 클라이언트 사이드에서 'chat message' 를 감지한다.

```html
<script>
  $(function () {
    var socket = io()
    $('form').submit(function (e) {
      e.preventDefault() // prevents page reloading
      socket.emit('chat message', $('#m').val())
      $('#m').val('')
      return false
    })
    socket.on('chat message', function (msg) {
      $('#messages').append($('<li>').text(msg))
    })
  })
</script>
```

![chat-example](./images/chat-example.png)

기본적인 채팅기능은 만들었지만, 실제 활용하기엔 조금 거리가 있다. 소켓서버와 채팅서버가 같이있고, 채팅방도 단 하나 뿐이다. 다음 예제에서는 koa와 함께 소켓서버를 따로 구축하고, 채팅 (frontend)과 분리해서 여러개의 채팅방을 만드는 예제를 해보려고 한다.
