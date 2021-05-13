---
title: 'HTTP1, HTTP2, HTTP3'
tags:
  - http
  - web
published: true
date: 2021-05-12 21:04:16
description: '공부할게 정말 많습니당'
---

## HTTP/1.1 vs HTTP/2

HTTP(Hypertext Transfer Protocol)는 1989년에 만들어진 월드 와이드 웹의 통신 표준이다. 1997년 HTTP/1.1이 릴리즈된 이래로 프로토콜이 거의 수정된 적이 없었다. 그러나 2015년에는 HTTP/2 버전이 나오면서 사용되기 시작했다. 이 버전에서는 모바일 플랫폼, 서버 집약적인 그래픽과 비디오를 다룰 때 레이턴시를 줄일 수 있는 여러가지 방법을 제공하였다. HTTP/2는 그 이후로 점점더 많은 인기를 얻어서 현재는 모든 웹사이트의 1/3 정도가 HTTP/2를 지원하는 것으로 알려져 있다. 

이 글에서는, HTTP/1.1과 HTTP2의 차이점을 알아보고, HTTP/2가 보다 효율적인 웹 프로토콜을 만들기 위해 채택한 기술적 변화를 살펴보자.

### 배경

HTTP/2가 HTTP/1.1에서 변경된 내용을 살펴보려면, 각 버전 별로 과거 어떻게 개발되었는지를 살펴볼 필요가 있다.

### HTTP/1.1

1989년 Timothy Berners-Lee가 월드 와이드 웹의 통신 표준으로 개발한 HTTP는 클라이언트 컴퓨터와 로컬 또는 원격 웹 서버간에 정보를 교환하는 최상위 애플리케이션 프로토콜이다. 이 프로세스에서는 클라이언트는 `GET` `POST`와 같은 메소드를 호출하여 텍스트 기반 요청을 보낸다. 이에 대한 응답으로 서버는 HTML 페이지나 리소스를 클라이언트를 보낸다.

예를 들어, www.example.com 도메인 웹사이트로 방문한다고 가정해보자. 이 URL로 이동하면, 웹 브라우저가 텍스트 기반 메시지 형식으로 HTTP 요청을 날린다.

```
GET /index.html HTTP/1.1
Host: www.example.com
```

이 요청은 `GET` 메소드를 사용하였으며, `Host:`뒤에 나열된 서버에 데이터를 요청한다. 이 요청에 대한 응답으로, `example.com` 서버는 이미지, 스타일시트, 기타 HTML 의 리소스 등과 함께 HTML 페이지를 리턴한다. 한가지 알아둬야 할 것은, 첫 번째 데이터 호출 시에 모든 리소스가 클라이언트로 반환되는 것은 아니다. 웹 브라우저가 화면의 HTML 페이지를 렌더링하는데 필요한 모든 데이터를 받을 떄까지 요청과 응답이 서버와 클라이언트를 왔다갔다 한다.

이런 요청과 응답의 교환은 transfer layer(일반적으로 TCP)와 network layer (IP) 의 최상단에 위치한 인터넷 프로토콜 스택의 단일 애플리케이션 계층으로 생각할 수 있다.

![Protocol Stack](https://assets.digitalocean.com/articles/cart_63893/Protocol_Stack.png)

### HTTP/2

HTTP/2는 압축, 멀리플렉싱, 우선순위 지정 등의 기술을 사용하여, 웹 페이지 로드 레이턴시를 줄이려는 목적으로 구글에서 개발한 SPDY 프로토콜로 시작된 기술이다. 그 이후로 2015년 5월 HTTP/2가 발표되었다. 처음 부터 많은 모던 브라우저들이 표준화 작업을 지원했다. 이러한 지원 덕분에, 많은 인터넷 사이트들이 2015년 이후로 이 프로토콜을 채택하기 시작했다.

기술적인 관점에서, HTTP/2와 HTTP/1.1을 구별하는 가장 중요한 기능 중 하나는 인터넷 프로토콜 스택에서 애플리케이션 계층의 일부로 간주되는 binary framing layer다. 모든 요청과 응답을 일반 텍스트 형식으로 관리하는 HTTP/1.1과는 다르게, HTTP/2는 모든 메시지를 이진 형식으로 캡슐화 하는 동시에 verb, 메서드, 헤더 등의 HTTP 문법을 유지한다. 애플리케이션 layer api는 여전히 전통적인 HTTP 형식으로 메시지를 만들지만, 그 하단의 레이어는 이러한 메시지를 이진수로 변환한다. 이렇게 하면 HTTP/2 이전에 작성된 웹 애플리케이션이 새 프로토콜과 상호작용 할 때 정상적으로 작동할 수 있다.

메시지를 이렇게 2진수로 변환하면, HTTP/2가 HTTP/1.1에서는 사용할 수 없는 데이터 전송에 대한 새로운 접근 방식을 시도할 수 있다. 

## Delivery Model

앞서 언급했던 것처럼 HTTP/1.1과 HTTP/2는 같은 문법을 공유한다. 두 프로토콜 모두 서버와 클라이언트 사이를 이동하는 요청은 `GET` `POST` 같은 익숙한 방법을 사용하여, 세더와 본문이 있는 전통적인 형식의 메시지로 목적지에 도달하도록 한다. 그러나 HTTP/1.1은 이것을 일반 텍스트로 전달하는 방면, HTTP/2는 이를 이진수로 코딩한다. 

### HTTP/1.1 - Pipelining and Head-of-Line Blocking

클라이언트가 HTTP GET 요청에서 받는 최초의 응답은 가끔씩 완전히 렌더링 할 수 있는 페이지 형태가 아닐 수 있다. 이 페이지에 추가로 필요한 리소스의 링크가 포함되어 있다. 클라이언트는 페이지를 다운로드를 한 이후에야 비로소 페이지의 전체 렌더링에 추가로 리소스가 필요한 것을 알게 된다. 이로 인해 클라이언트는 이러한 리소스를 추가로 요청해야 한다. HTTP/1.0에서는 클라이언트는 모든 새로운 요청을 위해 TCP 연결을 끊고 새로 연결을 만들어야 해서 시간과 리소스 측면에서 많은 비용이 들었다.

HTTP/1.1은 영구적 연결(persistent connection)과 파이프라인을 도입하여 이 문제를 해결한다. 영구적인 연결을 사용하면 TCP 연결을 닫으라고 직접 요청하지 않는 이상 계속해서 연결을 열어둔다. 이를 통해 클라이언트는 동일한 연결을 통해 여러 요청의 각각의 응답을 기다리지 않고 전송할 수 있다.따라서 1.1에서 크게 성능이 향상되었다.

하지만 안타깝게도 이 최적화 전략에는 피할 수 없는 병목현상이 존재한다. 동일한 대상으로 이동할 때 여러 데이터 패킷이 동시에 통과할 수 없기 때문에, 대기열 앞에 있는 요청이 이후의 모든 요청이 차단되어 버린다. 이는 HOL (Head of line) 블로킹으로 알려져 있으며, HTTP/1.1에서 연결 효율성을 최적화 하는데 있어 중요한 문재다. 별도의 병렬 TCP 연결을 추가하면 이러한 문제를 완화할 수는 있지만, 클라이언트와 서버간에 동시에 연결할 수 있는 TCP숫자는 제한이 있으며, 새로운 연결은 상당한 리소스를 필요로 한다.

### HTTP/2 - 이진 프레임 레이어의 장점

HTTP/2의 이진 프레임 레이어는 요청과 응답을 인코딩 하고 이를 더 작은 패킷으로 잘라 데이터 전송의 유연성을 향상 시킨다. 

HOL 블로킹의 영향을 줄이기 위해 여러개의 TCP 연결을 사용하는 HTTP/1.1과는 다르게, HTTP/2는 두 컴퓨터 사이에 단일 연결 개체를 설정한다. 이 연결에는 여러 데이터 스트림이 있다. 각 스트림은 요청/응답 형식의 여러메시지로 구성된다. 마지막으로 이러한 각 메시지는 프레임이라는 작은 단위로 분할 된다.

![connection](https://assets.digitalocean.com/articles/cart_63893/Streams_Frames.png)

가장 세분화된 레벨에서, 통신 채널은 각각 특정 스트림에 태그가 지정된 이진 인코딩 프레임 다발로 구성된다. 이렇게 만들어진 태그를 사용하면, 전송의 반대쪽 끝에서 다시 재조립할 수 있다. 인터리빙된 요청과 응답은, 멀티플렉싱이라는 프로세스 뒤에서 메시지를 차단하지 않고 병렬로 실행이 가능해진다. 멀티플렉싱은 다른 메시지가 완료될 때 까지 기다릴 필요가 없도록 하여, HTTP의 HOL 문제를 해결한다. 이는 서버 와 클라이언트가 동시에 요청과 응답을 보낼 수 있다는 것을 의미하므로, 제어 능력을 높이고 연결을 보다 효율적으로 관리할 수 있다.

> 통신에서 다중화(Multiplexing 혹은 MUXing)라는 용어는 두개 이상의 저수준의 채널들을 하나의 고수준의 채널로 통합하는 과정을 말하며, 역다중화(inverse multipleing, demultiplexing, demuxing) 과정을 통해 원래의 채널 정보들을 추출할 수 있다. 각각의 채널들은 미리 정의된 부호화 틀(coding scheme)을 통해 구분할 수 있다.

https://ko.wikipedia.org/wiki/%EB%8B%A4%EC%A4%91%ED%99%94_(%ED%86%B5%EC%8B%A0)

멀티플렉싱은 클라이언트가 여러 스트림을 병렬적으로 구성할 수 있게 해주기 때문에, 이들 스트림은 단일 TCP 연결만 사용하면 된다. 출처당 하나의 영구적인 접속은 네트워크 전체의 메모리와 처리 공간을 줄임으로써 HTTP/1.1에서 성능향상을 가져올 수 있게 된다. 네트워크 및 대역폭 활용도가 향상되어 전체 운영 비용이 절감된다.

클라이언트와 서버가 여러 요청과 응답에 대해 동일한 보안 세션을 재사용할 수 있으므로, 단일 TCP연결을 바탕으로 HTTPS 프로토콜의 성능도 향상된다. HTTPS에서는 TLS 또는 SSL 핸드셰이크 중에는 양 측이 모두 세션내내 단일 키를 사용하는 것에 동의 한다. 연결이 끊어지면 새로운 세션이 시작되므로, 추가 통신을 위해 새로운 키가 필요하다. 따라서 단일 연결을 유지하면, HTTPS 를 위해 필요한 리소스를 크게 줄일 수 있다. HTTP/2 규격에서 TLS 계층을 사용하도록 의무화 하지 않지만, 대부분의 브라우저는 HTTPS를 사용하는 HTTPS/2 만 지원한다.

이진 프렐임 레이어에 내재된 멀티플렉싱이 HTTP/1.1의 특정 문제를 해결하지만, 동일한 리소스를 기다리는 다중 스트림은 여전히 문제를 일으킬 수 있다.

### HTTP/2 - 스트리밍 우선순위
