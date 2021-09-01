---
title: 'Nodejs의 이벤트 루프 깊게 살펴보기'
tags:
  - nodejs
  - javascript
  - event-loop
published: true
date: 2021-08-31 21:38:06
description: '이벤트 루프는 4개의 큐, 그리고 2개의 중간 큐가 있습니다.'
---

## Table of Contents

## Overview

Nodejs가 다른 프로그래밍 플랫폼과 구별되는 특징은 I/O를 처리하는 방식이다. Nodejs를 소개할 때 마다 항상 반복해서 하는 얘기는 _구글 v8 자바스크립트 엔진 기반의 논블로킹, 이벤트 기반 플랫폼_ 라는 것이다. _논블로킹_, *이벤트 기반*이라는 것은 무슨 뜻일까? 이 모든 것에 대한 대답은 Nodejs의 중심인 이벤트 루프에 있다. 이벤트 루프는 무엇인지, 작동방식은 어떤지, 애플리케이션에 어떻게 영향을 미치는지, 어떻게 해야 최상의 결과를 얻을 수 있을까?

## 반응형 패턴

Nodejs는 **Event Demultiplexers** 및 **이벤트 큐**를 포함하고 있는 이벤트 기반 모델로 작동한다. 모든 I/O 요청은 완료/실패 또는 또다른 트리거를 발생시킨다. 이를 **이벤트** 라고 한다. 이러한 이벤트는 다음 알고리즘에 따라서 처리된다.

1. Event Demultiplexers는 I/O 요청을 받고, 이러한 요청을 적절한 하드웨어에 위임한다.
2. I/O 요청이 처리되면 (파일에 있는 데이터 읽기, 소켓에 있는 데이터 읽기 등) Event Demultiplexers는 처리해야할 특정 작업에 등록되어 있는 콜백 핸들러를 큐에 추가한다. 여기서 말하는 콜백을 이벤트라고 하고, 이벤트가 추가되는 큐를 이벤트 큐라고 한다.
3. 이벤트 큐에서 이벤트를 처리할 수 있는 경우, 이벤트를 수신한 순서대로 큐이 빌 때 까지 순차적으로 실행한다.
4. 이벤트 큐에 더이상 이벤트가 없거나, Event Demultiplexer에 더이상 보류 중인 요청이 없는 경우, 프로그램이 완료된다. 그렇지 않으면, 다시 첫 번째 단계 부터 프로세스가 계속된다.

이 전체 매커니즘을 조율하는 프로그램을 **이벤트 루프** 라한다.

![event-loop](https://miro.medium.com/max/1122/1*3fzASvL5gFrSC64hHKzQOQ.jpeg)

이벤트 루프는 단일 스레드이며, 반 무한 (semi-infinite) 루프다. 이것을 무한이 아닌 반 무한이라고 부르는 이유는, 더 이상 할일이 없는 시점에는 멈추기 때문이다. 개발자 관점에서 보자면, 여기에서 프로그램이 종료되는 것이다.

위 그림은, nodejs가 어떻게 작동하는지와, 이른바 [리액터 패턴](https://ko.wikipedia.org/wiki/%EB%B0%98%EC%9D%91%EC%9E%90_%ED%8C%A8%ED%84%B4) 이라고 불리우는 디자인 패턴의 주요 컴포넌트들을 보여 주고 있다. 하지만 실제로는 이것보다 훨씬 더 복잡하다.

## Event Demultiplexer

Event Demultiplexer라는 컴포넌트는 사실 실제로 존재하는 컴포넌트 개념이 아니다. 이는 리액터 패턴에 있어서 일종의 추상적인 개념이라고 볼 수 있다. Event Demultiplexer는 리눅스의 epoll, 맥과 같은 BSD 시스템에서는 kqueue, Solaris의 event ports, 윈도우의 IOCP (Input Output Completion Port) 등과 같이 서로 다른 이름으로 여러 시스템에 걸쳐 존재하고 있다. Nodejs는 이러한 구현을 활용하여 저수준 논블로킹, 비동기 하드웨어 I/O 기능을 사용한다.

### File I/O의 복잡성

그러나 안타깝게도, OS에서 제공하는 이 구현을 사용하여 모든 유형의 I/O를 수행할 수 있는 것은 아니다. 동일 OS 내부에서도, 서로 다른 유형의 I/O를 제공하는 데 있어서 복잡성이 존재한다. 일반적으로 네트워크 I/O 는 앞서 이야기한, epoll, kqueue, event ports, IOCP 등으로 구현할 수 있지만, 파일 I/O는 이보다 훨씬 복잡하다. 리눅스와 같은 일부 시스템의 경우, 파일 시스템 액세스에 필요한 완전한 비동기화 기능을 제공하지 않는다. [또한 macOS 시스템에서는 kqueue를 활용한 파일 시스템 이벤트 알림, 시그널링에 제한이 있다.](http://blog.libtorrent.org/2012/10/asynchronous-disk-io/) 따라서 완전한 비동기성을 제공하기 위해 모든 OS의 파일 시스템의 복잡성을 해결하는 것은 매우 어렵고, 해결하기도 거의 불가능하다.

### DNS의 복잡성

파일 I/O와 비슷하게, Node API에서 제공하는 특정 DNS 함수들에도 몇가지 복잡성이 존재하고 있다. NodeJS의 DNS 함수 중, `dns.lookup`와 같은 경우에는 `nsswitch.conf` `resolv.conf`, `/etc/hosts`와 같은 시스템 설정파일에 접근해야 하므로, 파일 시스템의 복잡성이 여기까지 적용된다고 볼 수 있다.

### 해결책

따라서 하드웨어 비동기 I/O 유틸리티로 직접 주소를 지정할 수 없는 I/O 함수를 지원하기 위해 `thread pool`의 개념이 도입되었다. 즉, 모든 I/O 함수가 쓰레드 풀에서 실행되지 않는다. (특정 함수만 쓰레드 풀에서 실행됨) NodeJS는 대부분의 I/O를 논블로킹 비동기 하드웨어 I/O를 사용하기 위하여 최선을 다했지만, 이를 사용하는 것이 차단되었거나, 해결하기 복잡한 위와 같은 유형의 문제에서는 쓰레드 풀을 사용한다.

> 사실 쓰레드 풀에서 실행되는 것은 I/O 뿐만 이 아니다. Node.js의 `crypto`내부에 있는 함수들 중 `crypto.pbkdf2`, 비동기 버전의 `crypto.randomBytes`와 `crypto.randomFill`, `zlib.*`는 CPU 집약적인 작업이어서 libuv의 쓰레드 풀에서 실행된다. 쓰레드 풀에서 실행되는 작ㄱ업들은 이벤트 루프를 블로킹하지 않는다.

### 종합하자면

살펴본 것처럼, 실제 세계에 존재하는 모든 서로다른 종류의 I/O 작업을 지원하는 것은, OS 마다 서로 다른 특징을 가지고 있기 때문에 매우 어렵다고 볼 수 있다. 일부 I/O는 비동기적인 특징을 유지하면서도 네이티브 하드웨어 구현을 활용하여 수행될 수도 있고, 비동기 특성을 보장하기 위해 쓰레드 풀에서 수행되는 경우도 있다.

> Nodejs가 쓰레드 풀에서 모든 I/O를 수행한다는 것은 거짓이다.

여러 플랫폼의 I.O를 지원하면서, 전체 프로세스를 제어하려면, 이러한 플랫폼간 복잡성을 캡슐화하고, 노드의 상위 계층에 일바회된 API를 노출하는 추상화된 계층이 있어야 한다.

그리고, 이를 수행하는 것이 바로,,,

![libuv](https://miro.medium.com/max/1400/1*PCRWGXEGI_bF2Rb3JxxBSg.png)

> libuv is cross-platform support library which was originally written for Node.js. It’s designed around the event-driven asynchronous I/O model.

> The library provides much more than a simple abstraction over different I/O polling mechanisms: ‘handles’ and ‘streams’ provide a high level abstraction for sockets and other entities; cross-platform file I/O and threading functionality is also provided, amongst other things.

libuv가 어떻게 구성되어 있는지 살펴보자. 아래 그림은 libuv 공식 홈페이지에서 가져왔다.

![libuv](http://docs.libuv.org/en/v1.x/_images/architecture.png)

`Event Demultiplexer` 는 앞서 언급한 것처럼 무언가 독립된 하나의 객체가 아니라, libuv에 의해 추상화 되고, NodeJS의 상위 계층에 노출되는 I/O 처리 API 모음이다. libuv가 제공하는 것은 이것 뿐만이 아니다. libuv는 nodejs 전체에 걸쳐 이벤트 루프, 이벤트 큐 매커니즘을 제공한다.

## 이벤트 큐

이벤트 큐는 모든 이벤트가 대기열에 들어가고, 그 대기열이 비어있을 때 까지 이벤트 루프에 의해 순차적으로 처리하는 데이터 구조여야 한다. 그러나 Nodejs에서 이러한 작업이 일어나는 동작은, 추상 리액터 패턴이 이를 설명하는 방식과 완전히 다르다. 어떻게 다를까?

> Nodejs에는 서로 다른 이벤트가 대기하는 한개 이상의 큐가 존재한다. 한 단계를 처리한 후, 다음 단계로 이동하기 전에 이벤트 루프는 중간 큐 남아 있는 항목이 없을 때까지, 두개의 중간 대기열을 처리한다.

Nodejs에는 얼마나 많은 큐가 있고, 이 큐들이 각각 어떤 동작을 하고 있을까?

- `Expired timers and intervals queue`: `setTimeout`, `setInterval`을 사용한 콜백
- `IO Events Queue`: 완료된 I/O 이벤트
- `Immediates Queue`: `setImmediate` 함수를 사용하여 추가된 콜백
- `Close Handlers Queue`: 모든 `close` 이벤트 핸들러

> 사실 일부는 큐 형태가아닌 다른 데이터 형태로 저장되어 있다 (타이머의 경우에는 min-heap)

이 4개의 메인 큐 이외에, 앞서 언급했던 `중간 큐`로 언급했던 2개의 큐가 Nodejs에서 처리된다. 이 큐는 libuv의 일부가 아니라 Nodejs의 일부다.

- `Next Ticks Queue`: `process.nextTick` 함수에 의해 추가된 콜백
- `Other Microtasks Queue`: Promise callback resolve와 같은 마이크로 태스크 작업

### 어떻게 동작하는가?

아래 그림을 보면, Nodejs는 타이머 대기열에 있는 만기된 타이머가 있는지 먼저 확인하면서 이벤트 루프를 시작한다. 그리고 처리할 총 아이템의 카운터 참조를 유지하면서 각 단계에서 각 큐를 실행한다. `close` 핸들러 큐를 처리한 이후에, 더이상 대기중인 큐가 없고 동작중인 작업이 없다면 루프를 빠져나가게 된다. 이벤트 루프에서 각 큐의 처리는 이벤트 루프의 한 단계로 볼수 있다.

![structure-of-eventloop](https://miro.medium.com/max/2000/1*2yXbhvpf1kj5YT-m_fXgEQ.png)

한가지 흥미로운 점은, 각 단계가 끝날 때 마다 중간 큐 (`next ticks queue`, `microtask queue`)에 현재 처리해야할 항목이 있는지 확인한다는 것이다. 이 중간 queue에 작업이 있을 경우, 이벤트 루프는 즉시 두개의 큐가 비워질 때 가지 해당 작업을 처리하게 된다. 그리고 이 두 큐가 비게 되면 다음 단계가 처리되기 시작한다.

### Next tick queue vs Other microtasks

Next tick queue는 다른 마이크로 태스크 큐에 비해 더 높은 우선 순위를 갖는다. 이 두 큐는 이벤트 루프의 각 단계 사이에서 실행되는데, 이는 libuv가 더 높은 레벨에 있는 nodejs와 각 단계가 끝날 때 마다 통신한다는 것을 의미한다.

이 중간 큐 규칙은 IO Starvation이라고 하는 새로운 문제를 야기한다. `process.nextTick`를 활용하여 다음 큐를 계속해서 채우는 경우, 이벤트 루프는 다음 으로 넘어가지 못하고, 다음 큐를 계속 기다리기만 하게 된다. 이 큐가 비워지지 않고는 다음 이벤트 루프를 넘어갈 수 없으므로, `IO Starvation`이 발생하게 된다.

![nodejs architecture](https://miro.medium.com/max/1400/1*-0Sa0i_g-gcL9sJqvecKEw.png)
