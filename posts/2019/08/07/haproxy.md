---
title: HAProxy
date: 2019-08-07 06:39:20
published: true
tags:
  - infrastructure
description:
  '## 로드밸런서 > 로드 밸런싱이란, 부하 분산을 위해서 가상 IP를 통해 여러 서버에 접속하도록 분배하는 기능을
  말한다.  로드 밸런싱에서 사용하는 주요 기술은  - NAT(Network Address Translation): 사설 IP 주소를 공인
  IP 주소로 바꾸는 데 사용하는 통신망의 주소 변조기이다. - DSR(Dynamic Source Rout...'
category: infrastructure
slug: /2019/08/07/haproxy/
template: post
---

## 로드밸런서

> 로드 밸런싱이란, 부하 분산을 위해서 가상 IP를 통해 여러 서버에 접속하도록 분배하는 기능을 말한다.

로드 밸런싱에서 사용하는 주요 기술은

- NAT(Network Address Translation): 사설 IP 주소를 공인 IP 주소로 바꾸는 데 사용하는 통신망의 주소 변조기이다.
- DSR(Dynamic Source Routing protocol): 로드 밸런서 사용 시 서버에서 클라이언트로 되돌아가는 경우 목적지 주소를 스위치의 IP 주소가 아닌 클라이언트의 IP 주소로 전달해서 네트워크 스위치를 거치지 않고 바로 클라이언트를 찾아가는 개념이다.
- Tunneling: 인터넷상에서 눈에 보이지 않는 통로를 만들어 통신할 수 있게 하는 개념으로, 데이터를 캡슐화해서 연결된 상호 간에만 캡슐화된 패킷을 구별해 캡슐화를 해제할 수 있다.

로드 밸런서는, 네트워크에서 IP 주소와 MAC주소를 바탕으로 목적지 IP주소를 찾아가고, 다시 출발지를 되돌아 오는 구조로 작동된다.

[출처: naver d2](https://d2.naver.com/helloworld/284659)

## HAProxy

![ha-proxy](https://i0.wp.com/foxutech.com/wp-content/uploads/2019/01/What-is-HAProxy-and-how-to-install-and-configure-in-Linux.png?fit=2000%2C1000&ssl=1)

HAProxy는 reserve proxy형태로 작동한다. 흔히 브라우저에서 사용하는 proxy는 클라이언트 앞에서 처리하는데, 이를 forward proxy라고 한다. 반대로 reserve proxy는 실제 서버 요청에 대해 서버 앞단에 존재하면서, 서버로 들어오는 요청을 대신 받아 서버에 전달하고, 요청한 곳에 그 결과를 다시 전달한다.

### 작동 흐름

1. 최초 접근 시 서버에 요청 전달
2. 응답 시 쿠키(cookie)에 서버 정보 추가 후 반환
3. 재요청 시 proxy에서 쿠키 정보 확인 > 최초 요청 서버로 전달
4. 다시 접근 시 쿠키 추가 없이 전달 > 클라이언트에 쿠키 정보가 계속 존재함(쿠키 재사용)

![haproxy-flow](https://d2.naver.com/content../../../images/2015/06/helloworld-284659-1.png)
