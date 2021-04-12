---
title: npm 'event-stream' 패키지, 비트코인 지갑을 노리는 악성코드에 감염
date: 2018-12-01 04:44:01
published: true
tags:
  - bitcoin
description: Malicious code in npm ‘event-stream’ package targets a bitcoin
  wallet and causes 8 million downloads in two months
  [원문](https://hub.packtpub.com/malicious-code-in-npm-event-stream-package-targets-a-b...
category: bitcoin
slug: /2018/11/30/malicious-code-in-npm-event-stream-package-targets-a-bitcoin-wallet/
template: post
---
Malicious code in npm ‘event-stream’ package targets a bitcoin wallet and causes 8 million downloads in two months

[원문](https://hub.packtpub.com/malicious-code-in-npm-event-stream-package-targets-a-bitcoin-wallet-and-causes-8-million-downloads-in-two-months/)

지난주 캘리포니아 주 CSUF 대학의 컴퓨터 사이언스 전공인 Ayrton Sparling는, 인기있는 npm 패키지인 [event-stream](https://github.com/dominictarr/event-stream)에 스트림이라는 악의적인 패키지가 포함되어 있다고 발표했다. 그는 EventStream의 repository 에 이 문제를 [공개했다.](https://github.com/dominictarr/event-stream/issues/116)

event stream npm 패키지는 원래 [Dominic Tarr](https://github.com/dominictarr)라는 개발자가 제작하여 유지 관리했다. 그러나 이 인기있는 패키지는 오랫동안 업데이트되지 않았다. 토머스 헌터의 포스팅에 따르면, “event stream의 소유권은 악의적인 사용자인 [right9ctrl](https://github.com/right9ctrl)에게 이관되었다. 악의적인 사용자는 패키지에 일련의 의미있는 기여를함으로써 원본 저자의 신뢰를 얻을 수 있었다.

악의적인 소유자는 flatmap stream이라는 악의적인 라이브러리를 이벤트 스트림 패키지에 종속성으로 추가했다. 이로 인해 모든 사용자가 이벤트 스트림 패키지 (3.3.6 버전 사용)를 다운로드하고 해당 기능을 호출했다. 악의적인 라이브러리의 다운로드는 2018년 9 월에 포함된 이후 약 8 백만 건의 다운로드가 이뤄졌다.

악의적인 패키지는 지정된 공격을 목표로 수행되었으며, [bitpay/copay](https://github.com/bitpay/copay)라는 오픈 소스 앱에 영향을 미친다. Copay는 데스크톱과 모바일 장치 모두를 위한 안전한 비트코인 지갑 플랫폼이다. 해당 모듈의 package.json (모듈에서 의존하고 있는 패키지 목록)에서 AES256 (암호화)를 디크립트 하는 패키지가 발견되었으며, 이는 copay앱을 타겟으로 하고 있는 것으로 보였다.

- https://twitter.com/dominictarr/status/1067186943304159233?ref_src=twsrc%5Etfw
- https://twitter.com/andrestaltz/status/1067157915398746114?ref_src=twsrc%5Etfw

이 악의적인 코드의 영향을받는 사용자는 이벤트 스트림의 버전 3.3.4를 사용하거나, 제거하는 것이 좋다.

사용자 응용 프로그램이 Bitcoin을 처리하면 지난 3 개월 동안 채굴되거나 이전된 비트 코인이 지갑에 들어 가지 않았는지 확인해야 한다.

그러나 응용 프로그램이 비트 코인을 다루지 않지만 확인이 필요한 경우, 의심스러운 활동에 대한 지난 3 개월 동안의 활동을 검사하는 것이 좋다. 이는 네트워크에서 의도하지 않은 대상으로 전송된 데이터를 분석하는 것이 필요할 것이다.

[참고](https://github.com/dominictarr/event-stream/issues/116)
