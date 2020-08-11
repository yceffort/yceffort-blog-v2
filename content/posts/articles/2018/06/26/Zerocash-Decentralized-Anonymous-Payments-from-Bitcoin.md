---
title: Zerocash) 비트코인 기반의 익명 분산 결제 시스템
date: 2018-06-26 12:02:03
published: true
tags:
  - blockchain
  - bitcoin
description: "Zerocash: Anonymous Distributed E-Cash from Bitcoin
  [원문](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6956581)  비트코인 자체는
  익명이 아니기 때문에, 이를 익명화 시키는데 많은 노력을 기울이고 있다. (mix라는 표현을 쓰고 있는데, 여기서는 간단하게 ..."
category: blockchain
slug: /2018/06/26/Zerocash-Decentralized-Anonymous-Payments-from-Bitcoin/
template: post
---
Zerocash: Anonymous Distributed E-Cash from Bitcoin

[원문](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6956581)

비트코인 자체는 익명이 아니기 때문에, 이를 익명화 시키는데 많은 노력을 기울이고 있다. (mix라는 표현을 쓰고 있는데, 여기서는 간단하게 세탁이라고 표현하겠다.)

> While Bitcoin is not anonymous itself, those with sufficient motivation can obfuscate their transaction history with the help of mixes (also known as laundries or tumblers).

이러한 세탁의 과정은, 사용자가 일정양의 화폐를 보내면 화폐 pool에서 같은 양의 다른 코인을 돌려준다. 그러나 이러한 세탁의 과정도 한계가 있다. 1) 세탁을 하기 위해서는 일정량의 코인이 있어야 하고, 2) 세탁된 코인도 추적이 가능하며 3) 세탁 과정에서 도난 당할 가능성도 있다. '무언가를 감추어야 하는' 사용자들에게 이러한 위험성은 충분히 감내 할 수 있을지도 모른다. 그러나 일반적인 유저들은 1) 자신의 소비습관을 다른 개인 사용자에게 알리고 싶지 않고 2) 자신의 개인정보 보호를 위해 특별한 노력을 기울이거나 위험을 감수하고 싶지도 않고 3) 그들의 사생활 침해를 인식하지 못하는 경우가 많다.

이 들의 개인정보를 보호하기 위해서는, 위험성이 없어야하고, 자동적으로 이웃이나, 동료나, 상인들에게 소비습관이나 잔고 등이 노출되지 말아야 한다. 그리고 익명의 거래는 화폐의 히스토리와 별개로 시장가치를 보증해주는 역할을 하므로, 사용자들에게 암호화폐가 유용하게 남을 수 있게 해준다.

### Zerocash

Zerocash은 비트코인에서 확장된 개념으로, 강력한 익명성을 보장한다. Zerocash은 화폐 확인을 하기 위해 디지털 서명을 하거나, 이중 지불 방지를 위해 중앙 은행을 필요로 하지 않는다. Zerocash은 이를 위해 [영지식증명](https://yceffort.github.io/2018/06/26/zero-knowledge-proof.html) 을 사용한다. Zerocash은 Zerocash 프로토콜을 활용하여 주기적으로 비트코인을 세탁하는 일을 한다. 일상적으로 이루어지는 거래는 비트코인을 통해 이뤄진다. 그 이유는 아래와 같다.

1. 성능: Zerocash을 교환하기 위해서는 이중 이산 로그 증명을 거쳐야 되는데, 이를 검증하는데 45kb가 넘는 데이터로 450ms 정도의 속도가 걸린다. 이러한 증명 작업은 네트워크를 통해 전파되고, 모든 노드들이 확인하고 그 뒤에는 원장에 영구적으로 저장된다. 이렇게 수반되는 비용은 비트코인 비용보다 더 크기 때문이다.
2. 기능성: Zerocash은 기본적인 전자화폐 기능만으로 구성되어 있기 때문에, 익명의 지불에 필요한 기능이 결여 되어 있다. (잔돈을 거슬러 주거나, 송금기능 등)

> 주: 하지만 이후에 [ZCash](https://z.cash/)라는, 비트코인과는 별개로 Zerocash의 기능을 온전히 이어받은 새로운 암호화폐가 등장하게 된다.

### zk_SNARK?

[zk_SNARK](https://z.cash/technology/zksnarks.html)는 `Zero-Knowledge Succinct Non-Interactive Argument of Knowledge`의 약자로, 특정 정보나 증명인과 검증인 사이에 정보 교환없이 특정 정보의 수요를 증명할 수 있는 증명 구조를 의미한다. 증명자는 실제로 숫자에 대한 정보를 공개하지 않고도 그러한 숫자를 알고 있음을 검증자에게 확신시킬 수 있다. '간결한' ZKP는 큰 프로그램의 내용의 경우에도 수백 밀리초 이내에 증명할 수 있다. 기존의 ZPK는 인증장치와 검증장치 사이에 여러 라운드로 통신을 거쳐야 했지만, 이 비대화형 구조에서는 피인증장치에서 검증장치로 보낸 단일 메시지로 구성된다. 블록체인에 게시할 만큼 비대화형의 짧은 ZPK를 생성하는 유일한 방법은 검증자와 증명자 사이에 공유되는 공통 참조 문자열을 초기 설정 단계에 생성하는 것이다. 이 공동 참조 문자는 시스템의 공용 매개변수로 참조하게 된다.

### 결론

탈중앙화 통화는 합법적인 금융거래를 수행할때, 사용자 개인정보를 다른 사람들로 부터 보호해야 한다. Zerocash는 사용자 정보, 거래금액, 잔고 등을 숨김으로 써 이러한 목표를 달성하고 있다. 그러나 이는 책임, 규제, 감독 등을 저해 한다는 비판을 받을 수 있다. 그러나 Zerocash가 기본적인 통화 시스템에서만 활용될 필요가 없다. 예를 들어, 사용자가 모든 거래에 대해 세금을 납부했는지를 거래를 공개하거나, 거래액을 보여주거나, 세금의 액수를 공개하지 않더라도 세금 납부 여부를 확인할 수 있다. 이를 통해 광범위한 규정 준수 및 규제 정책을 수동으로 검증할 수 있다.
