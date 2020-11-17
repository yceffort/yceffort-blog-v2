---
title: Bitcoin) BTCD와 bitcoin-cli (bitcoin core)의 차이
date: 2018-05-31 08:11:51
published: true
tags:
  - bitcoin
  - programming
description: 얼마전에 구글 컴퓨팅 엔진을 통해서 bitcoin-cli를 돌려봤었다. 그 때는 CPU 4개에, 램 16기가에, ssd
  100기가를 활용해서 약 3일에 걸쳐서 작업을 진행했다.  이러한 비슷한 작업을 GoLang으로 구현한 것이 바로
  BTCD다.  ![window-btcd](../images/window-btcd.png)  BTCD와 b...
category: bitcoin
slug: /2018/05/31/bitcoin-btcd-bitcoin-cli/
template: post
---
얼마전에 구글 컴퓨팅 엔진을 통해서 bitcoin-cli를 돌려봤었다.

그 때는 CPU 4개에, 램 16기가에, ssd 100기가를 활용해서 약 3일에 걸쳐서 작업을 진행했다.

이러한 비슷한 작업을 GoLang으로 구현한 것이 바로 BTCD다.

![window-btcd](../images/window-btcd.png)

BTCD와 bitcoin core (이제부터는 bitcoincli라고 하겠다)의 가장 큰 차이점이라고 한다면,** wallet 기능의 유무다.**

bitcoincli는 지갑의 기능도 함께 겸하고 있지만, BTCD는 그런 거 없이 오로지 bitcoin의 node를 validation하는 작업만 할 수 있다.

(btcwallet이나 btcgui로 지갑기능을 사용할 수 있다.)

이 차이는 태생적으로 어떤 목적을 가지고 개발됐냐의 차이에서 기인한다.

지갑의 기능이 없지만, BTCD는 bitcoincore에 비해 여러가지로 유용한 장점을 가지고 있다.

1.  컴파일 시간이 짧다.  
bitcoincli를 설치하고 컴파일 하기 위해 우분투에서 나는 다양한 라이브러리를 오랜시간에 걸쳐 설치해야만 했다. 하지만 컴파일이 굉장히 빠르게 된다는 GoLang의 특징 덕분에, 이러한 컴파일이 수초내로 최소화 된다.
2. 성능이 좋다.  
 bitcoincli는 c++ 기반으로 작성되었다. 때문에, RPC Request를 시스템에서 비용을 많이 소모하는 Thread를 열어서 handling 한다. 하지만 BTCD는 GoLang으로 작성되어 있기 때문에 GoLang의 GoRoutines를 활용하여 많은 양의 동시 request를 쉽고 빠르게 처리할 수 있다.
3. 다른 DB를 사용할 수 있다.  
 bitcoincli와 BTCD 모두 google의 key-value storage인 levelDB를 사용하고 있다. 이와 더불어 [BTCD는 다른 DB도 처리할 수 있도록 기능](https://github.com/btcsuite/btcd/tree/master/database)을 제공하고 있다.
4. HTTP post 요청과 웹소켓을 모두 지원한다.

Go라는 언어가 bitcoin이나 blockchain에서 처리해야하는 작업들을 Goroutine으로 굉장히 손쉽고 효율적으로 처리할 수 있기 때문에 관련 업종에서 Go를 core기능으로 많이 쓰는것으로 알고 있다.
