---
title: BigChainDB) 블록체인을 활용한 데이터 베이스
date: 2018-06-25 12:45:56
published: true
tags:
  - blockchain
description: BigChain DB 백서
  [원문](https://www.bigchaindb.com/whitepaper/bigchaindb-whitepaper.pdf)
  [Github](https://github.com/bigchaindb/bigchaindb)  BigChainDB는 2016년에 처음 소개된,
  블록체인을 기반으로한 데이터베이스다. BigChainDB 기존 ...
category: blockchain
slug: /2018/06/25/bigchain-database-whitepaper/
template: post
---
BigChain DB 백서

[원문](https://www.bigchaindb.com/whitepaper/bigchaindb-whitepaper.pdf)
[Github](https://github.com/bigchaindb/bigchaindb)

BigChainDB는 2016년에 처음 소개된, 블록체인을 기반으로한 데이터베이스다. BigChainDB 기존 데이터 베이스에 탈중앙화, 불변성 등 블록체인의 성격이 녹아있는 DB라고 할 수 있다.


특징\분류 | 일반적인 블록체인| 일반적인 분산형 DB | BigChainDB  
--|---|--
탈중앙화|o| |o|  
비잔티움 장애 허용|o| |o|  
불변성|o| |o|  
데이터 소유자가 컨트롤|o| |o|  
높은 트랜잭션 속도||o|o|  
낮은 지연||o|o|    
인덱싱, 구조화된 데이터 등||o|o|    

## BigChainDB의 장점.

### 1) 완전한 탈중앙화와 비잔티움 장애 허용

BigChainDB는 모든 네트워킹과 합의에 [Tendermint](https://Tendermint.com/)를 사용한다. 각각의 노드는 로컬 MongoDB 데이터베이스를 가지고 있고, 모든 통신은 Tendermint protocol을 사용한다. (Tendermint는 블록체인을 기반으로 하는 다수의 머신에서 응용프로그램을 안전하고 지속적으로 replication하는 소프트웨어다. 이 소프트웨어는 BFT 설계를 구현하였다. [참고](https://Tendermint.readthedocs.io/en/master/introduction.html)). 이로 인해 시스템은 BFT를 얻을 수 있게 된다. 또한 이로 인해, 해커가 이 들 노드 중 하나의 MongoDB를 오염시킨 다 하더라도, 최악의 경우 해당 로컬의 데이터베이스만 오염되거나 삭제되는 선에 서 끝나게 된다. 이 말은, 다른 노드들에게는 해킹의 영향이 미치지 않는 다는 뜻이다.

모든 노드가 BigChainDB를 사용하고, 모든 노드들이 각각 다른 주체에 대해 관리 된다면, 이는 한명의 관리자, 단일 관리점, 단일 장애지점이 없기 때문에 완전한 탈중앙화 네트워크라고 할 수 있다. 이상적으로는, 이러한 노드들이 많은 나라에 걸쳐 저장되어 있어야 하며, 이로 인해 법적관할권이나 호스팅에 영향을 받지 않게 된다. 어떤 노드가 실패해도 네트워크는 계속해서 운영될 수 있다. 최대 1/3까지 실패하게 되도 네트워크는 정상적으로 동작한다. (주: 이는 Tendermint에서 주장하는 것과 일치한다. 아마 Tendermint의 프로토콜을 인용한 것으로 보인다.)

### 2) 불변성

일단 BigChainDB 네트워크에 저장이 되면, 이는 일단 거의 변경되거나 지워지기 어렵다. 많약 지워지거나 변경된다면, 이는 추적가능하다. 이러한 불변성을 만들어내기 위해서, 가장 간단한 방법으로, BigChainDB API는 삭제나 변경 기능을 제공하지 않는다. 그리고 모든 노드들은 전체 데이터의 복사본을 MongoDB 데이터베이스에 저장해 둔다. 즉 , 한 노드의 오염이나 삭제는 다른 노드에 영향을 미치지 않는다. 마지막으로, 네트워크에서 일어나는 모든 트랜잭션은 암호로 서명되어 있다. 일단 트랜잭션이 저장되고 나면, 내용을 바꾼 다는 것은 곧 서명을 바꾼 다는 것이며, 이는 전체 네트워크에서 인지할 수 있게 된다. (공개키가 수정되더라도, 이는 추적가능하다. 왜냐하면 트랜잭션의 모든 블록은 노드에 의해 서명 되는데, 이러한 서명에 쓰인 공개키는 모든 노드들이 알기 때문이다.)

### 3) 소유자가 컨트롤하는 DB (Owner-Controlled Assets)

다른 블록체인과 마찬가지로, BigChainDB는 소유자가 자산에 대한 제어권을 가지고 있다. 자산 소유자 만이 이를 이존시킬 수 있다. (여기서 소유자란 일련의 특정 개인키를 가진 사람들을 의미한다.) 노드 운영자는 이를 제어할 수 없다. 또한 비트코인의 비트코인이나 이더리움의 이더 처럼 단하나의 asset 이 존재한다. 하지만 BigChainDB는 외부유저가 자신이 원하는 만큼 asset을 만드는 것을 허용한다. 그러나 사용자가 다른 사람이 만든 것 처럼 보이는 asset은 만들 수 없다.

예를 들어, Joe가 Joe Token이라는 이름으로 1,000개의 토큰을 발행하기로 결심했다고 치자. 그는 BigChainDB 내에서 CREATE 트랜잭션을 만들고, 자신의 개인키로 서명한다음, 네트워크로 전송할 것이다. 결과적으로 Joe는 다른 사람에게 보낼 수 있는 1,000개의 토큰이 생기게 된다. 만약 다른사람에게 40개의 토큰을 보내기 위해 BigChainDB 에 TRANSFER 트랜잭션을 생성하게 되면, Joe는 총 9,960개를 소유하게 된다.

BigChainDB는 이중지불을 막기 위해 모든 트랜잭션을 확인한다.


### 4) 높은 트랜잭션 속도

BigChainDB는 초당 많은 거래를 처리할 수 있도록 설계 되어 있다. 이는 Tendermint를 기반으로 했기에 가능한 사실이다. Tendermint를 기반으로 한 [Cosmos whitepaer](https://cosmos.network/resources/whitepaper)에 따르면,

> 5개 대륙에서 7개의 데이터센터를 바탕으로 64개의 노드를 클라우드로 구성한 결과, Tendermint Consensus는, 네트워크 지연이 1~2초 정도 있었음에도, 초당 수천개의 거래를 처리해 냈다.


### 5) 낮은 Latency 와 빠른

Tendermint를 기반으로 하였기 때문에, 거래가 포함된 새로운 블록을 만드는데 몇초만이 소요되며, 이는 미래에 변경되거나 삭제할 수 없다.

### 6) 인덱싱, 쿼리 구조의 데이터

BigChainDB의 노드들은 로컬에 MongoDB 데이터베이스를 갖고 있기 때문에, MongoDB의 모든 기능을 사용할 수 있다. 그리고 각각의 노드들은 자신의 노드를 REST Api나 GraphQL 로 최적화 시킬지 자유롭게 선택할 수 있다.

### 7) Sybil Tolerance

비트코인과 같은 일부 블록체인 네트워크는 누구나 네트워크에 붙을 수 있도록 허용해 두었다. 이는 익명의, 허위로 꾸며진 다수의 가짜 사용자가 네트워크에 붙어 공격할 수 있다는 것을 의미한다. (Sybil Attack) 비트코인은 이러한 공격을 확률적으로 굉장히 어렵게 만들었다. BigChainDB 내에서는 네트워크 참여자를 조직할 수 있으므로, 이러한 문제는 존재하지 않는다.


## BigChainDB의 활용범위

1. supply chain
2. 지적재산권 관리
3. [디지털트윈](http://www.itworld.co.kr/news/108997)과 IOT
4. 신원확인
5. 데이터 거버넌스
6. 불변성을 지닌 감사 추적


## BigChainDB는 어떻게 작동하는가?

### 1. BigChainDB 트랜잭션

트랜잭션은 아래와 같은 JSON String 으로 구성되어 있다. 각각의 트랜잭션에는 키와 값과 더불어, 트랜잭션이 어떻게 생성되었는지, 트랜잭션이 유효하기 위해서는 어떤 것을 확인해야 하는지 등의 내용이 포함되어 있다.  

~~~ json
{
  "id": "3667c0e5cbf1fd3398e375dc24f47206cc52d53d771ac68ce14ddf0fde806a1c",
  "version": "2.0",
  "inputs": [
    {
      "fulfillment": "pGSAIEGwaKW1LibaZXx7_NZ5-V0alDLvrguGLyLRkgmKWG73gUBJ2Wpnab0Y-4i-kSGFa_VxxYCcctpT8D6s4uTGOOF-hVR2VbbxS35NiDrwUJXYCHSH2IALYUUZ6529Qbe2g4G",
      "fulfills": null,
      "owners_before": [
        "5RRWzmZBKPM84o63dppAttCpXG3wqYqL5niwNS1XBFyY"
      ]
    }
  ],
  "outputs": [
    {
      "amount": "1",
      "condition": {
        "details": {
          "public_key": "5RRWzmZBKPM84o63dppAttCpXG3wqYqL5niwNS1XBFyY",
          "type": "ed25519-sha-256"
        },
        "uri": "ni:///sha-256;d-_huQ-eG-QQD-GAJpvrSsy7lLJqyNhtUAs_own7aTY?fpt=ed25519-sha-256&cost=131072"
      },
      "public_keys": [
        "5RRWzmZBKPM84o63dppAttCpXG3wqYqL5niwNS1XBFyY"
      ]
    }
  ],
  "operation": "CREATE",
  "asset": {
    "data": {
      "message": "Greetings from Berlin!"
    }
  },
  "metadata": null
}
~~~

### 2. 트랜잭션을 네트워크로 전송

트랜잭션이 만들어지면, 이를 HTTP Api를 활용하여 BigChainDB로 전송해야 한다. 이러한 요청은 하나이상의 노드에 전달되게 된다.

![BigChainDB-network](../images/bigchaindb-network.png)


### 3. 노드가 트랜잭션을 수신

BigChainDB는 WSGI/Gunicorn과 호환되는 파이썬 웹 프레임워크인 Flask를 사용한다. Flask에서는 이러한 요청을 받고, 이 트랜잭션이 유효 한지 확인 한다. 트랜잭션이 유효하지 않다면, 400에러를 내뱉는다. 유효하다면, 이를 Base64 기반으로 변환한다음, 새로운 정보를 포함시켜 JSON String으로 다시 만든다. 그리고 BigChainDB는 이 string 을 로컬 Tendermint 인스턴스에 HTTP POST요청으로 보낸다.

### 4. Tendermint Instacne에서 트랜잭션을 수신

Tendermint에서 트랜잭션을 수신과정에서 어떤일이 일어나는지 확인하기 위해서는, [Tendermint의 docs](http://Tendermint.readthedocs.io/projects/tools/en/master/using-tendermint.html#broadcast-api)를 참고할 필요가 있다.

Tendermint가 거래가 유효한지 확인해기 위해서, BigChainDB에 여러가지요소를 `CheckTX`를 통해 질의를 하게 된다. `CheckTX`에는 BigChainDB에서 상속받아 표현한 다양한 변수가 포함되어 있다. Tendermint는 새로운 블록(일련의 트랜잭션이 포함되어 있음) 을생성하고, 모든 노드가 비잔티움 장애 허용 방식으로 다음 블록에 동의 하는지 확인한다. Tendermint에서  BigChainDB에 새로운 트랜잭션을 보내면, BigChainDB는 다시한번 트랜잭션의 유효성을 확인한다. 그리고 유효하다면, Commit 메시지가 Tendermint에서 온다면 그때 비로소 MongoDB에 작성하게 된다. 트랜잭션을 MongoDB에 저장하기 전에, BigChainDB는 asset.data와 metadata를 MongoDB의 다른 컬렉션에 따로 저장하고 이를 지운다. 이렇게 따로 저장함으로서, 사용자는 MongoDB 에서 text search를 할 수 있게 된다.
