---
title: 이더리움 - 프라이빗 블록체인 만들기 (2)
date: 2018-07-04 12:00:48
tags:
  - ethereum
published: true
description:
  '[여기](/2018/07/03/ethereum-create-private-blockchain-1/)에서 이어집니다.
  자바스크립트 기반의 콘솔입니다.  ### 1. 어카운트 확인하기  ``` > eth.accounts
  ["0x44e74080949320292839b9a0df55e4459dd51434"] ```  아까 생성한 한계의 어카운트가
  보입니다.  ##...'
category: ethereum
slug: /2018/07/03/ethereum-create-private-blockchain-2/
template: post
---

[여기](/2018/07/03/ethereum-create-private-blockchain-1/)에서 이어집니다.

자바스크립트 기반의 콘솔입니다.

### 1. 어카운트 확인하기

```bash
> eth.accounts
["0x44e74080949320292839b9a0df55e4459dd51434"]
```

아까 생성한 한계의 어카운트가 보입니다.

### 2. 잔고 확인하기

```bash
> eth.getBalance(eth.accounts[0])
300000
```

첫 제네시스 블록을 만들때 alloc 했던 양 만큼 할당이 되어 있네요.

### 3. 단위 변경하기

그런데 저 단위는 사실 ether가 아니고 wei입니다. 아래 처럼 변환하면 됩니다. [참고](https://github.com/ethereum/wiki/wiki/JavaScript-API#web3fromwei)

```bash
>  web3.fromWei(eth.getBalance(eth.accounts[0], "ether"));
"0"
```

### 4. 채굴 시작하기

```bash
> miner.start()
INFO [07-03|15:33:29] Updated mining threads                   threads=0
INFO [07-03|15:33:29] Transaction pool price threshold updated price=18000000000
null
> INFO [07-03|15:33:29] Starting mining operation
INFO [07-03|15:33:29] Commit new mining work                   number=1 txs=0 uncles=0 elapsed=597.603µs
INFO [07-03|15:33:31] Generating DAG in progress               epoch=0 percentage=0 elapsed=413.009ms
INFO [07-03|15:33:31] Generating DAG in progress               epoch=0 percentage=1 elapsed=751.934ms
INFO [07-03|15:33:31] Generating DAG in progress               epoch=0 percentage=2 elapsed=1.091s
INFO [07-03|15:33:32] Generating DAG in progress               epoch=0 percentage=3 elapsed=1.437s
INFO [07-03|15:33:32] Generating DAG in progress               epoch=0 percentage=4 elapsed=1.769s
INFO [07-03|15:33:32] Generating DAG in progress               epoch=0 percentage=5 elapsed=2.142s
INFO [07-03|15:33:33] Generating DAG in progress               epoch=0 percentage=6 elapsed=2.480s
...
INFO elapsed=36.298s
INFO [07-03|15:34:07] Generating DAG in progress               epoch=0 percentage=94 elapsed=36.666s
INFO [07-03|15:34:07] Generating DAG in progress               epoch=0 percentage=95 elapsed=37.058s
INFO [07-03|15:34:08] Generating DAG in progress               epoch=0 percentage=96 elapsed=37.432s
INFO [07-03|15:34:08] Generating DAG in progress               epoch=0 percentage=97 elapsed=37.796s
INFO [07-03|15:34:08] Generating DAG in progress               epoch=0 percentage=98 elapsed=38.208s
INFO [07-03|15:34:10] Generating DAG in progress               epoch=0 percentage=99 elapsed=39.469s
INFO [07-03|15:34:10] Generated ethash verification cache      epoch=0 elapsed=39.471s
INFO [07-03|15:34:11] Successfully sealed new block            number=1 hash=48f248…d2232e
INFO [07-03|15:34:11] 🔨 mined potential block                  number=1 hash=48f248…d2232e
INFO [07-03|15:34:11] Commit new mining work                   number=2 txs=0 uncles=0 elapsed=597.26µs
INFO [07-03|15:34:11] Successfully sealed new block            number=2 hash=c5c768…b1e445
INFO [07-03|15:34:11] 🔨 mined potential block                  number=2 hash=c5c768…b1e445
INFO [07-03|15:34:11] Commit new mining work                   number=3 txs=0 uncles=0 elapsed=192.809µs
INFO [07-03|15:34:12] Successfully sealed new block            number=3 hash=163387…646389
INFO [07-03|15:34:12] 🔨 mined potential block                  number=3 hash=163387…646389
INFO [07-03|15:34:12] Commit new mining work                   number=4 txs=0 uncles=0 elapsed=163.111µs
INFO [07-03|15:34:12] Generating DAG in progress               epoch=1 percentage=0  elapsed=955.391ms
INFO [07-03|15:34:12] Successfully sealed new block            number=4 hash=cee703…d4001c
INFO [07-03|15:34:12] 🔨 mined potential block                  number=4 hash=cee703…d4001c
INFO [07-03|15:34:12] Mining too far in the future             wait=2s
INFO [07-03|15:34:12] Generating DAG in progress               epoch=1 percentage=1  elapsed=1.478s
INFO [07-03|15:34:13] Generating DAG in progress               epoch=1 percentage=2  elapsed=1.855s
INFO [07-03|15:34:13] Generating DAG in progress               epoch=1 percentage=3  elapsed=2.250s
INFO [07-03|15:34:14] Generating DAG in progress               epoch=1 percentage=4  elapsed=2.631s
INFO [07-03|15:34:14] Generating DAG in progress               epoch=1 percentage=5  elapsed=3.005s
INFO [07-03|15:34:14] Commit new mining work                   number=5 txs=0 uncles=0 elapsed=2.003s
INFO [07-03|15:34:14] Generating DAG in progress               epoch=1 percentage=6  elapsed=3.558s
INFO [07-03|15:34:15] Successfully sealed new block            number=5 hash=c7111e…cabeef
INFO [07-03|15:34:15] 🔨 mined potential block                  number=5 hash=c7111e…cabeef
INFO [07-03|15:34:15] Commit new mining work                   number=6 txs=0 uncles=0 elapsed=170.026µs
INFO [07-03|15:34:15] Generating DAG in progress               epoch=1 percentage=7  elapsed=4.427s
INFO [07-03|15:34:16] Successfully sealed new block            number=6 hash=230787…ba48a2
INFO [07-03|15:34:16] 🔗 block reached canonical chain          number=1 hash=48f248…d2232e
INFO [07-03|15:34:16] 🔨 mined potential block                  number=6 hash=230787…ba48a2
INFO [07-03|15:34:16] Commit new mining work                   number=7 txs=0 uncles=0 elapsed=141.75µs
INFO [07-03|15:34:16] Successfully sealed new block            number=7 hash=3a22b7…e4e015
INFO [07-03|15:34:16] 🔗 block reached canonical chain          number=2 hash=c5c768…b1e445
```

### 5. 외부에서 접속하기

```bash
deploy@jayg-blockchain2:~$ geth attach http://1.1.1.1:8123
Welcome to the Geth JavaScript console!

instance: Geth/PrivateNetwork/v1.8.11-stable-dea1ce05/linux-amd64/go1.10
coinbase: 0x44e74080949320292839b9a0df55e4459dd51434
at block: 25 (Tue, 03 Jul 2018 15:34:38 KST)
 modules: eth:1.0 miner:1.0 net:1.0 rpc:1.0 web3:1.0
```
