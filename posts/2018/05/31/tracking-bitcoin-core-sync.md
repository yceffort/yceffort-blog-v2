---
title: Bitcoin) Bitcoin-core의 Sync를 동기화 해보자.
date: 2018-05-31 09:01:45
published: true
tags:
  - programming
  - bitcoin
description:
  bitcoin-core를 설치했다면 bitcoind daemon 에서는 모든  block 정보를 동기화 하게 된다. 이는
  꽤나 많은 양이기 때문에, 동기화에 하루이틀 이상의 오랜 시간이 소요된다.  ![스크린샷 2016-04-20
  08.53.33](../images/bitcoin-sync-1.png)  지갑주소 공개합니다. 0...
category: programming
slug: /2018/05/31/tracking-bitcoin-core-sync/
template: post
---

bitcoin-core를 설치했다면 bitcoind daemon 에서는 모든  block 정보를 동기화 하게 된다.

이는 꽤나 많은 양이기 때문에, 동기화에 하루이틀 이상의 오랜 시간이 소요된다.

![스크린샷 2016-04-20 08.53.33](../images/bitcoin-sync-1.png)

지갑주소 공개합니다. 0.00001 BTC있는데 가져가쉴??

현재 Sync상황을 알기 위해선, 일단 내 bitcoin-core가 얼마나 동기화 는지 확인 해야 한다.

```
bitcoind getinfo bitcoind getblockcount
```

getinfo 명령어는 bitcoin daemon 의 전반적인 상황을 볼 수 있고, bitcoind getblockcount는 현재 동기화된 block의 count를 볼 수 가 있다.

281205개가 동기화가 되어 있다.  전체 block의 height는 [blockchain.info/ko/](https://blockchain.info/ko/) 여기에서 확인할 수 있다.

현재 약 408506개가 있는데, 절반정도 동기화되어 있는 걸 볼 수 있다.

이는 약 11시간에 걸쳐서 동기화한 결과다.

스펙은 아래와 같다.

Google Cloud Compute

n1-standard-2(vCPU 2개, 7.5GB 메모리)  
 Intel Haswell  
 SSD 75 GB

CPU 사용량 ( 사용량이 많이 튀는 시간 대는 내가 환경설정을 세팅할 때다.)

![스크린샷 2016-04-20 09.05.00](../images/bitcoin-sync-2.png)

용량

30여만개의 정보들은 약 20기가를 차지한다. (core 포함)

```
 savurself11@bitcoin:~/bitcoin$ df -h
 Filesystem Size Used Avail Use% Mounted on udev 3.7G 8.0K 3.7G 1%
 /dev tmpfs 749M 340K 748M 1%
 /run /dev/sda1 74G 20G 52G 28%
 / none 4.0K 0 4.0K 0%
 /sys/fs/cgroup none 5.0M 0 5.0M 0%
 /run/lock none 3.7G 0 3.7G 0%
 /run/shm none 100M 0 100M 0% /run/user
```

음.. 아마 8시간이 더 있으면 동기화가 완료 될 것 같다.

동기화가 완료되는 대로 다시 포스팅 해야지.

---

8시간후

서버를 켜 놓은지 약 20여 시간이 지났다.

그 사이에 컴퓨터 스펙을 업그레이드 했다. (CPU 4개, Ram 16gb)

실시간으로 프로세서의 상황을 모니터링하고 있었는데, 너무나 과부하가 걸리고 있어서 어쩔 수 없었다. (내돈 ㅠㅠ)

난 그저 내가 보낸 transaction 을 보고 싶었을  뿐인데, 이를 bitcoin-core로 보기위해서는 어마무시한 노력이 필요하다는 것을 알게 됐다.

```
 savurself11@bitcoin:~$ bitcoin-cli getinfo
```

```json
{
  "version": 90500,
  "protocolversion": 70002,
  "walletversion": 60000,
  "balance": 0.0,
  "blocks": 324389,
  "timeoffset": -1,
  "connections": 8,
  "proxy": "",
  "difficulty": 34661425923.97693634,
  "testnet": false,
  "keypoololdest": 1461075368,
  "keypoolsize": 101,
  "paytxfee": 0.0,
  "relayfee": 0.00001,
  "errors": ""
}
```

```
savurself11@bitcoin:~$ bitcoin-cli getblockcount 324398 savurself11@bitcoin:~$
```

전체 블록인 40만 8천여개 중에 이제 32만개를 돌파하고 있다. 이정도 추세라면 내일이 되어서야 완료될 성 싶다.

![스크린샷 2016-04-20 20.00.01](../images/bitcoin-sync-1.png)

--

정확히 이틀이 지났다.

어제 32만개라고 포스팅했는데, 간밤에 4만개 밖에 동기화 하지 못했다. ㅠㅠ

bitcoin-cli 의 현재 sync 상황을 확인해보자.

```
savurself11@bitcoin:~$ bitcoin-cli getblockcount
364511
```

364511 번 까지 왔다. 이 블록은 언제적 블록인지 알아보자.

```
savurself11@bitcoin:~$ bitcoin-cli getblockhash 364511 000000000000000014136f884dbf60e529a1cd296d3b321bcac22420c97be03d
```

블록 해쉬 정보만 있으면, 이 블록에 대한 정보를 가져올 수 있다.

```
bitcoin-cli getblock "000000000000000014136f884dbf60e529a1cd296d3b321bcac22420c97be03d"
```

transaction정보는 너무 많아서 생략한다.

```json
{
  "time": 1436418994,
  "nonce": 2453655801,
  "bits": "1816418e",
  "difficulty": 49402014931.22746277,
  "chainwork": "0000000000000000000000000000000000000000000847c569a240f670fc6820",
  "previousblockhash": "000000000000000006588b7d0aefa8045d5c6822e975b5d37558610f406880ac",
  "nextblockhash": "0000000000000000015b001aca1ba32cfedc54fe3deae0f4b724b321e3b7b425"
}
```



```
savurself11@bitcoin:~$ date -d @1436418994
Thu Jul 9 05:16:34 UTC 2015
```

2015년 7월 정보까지 따라왔다. 이 정도 추세면 내일 모레쯤이면 따라 잡을 수 있을 것 같기도하다.

---

동기화가 완료되었다 으헝헣ㅎ
그 사이에 google cloud compute engine(이하 gce) 의 쿠폰 유효기간이 다해 (내 300달러 ㅠㅠ) 과금이 되고 있었다.

4월 21 경에 서버를 돌리다가 과부하 되는 cpu 가 안쓰러워 업그레이드를 했고

![Compute Engine study workspace](../images/bitcoin-sync-4.png)

5시간 전에는 70기가에서 100기가로 업그레이드를 했다.

**참고로 디스크 점유율이 90%를 넘어가면 동기화를 정지한다.**

이제 내가 보낸 비트코인정보를 내 서버에서 볼 수 있게 되었다.

내 30만원 ^^

2018년 현재: 거래량이 많아진 지금은 더 오랜 시간이 소요될 것.
