---
title: 이더리움 - 프라이빗 블록체인 만들기 (1)
date: 2018-07-03 07:51:20
published: true
tags:
  - ethereum
description: "Geth client 설치부터 사설 블록체인 시작까지 본 포스팅은 ubuntu 16.x 버전을 기준으로
  작성되었습니다.  ### 1. 이더리움 설치  ``` sudo apt-get install software-properties-common
  sudo add-apt-repository -y ppa:ethereum/ethereum sudo apt-get u..."
category: ethereum
slug: /2018/07/03/ethereum-create-private-blockchain-1/
template: post
---
Geth client 설치부터 사설 블록체인 시작까지

본 포스팅은 ubuntu 16.x 버전을 기준으로 작성되었습니다.

### 1. 이더리움 설치

```
sudo apt-get install software-properties-common
sudo add-apt-repository -y ppa:ethereum/ethereum
sudo apt-get update
sudo apt-cache madison geth
sudo apt-get -y install ethereum
```

### 2. geth 실행 확인해보기

```
deploy@jayg-blockchain2:~$ geth
INFO [07-03|10:55:27] Maximum peer count                       ETH=25 LES=0 total=25
INFO [07-03|10:55:27] Starting peer-to-peer node               instance=Geth/v1.8.11-stable-dea1ce05/linux-amd64/go1.10
INFO [07-03|10:55:27] Allocated cache and file handles         database=/home/deploy/.ethereum/geth/chaindata cache=768 handles=1024
INFO [07-03|10:55:27] Initialised chain configuration          config="{ChainID: 1 Homestead: 1150000 DAO: 1920000 DAOSupport: true EIP150: 2463000 EIP155: 2675000 EIP158: 2675000 Byzantium: 4370000 Constantinople: <nil> Engine: ethash}"
INFO [07-03|10:55:27] Disk storage enabled for ethash caches   dir=/home/deploy/.ethereum/geth/ethash count=3
INFO [07-03|10:55:27] Disk storage enabled for ethash DAGs     dir=/home/deploy/.ethash               count=2
INFO [07-03|10:55:27] Initialising Ethereum protocol           versions="[63 62]" network=1
INFO [07-03|10:55:27] Loaded most recent local header          number=0 hash=d4e567…cb8fa3 td=17179869184
INFO [07-03|10:55:27] Loaded most recent local full block      number=0 hash=d4e567…cb8fa3 td=17179869184
INFO [07-03|10:55:27] Loaded most recent local fast block      number=0 hash=d4e567…cb8fa3 td=17179869184
INFO [07-03|10:55:27] Loaded local transaction journal         transactions=0 dropped=0
INFO [07-03|10:55:27] Regenerated local transaction journal    transactions=0 accounts=0
INFO [07-03|10:55:27] Starting P2P networking
INFO [07-03|10:55:29] UDP listener up                          self=enode://78fe76020fb45f87bc6633033c9a176893601f10b45e7347b34a9f0036236b72713b9dd2fb29249d95b9e64cc7da50e94899a87357e5dd1c47f9837abe16976b@[::]:30303
INFO [07-03|10:55:29] RLPx listener up                         self=enode://78fe76020fb45f87bc6633033c9a176893601f10b45e7347b34a9f0036236b72713b9dd2fb29249d95b9e64cc7da50e94899a87357e5dd1c47f9837abe16976b@[::]:30303
INFO [07-03|10:55:29] IPC endpoint opened                      url=/home/deploy/.ethereum/geth.ipc
```

정상적으로 설치 된 것 같습니다.

### 3. Account 만들기

```
deploy@jayg-blockchain3:~$ geth --datadir ./ethereum/data/ account new
INFO [07-03|10:42:38] Maximum peer count                       ETH=25 LES=0 total=25
Your new account is locked with a password. Please give a password. Do not forget this password.
Passphrase:
Repeat passphrase:
Address: {44e74080949320292839b9a0df55e4459dd51434}
```

암호를 입력하면 어카운트가 생성 됩니다.

### 4. Account 확인하기

```
deploy@jayg-blockchain3:~$ geth --datadir ./ethereum/data/ account list
INFO [07-03|10:42:57] Maximum peer count                       ETH=25 LES=0 total=25
Account #0: {44e74080949320292839b9a0df55e4459dd51434} keystore:///home/deploy/ethereum/data/keystore/UTC--2018-07-03T01-42-40.966434322Z--44e74080949320292839b9a0df55e4459dd51434
```

### 5. Genesis block 만들기

Genesis Block은 블록체인에서 가장 첫번째로 생성되는 블록으로, 이전 블록에 대한 정보를 갖고 있지 않은 유일한 블록입니다. 이더리움은 이 블록에 많은 것들을 저장할 수 있도록 다양하나 옵션을 지원하며, 원하는 옵션을 구현함으로써 사설 블록체인 네트워크를 구축할 수 있습니다. 네트워크에 참여하는 노드들은 모두 이 블록을 가지고 있어야 참여할 수 있습니다.

```json
{
  "config": {
    "chainId": 15,
    "homesteadBlock": 0,
    "eip155Block": 0,
    "eip158Block": 0
  },
  "difficulty": "20",
  "gasLimit": "2100000",
  "alloc": {
    "44e74080949320292839b9a0df55e4459dd51434": { "balance": "300000" }
  }
}
```

- `config`: 이더리움관련 설정이 들어 있습니다.

- `config.chainId`: chain id는 현재 chain을 구별하는 값이며, [replay attack](https://en.wikipedia.org/wiki/Replay_attack)으로 부터 보호해주는 역할을 합니다. replay attack이란, 네트워크 공격의 한 종류로, 유효한 데이터 전송을 악의적으로 반복시키거나 지연시키는 공격의 일종입니다.

- `config.homesteadBlock`: homestead는 이더리움의 4단계 로드맵 중 두번째 메이저 단계입니다. [여기](http://news.joins.com/article/22016484)를 참조하시면 됩니다. 0은 여기서 true 정도를 의미한다고 생각하시면 됩니다.

- `config.epi155Block`: eip는 Ethereum Imporvement Proposal의 약자로, 개발자들이 이더리움을 업그레이드 하기 위해 제안된 아이디어를 의미합니다. 여기서느 epi155를 채택한다는 뜻이 되겠네요. 155 는 chainId와 마찬가지로, replay attack을 막기 위한 설정입니다. [여기](https://github.com/ethereum/EIPs/blob/master/EIPS/eip-155.md)를 참조하세요.

- `config.eip158Block`: state clearing입니다. 어카운트에서 상태 변경이 이루어지고, 이 변경으로 인해 계정 의 상태가 nonce=0, balance=0, code 및 storage가 빈 값이 되면 어카운트를 삭제한다는 것을 의미합니다. [여기](https://github.com/ethereum/EIPs/blob/master/EIPS/eip-158.md)를 참조하세요.

위 4가지 설정은 사설 블록체인을 만들 때 기본적으로 동일한 설정입니다.

- `difficulty`: 채굴 난이도 입니다. 값이 클 수록 채굴 난이도가 상승하고, 채굴에 오래 걸리니까 낮은 값으로 정해두었습니다.
- `gasLimit`: 블록당 가스(수수료)의 제한입니다. 몇 개의 거래를 하나의 블록에 담을 수 있는 지 결정하는데 필요한 옵션이고, 클 수록 거래를 & 테스트를 많이 할 수 있으므로 씨게 잡아둡시다.
- `alloc`: 블록생성과 동시에, 여기에 주소를 적어두면 이더를 원하는 만큼 이더를 보낼 수 있습니다.

#### 그 밖에 안넣은 값

- `parentHash`: 제네시스 블록은 부모가 없기 때문에 (ㅠㅠ), 넣지 않았습니다.
- `coinbase`: 블록 채굴시 주어지는 보상입니다. 어차피 내맘대로 alloc을 하는데, 채굴 보상이 의미가 없죠.
- `nonce`, `mixhash`: 이 두 개는 블록이 제대로 채굴되었는지 증명해주는 옵션입니다. 블록체인의 증명을 위해서는 mixhhash와 nonce가 조합된 hash 값이 일정한 수 이하인 nonce를 찾는데, 가장 최근에 추가된 블록의 헤더의 해시가 nonce 값의 조합으로 일정 수를 찾아 작업증명을 완료 하게 됩니다. 이를 가지고 있는 이유는, 공격자가 잘못된 nonce로 블록을 만들 경우, 이것이 위조되었는지 빠르게 확인하기 위해서 입니다. mixhash는 nonce를 찾기 위한 중간 계산 값입니다. 두 옵션 모두 '블록이 제대로 만들었는지 증명하는 용도' 로 이해하면 됩니다. 이 역시 제네시스 블록에서는 의미가 없는데, (어차피 채굴이 아니고 내가 만든거니깐) 랜덤한 값을 넣어서 다른 누군가 우연히 똑같은 제네시스블록으로 체인을 연결하지 않도록 할 수도 있습니다.
- `timestamp`: 해당 블록이 취득된 시점을 의미하는 값으로, 유닉스 타임스탬프 값이 들어갑니다. 어차피 최초의 블록이므로, 0 (0x00) 으로 설정해도 됩니다. 이는 블록간의 순서 및 난이도 조절 (간격이 짧으면 쉽고, 길면 어렵고)을 위해 쓰입니다.

파일을 생성하고, 제네시스 블록으로 시작합니다.

```
deploy@jayg-blockchain3:~$ touch genesis.json
deploy@jayg-blockchain3:~$ geth --datadir ./ethereum/data/ init ./genesis.json
INFO [07-03|11:57:49] Maximum peer count                       ETH=25 LES=0 total=25
INFO [07-03|11:57:49] Allocated cache and file handles         database=/home/deploy/ethereum/data/geth/chaindata cache=16 handles=16
INFO [07-03|11:57:49] Writing custom genesis block
INFO [07-03|11:57:49] Persisted trie from memory database      nodes=1 size=143.00B time=71.193µs gcnodes=0 gcsize=0.00B gctime=0s livenodes=1 livesize=0.00B
INFO [07-03|11:57:49] Successfully wrote genesis state         database=chaindata                                 hash=e41c72…24d37b
INFO [07-03|11:57:49] Allocated cache and file handles         database=/home/deploy/ethereum/data/geth/lightchaindata cache=16 handles=16
INFO [07-03|11:57:49] Writing custom genesis block
INFO [07-03|11:57:49] Persisted trie from memory database      nodes=1 size=143.00B time=52.505µs gcnodes=0 gcsize=0.00B gctime=0s livenodes=1 livesize=0.00B
INFO [07-03|11:57:49] Successfully wrote genesis state         database=lightchaindata                                 hash=e41c72…24d37b

```

### 6. 이더리움 네트워크 실행하기

```
deploy@jayg-blockchain3:~$ geth --identity 'PrivateNetwork' --datadir ./ethereum/data/ -port '33333' --rpc --rpcaddr 0.0.0.0 --rpcport '8123' --rpccorsdomain '*' --nodiscover --networkid 1900 --nat 'any' --rpcapi 'db,eth,net,web3,miner' console
INFO [07-03|12:07:27] Maximum peer count                       ETH=25 LES=0 total=25
INFO [07-03|12:07:27] Starting peer-to-peer node               instance=Geth/PrivateNetwork/v1.8.11-stable-dea1ce05/linux-amd64/go1.10
INFO [07-03|12:07:27] Allocated cache and file handles         database=/home/deploy/ethereum/data/geth/chaindata cache=768 handles=1024
INFO [07-03|12:07:27] Initialised chain configuration          config="{ChainID: 15 Homestead: 0 DAO: <nil> DAOSupport: false EIP150: <nil> EIP155: 0 EIP158: 0 Byzantium: <nil> Constantinople: <nil> Engine: unknown}"
INFO [07-03|12:07:27] Disk storage enabled for ethash caches   dir=/home/deploy/ethereum/data/geth/ethash count=3
INFO [07-03|12:07:27] Disk storage enabled for ethash DAGs     dir=/home/deploy/.ethash                   count=2
INFO [07-03|12:07:27] Initialising Ethereum protocol           versions="[63 62]" network=1900
INFO [07-03|12:07:27] Loaded most recent local header          number=0 hash=e41c72…24d37b td=20
INFO [07-03|12:07:27] Loaded most recent local full block      number=0 hash=e41c72…24d37b td=20
INFO [07-03|12:07:27] Loaded most recent local fast block      number=0 hash=e41c72…24d37b td=20
INFO [07-03|12:07:27] Regenerated local transaction journal    transactions=0 accounts=0
INFO [07-03|12:07:27] Starting P2P networking
INFO [07-03|12:07:27] RLPx listener up                         self="enode://08a5d152bedf418cc043b439737bb8f2203e0da33892cf7f1779bed890714f02cb7202394a7f95c12b69fd9696872aeea7ae5071d51b94d637b5c3e48723bd9d@[::]:33333?discport=0"
INFO [07-03|12:07:27] IPC endpoint opened                      url=/home/deploy/ethereum/data/geth.ipc
INFO [07-03|12:07:27] HTTP endpoint opened                     url=http://0.0.0.0:8123                 cors=* vhosts=localhost
Welcome to the Geth JavaScript console!

instance: Geth/PrivateNetwork/v1.8.11-stable-dea1ce05/linux-amd64/go1.10
INFO [07-03|12:07:27] Etherbase automatically configured       address=0x44E74080949320292839B9A0df55e4459dD51434
coinbase: 0x44e74080949320292839b9a0df55e4459dd51434
at block: 0 (Thu, 01 Jan 1970 09:00:00 KST)
 datadir: /home/deploy/ethereum/data
 modules: admin:1.0 debug:1.0 eth:1.0 miner:1.0 net:1.0 personal:1.0 rpc:1.0 txpool:1.0 web3:1.0
```

짜잔
