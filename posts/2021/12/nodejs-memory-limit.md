---
title: 'nodejs의 메모리 제한'
tags:
  - javascript
  - nodejs
published: true
date: 2021-12-13 19:21:45
description: '어디서 새고 있을까 내 메모리는'
---

## V8 가비지 콜렉션

힙은 메모리 할당이 필요한 곳이고, 이는 여러 `generational regions`로 나뉜다. 이 `region`들은 단순히 `generations`이라고 불리우고, 이 객체들은 라이프 사이클 동안 같은 세대 (generation)을 공유한다.

여기에는 `young generation`과 `old generation`이 있다. 그리고 `young generation`의 `young objects`는 또다시 `nursery`(유아)와 `intermediate`(중간) 세대로 나뉜다. 이 객체들이 가비지 컬렉션에서 살아남게 되면, `older generation`에 합류하게 된다.

![generation](https://v8.dev/_img/trash-talk/02.svg)

이 `generation` 가설의 기본 원리는 대부분의 객체가 older로 넘어가기 전에 죽는다. (가비지 콜렉팅 당한다)는 것이다. V8 가비지 컬렉터는 이러한 기본적인 가정을 기반으로 설계 되어 있으며, 여기에서 살아남은 객체만 승격하게 된다. 객체는 살아남으면서 다음 영역으로 복사되고, 그리고 결국엔 `old generation`이 되는 것이다.

node에서 메모리가 소비되는 영역은 크게 세군데로 볼 수 있다.

- code
- call stack: 숫자, 문자열, boolean 과 같은 primitive values 또는 함수
- heap memory

우리는 여기에서 힙 메모리를 중점적으로 볼 것이다.

가비지 콜렉터에 대해 간단히 알아봤으니, 힙에 메모리를 할당해보자.

```javascript
function allocateMemory(size) {
  // Simulate allocation of bytes
  const numbers = size / 8
  const arr = []
  arr.length = numbers
  for (let i = 0; i < numbers; i++) {
    arr[i] = i
  }
  return arr
}
```

지역 변수는 함수 호출이 call stack에서 끝나는 즉시 `young generation`에 있다가 사라지게 된다. 숫자와 같은 기본형 변수들은 힙에 도달하지 못하고 대신 호출 스택에서 할당된다. `arr`의 경우 힙에 들어가서 가비지 콜렉션에서 살아남을 수 있다.

## 힙 메모리에 제한이 있을까?

이제 노드 프로세스를 최대 용량으로 밀어넣고, 힙 메모리가 언제쯤 고갈되는지 살펴보자.

```javascript
const memoryLeakAllocations = []

const field = 'heapUsed'
const allocationStep = 10000 * 1024 // 10MB

const TIME_INTERVAL_IN_MSEC = 40

setInterval(() => {
  const allocation = allocateMemory(allocationStep)

  memoryLeakAllocations.push(allocation)

  const mu = process.memoryUsage()
  // # bytes / KB / MB / GB
  const gbNow = mu[field] / 1024 / 1024 / 1024
  const gbRounded = Math.round(gbNow * 100) / 100

  console.log(`Heap allocated ${gbRounded} GB`)
}, TIME_INTERVAL_IN_MSEC)
```

위 코드는 40ms 간격으로 10메가바이트를 계속 할당하므로, 가비지 콜렉팅에 필요한 시간이 남아있는 객체들을 `old generation`으로 빠르게 승격시킬 수 있다. `process.memoryUsage`는 현재 힙 사용률에 대한 지표를 수집할 수 있는 도구다. 힙 할당량이 커지면, `heapUsed` 필드에서 현재 힙 사이즈를 추적한다.

결과는 실행환경에 따라 다르다. 16gb 메모리가 있는 내 맥에서는 다음과 같은 결과가 나왔다.

```
...
Heap allocated 3.95 GB
Heap allocated 3.96 GB
Heap allocated 3.97 GB
Heap allocated 3.98 GB
Heap allocated 3.99 GB
Heap allocated 4 GB

<--- Last few GCs --->

[88809:0x130008000]    23137 ms: Scavenge (reduce) 4085.6 (4094.2) -> 4085.6 (4094.2) MB, 1.6 / 0.0 ms  (average mu = 0.855, current mu = 0.691) allocation failure
[88809:0x130008000]    23449 ms: Mark-sweep (reduce) 4095.4 (4104.0) -> 4095.3 (4104.0) MB, 274.1 / 0.0 ms  (+ 138.5 ms in 153 steps since start of marking, biggest step 6.2 ms, walltime since start of marking -558038699 ms) (average mu = 0.740, current m

<--- JS stacktrace --->

FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory
```

여기에서 가비지 콜렉터는 `heap out of memory` 예외를 던지기 전에 마지막 수단으로 메모리 압축을 시도하는 것을 볼 수 있다. 이 프로세스는 4.1gb까지 도달했고, 23.1초 정도가 소요 되었다.

## 메모리 할당량 늘리기

`--max-old-space-size` 파라미터를 사용하면 크기를 늘릴 수 있다.

```bash
node index.js --max-old-space-size=8000
```

위 커맨드에서는 최대 제한을 8gb로 설정했다. 이 크기를 설정할 때는 조심해야 한다. RAM에 물리적으로 사용가능한 공간을 설정해두는 것이 좋다. 물리적 메모리가 부족하면, 프로세스는 가상 메모리를 통해 디스크 공간을 확보하기 시작한다. 이 제한을 너무 높게 설정하면 PC가 손상될 수 있다.

```
...
Heap allocated 7.8 GB
Heap allocated 7.8 GB
Heap allocated 7.81 GB

<--- Last few GCs --->

[89239:0x148008000]    51777 ms: Mark-sweep (reduce) 7992.0 (8006.7) -> 7991.8 (8006.7) MB, 2770.5 / 0.0 ms  (+ 106.4 ms in 97 steps since start of marking, biggest step 8.0 ms, walltime since start of marking -558036240 ms) (average mu = 0.302, current m[89239:0x148008000]    54751 ms: Mark-sweep (reduce) 8001.7 (8016.5) -> 8001.6 (8016.5) MB, 2968.3 / 0.0 ms  (average mu = 0.171, current mu = 0.002) allocation failure scavenge might not succeed


<--- JS stacktrace --->

FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory
```

프로덕션에서는 메모리가 부족해지는 데에는 1분도 채 걸리지 않을 수 있다. 이것이 메모리 소비량을 계속해서 모니터링하고 파악해야 하는 이유 중 하나다. 메모리 소비량은 시간이 지남에 따라 점차 느리게 증가할 수 있고, 문제가 있다는 것을 알 때 까지 며칠이 더 걸릴 수 잇다. 프로세스가 계속 충돌하고, 메모리 부족 예외가 로그에 표시되면 코드에서 메모리 누수가 발생한 것일 수 있다.

또한 프로세스는 더 많은 데이터로 작업 하기 때문에 더많은 메모리를 소비할 수 있다. 리소스 사용량이 계속 증가하면 이를 마이크로서비스로 분리해야 할 수도 있다. 마이크로 서비스로 분리하면 메모리 부담을 줄이고, 노드를 수평으로 확장할 수 있다.

## nodejs의 메모리 누수를 추적하는 방법

`process.memoryUsage` 함수내 `heapUsed` 변수는 유용하다. 메모리 누수를 디버깅하는 한가지 방법은 메모리 지표를 다른 도구에 넣어두는 것이다. 그러나 이 구현은 정교하지 않아서 분석을 할 때는 수동으로 해야 한다.

```javascript
const path = require('path')
const fs = require('fs')
const os = require('os')

const start = Date.now()
const LOG_FILE = path.join(__dirname, 'memory-usage.csv')

fs.writeFile(LOG_FILE, 'Time Alive (secs),Memory GB' + os.EOL, () => {}) // fire-and-forget
```

힙 할당 지표를 메모리에 저장하지 않기 위해 데이터를 쉽게 사용할 수 있도록 csv 파일에 쓰도록 처리한다. 만약 점진적으로 메모리 지표를 가져오기 위해서는 위 테스트 코드 `console.log` 상단에 아래 코드를 붙여 두면 된다.

```javascript
const elapsedTimeInSecs = (Date.now() - start) / 1000
const timeRounded = Math.round(elapsedTimeInSecs * 100) / 100

s.appendFile(LOG_FILE, timeRounded + ',' + gbRounded + os.EOL, () => {}) // fire-and-forget
```

이 코드를 사용하면 시간이 지남에 따라, 힙 사용이 증가한다면 메모리 누수를 디버깅할 수 있다.

### index.js

```javascript
function allocateMemory(size) {
  // Simulate allocation of bytes
  const numbers = size / 8
  const arr = []
  arr.length = numbers
  for (let i = 0; i < numbers; i++) {
    arr[i] = i
  }
  return arr
}

const path = require('path')
const fs = require('fs')
const os = require('os')

const memoryLeakAllocations = []

const field = 'heapUsed'
const allocationStep = 10000 * 1024 // 10MB

const TIME_INTERVAL_IN_MSEC = 40

setInterval(() => {
  const allocation = allocateMemory(allocationStep)

  memoryLeakAllocations.push(allocation)

  const mu = process.memoryUsage()
  // # bytes / KB / MB / GB
  const gbNow = mu[field] / 1024 / 1024 / 1024
  const gbRounded = Math.round(gbNow * 100) / 100

  const start = Date.now()
  const LOG_FILE = path.join(__dirname, 'memory-usage.csv')

  const elapsedTimeInSecs = (Date.now() - start) / 1000
  const timeRounded = Math.round(elapsedTimeInSecs * 100) / 100

  s.appendFile(LOG_FILE, timeRounded + ',' + gbRounded + os.EOL, () => {})
  console.log(`Heap allocated ${gbRounded} GB`)
}, TIME_INTERVAL_IN_MSEC)
```

![memory-usage](./images/memory-usage.png)

메모리 누수 감지 코드를 재사용할 수 있게 만드는 방법 중 하나는, 이 누수 감지 코드가 메인 루프 내부에 존재할 필요가 없으므로 이 코드를 자체 간격으로 실행될 수 있도록 래핑하는 것이다.

```javascript
setInterval(() => {
  const mu = process.memoryUsage()
  // # bytes / KB / MB / GB
  const gbNow = mu[field] / 1024 / 1024 / 1024
  const gbRounded = Math.round(gbNow * 100) / 100

  const elapsedTimeInSecs = (Date.now() - start) / 1000
  const timeRounded = Math.round(elapsedTimeInSecs * 100) / 100

  fs.appendFile(LOG_FILE, timeRounded + ',' + gbRounded + os.EOL, () => {}) // fire-and-forget
}, TIME_INTERVAL_IN_MSEC)
```

이는 운영용 코드로는 쓸 수 없지만, 적어도 로컬 에서 메모리 누수를 디버깅하는 방법을 보여주었다.실제 구현에서는 서버 디스크 공간이 부족하지 않도록 하는 설정, 비주얼, 알림, 로그 rotate 등이 필요하다.

## 프로덕션 코드에서 메모리 누수 추적하기

위 코드를 프로덕션에서 쓰는 것은 무리 일 것이다. 프로덕션에서는 [PM2와 같은 데몬 프로세스](https://pm2.keymetrics.io/docs/usage/restart-strategies/)를 활용하여 추적할 수 있을 것이다.

```bash
pm2 start index.js --max-memory-restart 8G
```

또다른 도구로는 [node-memwatch](https://github.com/lloyd/node-memwatch)가 있다. 이 라이브러리는 메모리 누수가 발생하면 특정 코드를 실행시킬 수 있다.

```javascript
const memwatch = require('memwatch')

memwatch.on('leak', function (info) {
  // event emitted
  console.log(info.reason)
})
```
