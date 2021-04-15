---
title: 'nodejs의 멀티쓰레딩과 worker threads'
tags:
  - nodejs
  - javascript
published: true
date: 2021-04-15 17:09:10
description: '그 놈의 싱글스레드'
---

nodejs 10.5.0 버전 이후에서 부터는 `worker_threads`가 가능해졌고, 12 LTS 부터 stable로 자리잡았다. 이 `worker_threads`는 무엇이고 어떤 역할을 하는 것일까?

## 싱글스레드 세상

자바스크립트는 브라우저에서 실행되는 단일 스레드 프로그래밍 언어로 설계되었다. 단일 스레드 라는 것은, 하나의 프로세스 (브라우저 또는 모던 브라우저의 경우 하나의 탭)에서 하나의 명령어 집합만 실행된다는 것을 의미한다.

이러한 설계는 개발자들이 언어를 사용하는데 있어서 더 쉽게 할 수 있는 요소가 되었다. 자바스크립트는 처음에 웹 페이지, 폼 유효성 검사 등의 상호작용을 추가하는데만 사용되었었는데, 이 정도 작업으로는 멀티스레딩의 복잡함이 전혀 필요하지 않았다.

그러나 nodejs의 창시자(Ryan Dahl)는 이러한 한계를 기회로 보았다. 그는 비동기 IO 를 기반으로 서버측 플랫폼을 구현하고 싶어했다. 즉, 쓰레드가 필요하지 않았다. 동시성은 매우 풀기 어려운 문제가 될 수 있다. 동일한 메모리에 접근하려고 하는 스레드가 많아지면 재현 및 수정이 매우 어려운 race condition 문제가 발생할 수 있다.

## nodejs는 싱글스레드 인가?

그래서, nodejs 애플리케이션은 단일 스레드일까? 어느정도는 그렇다.

사실 우리는 병렬로 실행할 수 있지만, 우리는 쓰레드를 만들지도 않고 이를 동기화 하지도 않는다. 가상머신과 운영체재가 IO를 병렬로 실행하며, 자바스크립트 코드로 데이터를 다시 전송해야 할 때, 자바스크립트 부분은 단일 쓰레드로 실행된다.

즉, 자바스크립트 코드를 제외한 모든 것이 병결로 실행된다. 자바스크립트 코드의 동시식 블록은 항상 한번에 하나씩 실행된다.

```javascript
let flag = false
function doSomething() {
  flag = true
  // flag를 수정하지 않는 더 많은 코드...

  // 우리는 flag가 true라는 것은 확신할 수 있다.
  // 이 코드 블록이 동기로 실행되는 이상,
  // 이 코드블록이 flag 값을 바꿀일이 없다.
}
```

우리가 처리하는 것이 모두 비동기 IO로 이루어져있다면 이러한 방식은 매우 훌륭하다고 볼 수 있다. 코드는 빠르게 실행되고, 데이터를 파일과 스트림에 전달하는 동기 블록은 작은 부분으로 구성된다. 자바스크립트 코드는 너무 빨라서, 다른 자바스크립트의 실행을 막지 못한다. 자바스크립트 코드가 실행될 때 보다, IO 이벤트가 발생할 때 까지 기다리는 시간이 훨씬 더 많다. 아래 예를 살펴보자.

```javascript
// 2
db.findOne('SELECT ... LIMIT 1', function (err, result) {
  if (err) return console.error(err)
  console.log(result)
})

// 1
console.log('Running query')

// 3
setTimeout(function () {
  console.log('Hey there')
}, 1000)
```

데이터베이스로 실행한 이 쿼리는 몇 분 정도 걸릴 수 있지만, `Running query` 메시지는 쿼리를 호출한 후 즉시 표시된다. 그리고 잠시 후 쿼리가 여전히 실행중인지 아닌지를 확인 한후, 잠시후에 `Hey there` 메시지를 볼 수 있다.

nodejs 애플리케이션은 함수를 호출할 뿐, 다른 코드의 실행을 차단하지 않는다. 조회사 완료되면 콜백을 통해 알림을 받고, 결과를 받는다.

## CPU 집약적인 작업

만일 큰 데이터를 가지고 메모리에서 복잡한 계산을 수행하는 것 과 같이, 동기 집약적인 작업을 수행해야하는 경우엔 어떻게 될까? 그렇게 되면 많은 시간이 걸리게되고, 나머지 코드를 차단하는 동기 코드블록이 생길 수도 있다.

동기 코드 실행에 10초가 걸린다고 상상해보자. 웹서버를 실행중이면, 이 작업 때문에 다른 요청이 최소 10초동안 차단된다. 100ms이상 걸리는 작업은 문제를 유발할 수 있다.

자바스크립트와 nodejs는 CPU 바인딩 작업에 사용할 수 없었다. 자바스크립트는 단일 스레드이기 때문에, 브라우저의 UI는 멈추게되고, Nodejs의 IO 이벤트를 넣게 된다.

```javascript
db.findAll('SELECT ...', function (err, results) {
  if (err) return console.error(err)

  // 겁나 오래 걸리는 작업
  for (const encrypted of results) {
    const plainText = decrypt(encrypted)
    console.log(plainText)
  }
})
```

조회가 완료되면 콜백이 실행된다. 콜백이 실행 끝날 때 까지 자바스크립트 코드가 실행되지 않는다.

일반적으로, 앞서 이야기 한 것 처럼 코드는 일반적으로 매우 작고 빠르다. 그러나 위의 예제 코드에서는, 많은 결과가 나오고, 이에 따른 많은 계산이 필요하다. 이는 몇 초 정도 걸릴 수 있으며, 이 기간 동안은 다른 자바스크립트 실행이 대기 중이 기 때문에, 동일한 애플리케이션에서 서버를 실행하는 경우 해당 시간 동안 모든 사용자가 블로킹 될 수 있다.

## 자바스크립트에서 쓰레드가 없는 이유

그래서, 많은 사람들이 nodejs 코어에 새 모듈을 추가하여 스레드를 만들고 동기화 할 수 있어야 한다고 생각한다.

만약, nodejs에서 쓰레드가 추가된다면, 이는 언어의 본질을 바꿔버리는 것이다. 단순히 클래스 또는 함수 추가 만으로 쓰레드를 만들 수는 없다. 언어를 바꿀 필요가 있다. 멀티스레딩을 지원하는 언어에는 스레드간 협력이 가능하도록 `synchronized`와 같은 키워드가 있다.

예를 들어, 자바에서는 일부 숫자 유형의 경우 원자형이 아니다. 액세스를 동기화 하지 않는다면, 두개의 스레드가 변수의 값을 변경할 수 있다. 결과적으로 두개의 스레드가 변수에 액세스하고, 하나의 스레드에 의해 몇 바이트가 변경되고, 다른 스레드에 의해 몇 바이트가 변경되므로 유효한 값을 얻지 못할 것이다.

## 일단 간단한 해결책

nodejs는 이벤트 큐의 다음 코드블록을, 이전 코드블록의 실행이 완료될 때까지 평가하지 않는다. 그래서 할 수 있는 간단한 방법 중 하나는, 코드를 작은 동기식 코드로 나누고, nodejs에 이 작업이 끝났다고 전달한 이후에 큐에서 보류 중인 것들을 계속해서 실행할 수 있도록 하는 것이다.

```javascript
const arr = [
  /*겁나 큰 배열*/
]
for (const item of arr) {
  // 무거운 작업
}
// 저 포문이 끝날 때 까지 여기는 오지 못함
```

이를 작은 청크로 나눠서 실행해보자.

```javascript
const crypto = require('crypto')

const arr = new Array(200).fill('something')
function processChunk() {
  if (arr.length === 0) {
    // code that runs after the whole array is executed
    // 모든 배열이 실행이 끝난 뒤에 실행됨
  } else {
    console.log('processing chunk')
    /// 10개만 추출
    const subarr = arr.splice(0, 10)
    for (const item of subarr) {
      // 오래 걸리는 작업
      doHeavyStuff(item)
    }
    // 다음 큐로 작업을 밀어넣음
    setImmediate(processChunk)
  }
}

processChunk()

function doHeavyStuff(item) {
  crypto
    .createHmac('sha256', 'secret')
    .update(new Array(10000).fill(item).join('.'))
    .digest('hex')
}

// 다른 작업도 가능한지 확인하기 위한 함수
let interval = setInterval(() => {
  console.log('tick!')
  if (arr.length === 0) clearInterval(interval)
}, 0)
```

이제 `setImmediate(callback)`가 실행될 때마다 10개씩 작업을 처리하게 되며, 이외에 무언가 작업할게 생기게 되면 이 작업 사이에 처리하게 된다.

그러나 보다시피 코드가 더 복잡해졌다. 또한 알고리즘은 이보다 더 복잡하기 때문에 어디서 적절히 `setImmediate()`를 배치해야 할지 알 수 없다. 게다가 이제 코드는 비동기 식이며, 다른 외부 라이브러리에 의존하게 되면 실행을 더 작은 청크로 분할해서 실행하기 어려워 질 수도 있다.

## 백그라운드 프로세스

`setImmediate()`는 간단한 시나리오에서는 쓸만했지만, 아주 적당한 해결책이라 보기 어렵다.

쓰레드 없이 프로세스를 병렬로 처리하는 것이 가능할까? 그렇다. 우리에게 필요한 것은 충분한 CPU와 시간을 활용해서 결과를 애플리케이션으로 되돌 릴 수 있는 일종의 백그라운드 처리이다.

```javascript
// `script.js`를 별도의 환경에서 실행하여 메모리를 공유하지 않는다.
const service = createService('script.js')
// 여기에 데이터를 넘기고 결과를 받는다.
service.compute(data, function (err, result) {
  // 결과
})
```

사실 우리는 이미 nodejs에서 백그라운드 처리를 할 수 있다. 프로세스를 fork 하여 메시지를 전달하는 방식으로 구현이 가능하다. 메인 프로세스에서 하위 프로세스로 이벤트를 주고 받는 방식으로 통신이 가능하다.

메모리 공유는 없다. 서로 교환되는 데이터는 모두 복제된 데이터이며, 한쪽 데이터를 변경한다고 다른 쪽에 변경이 일어나지는 않는다.

그러나 이는 해결책이긴 하지만, 이상적인 해결책은 아니다. 프로세스를 만드는 것은 리소스 측면에서 많은 비용이 든다. 그리고 느리다. 프로세스가 메모리를 공유하지 않기 때문에, 많은 메모리를 사용하여 새 가상 시스템을 처음부터 실행해야 한다.

물론 동일한 포크 프로세스를 재사용할 수 있다. 그러나 포킹된 프로세스 내에서 동시에 실행되는 서로 다른 과중한 워크로드를 전송하면 두가지 문제가 발생한다.

먼저, 메인 애플리케이션은 차단하지 않을 수 도 있지만, 포크된 프로세스는 한번에 하나의 작업만 처리할 수 있다. 10초가 걸리는 작업과 1초가 걸리는 작업이 순서대로 대기하는 경우, 두번쨰 작업을 실행하기 위해 10초를 기다리는 것은 이상적이지 않다.

또 다른 문제로, 한작업이 프로세스가 중단되면 동일하나 프로세스에 전송되는 모든 작업이 완료되지 않은 채로 남아있게 된다. 이러한 문제를 해결하기 위해서는 포크가 한 개가 아니고 여러개가 있어야 한다. 그러나 각 프로세스마다 모든 가상 시스템 코드가 메모리에 중복되므로, 프로세스당 몇 MBs의 처리시간과 부팅시간을 계산해서 포크되는 프로세스의 개수를 제한해야 한다.

따라서 데이터베이스 연결과 마찬가지로, 사용할 수 있는 일종의 프로세스 풀이 필요하고, 각 프로세스마다 한번에 작업을 실행하고, 작업이 완료된 후 프로세스를 다시 사용해야 한다. 구현이 매우 복잡해보인다. [worker-farm](https://github.com/rvagg/node-worker-farm)이라는 것을 사용해보자.

```javascript
// main app
const workerFarm = require('worker-farm')
const service = workerFarm(require.resolve('./script'))

service('hello', function (err, output) {
  console.log(output)
})

// script.js
// 여기는 포크 프로세스에서 실행됨
module.exports = (input, callback) => {
  callback(null, input + ' ' + world)
}
```

## 문제 해결?

뭐, 문제는 해결되었지만 멀티스레드 솔루션보다 훨씬 더 많은 메모리를 쓰고 있다. 스레드들은 포크 프로세스에 비해 리소스 측면에서 매우 가볍다. 이러한 이유 때문에 Worker Thread가 탄생하게 되었다.

Worker Thread에는 분리된 컨텍스트가 있다. 메시지 패싱을 활용하여 메인 프로세스와 정보를 교환하기 때문에 레이스 컨디션 문제를 해결할 수 있다. 또한 이들은 같은 프로세스에 존재하기 때문에 더 적은 메모리를 쓴다.

Worker Thread와 메모리도 공유할 수 있다. 이러한 용도로 많이 사용되는 객체 `SharedArrayBuffer`를 활용하여 객체를 전달할 수 있다. 많은 양의 데이터를 활용하여 CPU 집약적인 작업을 수행할 때만 이 기능을 사용해야 한다.

## Worker Thread 예제

노드 10버전 이상을 사용하고 있다면 `worker_threads`를 쓸 수 있다. 그러나 11.7버전 이하에서는 `--experimental-worker`를 추가해야 사용이 가능하다.

한가지 명심해야할 것은, 아무리 프로세스 포킹보다 저렴하다 하더라도 너무 많이 워커를 생성하면 리소스를 많이 사용할 수도 있다는 것이다. 이 경우에는, 워커 풀응ㄹ 만들기를 권장한다. 이 워커 풀도 마찬가지로 직접 구현하는대신, 워커 풀을 구현한 다른 패키지를 npm에서 찾을 수 있다.

간단한 예제에서 시작해보자. 먼저 Worker thread를 만드는 메인 파일을 구현하고, 데이터를 넘긴다. API는 이벤트 드리븐이지만 Promise로 감싸서 워커로부터 첫번째 메시지를 받는다면 `resolve`하도록 한다.

```javascript
// index.js
const { Worker } = require('worker_threads')

function runService(workerData) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('./service.js', { workerData })
    worker.on('message', resolve)
    worker.on('error', reject)
    worker.on('exit', (code) => {
      if (code !== 0) reject(new Error(`Worker stopped with exit code ${code}`))
    })
  })
}

async function run() {
  const result = await runService('world')
  console.log(result)
}

run().catch((err) => console.error(err))
```

보시다시피, 파일 이름과 데이터를 argument로 넘기는 것 만으로도 쉽게 구현이 가능하다. 이 데이터는 복제되었다. 그런 다음 메시지 이벤트를 리슨하여 Worker Thread가 메시지를 보낼 때 까지 기다린다.

```javascript
const { workerData, parentPort } = require('worker_threads')

// 여기에서 무거운 작업을 동기로 메인 스레드를 방해하지 않으면서 처리할 수 있다.
parentPort.postMessage({ hello: workerData })
```

여기서는 메인 애플리케이션이 보낸 `workerData`와 메인 애플리케이션으로 정보를 돌려보내는 방법 이 두가지가 필요하다. 작업이 끝나면 `parentPort.postMessage`를 통해서 결과를 보낼 수 있다.

이게 접누다. 이는 간단한 예제지만, 우리는 더 복잡한 것들을 할 수 있다. 예를 들어 피드백을 보내야 하는 경우, Worker thread에서 실행 상태를 나타내는 메시지를 여러개 보낼 수도 있다. 아니면 일부 결과만 보낼 수 있다. 수천개의 이미지를 처리한다고 가정해보자. 처리된 이미지별로 보낼 수 있지만, 이 모든작업이 처리될 때 까지 기다리는 것은 좋지 않다.

https://nodejs.org/docs/latest-v10.x/api/worker_threads.html

## 웹 워커?

아마도 Web Worker API에 대해 들어본 적이 있을 것이다.

- https://developer.mozilla.org/ko/docs/Web/API/Web_Workers_API
- https://caniuse.com/webworkers

이는 웹에서 지원하는 api고, 모던 브라우저에서 잘 지원되고 있다. (심지어 IE11에서도?!) API는 요구사항과 기술조건들이 제각각이지만, 브라우저 런타임에서 유사한 문제를 충분히 해결할 수 있다. 웹 애플리케이션에서 암호화, 압축/압축해제, 이미지조작, 컴퓨터 비전(얼굴인식) 등을 수행하는 하는 경우 유용하다.

## 결론

Web Worker는 Nodejs 애플리케이션에서 CPU 집약적인 작업을 할 때 유용하다. 이는 마치 공유메모리가 없는 쓰레드로, 레이스 컨디션과 같은 문제를 피할 수 있다. `worker_threads`는 nodejs 12 LTS에서 부터 정식 지원하므로, 한번 사용해봄직하다.
