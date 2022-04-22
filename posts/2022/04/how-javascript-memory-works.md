---
title: '자바스크립트 메모리는 어떻게 이루어져있는가'
tags:
  - javascript
published: true
date: 2022-04-21 22:10:10
description: '메모리, 그리고 메모리'
---

## 거의 대부분은 힙에 존재한다.

**일반적으로 원시값은 스택에, 객체는 힙에 할당된다고 알려져있지만, 이와 반대로 자바스크립트의 모든 원시값도 힙에 할당되어 있다.** 이는 굳이 V8 소스코드를 까지 않고도 알 수 있는 방법이 존재한다.

1. 먼저 자신의 로컬 머신에 `node --v8-options` 명령어를 날려보자. 여기에는 다양한 노드 v8과 관련된 옵션값이 존재한다. 그 중에 `stack-size`라는 옵션을 활용하면, 로컬머신의 V8 스택 사이즈를 확인할 수 있다. 나는 최신버전의 (20220422 기준) 노드 v16.6.2를 사용하고 있는데, 864kb가 나왔다.
2. 자바스크립트 파일을 생성한다음, 엄청나게 큰 string을 만들고, `process.memoryUsage().heapUsed`를 활용해서 힙을 얼마나 차지하고 있는지 확인해보자.

```javascript
function memoryUsed() {
  const mbUsed = process.memoryUsage().heapUsed / 1024 / 1024
  console.log(`Memory used: ${mbUsed} MB`)
}

function memoryUsed() {
  const mbUsed = process.memoryUsage().heapUsed / 1024 / 1024
  console.log(`Memory used: ${mbUsed} MB`)
}

console.log('before')
memoryUsed() // Memory used: 4.7296905517578125 MB

const bigString = 'x'.repeat(10 * 1024 * 1024)
console.log(bigString) // 컴파일러가 bitString을 최적화하여 날려먹지 않게 필요하다.

console.log('after')
memoryUsed() // Memory used: 14.417839050292969 MB
```

엄청나게 큰 10mb짜리 string을 선언한 뒤 확인해보니, 해당 문자열 만큼의 약10mb의 차이가 있는 것을 확인할 수 있었다. 앞서 언급했던 스택 사이즈의 경우, 오직 864kb 밖에 없었다. 그렇다. 스택에는 이렇게 큰 문자열을 저장할 공간이 없다.

## 자바스크립트의 원시 값은 대부분 재활용 된다.

##
