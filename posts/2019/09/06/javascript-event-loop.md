---
title: 자바스크립트의 이벤트루프, 태스크, 그리고 마이크로 태스크
date: 2019-12-27 10:15:14
published: true
tags:
  - javascript
  - event-loop
description:
  "## 자바스크립트는 단일 스레드 기반의 언어 자바스크립트는 '단일 스레드' 기반의 언어다. 즉, 스레드가 하나이기
  때문에 동시에 하나의 작업만 처리할 수 있다. 그러나 자바스크립트가 사용되는 웹을 곰곰히 생각해보면 동시에 여러개의 작업을 처리하는 모습을
  볼 수 있다. 스레드가 하나인 자바스크립트는 동시성을 어떻게 처리할까? 먼저 브라우저 구동환경을 살펴보..."
category: javascript
slug: /2019/09/06/javascript-event-loop/
template: post
---

## 자바스크립트는 단일 스레드 기반의 언어

자바스크립트는 '단일 스레드' 기반의 언어다. 즉, 스레드가 하나이기 때문에 동시에 하나의 작업만 처리할 수 있다. 그러나 자바스크립트가 사용되는 웹을 곰곰히 생각해보면 동시에 여러개의 작업을 처리하는 모습을 볼 수 있다. 스레드가 하나인 자바스크립트는 동시성을 어떻게 처리할까? 먼저 브라우저 구동환경을 살펴보자.

![browser](https://miro.medium.com/max/1600/1*iHhUyO4DliDwa6x_cO5E3A.gif)

![nodejs](https://image.toast.com/aaaadh/real/2018/techblog/Bt5ywJrIEAAKJQt.jpg)

위 이미지에서, 자바스크립트 엔진은 메모리 할당을 관리하는 heap과 call stack만 존재하는 것을 알 수 있다. 즉, 동시성에 대한 처리는 자바스크립트 외부에서 처리하고 있음을 알 수 있다. 즉, 정리해서 말하면 자바스크립트는 단일 스레드기반의 언어라서, 단일 호출 스택을 사용하지만, 실제로 자바스크립트를 이용하는 환경 (브라우저, Nodejs)에서는 여러개의 스레드를 활용하며, 이러한 환경을 자바스크립트 엔진과 상호 연동하기 위해서 사용하는 것이 바로 **이벤트 루프**다.

## 단일 호출 스택, Run-to-Completion

자바스크립트의 함수가 실행되는 방식을 `Run-to-Completion`, 하나의 함수가 실행되면 이게 끝날 때까지는 다른 어떤 작업도 끼어들지 못함을 의미한다. 자바스크립트는 하나의 호출 스택을 사용하며, 현재 스택에 쌓여있는 함수들이 모두 실행되기 전까지는 다른 어떠한 함수도 실행될 수 없다.

```javascript
function delay() {
  for (var i = 0; i < 10000; i++);
}
function hi3() {
  delay()
  hi2()
  console.log('hi3!') // (3)
}
function hi2() {
  delay()
  console.log('hi2!') // (2)
}
function hi1() {
  console.log('hi1!') // (4)
}

setTimeout(hi1, 10) // (1)
hi3()
```

이 함수들이 실행되는 순서를 살펴보자.

[여기](http://latentflip.com/loupe/?code=ZnVuY3Rpb24gZGVsYXkoKSB7CiAgZm9yICh2YXIgaSA9IDA7IGkgPCAxMDAwMDsgaSsrKTsKfQpmdW5jdGlvbiBoaTMoKSB7CiAgZGVsYXkoKTsKICBoaTIoKTsKICBjb25zb2xlLmxvZygiaGkzISIpOyAvLyAoMykKfQpmdW5jdGlvbiBoaTIoKSB7CiAgZGVsYXkoKTsKICBjb25zb2xlLmxvZygiaGkyISIpOyAvLyAoMikKfQpmdW5jdGlvbiBoaTEoKSB7CiAgY29uc29sZS5sb2coImhpMSEiKTsgLy8gKDQpCn0KCnNldFRpbWVvdXQoaGkxLCAxMCk7IC8vICgxKQpoaTMoKTs%3D!!!PGJ1dHRvbj5DbGljayBtZSE8L2J1dHRvbj4%3D)를 살펴보세용 .

setTimeout이 얼마나 일찍 끝났건 간에, 다른 작업들이 먼저 콜 스택에 들어갔으므로, `hi1`은 절대 먼저 실행되지 않는다. 근데 어디서 이 setTimout에 있는 `hi1()`를 잡아다가 다시 실행해줬을까? 이를 도와주는 것이 태스크 큐와 이벤트 루프다. 태스크 큐는 콜백 함수들이 대기하는 큐(FIFO) 형태의 배열이고, 이벤트 루프는 콜 스택이 비워질 때 마다 콜백함수에서 꺼내와서 실행하는 역할을 한다.

10ms가 지난 후에, `hi1()`은 바로 실행되지 안혹, 태스크 큐에 추가한다. 이벤트루프는 현재 실행중인 모든 태스크가 끝나자마자 큐에서 대기중인 첫번째 태스크인 `hi1()`을 실행해서, 콜스택에 추가한다.

- 비동기 api들은 작업이 완료되면 콜백함수를 태스크 큐에 추가한다
- 이벤트 루프는 현재 실행중인 태스크가 없을때 태스크 큐에서 FIFO형식으로 큐를 꺼내와서 실행한다.

렌더링 엔진의 경우에도 마찬가지로, 자바스크립트 엔진과 동일한 태스크 큐를 사용한다.

## 마이크로 태스크

```javascript
console.log('script start')

setTimeout(function () {
  console.log('setTimeout')
}, 0)

Promise.resolve()
  .then(function () {
    console.log('promise1')
  })
  .then(function () {
    console.log('promise2')
  })

console.log('script end')
```

여기서 `Promise`가 setTimeout보다 먼저 실행되는데, 그 이유는 `Promise`가 마이크로 태스크에 등록되기 때문이다. 마이크로 태스크는 일반 태스크 보다 더 높은 우선순위를 갖으며, 태스크 큐에 대기중인 것이 있다고 하더라도 마이크로태스크에 있는 것이 우선해서 실행된다. 마이크로 태스크의 잡은 태스크 큐보다 우선하기 때문에, 시간이 오래 걸릴 경우 렌더링 엔진이 작동하지 못하고(일반 태스크에 있으므로) 렌더링이 느려지는 현상이 발생할 수도 있다.
