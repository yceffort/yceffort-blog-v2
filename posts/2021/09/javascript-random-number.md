---
title: '자바스크립트에서 랜덤한 숫제 생성하기'
tags:
  - javascript
  - nodejs
published: true
date: 2021-09-01 21:19:35
description: ''
---

## Table of Contents

## Introduction

애플리케이션을 개발하다보면, 안전하게 난수를 생성해야 하는 경우가 있다. 예를 들어 주사위 게임이나 추첨, private key 생성 등등, 안전하게 난수를 생성하는 방법을 알아두어야 한다.

일단, 자바스크립트에는 `Math` 객체에 `random()`이라는 메소드가 존재한다. 이 메소드를 사용하면, 랜덤한 숫자를 생성할 수 있다.

`Math.random()`에서는 0이상 1미만의 부동 소수점 난수를 리턴한다. $$0 \geq x \lt 1$$

이 메소드를 사용하여 특정 범위의 랜덤한 숫자를 생성하는 다양한 방법이 있지만, 사실 `Math.random()`은 실제로 랜덤한 숫자롤 생성한다고 보기 어렵다. 이는 [유사 난수](https://ko.wikipedia.org/wiki/%EC%9C%A0%EC%82%AC%EB%82%9C%EC%88%98) 이기 떄문이다. 알려진 것 처럼, [컴퓨터로 유사 난수가 아닌 진짜 난수를 생성하는 것은 어렵다.](https://en.wikipedia.org/wiki/Random_number_generation#Computational_methods)

일반적으로, `Math.random()`으로 생성한 유사 난수는 대부분의 경우 충분한 답이 될 수 있지만, 암호학적으로 안전한 난수를 생성할 필요도 존재한다. 즉, 패턴을 통해서 쉽게 추측할 수 없거나, 시간이 지나도 반복되지 않는 진짜 난수가 필요하다는 것이다.

## 자바스크립트에서 `Math.random()`을 사용해야 하는 경우

`Math.random()`은 이른바 '시드'라고 하는 내부의 숨겨진 값에서 만들어지는 비 암호화 랜덤 숫자를 리턴한다. 시드는 지정된 범위에서 균일하게 생성된 숨겨진 숫자 시퀀스의 시작점이다.

이 메소드의 가장 간단한 사용예제로는, 0과 1사이의 랜덤한 부동 소수점을 만드는 것이다.

```javascript
const randomNumber = Math.random()
console.log(randomNumber) // 0.10150112695188218
```

이 랜덤한 숫자에 다른 숫자를 곱해서 원하는 크기의 결과를 만들어 낼 수 있다.

```javascript
const max = 6
const randomNumber = Math.floor(Math.random() * max)
console.log(randomNumber) // 3
```

또 다른 사용사례로는, `Math.floor`를 사용하여 특정 범위내의 난수를 생성하는 것이다. `floor`는 특정 숫자 보다 작거나 같은 숫자를 리턴한다.

```javascript
const max = 4
const min = 2

const result1 = Math.random() * (max - min)
console.log(result1) // 0.30347479463943516

const result2 = Math.random() * (max - min) + min
console.log(Math.floor(result2)) // 2
```

## `Math.random()`의 보안 취약점

`Math.random()`은 앞서 언급한 것처럼 보안적인 측면에서 단점이 있다. MDN의 문서에 따르면 `Math.random()`는 암호학적으로 안전한 난수를 생성해주지 않는다. 따라서 프로그램의 보안과 관련된 로직에서는 `Math.random()`을 사용하지 않는 것이 좋다.

> Note: Math.random() does not provide cryptographically secure random numbers. Do not use them for anything related to security. Use the Web Crypto API instead, and more precisely the window.crypto.getRandomValues() method.

그 원인은 아래와 같다.

- 균일한 분포 내에서 랜덤 정수를 생성하는데 사용되는 로직이 부적절하고 일반적으로 편향되어 있음
- 사용해야할 임의의 비트/바이트 수가 브라우저 별로 일치 하지 않음
- 무작위 결과값은 항상 일관되게 다시 생성하기 어려우므로, 이는 본질적으로 비결정적이고 불규칙함
- 빌트인 시드가 변조될 수 있으므로 무결성 측면에서 부적합
