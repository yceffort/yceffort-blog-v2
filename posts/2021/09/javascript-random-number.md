---
title: '자바스크립트에서 안전하게 난수 생성하는 방법'
tags:
  - javascript
  - nodejs
published: true
date: 2021-09-01 21:19:35
description: 'Math.random()도 잘못 사용하는 경우가 더러 있음'
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

이러한 문제들 때문에, [월드와이드웹 컨소시움](https://www.w3.org/)은 [Web Crypto API](https://www.w3.org/TR/WebCryptoAPI/)를 만들어 공개하였다. 이 기능은 [대부분의 브라우저에서 사용할 수 있다.](https://caniuse.com/cryptography)

## Web Crypto API

`Web Crypto API`는 `window.crypto`를 통해 엑세스할 수 있는 다양한 암호화 관련 메소드와 함수를 제공한다. 브라우저에서는, `crypto.getRandomValues(Int32Array)`를 사용하여 암호학적인 난수를 생성할 수 있다.

```javascript
var array = new Uint32Array(10);
window.crypto.getRandomValues(array);

console.log("나의 행운의 숫자들:");
for (var i = 0; i < array.length; i++) {
    console.log(array[i]);
}
// 나의 행운의 숫자들:
// 4213312451
// 4055435872
// 1248983520
// 2190329984
// 3226059214
// 1665817179
// 745131913
// 3947493810
// 218658595
// 2076931579
```

Nodejs에서는 표준 web crypto api가 제공된다. `require('crypto').randomBytes(size)`를 사용하면, node에 있는 native 암호화 모듈을 사용하여 난수를 생성할 수 있다.

```javascript
const randomBytes = require('crypto').randomBytes(2)
const number = parseInt(randomBytes.toString("hex"), 16)

console.log(number) // 40358
```

Web Crypto API에 사용되는 의사 난수 생성 알고리즘 (psuedo-random number generator algorithm, PRNG)는 브라우저에 따라서 다를 수 있다. 

## Web Crypto API 활용하기

`Crypto.getRandomValues()` 메소드는 암호학적으로 강력한 난수를 리턴한다. 대부분의 웹 브라우저에서 사용할 수 있으며, 구현 방식에 따라 차이가 있을 수 있지만 엔트로피가 충분한 시드를 사용해야 한다. 이는 성능과 보안에 부정적인 영향을 미치지 않기 위함이다.

`getRandomValues()`는 crypto 인터페이스 중 유일하게 안전하지 않는 컨텍스트에서 사용할 수 있는 메소드다. 따라서 여기서 얻은 암호화 키는 안전한 결과가 아닐 수도 있으므로, 암호화 키를 생성할 때 이 메소드를 사용하지 않는 것이 좋다. 이 경우에는, `generateKey()` 메소드를 사용하는 것이 좋다.

### 문법

`Web Cryptography API`는 바이트 시퀀스를 나타내는 입력으로 `ArrayBuffer`, `TypedArray`를 인수로 받는다.

```javascript
cryptoObj.getRandomValues(typedArray);
```

`typedArray`는 정수 기반의 `TypedArray`객체다. 이 외에도 `Int8Array`, `Uint8Array`, `Int16Array`, `Uint16Array`,  `Int32Array`, `Uint32Array`가 될 수 있다. 이 배열이 이제 랜덤한 난수로 채워지게 된다.

## 난수 생성하기

보안 목적으로 필요한 모든 임의의 값 (공격자에게 공격 받을 수 있는 가능성이 있는 모든 값)는 암호학적으로 안전한 의사 난수 생성기 ([Cryptographically Secure Pseudo-Random Number Generator, CSPRNG](https://en.wikipedia.org/wiki/Cryptographically-secure_pseudorandom_number_generator))를 사용하여 생성해야 한다.

이를 활용할 수 있는 분야로는 토큰 확인 또는 리셋, 복권 번호, API 키, 암호 생성, 암호화 키 등이 있다.

가장 안전하게 생성할 수 있는 방법은 무엇일까? 가장 좋은 방법은 보안상으로 잘 설계 되어 있는 라이브러리를 활용하는 것이다. Nodejs를 기준으로 살펴보면, 

- [random-number-csprng](https://www.npmjs.com/package/random-number-csprng)
- API Key, 토큰에는 [uuid](https://www.npmjs.com/package/uuid), `uuid.v4`

