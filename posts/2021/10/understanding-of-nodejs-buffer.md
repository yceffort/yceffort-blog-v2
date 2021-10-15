---
title: 'nodejs의 버퍼 이해하기'
tags:
  - javascript
  - nodejs
published: true
date: 2021-10-15 23:48:00
description: 'nodejs도 본격적으로 해보고 싶네여'
---

## Table of Contents

## 버퍼란 무엇인가

Nodejs에서 buffer는 raw 바이너리 데이터를 저장할 수 있는 특수한 유형의 객체다. 버퍼는 일반적으로 컴포터에 할당된 메모리 청크 - 일반적으로 RAM -을 나타낸다. 일단 버퍼크기를 설정하게 되면, 이후에는 변경할 수 없다.

버퍼는 바이트를 저장하는 단위라고 볼 수 있다. 그리고 바이트는 8비트 순서로 이루어져있다. 비트는 컴퓨터의 가장 기본적인 저장 단위이며, 0 또는 1로 이루어져 있다.

Nodejs는 버퍼클래스를 전역 스코프에 expose 하므로 `import`나 `require`를 할 필요가 없다. 이 클래스를 활용하면 raw 바이너리를 조작할 수 있는 함수와 추상화등을 얻을 수 있다.

nodejs의 버퍼를 먼저 살펴보자.

```javascript
<Buffer 79 63 65 66 66 6f 72 74>
```

위 예제에서는, 7쌍의 문자를 볼 수 있다. 각 쌍은 버퍼에 저장된 바이트를 나타낸다. 이 버퍼의 크기는 7이다. 그런데, 아까 0과 1로 이루어졌다고 했는데, 보이는 건 그렇지가 않다. 🤔 그 이유는 nodejs는 16진수를 활용하여 바이트를 표현하기 때문이다. 그러므로 모든 마이트는 0~9, a~f를 활용한 두자리로만 나타낼 수 있다.

근데 버퍼는 왜 필요한걸까? 버퍼가 도입되기 이전에는 자바스크립트에서는 이 바이너리 데이터를 처리할 방법이 마땅히 없었다. 속도가 느리고, 바이너리를 처리할 전문적인 도구가 없기 때문에 기존에는 문자열 (string)과 같은 원시 값들을 이용해야 했다. 버퍼는 비트와 바이트를 조금더 쉽게, 그리고 성능에도 유리한 방법으로 조작할 수 있도록 제공되고 있다.

## 버퍼 사용해 보기

버퍼로 무엇을 할 수 있는지 살펴보자.

이제 곧 버퍼에 대해서 알아볼 텐데, 이 처리하는 방식이 자바스크립트의 배열과 유사하다는 것을 알 수 있다. `slice` 라던가 `concat`가 있다거나, `length`로 길이를 구할 수 있다거나. 버퍼는 또한 이터러블 하며 `for-of`에서도 사용할 수 있다.

### 버퍼 생성하기

버퍼를 생성하는 방법은 크게 세가지가 있다.

- `Buffer.from()`
- `Buffer.alloc()`
- `Buffer.allocUnsafe()`

> 과거에는 constructor를 사용하는 방법도 있었지만, 이 방법은 deprecated 되었으므로 쓰지 않는 것이 좋다.

#### `Buffer.from`

이 방법은 buffer를 만드는 가장 뚜렷한 방법이다. string, 배열, `ArrayBuffer` 혹은 또다른 버퍼 인스턴스를 인수로 받을 수 있다. 무엇을 넘겨주냐에 따라서, `Buffer.from()`은 버퍼를 약간씩 다른 방법으로 만든다.

일단 문자열을 넘기면, 그 문자열을 담고 있는 새로운 버저 객체를 만들어 낸다. 기본적으로, 문자열을 `utf-8`로 인코딩한다. [가능한 인코딩 타입들](https://nodejs.org/api/buffer.html#buffer_buffers_and_character_encodings)

```javascript
// utf8로 생성
Buffer.from('yceffort')
// <Buffer 79 63 65 66 66 6f 72 74>

Buffer.from('19881024', 'hex')
// <Buffer 19 88 10 24>
```

또한 바이트의 배열을 파라미터로 넘길 수 있다.

```javascript
Buffer.from([0x79, 0x63, 0x65, 0x66, 0x66, 0x6f, 0x72, 0x74])
// <Buffer 79 63 65 66 66 6f 72 74>
```

> `0xNN`은 `0x` 뒤의 값이 16 진수라는 것을 의미한다.

만약 `Buffer.from()`에 또다른 버퍼를 넘긴다면, nodejs는 해당 버퍼를 복제해서 또다른 버퍼를 만든다. 그리고 이렇게 생성된 새로운 버퍼는 메모리의 다른 공간에 저장해두기 때문에, 독립적으로 수정할 수 있다.

```javascript
const buffer1 = Buffer.from('yceffort')
const buffer2 = Buffer.from(buffer1)

buffer2[0] = 0x63

console.log(buffer1.toString()) // yceffort
console.log(buffer2.toString()) // cceffort
```
