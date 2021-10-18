---
title: 'nodejs의 버퍼 이해하기'
tags:
  - javascript
  - nodejs
published: true
date: 2021-10-15 23:48:00
description: 'nodejs로 백엔드하는 회사 찾습니다'
---

## Table of Contents

## 버퍼란 무엇인가

Nodejs에서 buffer는 raw 바이너리 데이터를 저장할 수 있는 특수한 유형의 객체다. 버퍼는 일반적으로 컴포터에 할당된 메모리 청크, 일반적으로 RAM을 나타낸다. 일단 버퍼크기를 설정하게 되면, 이후에는 변경할 수 없다.

버퍼는 바이트를 저장하는 단위라고 볼 수 있다. 그리고 바이트는 8비트 순서로 이루어져있다. 비트는 컴퓨터의 가장 기본적인 저장 단위이며, 0 또는 1로 이루어져 있다.

Nodejs는 버퍼클래스를 전역 스코프에 expose 하므로 `import`나 `require`를 할 필요가 없다. 이 클래스를 활용하면 raw 바이너리를 조작할 수 있는 함수와 추상화등을 얻을 수 있다.

nodejs의 버퍼를 먼저 살펴보자.

```bash
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

#### Buffer.alloc

`.alloc()` 메소드는 데이터를 채울 필요가 없는 빈 버퍼를 생성하고 싶을 때 유용하다. 기본적으로, 숫자를 인수로 받으며 받은 숫자만큼의 빈 사이즈의 버퍼를 생성한다.

```javascript
Buffer.alloc(7)
// <Buffer 00 00 00 00 00 00 00>
```

이렇게 생성한 버퍼에, 원하는 데이터를 채울 수 있다.

```javascript
const buffer = Buffer.alloc(1)
buffer[0] = 0x78
buffer.toString('utf-8')
// 'x'
```

#### Buffer.allocUnsafe()

`allocUnsafe`를 사용하면 버퍼안의 내용을 검사하고 0으로 채우는 기본적인 작업을 스킵한다. 버퍼는 이전 데이터를 포함 할 수 있는 메모리 ("unsafe" 이라는 뜻은 여기에서 나온 것이다) 영역에 할당한다. 예를 들어, 다음 코드를 실행하면 실행 시마다 매번 일부 랜덤 데이터를 프린트 하게 된다.

```javascript
const buffer = Buffer.allocUnsafe(100)
buffer.toString('utf-8')
// '\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
```

도대체 쓸 곳이 없어 보이는 이 메소드는 안전하게 할당된 버퍼를 복사하는 케이스에 사용해 봄직하다. 복사된 버퍼를 완전히 덮어 써버리기 때문에 모든 이전 바이트가 예측 가능한 데이터로 대체할 수 있게 된다.

```javascript
const originalBuffer = Buffer.from('hello, yceffort')
const copyBuffer = Buffer.allocUnsafe(originalBuffer.length)
originalBuffer.copy(copyBuffer)
copyBuffer.toString()
// 'hello, yceffort'
```

일반적으로, `allocUnsafe()`는 오로지 적절한 이유가 있을 때만 사용하는 것이 좋다. (성능 최적화 라던가) 이 메소드를 사용할 때는, 내부에 예측하지 못한 데이터로 채워지지 않도록 꼭 적절한 데이터로 채워야 한다. 그렇지 않으면 원치 않는 정보가 밖으로 새내어갈 수 도 있다.

### Buffer를 쓰기

`Buffer.write()`를 사용하여 일반적으로 버퍼에 데이터를 쓰는 작업을 진행한다. 기본적으로, `utf-8`로 별도 오프셋 없이(버퍼 맨처음 부터) 작성된다. 이 메소드를 쓰면, 버퍼를 사용하는데 들었던 바이트를 리턴한다.

```javascript
const buffer = Buffer.alloc(7)

buffer.write('yceffort')
buffer.write('babo yceffort')

buffer.toString()
// 'babo yc', 7로 생성했기 때문에 이후 데이터는 잘린다.
```

한가지 명심해야할 것은, 모든 글자가 하나의 바이트에 저장되지 않는 다는 것이다.

```javascript
const wrongEmojiBuffer = Buffer.alloc(1)
wrongEmojiBuffer.write('🥸')
wrongEmojiBuffer.toString()
// \x00'
```

utf-8 인코딩은 최대 4바이트의 문자를 지원한다. 버퍼 크기는 이후에 수정할 수 없으므로, 항상 버퍼에 작성하는 내용과 버퍼의 크기와 작성하려는 콘텐츠의 크기를 항상 염두해 두어야 한다.

```javascript
const emojiBuffer = Buffer.alloc(4)
emojiBuffer.write('🥸')
emojiBuffer.toString()
// '🥸'
```

버퍼를 쓰는 또다른 방법은 버퍼의 특정위치에 바이트를 추가하는, 즉 배열에 요소를 넣는것 과 같은 방법을 사용하는 것이다. 1바이트를 초과하는 데이터는 버퍼의 각 위치에서 분해해서 설정해야 한다.

```javascript
const buff = Buffer.alloc(5)

buff[0] = 0x68 // 0x68 is the letter "h"
buff[1] = 0x65 // 0x65 is the letter "e"
buff[2] = 0x6c // 0x6c is the letter "l"
buff[3] = 0x6c // 0x6c is the letter "l"
buff[4] = 0x6f // 0x6f is the letter "o"

console.log(buff.toString())
// hello

// 2바이트 이상의 글자를 한 버퍼에 넣는 다면 실패한다.
buff[0] = 0xc2a9

console.log(buff.toString())
// --> '�ello'

buff[0] = 0xc2
buff[1] = 0xa9

console.log(buff.toString())
// ©llo
```

배열 처럼 쓸 수 있는 건 재밌는 일이지만, 가능하면 `Buffer.from`을 사용하는 것이 좋다. 입력 값의 길이를 관리하는 것은 굉장히 어렵고, 코드의 복잡성을 키울 수 있다. `from()`을 사용하면 별 걱정 없이 쓸 수 있으며, 입력이 너무 큰 경우 입력이 잘 되지 않는 지등을 확인하여 처리할 수 있다.

### 버퍼 순회하기

모던 자바스크립트에서 순회를 하는 방식으로, 버퍼도 마찬가지로 순회 할 수 있다.

```javascript
const buffer = Buffer.from('yceffort')
for (const b of buffer) {
  console.log(b.toString(16))
}
// 79
// 63
// 65
// 66
// 66
// 6f
// 72
// 74
```

`.entries()` `.values()` `.keys()`도 물론 가능하다.

```javascript
const buffer = Buffer.from('yceffort')
const copyBuffer = Buffer.alloc(buffer.length)
for (const [index, b] of buffer.entries()) {
  copyBuffer[index] = b
}

console.log(copyBuffer.toString())
// yceffort
```

## 버퍼와 TypedArrays

자바스크립트에서는, 메모리를 `ArrayBuffer` 클래스를 사용하여 할당 할 수 있다. 이 `ArrayBuffer` 객체를 직접 조작하는 경우는 거의 없다. 대신 이 `ArrayBuffer`를 참조하는 "view" 객체 집합을 사용한다. 이 객체 집합으로는 다음과 같은 것들이 있다.

- `Int8Array`
- `Uint8Array`
- `Uint8ClampedArray`
- `Int16Array`
- `Uint16Array`
- `Int32Array`

> https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/TypedArray#typedarray_objects

그리고 `TypedArray`가 있다. 이는 위에 나열된 모든 뷰 객체를 포괄하는 용어다. 모든 뷰 객체는 프로토타입을 통해 `TypedArray` 메소드를 상속한다. `TypedArray` 생성자는 글로벌로 노출되지 않으므로 `new TypedArray()`와 같은 방법을 사용해야 한다.

nodejs에서는 Buffer 클래스로 생성된 객체도 `Unit8Array` 인스턴스다. 여기에는 아주 작은 차이점이 존재한다.

> https://nodejs.org/api/buffer.html#buffer_buffers_and_typedarrays
