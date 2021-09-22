---
title: '자바스크립트 String'
tags:
  - typescript
published: true
date: 2020-09-21 22:12:37
description: '자바스크립트의 String'
category: typescript
template: post
---

```javascript
const hello = 'hello'
hello.length // 5

const bow = '🙇‍♂️'
bow.length // 5 ???
```

첫 번째 `hello`의 길이는 이해가 되지만, 두번째 `🙇‍♂️` 의 길이는 왜 5일까?

이를 알기 위해서는 [javascript string의 정의](https://tc39.es/ecma262/#sec-ecmascript-language-types-string-type)를 참고할 필요가 있다.

> The String type is the set of all ordered sequences of zero or more 16-bit unsigned integer values ("elements") up to a maximum length of 253 - 1 elements. The String type is generally used to represent textual data in a running ECMAScript program, in which case each element in the String is treated as a UTF-16 code unit value. Each element is regarded as occupying a position within the sequence. These positions are indexed with nonnegative integers. The first element (if any) is at index 0, the next element (if any) at index 1, and so on. The length of a String is the number of elements (i.e., 16-bit values) within it. The empty String has length zero and therefore contains no elements.

> String type은 최대 길이 $$2^53 - 1$$ 의 원소들로 이루어진 0 이상의 16비트 미부호 정수 값("요소")의 순서로 구성된 모든 시퀀스 집합이다. 문자열 유형은 일반적으로 실행 중인 ECMAScript 프로그램에서 텍스트 데이터를 나타내기 위해 사용되며, 이 경우 문자열의 각 요소는 UTF-16 코드 단위 값으로 처리된다. 각 원소는 순서에서 위치를 차지하는 것으로 간주된다. 이러한 위치는 음이 아닌 정수로 나타난다. 첫 번째 요소(있는 경우)는 index 0에 있고, 다음 요소(있는 경우)는 index 1에 있는 등.. 으로 이루어진다. 문자열의 길이는 문자열 내의 요소 수(즉, 16비트 값)이다. 빈 문자열은 길이가 0이므로 요소를 포함하지 않는다.

자바스크립트는 문자열을 16비트로 표현하고 있고, 문자열의 길이는 이 16비트 요소들의 길이 인 것이다. [UTF-16](https://ko.wikipedia.org/wiki/UTF-16)

실제로, string을 UTF-16으로 변환하는 방법이 [여기](https://developers.google.com/web/updates/2012/06/How-to-convert-ArrayBuffer-to-and-from-String)에 나와 있다.

그리고 이러한 UTF-16 코드 유닛은 `0x0000` 부터 `0xFFFF`까지 존재한다. 예를 들어서, 아까 `hello`는

```javascript
const hello = '\u0048\u0065\u006C\u006C\u006F'

hello === 'Hello' // true
hello.length // 5
```

이 `\u0048\u0065\u006C\u006C\u006F`가 자바스크립트가 string을 보는 방법이다. 그리고 이것이 앞서 언급한 코드 시퀀스의 집합이다.

한가지 알아둬야 할 것은, 이렇게 UTF-16의 한 비트로 표현할 수 있는 문자들은 [Basic Multilangual Plane]()에 한정되어 있다는 것이다. 그 외에 언어들에는

- [Supplementary Multilingual Plane](<https://en.wikipedia.org/wiki/Plane_(Unicode)#Supplementary_Multilingual_Plane>)
- [Supplementary Ideographic Plane](<https://en.wikipedia.org/wiki/Plane_(Unicode)#Supplementary_Ideographic_Plane>)
- [Tertiary Ideographic Plane](<https://en.wikipedia.org/wiki/Plane_(Unicode)#Tertiary_Ideographic_Plane>)

위 언어들은 따로나눌수 없는 UTF-16 의 쌍으로 이루어져 있다. 예를 들어 `😀`는 `0x1F600`인데, 이는 16비트로 표현할 수 있는 범위를 넘어섰으므로, `0xD83D0xDE00`로 표현한다.

어쨌든, `length`의 경우 하나의 16비트 진수를 길이로 표현하므로, 😀의 길이는 2가 된다.

그러나 한가지 다른 예제가 있다.

```javascript
const message = 'hello'
const smile = '😀'

[...message].length // => 5
[...smile].length // => 1
```

`string iterator`의 경우에는, UTF-16의 쌍 `surrogate pair`를 따로 분리하지 않고, 한개의 유닛으로 분리한다. 따라서 이것의 길이는, 사람 눈에 보이는 것처럼 1이 된다.

요약하자면, 자바스크립트의 string을 볼 때, 보여지는 그대로 보는 것이 가장 간단한 방법이다. 이 방법은 영문자, 숫자, 아스키 코드에만 유효하다.

그러나, 엄격하게 보자면 자바스크립트 문자열은 UTF-16 코드의 나열로 이루어져 있다. `string.length`는 이러한 코드의 길이를 보고 판단한다.
