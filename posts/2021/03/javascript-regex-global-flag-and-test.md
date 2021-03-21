---
title: '알쏭 달쏭한 자바스크립트 정규식'
tags:
  - regex
  - javascript
published: true
date: 2021-03-21 14:42:10
description: '정규식을 자유자재로 써야 간지인데'
---

자바스크립트 정규식으로 숫자를 찾는 방법은 보통 아래와 같다.

```javascript
const str = 'hello world, 123'
const digitRegex = /\d+/g

digitRegex.test(str) //true
```

그런데 코딩을 하던 중 한가지 이상한 일이 발생했는데, 대략 아래와 같은 모습이었다.

```javascript
const digitRegex = /\d+/g

const result1 = digitRegex.test('hello 123')
const result2 = digitRegex.test('123')
const result3 = digitRegex.test('123')

console.log(result1) // true
console.log(result2) // false ???????????
console.log(result3) // true
```

> 정규 표현식에 전역 플래그를 설정한 경우, test() 메서드는 정규 표현식의 lastIndex (en-US)를 업데이트합니다. (RegExp.prototype.exec()도 lastIndex 속성을 업데이트합니다.)
>
> test(str)을 또 호출하면 str 검색을 lastIndex부터 계속 진행합니다. lastIndex 속성은 매 번 test()가 true를 반환할 때마다 증가하게 됩니다.
>
> 참고: test()가 true를 반환하기만 하면 lastIndex는 초기화되지 않습니다. 심지어 이전과 다른 문자열을 매개변수로 제공해도 그렇습니다!

https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/RegExp/test

조금더 자세히 살펴보자면, 대략 다음과 같다는 뜻이다.

```javascript
const digitRegex = /\d+/g

console.log(digitRegex.lastIndex) // 0
const result1 = digitRegex.test('hello 123')

// test로 digitRegex의 index (9) 를 찾았으므로 업데이트 함
console.log(digitRegex.lastIndex) // 9
// 9 부터 다시 찾음. 그런데 못찾았으므로 초기화가 진행됨.
const result2 = digitRegex.test('123')

console.log(digitRegex.lastIndex) // 0
// 0 부터 다시 찾아서 숫자를 찾음.
const result3 = digitRegex.test('123')
```

정확히 왜 이렇게 구현했는지에 대해서는 찾을 수 없었다. 추측건데 정규식에 전역플래그가 있다는 것은 검색하려는 대상 str도 전역으로 쓰이는 용도일 것이고, 그 때문에 검색하는 str이 아예 다른 것이 온다는 가정을 하지 않았기 때문이 아닐까? (뭐래는거야?)

어쨌든 간에, 이를 올바르게 동작하게 만들기 위해서는 `search`를 쓰면 된다. 주의할 점은 `.search`는 string에 있는 메소드라는 것.

```javascript
const digitRegex = /\d+/g

const result1 = 'hello 123'.search(digitRegex) > -1
const result2 = '123'.search(digitRegex) > -1
const result3 = '123'.search(digitRegex) > -1
const result4 = '바보'.search(digitRegex) > -1

console.log(result1) // true
console.log(result2) // true
console.log(result3) // true
console.log(result4) // false
```
