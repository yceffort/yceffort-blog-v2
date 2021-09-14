---
title: '자바스크립트에서의 정규식, 이론부터 조심해야 할 것 까지'
tags:
  - javascript
published: true
date: 2021-09-14 00:05:25
description: '아직도 정규식이랑 안친함'
---

## Table of Contents

## 정규식은 무엇인가

정규식은 string 데이터의 패턴을 설명하는 방식이다. 다양한 언어에서 사용가능하며, 정규식을 사용하면 이메일 주소나, 암호와 같은 일련의 문자에서 패턴을 확인하여 해당 정규식에 정의된 패턴과 일치하는지 확인하고, 실행 가능한 정보를 얻을 수 있다.

## 정규식 만들기

자바스크립트에서 정규식을 만드는 방법은 두가지가 있다. `RegExp` 생성자를 사용하여 만들거나, `/` 를 사용하여 패턴을 감싸는 방법이 있다.

### 정규식 생성자

문법 : `new RegExp(patter[, flags])

```javascript
var regexConst = new RegExp('abc')
```

### 정규식 리터럴

문법: `/pattern/flags`

```javascript
var regexLiteral = /abc/
```

- `flags`는 옵셔널한 값인데, 이후에 다룬다.

정규식을 동적으로 만들고 싶은 경우가 있는데, 이 경우에는 정규식 리터럴을 사용할 수 없으므로 정규식 생성자를 사용해야 한다.

둘 중에 어떤 방법을 선택하던지 똑같은 정규식이 된다. 두 정규식 객체 모두 메서드와 속성이 동일하다.

> 슬래시는 패턴을 묶는데 사용되므로, 정규식의 일부로 사용하기 위해서는 `\/`와 같이 백슬래시로 이스케이프 해야 한다.

## 정규식 메소드

정규식 테스트를 위해 주로 사용하는 메소드가 두가지 있다.

### ReExp.prototype.test()

이 메서드는 정규식과 일치하는 항목이 있는지 여부를 테스트하는데 사용된다.

```javascript
var regex = /hello/
var str = 'hello world'
var result = regex.test(str)
console.log(result) // true
```

### RegExp.prototype.exec()

이 메소드는, 일치하는 모든 그룹을 배열로 리턴한다.

```javascript
var regex = /hello/
var str = 'hello world'
var result = regex.exec(str)
console.log(result)
// [ 'hello', index: 0, input: 'hello world', groups: undefined ]
// 'hello' -> 패턴에 일치하는 것
// index: -> 시작 index
// input: -> 실제 넘겨 받은 문자열
```

## 간단한 정규식 패턴

리터럴 텍스트와, 테스트 문자열이 일치하는지 확인하는 가장 간단한 패턴이다.

```javascript
var regex = /hello/
console.log(regex.test('hello world'))
// true
```

## 특수문자

사실 간단한 정규식 패턴은 별로 사용할일이 없다. 복잡한 경우를 다룰 때, 정규 표현식을 어떻게 쓰는지 살펴포자.

예를 들어, 특정 이메일 주소를 확인하는 것이 아니라 여러 이메일 주소를 확인하고자 한다. 여기에서는 특별한 문자가 등장한다. 정규 표현식을 완전히 이해하기 위해서는, 이것들을 암기해야 한다.

### Flags

정규 표현식에는 다섯가지의 옵셔널 플래그 또는 한정자가 있다. 그 중 가장 유명한 두개는 아래와 같다.

- `g`: 글로벌 검색
- `i`: 대소문자 구별을 안함

플래그와 단일 정규식을 결합할 수 있다.

정규식 리터럴 - `/pattern/flogs`

```javascript
var regexGlobal = /abc/g
console.log(regexGlobal.test('abc abc')) // true
var regexInsensitive = /abc/i
console.log(regexInsensitive.test('Abc')) // true
```

정규식 생성자 - `new RegExp('pattern', 'flags')`

```javascript
var regexGlobal = new RegExp('abc', 'g')
console.log(regexGlobal.test('abc abc')) // true
var regexInsensitive = new RegExp('abc', 'i')
console.log(regexInsensitive.test('Abc')) // true
```

### 문자열 그룹

#### `[xyz]`

특정한 위치에 서로 다른 문자를 일치시키는 방법으로, 괄호안에 있는 문자의 문자열에 있는 모든 단일 문자를 확인한다.

```javascript
var regex = /[bt]ear/
console.log(regex.test('tear'))
// returns true
console.log(regex.test('bear'))
// return true
console.log(regex.test('fear'))
// return false
```

#### `[^xyz]`

괄호안에 있는 것과 일치하지 않는 것만 확인한다.

```javascript
var regex = /[^bt]ear/
console.log(regex.test('tear'))
// returns false
console.log(regex.test('bear'))
// return false
console.log(regex.test('fear'))
// return true
```

#### `[a-z]`

모든 알파벳을 특정 위치에서 일치시키고 싶다면, 모든 문자를 쓰는 대신 이처럼 범위를 쓰면 된다. `[a-h]`는 a~h를 의미한다. `[0-9]`를 사용하여 숫자를 찾거나, `[A-Z]`를 사용하여 대문자만 찾을 수도 있다.

```javascript
var regex = /[a-z]ear/
console.log(regex.test('fear'))
// returns true
console.log(regex.test('tear'))
// returns true
```

#### 메타 문자

특수한 의미를 가진 것들을 의미한다. 매우 다양하지만, 일단 중요한 것은 아래와 같다.

- `\d`: 숫자 (`[0-9]`와 같음)
- `\w`: 글자, 숫자, `_`를 포함. (`[a-zA-Z0–9_]`와 같음)
- `\s`: 공백 문자 (스페이스, 탭)
- `\t`: 탭 문자
- `\b`: 단어의 시작이나 끝에 일치하는 단어를 찾는다. 단어 바운더리 라고도 불리운다.
- `.`: 새 줄(`\n`)을 제외한 모든 문자와 일치
- `\D`: `\d`와 같음
- `\W`: `\w`와 정반대
- `\S`: `\s`와 정반대

#### Quantifiers

Quantifiers는 정규식에서 특별한 의미를 갖는 기호를 의미한다.

- `+`: 이전 식과 1회 이상 일치

  ```javascript
  var regex = /\d+/
  console.log(regex.test('8'))
  // true
  console.log(regex.test('88899'))
  // true
  console.log(regex.test('8888845'))
  // true
  ```

- `*`: 이전 식을 0회 이상ㅇ리치

  ```javascript
  var regex = /go*d/
  console.log(regex.test('gd'))
  // true
  console.log(regex.test('god'))
  // true
  console.log(regex.test('good'))
  // true
  console.log(regex.test('goood'))
  // true
  ```

- `?`: 이전식을 0, 1번 일치

  ```javascript
  var regex = /goo?d/
  console.log(regex.test('god'))
  // true
  console.log(regex.test('good'))
  // true
  console.log(regex.test('goood'))
  // false
  ```

- `^`: 문자열의 시작과 일치. 문자열 뒤에 오는 정규식은 테스트 문자열의 시작에 있어야 한다. 즉, `^`은 문자열의 시작과 일치해야 한다.

  ```javascript
  var regex = /^g/
  console.log(regex.test('good'))
  // true
  console.log(regex.test('bad'))
  // false
  console.log(regex.test('tag'))
  // false
  ```

- `$`: 문자열의 끝, 즉 문자열 앞에 와야하는 정규식과 일치한다.

  ```javascript
  var regex = /.com$/
  console.log(regex.test('test@testmail.com'))
  // true
  console.log(regex.test('test@testmail'))
  // false
  ```

- `{N}`: 이전 정규식과 N번 일치

  ```javascript
  var regex = /go{2}d/
  console.log(regex.test('good'))
  // true
  console.log(regex.test('god'))
  // false
  ```

- `{N,}`: 최소 N번 이상 이전 정규식과 일치

  ```javascript
  var regex = /go{2,}d/
  console.log(regex.test('good'))
  // true
  console.log(regex.test('goood'))
  // true
  console.log(regex.test('gooood'))
  // true
  ```

- `{N,M}`: 최소 N번 이상 M번 미만으로 정규식과 일치

  ```javascript
  var regex = /go{1,2}d/
  console.log(regex.test('god'))
  // true
  console.log(regex.test('good'))
  // true
  console.log(regex.test('goood'))
  // false
  ```

- `X|Y`: `X` 또는 `Y`와 일치

  ```javascript
  var regex = /(green|red) apple/
  console.log(regex.test('green apple'))
  // true
  console.log(regex.test('red apple'))
  // true
  console.log(regex.test('blue apple'))
  // false
  ```

  ```javascript
  var regex = /a+b/ // This won't work
  var regex = /a\+b/ // This will work
  console.log(regex.test('a+b')) // true
  ```

#### 고오급

- `(x)`: `x`와 일치하고, 이 일치항목을 기억한다. 이를 캡쳐 그룹이라고 한다. 정규식 내 하위 식을 만드는 데에도 사용된다.

  ```javascript
  var regex = /(foo)bar\1/
  console.log(regex.test('foobarfoo'))
  // true
  console.log(regex.test('foobar'))
  // false
  ```

  `\1`는 괄호안의 첫번쨰 하위 표현식과 일치하는 항목을 기억하고, 이를 사용한다.

- `(?:x)`: x와 일치하는 것을 찾고, 그리고 이를 기억하지 않는다. 이는 논 캡쳐 그룹이라고 한다. `\1`는 작동하지않지만, `\1`과 일치하게 된다.

  ```javascript
  var regex = /(?:foo)bar\1/
  console.log(regex.test('foobarfoo'))
  // false
  console.log(regex.test('foobar'))
  // false
  console.log(regex.test('foobar\1'))
  // true
  ```

- `x(?=y)`: x가 y뒤에 올 경우 일치시킨다. 이를 positive look ahead라고 도 한다.

  ```javascript
  var regex = /Red(?=Apple)/
  console.log(regex.test('RedApple'))
  // Apple앞에있는 Red만 일치
  // true
  ```

## 실제 사용법

### 숫자 10개와 일치하는 정규식

```javascript
var regex = /^\d{10}$/
console.log(regex.test('9995484545'))
```

위 정규식을 하나씩 파해쳐 보자.

1. 일치 항목이 전체 문자열에 걸쳐서 있어야 한다면 (= 전체 문자열과 같아야 한다면) `^`, `$`를 사용하면 된다.
2. `\d`는 숫자만 허용한다
3. `{10}`은 이전 표현식을 10번 일치하는 것을 의미하므로, 여기서는 숫자 10개 일치를 의미한다.

### 날짜 `DD-MM-YYYY`또는 `DD-MM-YY`

```javascript
var regex = /^(\d{1,2}-){2}\d{2}(\d{2})?$/
console.log(regex.test('01-01-1990'))
// true
console.log(regex.test('01-01-90'))
// true
console.log(regex.test('01-01-190'))
// false
```

위 정규식을 하나씩 파해쳐 보자.

1. `^`와 `$`로 문자열 전체를 일치시키는 것만 찾는다.
2. `(`는 첫번째 하위 표현을 의미한다.
3. `\d{1, 2}` 숫자 1~2개
4. `-`: 하이픈 일치
5. `)`: 첫번째 하위 표현 종료
6. `{2}` 첫번째 표현과 정확히 2개 일치하는 경우
7. `\d{2}`: 정확히 두개의 숫자
8. `(\d{2})?`: 두개의 숫자. 그러나 옵셔널 이므로, 년도는 2개나 4개가 가능해진다.


## 조심해야 할 것

### lookbehind 문법은 사파리와 익스플로러에서 쓸 수 없다

[여기](https://yceffort.kr/2020/03/regex-formatting-number)에서도 한번 언급했던 문제. `x(?<=y)` `x(?<!y)`와 같은 lookbehind문법은 [사파리와 익스플로러에서는 지원하지 않으므로](https://yceffort.kr/2020/03/regex-formatting-number), 다른 방법으로 처리해야한다.

### Catastrophic Backtracking

정규식에는 두가지 알고리즘이 존재한다.

- Deterministic Finite Automaton (DFA): 문자열의 문자를 한번만 확인한다.
- Nondeterministic Finite Automaton (NFA): 최적의 일치를 찾을 때 까지 여러번 확인한다.

여기에서 자바스크립트는 NFA 알고리즘을 사용하고 있는데, NFA의 동작으로 인해 Catastrophic Backtracking 가 일어날 수 있다.

무슨말인지 잘 모르겠으니 아래 정규식을 살펴보자.

```javascript
/(g|i+)+t/
```

매우 간단한 정규식이지만, 매우 무거운 정규식이기도 하다.

- `(g|i+)`: 주어진 문자얄이 `g`로 시작하는지, 또는 `i`가 하나이상 있는지 확인한다.
- `+`: 이전 그룹이 한개이상 존재하는지 확인한다.
- `t`: 문자열은 `t`로 끝나야 한다.

