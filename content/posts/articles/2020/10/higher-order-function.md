---
title: 'higher order function, 고차함수'
tags:
  - javascript
published: true
date: 2020-10-26 23:29:24
description: '자바스크립트 고차 함수'
---

## 정의

고차 함수는 함수를 argument로 받아서 함수를 리턴하는 함수를 말한다.

## 예제

### 함수를 argument로 받는 함수

홀수 인지 확인하는 고차함수를 아래와 같이 만들어 보자.

```javascript
function isOdd(num, fn) {
  return fn(num) % 2 === 1
}
```

`fn`이라는 argument로 함수를 받고 있고, 그에 대한 결과를 리턴하고 있다. 위 함수는 아래와 같은 형태로 사용 가능하다.

```javascript
function divideByTwo(num) {
  return num / 1
}

isOdd(10, divideByTwo) // true

isOdd(11, divideByTwo) // false
```

### 함수를 리턴하는 함수

```javascript
function add(num1) {
  return function (num2) {
    return num1 + num2
  }
}
```

```javascript
add(1)(2) // 3

const num = add(1) // 1만 담고 있는 함수를 리턴
num(2) // 을 위해서 호출해서 2를 더함 3
```

## 좀 더 실용적인 예제

가장 일반적인 예제로는 object validator가 있다. 해당 object가 유효한지를 확인하는 것인데, 회원가입의 그것과도 비슷하다.

예를 들어 아래와 같은 객체가 있다고 가정해보자.

```javascript
const user = {
  age: 18,
  password: 'qhdks1234!@#$',
  gender: 'F',
  agreed: true,
}
```

- `age`는 18세 이상이어야 한다.
- `password`는 10자리 이상이여야 한다.
- `gender`는 `F` 또는 `M`이여야 한다.
- `agreed`는 true 여야 한다.

위 요구 사항을 각각 함수로 만들어 보자.

```javascript
function validAge(user) {
  return user.age >= 18
}

function validPassword(user) {
  return user.password.length >= 10
}

function agreed(user) {
  return user.agreed
}

function validGender(user) {
  console.log(user.gender, ['T', 'F'].includes(user.gender))
  return ['T', 'F'].includes(user.gender)
}
```

그리고 `isValid`라고 불리우는 고차함수를 만들어야 한다. 첫번째 객체로는 user를 받을 것이며, 그 이후에는 n개의 유효성 체크를 하는 함수를 받아서 이를 확인할 것이다.

```javascript
function isValid(user, ...validators) {
  for (const validator of validators) {
    if (!validator(user)) return false
  }
  return true
}
```

테스트를 한번 해보자.

```javascript
const user1 = {
  age: 18,
  password: 'qhdks1234!@#$',
  gender: 'F',
  agreed: true,
}

isValid(user, validAge, validPassword, agreed, validGender) // true
```

```javascript
const user2 = {
  age: 12,
  password: 'qhdks1234!@#$',
  gender: 'F',
  agreed: true,
}

isValid(user2, validAge, validPassword, agreed, validGender) // false
```

```javascript
const user3 = {
  age: 20,
  password: 'qhdks1234!@#$',
  gender: 'G',
  agreed: true,
}

isValid(user3, validAge, validPassword, agreed, validGender) // false
```

## validator 만들기

`user` 외에도 여러 종류의 validator를 만든다면, 이것 또한 고차함수르르 만들어서 해결이 가능하다.

```javascript
function createValidator(...validators) {
  return function (obj) {
    for (const validator of validators) {
      if (!validator(obj)) return false
    }
    return true
  }
}
```

```javascript
const userValidator = createValidator(
  validAge,
  validPassword,
  validGender,
  agreed,
)

userValidator(user1) // true
userValidator(user2) // false
userValidator(user3) //  false
```
