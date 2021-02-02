---
title: 'null과 undefined의 차이, 그리고 역사'
tags:
  - javascript
published: true
date: 2021-02-02 20:57:53
description: '이런 것 또한 매력이라면 매력이 아니다'
---

## ECMAScript 언어 명세 상의 차이

- `undefined`: 변수에 값이 할당되지 않았을 때 사용된다. https://tc39.es/ecma262/#sec-undefined-value primitive value used when a variable has not been assigned a value

- `null`: 의도적으로 어떤 객체의 값이 비어 있다는 것을 나타낼 때 사용된다. primitive value used when a variable has not been assigned a value primitive value that represents the intentional absence of any object value

다른 언어와 다르게, 두개의 non-value가 있는 것은 자바스크립트 설계의 실수라고 한다. 그러나 이것이 지금까지 사용되고 있는 것은 자바스크립트가 구버전과의 호환성을 절대로 깨뜨리지 않는 다는 원칙이 있기 때문이다. [비슷한 사례로 `typeof null`이 있다.](https://2ality.com/2013/10/typeof-null.html)

자바스크립트가 많은 영감을 받은 자바에서는, 변수의 정적 타입에 따라 값을 초기화 한다.

- 객체 타입의 변수는 `null`로 초기화 한다.
- 원시 타입의 변수는 각자의 초기 값으로 초기화 한다. 숫자의 경우 0 이다.

자바스크립트의 경우에는, 변수가 객체가 될 수도 있고, 원시값이 될 수도 있다. 그러므로 만약 `null`이 "객체가 아니다" 라면, 자바스크립트는 "객체도 아니고 원시값도 아닌" 초기 값이 필요해 진다. 그것이 바로 `undefined` 다.

## undefined가 나오는 경우

```javascript
// 초기 값을 주지 않은 경우
let var

// 객체에서 정의 하지 않은 속성에 접근하는 경우
const obj = {}
obj.prop

// 함수의 리턴이 없는 경우
function func() {}
assert.equal(func(), undefined)

// 리턴은 있는데 아무것도 안하는 경우
function func() {
  return
}
assert.equal(func(), undefined)

function func(x) {
  assert.equal(x, undefined)
}

// 정의 되지 않은 파라미터
func()

undefined?.prop
null?.prop
```

## null이 나오는 경우

```javascript
// 프로토타입의 종점
Object.getPrototypeOf(Object.prototype) // null

// 정규식
/a/.exec('x') // null

// JSON은 undefined를 지원 하지 않는다.
JSON.stringify({a: undefined, b: null}) // "{"b":null}"
```

## undefined와 null을 특별하게 처리하는 연산자

### 함수 파라미터의 기본값

함수 파라미터의 기본값은 두가지 경우에서만 사용된다.

- 파라미터가 존재하지 않는 경우
- 파라미터의 값이 `undefined`인 경우

```javascript
function func(arg = 'abc') {
  return arg
}
console.log(func()) // abc
console.log(func(undefined)) // abc
console.log(func(null)) // null
```

### 분해연산자의 기본값과 undefined

위의 기본값 예제와 동일하게 동작한다.

```javascript
const [a = 'a'] = [] // a
const [b = 'b'] = [undefined] // b
const { prop: c = 'c' } = {} // c
const { prop: d = 'd' } = { prop: undefined } //d
const [e = 'e'] = [null] // null
const { prop: f = 'f' } = { prop: null } // null
```

### 옵셔널 체이닝

값이 nullish한 경우 (`null` `undefined`) `undefined`를 리턴한다.

```javascript
function getProp(obj) {
  return obj?.prop
}
getProp({ prop: 123 }) // 123
getProp(undefined) // undefined
getProp(null) // undefined
```

### undefined, null, nullish coalescing (null 병합 연산자)

`||` 와는 다르게 `nullish` 한 값에 대해서만 대응한다.

```javascript
undefined ?? 'default value' // default value
null ?? 'default value' // default value
0 ?? 'default value' // 0
'' ?? 'default value' // ''
```

null 병합 할당 현산자인 `??=` 도 마찬가지로 동작한다.

## 결론(?)

명시적으로 값이 없다는 걸 나타내고 싶다면, `null`을 쓰는게 좋다. `undefined`는 기본값 할당, JSON에서 생략되는 등 의도치 않은 동작을 나타낼 수 있다. 내가 의도적으로 빈값을 두고 싶다면, `null`이 나은 것 같다. 그렇다고 `undefined`가 내가 짠 코드에서 안나오는 것은 아닐 것이다. 말하고 싶은 것은 의도적으로 빈값을 집어 넣을 때 둘중에 `null`을 더 선호한다 정도일 것이다. 그러나 이건 어디까지나 내취향의 문제일 뿐, 적절히 헷갈리지만 않게 쓰면 될 것 같다.
