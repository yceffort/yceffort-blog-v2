---
title: '나만의 자바스크립트 polyfill 만들고 공부하기'
tags:
  - javascript
published: true
date: 2021-02-15 21:50:50
description: '어디 재밌는 글 없나'
---

자바스크립트는 새로운 feature 가 제안 되어도, 항상 이전 버전과의 하위호환이 깨지지 않는 선에서 지원이 가능해야 한다. 따라서, 모든 새로운 기능들은 polyfill로 지원이 가능하다. 그러나 이걸 가져다 써보기만 해봤지, 직접 만들어 본적은 없는 것 같다. 그래서 한번 직접 만들어 보려고 한다.

tc39문서를 보다가 새롭게 보게 된 feature 중 하나가 바로 이것이다.

## `Array.prototype.at`

- https://tc39.es/proposal-relative-indexing-method/
- https://github.com/tc39/proposal-relative-indexing-method

`array[0]` 과 비슷 해보이지만, 이것은 파이썬의 그것과 비슷하게 음수까지 지원해 준다. 예컨데, `[-1]`로 인덱싱을 하면 가장 마지막 것을 불러오는 것이다. 이 기능을 한번 polyfill로 구현해보자.

```javascript
function at(n) {
  n = parseInt(n, 10) || 0

  // 음수 일 경우 길이 만큼 더한다. 이렇게 하면 음수 인덱싱을 대응할 수 있다.
  if (n < 0) {
    n += this.length
  }

  // 인덱싱에 대응할 수 없을 경우 undefined를 리턴한다.
  if (n < 0 || n >= this.length) {
    return undefined
  }

  return this[n]
}

for (let T of [Array, String]) {
  Object.definedProperty(T.prototype, "at", { 
    value: at,
    writable: true,
    enumerable: false,
    configurable: true})
}
```

여기서 배울 수 있는 것은 다음과 같다.

### this

정적으로 결정되는 스코프와 다르게, `this`는 어떻게 호출되었는지에 따라서 달라진다. 메서드 this는, 메서드를 호출한 객체, 즉 `.` 연산자 앞에서 호출한 객체가 바인딩 된다. 

```javascript
let arr = [1,2,3]
arr.at(1)
```

우리는 위와 같은 방식으로 `.at()`을 호출하기 때문에, `at` 함수 내부의 `this`는 `at()`을 호출한 객체인 리스트가 될 것이다.

### prototype

`prototype` 프로퍼티는 함수 객체만이 지닌 프로퍼티다. (그리고 `Array`, `String`은 함수다. 사실 당연 한 거아님?) 이는 생성자 함수가 생성할 인스턴스의 프로토타입을 가리킨다. 생성자 함수가 자신이 생성할 객체의 프로토타입을 할당해주기 위해 사용하는 것으로, 여기에 있는 메소드들은 향후 새롭게 생성되는 객체들의 `__proto__` 또는 `Object.getPrototypeOf`로 접근할 수 있다. 

> 모든 Array 인스턴스는 Array.prototype을 상속합니다. 다른 생성자와 마찬가지로, Array() 생성자의 프로토타입을 수정하면 모든 Array 인스턴스도 수정의 영향을 받습니다. 예를 들면, 새로운 메서드와 속성을 추가해 모든 Array를 확장할 수 있으므로, 폴리필에 쓰입니다.

![array-prototype](./images/array-prototype.png)

```javascript
Array.prototype === Object.getPrototypeOf([1, 2, 3]) // true
```

## `definedProperty`

`definedProperty`를 알기 위해서는, property attribute를 알아야 한다. 이는 프로퍼티의 상태를 의미하며, 데이터 프로퍼티와 접근자 프로퍼티가 있다.

- 데이터 프로퍼티: 일반적인 프로퍼티로, 키와 값으로 구성되어 있다.
- 접근자 프로퍼티: 자체적으로 값을 가지고 있지는 않지만, 다른 데이터 프로퍼티의 값을 읽거나 지정할 때 호출되는 접근자 함수로 구성된 프로퍼티

여기에서 우리가 사용해야할 것은 데이터 프로퍼티로, `value` `writable` `enumerable` `configurable`을 값을 가진다. 그리고 `definedProperty`를 통해서, 새로운 프로퍼티를 정의할 수 있다. 나는 `Array` 와 `String`에 각각 지정했다.

## 공식 polyfill과 다른점

### 1. Math.trunc vs parseInt

숫자를 처리하는데 있어 나는 `parseInt`를 썼지만, 저 친구는 `Math.trunc`를 사용했다. 이는 그냥 소수점 이하 단위를 버리는 함수다. 굳이 굳이 비교하면 뭔차이가 있을까 하고 찾아봤는데

https://www.samanthaming.com/tidbits/55-how-to-truncate-number/ 

> parseInt is mainly used for a string argument. So if you're dealing with numbers, it's way better to use Math.trunc().

뭔가 성능에 있어서 차이가 있나보다. 아쉽게도 `jsPerf`가 뻗어있어서 (2021.02.15 기준) 알수가 없었지만, 아무튼 그런가 보다. 시간나면 해보자.

### 2. 형식화 배열

자바스크립트의 `Indexed Collections`에는 다음과 같은 것들이 있다.

https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects#indexed_collections

- Array
- Int8Array
- Uint8Array
- Uint8ClampedArray
- Int16Array
- Uint16Array
- Int32Array
- Uint32Array
- Float32Array
- Float64Array
- BigInt64Array
- BigUint64Array

https://developer.mozilla.org/ko/docs/Web/JavaScript/Typed_arrays

> JavaScript 형식화 배열(typed array)은 배열같은 객체이고 원시(raw) 이진 데이터에 액세스하기 위한 메커니즘을 제공합니다. 이미 아시다시피, Array 객체는 동적으로 늘었다 줄고 어떤 JavaScript 값이든 가질 수 있습니다. JavaScript 엔진은 이러한 배열이 빨라지도록 최적화를 수행합니다. 그러나, audio 및 video 조작과 같은 기능 추가, WebSocket을 사용한 원시 데이터에 액세스 등 웹 어플리케이션이 점점 더 강력해짐에 따라, 빠르고 쉽게 형식화 배열의 원시 이진 데이터를 조작할 수 있게 하는 것이 JavaScript 코드에 도움이 될 때가 있음이 분명해 졌습니다.

사실 자바스크립트의 배열은 엄밀히 말하면 자료구저의 배열이 아니다. 자료구조의 배열은, 동일한 크기의 메모리 공간이 빈틈없이 연속적으로 나열되어 있어야 한다. (밀집 배열) 그러나 자바스크립트 배열은 희소배열이다. 자바스크립트는 동일한 크기를 연속적으로 확보하는 것이 불가능하므로 (무슨타입이 올줄 알고?) 배열의 동작을 흉내내서 만들었다. 그러나 위에 언급된 `Array`를 제외한 나머지 배열들을 `.from()`으로 만들경우, 진짜 `dense array`, 밀집 배열을 만들 수 있다.

> When Array.from() gets an array-like which isn't an iterator, it respects holes. TypedArray.from() will ensure the result is dense.

물론 Array로도 밀집 배열을 만들 수 있다. https://2ality.com/2012/06/dense-arrays.html

아무튼 지간에 결론은, 자바스크립트에는 저런 배열들도 있으므로, 저런 배열의 prototype에도 마찬가지로 추가를 해줬어야 했다.