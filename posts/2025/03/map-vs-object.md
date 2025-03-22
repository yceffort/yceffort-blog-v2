---
title: '맵과 객체 중 무엇을 언제 쓰는 것이 좋을까?'
tags:
  - javascript
  - book
published: true
date: 2025-03-21 22:38:46
description: 'Map 도 씁시다'
---

## Table of Contents

## 개요

[Map](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Map)은 ES6에서 부터 추가된 새로운 데이터 타입입니다. Map은 키와 값의 쌍을 저장하며, 키는 유일해야 한다는 특징을 지니고 있습니다. 얼핏 보면 객체와 비슷해보이지만, 용도와 특징에 맞게 잘활용한다면 때로는 객체보다 더 좋은 선택이 될 수도 있습니다. 그러나 대부분 자바스크립트 프로젝트를 진행하고 나면, 의식적으로 Map 을 사용하는 일은 드뭅니다. 이 글에서는 Map과 객체의 차이점과 어떤 상황에서 어떤 것을 사용하는 것이 좋을지 알아보겠습니다.

## 객체

객체는 자바스크립트 생태계에서 가장 널리 쓰이는 데이터 타입으로, 객체는 키와 값의 쌍을 저장하는 자료구조입니다. 원시값이 아닌 모든 값은 객체이며, 대부분의 자바스크립트 값 (배열 ,함수) 등도 내부적으로 객체를 상속받는 구조로 되어 있는, 자바스크립트 프로그래밍에서 가장 중요한 개념입니다. 주요 특징은 다음과 같습니다.

- 프로퍼티를 동적으로 추가하거나 삭제할 수 있습니다.
- 키로는 오직 문자열과 심볼만 가능합니다.
- 순서가 보장되지 않으며, 기본적으로 이터러블(iterable)이 아닙니다.
- 내부적으로 `Object.prototype`을 상속받기 때문에, `toString`, `hasOwnProperty` 같은 메서드를 사용할 수 있습니다.

```javascript
// 1. 객체 생성
const person = {
  name: 'Alice',
  age: 25,
}

// 2. 프로퍼티 추가
person.city = 'Seoul'

// 3. 프로퍼티 수정
person.age = 26

// 4. 프로퍼티 삭제
delete person.city

// 5. 프로퍼티 접근
console.log(person.name) // 'Alice'
console.log(person['age']) // 26

// 6. 순회(기본적으로는 for...in 또는 Object.keys 등을 사용)
for (const key in person) {
  // 주의: 프로토타입 체인에 있는 것도 나올 수 있으므로 hasOwnProperty 등을 확인
  if (person.hasOwnProperty(key)) {
    console.log(key, person[key])
  }
}

// 결과 예시:
// name 'Alice'
// age 26
```

## Map

Map 역시 키와 값의 쌍을 저장하는 자료구조입니다. 객체와 비슷해보이지만, 다음과 같은 차이점이 있습니다.

- 객체와 다르게 키 타입에 제한이 없고, 어떤 자료형(문자열, 숫자, 객체, 함수 등)이든 키로 사용할 수 있습니다.
- 키-값 쌍을 **추가(set)** 하거나 **삭제(delete)** 할 수 있고, 한 번에 모두 **제거(clear)** 할 수도 있습니다.
- 삽입 순서가 보장되며, 이터러블한 자료구조입니다. 따라서 for...of나 전개 연산자 등을 사용해 손쉽게 순회할 수 있습니다, size 프로퍼티로 원소 수를 바로 확인할 수 있습니다.
- 내부적으로 해시 구조를 사용해, 대규모 데이터에서도 빠른 키 조회/추가/삭제를 지원하도록 설계되었습니다.

```js
// 1. Map 생성
const map = new Map()

// 2. 다양한 타입의 키 사용
const objKey = {id: 1}
const funcKey = function () {}

map.set('name', 'Alice') // 문자열 키
map.set(123, 'Number Key') // 숫자 키
map.set(objKey, 'Object Key')
map.set(funcKey, 'Function Key')

// 3. 값 조회
console.log(map.get('name')) // 'Alice'
console.log(map.get(123)) // 'Number Key'
console.log(map.get(objKey)) // 'Object Key'

// 4. 삭제
map.delete(123)
console.log(map.has(123)) // false

// 5. 전체 삭제
// map.clear(); // 모든 키-값 삭제

// 6. 반복/순회
for (const [key, value] of map) {
  console.log(key, value)
}
// 삽입 순서대로 출력됨
// 'name' 'Alice'
// { id: 1 } 'Object Key'
// [Function: funcKey] 'Function Key'

// 7. 사이즈 확인
console.log(map.size) // 3
```

## 기본적인 차이를 표로 비교

| 구분                   | **Object**                                                                                                                         | **Map**                                                                                     |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **키 타입**            | 문자열(String), 심볼(Symbol)만 가능<br/>(숫자, 객체 등을 키로 사용하면 내부적으로 문자열 변환)                                     | 모든 타입 가능<br/>(문자열, 숫자, 객체, 함수 등 어떤 자료형도 키로 사용 가능)               |
| **이터러블 여부**      | 기본적으로 이터러블하지 않음<br/>`for...in`은 사용 가능하나, 프로토타입 체인까지 순회될 수 있음                                    | 기본적으로 **이터러블**<br/>`for...of`로 손쉽게 순회 가능                                   |
| **순서 보장**          | 프로퍼티 순서가 보장되지 않거나, 숫자 문자열 키가 우선 정렬되는 등<br/>브라우저마다 일부 일관성이 있긴 하지만 완전히 보장되진 않음 | 삽입된 순서를 그대로 유지                                                                   |
| **크기 확인 (size)**   | 전용 프로퍼티 없음<br/>`Object.keys(obj).length` 등 별도 방법으로 O(n)에 구해야 함                                                 | `map.size` 프로퍼티로 O(1)에 확인 가능                                                      |
| **프로토타입 상속**    | `Object.prototype` 내장 메서드(`toString`, `hasOwnProperty` 등)를 상속받아 사용<br/>사용자 정의 프로퍼티와 충돌 가능성 존재        | `Map` 자체 메서드 집합만 사용<br/>사용자 정의 키와 충돌 위험 적음                           |
| **프로퍼티 추가·삭제** | 점(`.`) 또는 대괄호(`[]`) 표기법으로 추가·삭제 가능<br/>전체 삭제 시 별도의 반복문이나 여러 번 `delete` 호출 필요                  | `map.set(key, value)`, `map.delete(key)`, `map.clear()`로 간편히 관리 가능                  |
| **주요 사용 예**       | - 고정된 구조의 **레코드(Record)**<br/>- 프로퍼티가 많지 않은 간단한 데이터<br/>- Config, Options 등                               | - **동적으로 키가 자주 바뀌는 해시맵**<br/>- 대규모 데이터에서 빠른 조회·삭제가 필요한 경우 |

## 해시 맵

두 데이터 타입중 어떤 것이 더 적절한지를 알기 위해선, 먼저 해시맵 데이터 타입에 대해서 알아야 합니다.

### 해시 맵의 정의

해시 맵은 키와 값을 한 쌍에 저장하는 데이터 구조로, '해싱' 이라는 기법을 사용하여 빠르게 데이터 검색, 삭제, 삽입을 가능하게 합니다. 이 기법을 사용하면, 평균적으로 O(1)의 시간 복잡도로 원하는 데이터를 조회할 수 있습니다. 해시 맵은 다음과 같은 방식으로 동작합니다.

1. 해시 함수: 키를 받으면, 그 키를 특정 숫자로 매핑해주는 함수를 일컫습니다.
2. 해시 값을 기반으로 인덱스 결정: 1에서 받은 키를 바탕으로 인덱스를 구합니다.
3. 충돌 해결: 2에서 구한 키가 겹치는 경우가 있는데, 이를 충돌이라고 합니다. 이 충돌을 해결하는 방법으로는 크게 두가지가 있습니다.
   1. 체이닝: 같은 인덱스에 링크드 리스트를 만들어 키-값을 쌍으로 보관
   2. 오픈 어드레싱: 충돌이 발생하면 다른 빈 공간을 찾아서 삽입
4. 검색, 삽입, 삭제: 키에 대해 해시 함수를 적용하여 인덱스를 찾고, 저장한 값에 바로 접근

해시 맵은 다음과 같은 장점을 지닙니다.

- 빠른 검색, 삽입, 삭제: 해시 함수를 통해 O(1)의 시간 복잡도로 데이터를 관리할 수 있습니다.
- 다양한 타입의 키 사용 가능: 문자열, 숫자, 객체, 함수 등 어떤 자료형도 키로 사용 가능합니다.

반면에 다음과 같은 단점도 존재합니다.

- 해시 함수가 적절하지 않거나 충돌이 많아지면 O(n) 까지도 성능이 떨어질 수 있습니다.
- 해시 함수를 계산하는데 오버헤드가 존재합니다.

그러나 일반적으로 자바스크립트의 경우, 해시 함수를 직접 구현할 일이 거의 없습니다. 이미 v8 엔진 등 내부 엔진에서 해시 알고리즘을 자체적으로 구현하여 사용자에게 제공하고 있기 때문입니다.

### 해시 맵이 필요한 경우

해시 맵은 다음과 같은 상황에서 유용하게 사용될 수 있습니다.

- 대용량 데이터에서 빠르게 삽입, 삭제가 필요한 경우
- 키 값 쌍을 관리해야 하는데, 키의 수가 동적으로 계속해서 변하는 경우
- 중복 여부를 효율적으로 확인해야 하는 경우
- 빈번하게 키에 접근해야 하는 경우
- 키를 보다 안전하게 저장하고 싶은 경우

## 해시 맵으로 객체가 부적절한 이유

해시 맵은 프로그래머에게 꼭 필요한 기능이지만, 자바스크립트 생태계에서 객체를 해시맵으로 쓰기에는 다음과 같은 한계가 존재합니다.

### 제한된 키

객체는 키로 문자열과 심볼만 사용할 수 있습니다. 만약 객체를 해시맵으로 사용하려면, 키로 문자열이나 심볼만 사용해야 합니다. 그러나 해시맵은 어떤 자료형이든 키로 사용할 수 있어야 합니다. 그러나 객체에서는 문자열과 심볼 외에 다른 자료 형이 오는 경우, 내부적으로 `toString()`을 호출하여 문자열로 변환을 해버립니다.

```js
const foo = []
const bar = {}
const obj = {[foo]: 'foo', [bar]: 'bar'}

console.log(obj)
// 결과: {"": 'foo', "[object Object]": 'bar'}
```

### 상속

해시맵 사용을 위해 객체를 사용한다고 가정해봅시다.

```js
const hashMap = {}
```

이 해시맵 (객체) 는 정말로 아무것도 없는 빈 상태일까요? 그렇지 않습니다. 이미 객체라는 사실 때문에 `Object.prototype`에서 상속 받은 각종 메서드를 지닌 상태로 시작합니다.

```js
// 객체 생성
const hashMap = {}
// 내부적으로 상속받은 객체 존재
console.log(Object.getPrototypeOf(hashMap))
// hasOwnProperty, isPrototypeOf, propertyIsEnumerable, toLocaleString, toString, valueOf....
console.log('toString' in hashMap) // true
```

이는 자바스크립트의 특징인 프로토타입 상속으로 인해 발생하는 현상으로, 객체 자신의 프로퍼티와 체인을 통해 상속된 프로퍼티가 뒤섞이게 됩니다. 이 때문에 사요앚가 만든 프로퍼티와, 상속된 프로퍼티를 구분하려면 [hasOwnProperty](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Object/hasOwnProperty) 메서드를 사용해서 반드시 확인해야 합니다.

뿐만 아니라, 섣부르게 `Object.prototype`을 수정하면, 모든 객체에 영향을 미치게 되므로 프로토타입 오염 공격을 맞닥드릴 수 도 있으며, 원치 않는 부작용이 발생할 수 있습니다.

### 이름 충돌

객체를 해시맵으로 사용할 때, 키로 사용하는 문자열이 중복되는 경우가 발생할 수 있습니다. 이는 객체의 프로퍼티 이름이 중복되는 경우, 마지막에 선언된 프로퍼티가 이전 프로퍼티를 덮어쓰게 됩니다. 이는 객체의 프로퍼티 이름이 중복되는 경우, 마지막에 선언된 프로퍼티가 이전 프로퍼티를 덮어쓰게 됩니다.

예를 들어 다음과 같은 코드가 있다고 가정해봅시다.

```js
function foo(obj) {
  // 만약 obj가 { hasOwnProperty: 'foo' } 라면?
  for (const key in obj) {
    // ???????
    if (obj.hasOwnProperty(key)) {
    }
  }
}
```

위 코드는 객체를 인수로 받아, 객체의 프로퍼티를 순회하는 함수입니다. 그러나 만약 객체의 프로퍼티 이름이 말그대로 `hasOwnProperty` 가 있는 경우, 이 함수는 제대로 동작하지 않습니다. 이는 `hasOwnProperty`가 `Object.prototype`에 있는 메서드이기 때문에, 객체의 프로퍼티로 사용되면서 함수가 제대로 동작하지 않게 됩니다.

이를 진짜 제대로 방어하기 위해서는 다음과 같이 진짜 `Object.prototype.hasOwnProperty` 메서드를 사용해야 합니다.

```js
function foo(obj) {
  for (const key in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, key)) {
    }
  }
}
```

### 불편한 메소드

해시 맵을 사용할 때 써야하는 주요 메소드를 살펴보면 다음과 같으며, 객체에서는 아래 메소드를 사용하는데 애로 사항이 존재합니다.

- `size`: 객체에는 알고 싶은 종류에 따라서 크기를 가져올 수 있는 메소드가 3개나 존재합니다. 그리고 사실 직접 프로퍼티를 순회해서 길이를 확인해야 하므로, O(n)의 시간 복잡도가 발생합니다.
  - `Object.keys(obj).length`: 열거 가능한 프로퍼티의 개수를 반환합니다.
  - `Object.getOwnPropertyNames(obj).length`: 객체 자신의 프로퍼티 중 열거 가능한 프로퍼티의 개수를 반환합니다.
  - `Object.getOwnPropertySymbols(obj).length`: 객체 자신의 심볼 프로퍼티의 개수를 반환합니다.
- `clear`: 객체를 초기화하는 메소드가 없습니다. 객체를 초기화하려면, 객체를 다시 생성하거나 `delete`를 사용하는 수밖에 없습니다.

- 순회: 순회 역시 `size`와 비슷한 문제가 존재합니다.

  - `for...in`: 객체의 프로퍼티를 순회할 때, 프로토타입 체인까지 순회하므로 `hasOwnProperty`를 사용해야 합니다.

  ```js
  Object.prototype.foo = 'bar'
  const obj = {bar: 1}
  for (const key in obj) {
    console.log(key) // foo, bar
  }
  ```

  - `for..of`: 객체는 이터러블이 아니므로 사용할 수 없습니다.
  - `Object.keys`, `Object.values`, `Object.entries`: 이 메소드들은 객체의 프로퍼티를 배열로 반환해주는 메소드라서 사용가능하지만, 어쨌든 배열을 한번 더 만드는 비용이 존재합니다.

  이와 더불어 순서가 보장되지 않는 다는 점도 중요합니다.

  ```js
  const obj = {}

  obj.foo = '첫번째'
  obj[2] = '두번째'
  obj[1] = '세번째'

  // {1: '세번째', 2: '두번째', foo: '첫번째'}
  ```

- 존재여부 확인하기: 일반적으로 객체에서는 `.`이나 `[]` 을 사용해 `undefined`를 반환하는지 확인합니다만, 실제로 값이 `undefined`로 인 경우에도 동일하게 반환합니다.

  ```js
  const obj = {a: null, b: undefined}
  obj.b === obj.c // true
  ```

  이러한 문제로 인해 일반적으로는 `in`연산자를 사용합니다만, 이 역시 프로토타입 문제가 있으므로 제대로 확인하기 위해서는 `Object.prototype.hasOwnProperty`를 사용해야 합니다.

  ```js
  const obj = {a: null, b: undefined}
  'b' in obj // true
  'c' in obj // false
  Object.prototype.hasOwnProperty.call(obj, 'b') // true
  Object.prototype.hasOwnProperty.call(obj, 'c') // false
  ```

반면 `Map`은 개발자 의도대로 동작하는 메소드를 각각 제공합니다.

- `Map.prototype.size`: 맵의 크기 확인
- `Map.prototype.clear`: 맵 초기화
- `Map.prototype.keys`, `Map.prototype.values`, `Map.prototype.entries`: 각각 맵의 키, 값, 키-값 쌍을 반환
- `Map.prototype.has`: 키 존재 여부 확인

## 벤치 마크로 성능 비교

실제 벤치마크에서 객체와 맵의 성능을 비교해보겠습니다. 벤치마크로는 deno 를 사용하였습니다.

```ts
const DATA_SIZE = 100_000

function generateRandomStrings(n: number): string[] {
  const result: string[] = []
  for (let i = 0; i < n; i++) {
    const rand = Math.random().toString(36).slice(2, 12)
    result.push(rand)
  }
  return result
}
```

### 삽입

```ts
Deno.bench({
  name: 'Object Insertion',
  group: 'Insertion',
  baseline: true,
  fn: () => {
    const keys = generateRandomStrings(DATA_SIZE)
    const obj: Record<string, number> = {}

    for (let i = 0; i < DATA_SIZE; i++) {
      obj[keys[i]] = i
    }
  },
})

Deno.bench({
  name: 'Map Insertion',
  group: 'Insertion',
  fn: () => {
    const keys = generateRandomStrings(DATA_SIZE)
    const map = new Map<string, number>()

    for (let i = 0; i < DATA_SIZE; i++) {
      map.set(keys[i], i)
    }
  },
})
```

```bash
    CPU | Apple M3 Pro
Runtime | Deno 2.2.5 (aarch64-apple-darwin)

benchmark                           time/iter (avg)        iter/s      (min … max)           p75      p99     p995
----------------------------------- ----------------------------- --------------------- --------------------------

group Insertion
Object Insertion                            47.2 ms          21.2 ( 41.0 ms …  72.5 ms)  48.8 ms  72.5 ms  72.5 ms
Map Insertion                               20.3 ms          49.3 ( 19.1 ms …  24.0 ms)  20.3 ms  24.0 ms  24.0 ms

summary
  Object Insertion
     2.33x slower than Map Insertion
```

객체가 맵보다 2.33배 느리게 삽입되었습니다.

### 순회

```ts
Deno.bench({
  name: 'Object Iteration (for...in)',
  group: 'Iteration',
  baseline: true,
  fn: () => {
    const keys = generateRandomStrings(DATA_SIZE)
    const obj: Record<string, number> = {}
    for (let i = 0; i < DATA_SIZE; i++) {
      obj[keys[i]] = i
    }

    let sum = 0
    for (const k in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, k)) {
        sum += obj[k]
      }
    }
  },
})

Deno.bench({
  name: 'Map Iteration (for...of)',
  group: 'Iteration',
  fn: () => {
    const keys = generateRandomStrings(DATA_SIZE)
    const map = new Map<string, number>()
    for (let i = 0; i < DATA_SIZE; i++) {
      map.set(keys[i], i)
    }

    // 순회
    let sum = 0
    for (const [_, value] of map) {
      sum += value
    }
  },
})
```

```bash
    CPU | Apple M3 Pro
Runtime | Deno 2.2.5 (aarch64-apple-darwin)

benchmark                           time/iter (avg)        iter/s      (min … max)           p75      p99     p995
----------------------------------- ----------------------------- --------------------- --------------------------

group Iteration
Object Iteration (for...in)                 56.0 ms          17.9 ( 46.2 ms …  83.6 ms)  56.1 ms  83.6 ms  83.6 ms
Map Iteration (for...of)                    20.3 ms          49.4 ( 18.5 ms …  25.1 ms)  20.9 ms  25.1 ms  25.1 ms

summary
  Object Iteration (for...in)
     2.77x slower than Map Iteration (for...of)

```

객체가 맵보다 2.77배 느리게 순회되었습니다.

### 삭제

```ts
Deno.bench({
  name: 'Object Deletion (delete operator)',
  group: 'Deletion',
  baseline: true,
  fn: () => {
    const keys = generateRandomStrings(DATA_SIZE)
    const obj: Record<string, number> = {}
    for (let i = 0; i < DATA_SIZE; i++) {
      obj[keys[i]] = i
    }

    for (let i = 0; i < DATA_SIZE; i++) {
      delete obj[keys[i]]
    }
  },
})

Deno.bench({
  name: 'Map Deletion (map.delete)',
  group: 'Deletion',
  fn: () => {
    const keys = generateRandomStrings(DATA_SIZE)
    const map = new Map<string, number>()
    for (let i = 0; i < DATA_SIZE; i++) {
      map.set(keys[i], i)
    }

    for (let i = 0; i < DATA_SIZE; i++) {
      map.delete(keys[i])
    }
  },
})
```

```bash
Check file:///Users/USER/private/deno-study/helloworld/bench-map-object.ts
    CPU | Apple M3 Pro
Runtime | Deno 2.2.5 (aarch64-apple-darwin)

file:///Users/USER/private/deno-study/helloworld/bench-map-object.ts

benchmark                           time/iter (avg)        iter/s      (min … max)           p75      p99     p995
----------------------------------- ----------------------------- --------------------- --------------------------

group Deletion
Object Deletion (delete operator)           50.5 ms          19.8 ( 41.2 ms …  74.8 ms)  52.4 ms  74.8 ms  74.8 ms
Map Deletion (map.delete)                   26.1 ms          38.3 ( 21.8 ms …  42.0 ms)  27.1 ms  42.0 ms  42.0 ms

summary
  Object Deletion (delete operator)
     1.93x slower than Map Deletion (map.delete)
```

객체가 맵보다 1.93배 느리게 삭제되었습니다.

### 무작위 값 검색

```ts
const DATA_SIZE = 1_000_000

// 일부 키만 존재하고, 나머지는 존재하지 않도록 비율을 조정
// 70%는 실제로 존재하는 키, 30%는 존재하지 않는 키
const EXISTING_RATIO = 0.7

function generateRandomStrings(n: number): string[] {
  const result: string[] = []
  for (let i = 0; i < n; i++) {
    const rand = Math.random().toString(36).slice(2, 12)
    result.push(rand)
  }
  return result
}

function generateMixedLookupKeys(
  allKeys: string[],
  totalLookups: number,
  existingRatio: number,
): string[] {
  const existingCount = Math.floor(totalLookups * existingRatio)
  const nonExistingCount = totalLookups - existingCount
  const lookupKeys: string[] = []

  // 1) 실제로 존재하는 키 중 무작위로 existingCount개를 샘플
  for (let i = 0; i < existingCount; i++) {
    const randomIndex = Math.floor(Math.random() * allKeys.length)
    lookupKeys.push(allKeys[randomIndex])
  }

  // 2) 전혀 없는 무작위 키 생성
  const nonexistentKeys = generateRandomStrings(nonExistingCount)
  for (const key of nonexistentKeys) {
    lookupKeys.push(key)
  }

  // 3) 전체를 무작위로 섞기 (Fisher–Yates shuffle)
  for (let i = lookupKeys.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[lookupKeys[i], lookupKeys[j]] = [lookupKeys[j], lookupKeys[i]]
  }

  return lookupKeys
}

const keys = generateRandomStrings(DATA_SIZE)
const obj: Record<string, number> = {}
const map = new Map<string, number>()

for (let i = 0; i < DATA_SIZE; i++) {
  obj[keys[i]] = i
  map.set(keys[i], i)
}

const lookupKeys = generateMixedLookupKeys(keys, DATA_SIZE, EXISTING_RATIO)

Deno.bench({
  name: 'Object Random Lookup (mixed existing/non-existing)',
  group: 'RandomLookup',
  fn: () => {
    let found = 0
    for (let i = 0; i < DATA_SIZE; i++) {
      const k = lookupKeys[i]
      if (obj[k] !== undefined) {
        found++
      }
    }
  },
})

Deno.bench({
  name: 'Object Random Lookup with hasOwnProperty (mixed existing/non-existing)',
  group: 'RandomLookup',
  fn: () => {
    let found = 0
    for (let i = 0; i < DATA_SIZE; i++) {
      const k = lookupKeys[i]
      if (Object.prototype.hasOwnProperty.call(obj, k)) {
        found++
      }
    }
  },
})

Deno.bench({
  name: 'Map Random Lookup (mixed existing/non-existing)',
  group: 'RandomLookup',
  fn: () => {
    let found = 0
    for (let i = 0; i < DATA_SIZE; i++) {
      const k = lookupKeys[i]
      if (map.has(k)) {
        found++
      }
    }
  },
})
```

```bash
    CPU | Apple M3 Pro
Runtime | Deno 2.2.5 (aarch64-apple-darwin)

file:///Users/USER/private/deno-study/helloworld/bench-map-object-lookup.ts

benchmark                                                                time/iter (avg)        iter/s      (min … max)           p75      p99     p995
------------------------------------------------------------------------ ----------------------------- --------------------- --------------------------

group RandomLookup
Object Random Lookup (mixed existing/non-existing)                               56.7 ms          17.6 ( 56.3 ms …  58.3 ms)  56.7 ms  58.3 ms  58.3 ms
Object Random Lookup with hasOwnProperty (mixed existing/non-existing)           53.4 ms          18.7 ( 53.1 ms …  54.3 ms)  53.5 ms  54.3 ms  54.3 ms
Map Random Lookup (mixed existing/non-existing)                                 113.4 ms           8.8 (109.7 ms … 126.3 ms) 114.1 ms 126.3 ms 126.3 ms

summary
  Object Random Lookup with hasOwnProperty (mixed existing/non-existing)
     1.06x faster than Object Random Lookup (mixed existing/non-existing)
     2.12x faster than Map Random Lookup (mixed existing/non-existing)
```

여기서는 다소 예상과 다르게 객체가 맵보다 빠르게 검색되었습니다. 자바스크립트 엔진은 객체 속성 접근을 최적화 하기 위해 인라인 캐싱, 히든 클래스 등 다양한 최적화 기법을 사용하고 있습니다. 또한 심볼과 키만 사용가능 한 객체의 경우에는 조금더 최적화 할 여지가 있을 수 있고, 나아가 객체는 아무래도 오랜시간 사용된 자바스크립트의 기본 데이터 타입이기 때문에, 엔진 내부에서 더 최적화가 되어있을 수 있습니다.

> - 인라인캐싱: 동일한 속성을 접근하는 작업이 반복되면, 이를 캐싱하여 빠르게 접근할 수 있도록 하는 기법
> - 히든클래스: 객체에 어떤 프로퍼티들이 있고, 어떤 순서대로 추가되었는지 등을 기록하여, 빠르게 접근할 수 있도록 하는 기법

## 실제 예제

서버개발이 아닌 프론트엔드 개발이라면 일반적으로 객체를 더 많이 사용하게 되고, 아마 그렇기 때문에 맵에 대한 필요성을 크게 느끼지 못하실 수도 있습니다. 그렇다면 프론트엔드 개발에서 맵이 필요한 상황, 즉 추가와 삭제, 그리고 검색이 빈번하게 일어나며 동적으로 계쏙해서 바뀌는 경우에는 무엇이 이있을까요? 바로 `EventEmitter`입니다.

`EventEmitter`는 이름 그대로 이벤트를 발생시키고(listen), 구독(subscribe), 해제(unsubscribe)하는 기능을 제공하는 객체이자 패턴을 말합니다. 주로 Node.js나 브라우저 환경에서 이벤트 기반으로 동작하기 위해 쓰이며, 이는 간단히 말해 이벤트 이름에 따라 콜백을 등록하는 것을 의미합니다.

```ts
eventEmitter.on('data', onDataReceived)
eventEmitter.on('error', onError)
eventEmitter.emit('data', {
  // do something
})
```

이벤트를 관리하는데 맵이 사용되어야 하는 이유는 다음과 같습니다.

- 동적으로 늘어나는 키: 이벤트 이름은 작성시점에만 국한될 뿐 아니라 런타임에서도 확장될수 있습니다.
- 다양한 타입의 키: 대부분의 이벤트 명은 문자열이지만, 특수한 경우에는 심볼이나 다른 타입의 키를 사용할 수도 있습니다.
- 빈번한 추가 및 삭제: 이벤트는 동적으로 추가되거나 삭제되는 경우가 많은데, 이때 맵은 `set`, `delete` 메소드를 통해 간편하게 관리할 수 있습니다.
- 이름 충돌이 없음: 객체와 다르게 프로로타입과 이름이 충돌될 걱정을 하지 않아도 됩니다.
- 순회 및 사이즈 계산에 유리: 등록된 이벤트를 순회해야할 때, 맵은 이터러블이므로 `for...of`문을 간단하게 사용할 수 있으며, `size` 프로퍼티로 사이즈를 쉽게 확인할 수 있습니다.

```ts
type Listener = (...args: unknown[]) => void

export class EventEmitter {
  private events: Map<string, Listener[]>

  constructor() {
    this.events = new Map<string, Listener[]>()
  }

  public on(eventName: string, listener: Listener): this {
    if (!this.events.has(eventName)) {
      this.events.set(eventName, [])
    }
    this.events.get(eventName)!.push(listener)
    // 메서드 체이닝 지원
    return this
  }

  public emit(eventName: string, ...args: unknown[]): void {
    const listeners = this.events.get(eventName)
    if (!listeners) return // no listeners

    for (const fn of listeners) {
      fn(...args)
    }
  }

  public off(eventName: string, listener: Listener): void {
    const listeners = this.events.get(eventName)
    if (!listeners) return

    // 특정 리스너 함수만 찾아서 제거
    const idx = listeners.indexOf(listener)
    if (idx !== -1) {
      listeners.splice(idx, 1)
    }
    // 더 이상 리스너가 없다면, 맵에서 전체 이벤트 삭제
    if (listeners.length === 0) {
      this.events.delete(eventName)
    }
  }

  public clearEvent(eventName: string): void {
    // 해당 이벤트 자체를 통째로 제거
    this.events.delete(eventName)
  }
}
```

```ts
// EventEmitter.ts
import {EventEmitter} from './EventEmitter' // 위 구현을 임의 파일명으로 저장했다고 가정

// 1) EventEmitter 인스턴스 생성
const emitter = new EventEmitter()

// 2) 콜백 함수 정의
function onDataReceived(data: unknown) {
  console.log('[onDataReceived] data:', data)
}

function onError(err: Error) {
  console.error('[onError]', err)
}

// 3) 이벤트 리스너 등록
emitter.on('data', onDataReceived)
emitter.on('error', onError)

// 4) 이벤트 발생 (emit)
emitter.emit('data', {message: 'Hello, world!'})
// 출력: [onDataReceived] data: { message: 'Hello, world!' }

// 5) 특정 리스너 해제
emitter.off('data', onDataReceived)

// 더 이상 'onDataReceived'는 호출되지 않음
emitter.emit('data', {message: 'Should NOT be logged!'})

// 6) 이벤트 전체 삭제
emitter.clearEvent('error')

// 'error' 이벤트가 완전히 제거되었으므로 emit해도 아무 일도 일어나지 않음
emitter.emit('error', new Error('Test error'))
```

## 결론

- 작성 시점에 프로퍼티가 고정적으로 정해져있고, 적은 수의 프로퍼티를 다루는 경우에는 객체가 적합합니다.
- 키의 개수가 가변적이고, 반복적으로 추가 및 삭제가 발생하며, 작성 시점에 어떤 프로퍼티가 들어올지 모르는 경우, 그리고 해시맵을 다루기 위한 안정적인 메소드가 필요한 경우에는 맵이 더 적합합니다.
- 대부분의 경우 맵이 더 성능이 빠르지만, 자바스크립트의 오랜 최적화 노력 덕분에 일부 상황에서는 객체가 더 빠를 수도 있습니다.
- 프론트엔드 개발에서도 Map을 쓸 수 있습니다. ES6 이상을 지원하는 브라우저에서는 Map을 사용할 수 있으며, 이벤트 관리와 같은 상황에서 유용하게 사용할 수 있습니다.
