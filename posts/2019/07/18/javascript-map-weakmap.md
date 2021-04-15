---
title: Javascript Set 그리고 Map
date: 2019-07-18 08:30:52
published: true
tags:
  - javascript
description:
  '## 근데 사실.. 컬렉션 필요없지 않을까 자바스크립트에서 일반적인 Object는 key-value쌍을 끊임 없이
  추가할 수 있는 형태로 구성되어 있다. 그래서 사실 컬렉션이 필요하지 않은 것 처럼 보일 수도 있다. 그러나 이따금씩 object로 부족할
  때가 있다.  - key 충돌 위험이 존재하는 경우 - 문자열/심볼 이외의 키 값이 필요한 경우 - 객...'
category: javascript
slug: /2019/07/18/javascript-map-weakmap/
template: post
---

## 근데 사실.. 컬렉션 필요없지 않을까

자바스크립트에서 일반적인 Object는 key-value쌍을 끊임 없이 추가할 수 있는 형태로 구성되어 있다. 그래서 사실 컬렉션이 필요하지 않은 것 처럼 보일 수도 있다. 그러나 이따금씩 object로 부족할 때가 있다.

- key 충돌 위험이 존재하는 경우
- 문자열/심볼 이외의 키 값이 필요한 경우
- 객체에 얼마나 많은 속성이 있는지 알아낼 수 있는 효과적인 방법이 필요한 경우
- 객체가 iterable하지 않음. 따라서 `for..of` 나 `...`를 사용할 수 없음

es6에 추가된 컬렉션들은 따라서 멤버 데이터를 드러내기 위해 property를 사용하지 않는다. (`obj.key`, `obj[key]` 불가능.) 그리고 이들에는 자유롭게 메소드를 추가할 수 있다.

## Set

`Set`은 value로 이루어진 컬렉션이다. 그리고 수정가능하다. 배열과 같을 것 같지만, 다르다.

```javascript
let faces = new Set('😀 😁 😂 🤣 😃 😄 😅')
faces.size // 8
faces.add('😂')
```

- 일단 (당연하게도) Set에는 같은 value가 중복으로 포함될 수 없다. 기존에 있는걸 추가해도 아무런 변화가 없다.
- Set은 어떤데이터가 자신의 멤버인지 빠르게 확인하기 위한 목적으로 사용한다
- 그러나 Set은 index로 값을 조회할 수는 없다.

### set 으로 할 수 있는 것

- `new Set()`: 비어있는 set 생성
- `set.size`: set 데이터 개수 조회
- `set.has(value)`: `value`가 `set`에 존재하는지 조회
- `set.add(value)`
- `set.delete(value)`
- `sets[Symbol.iterator]()`: set 안의 값을 순회할 수 있는 새로운 이터레이러를 리턴한다. set을 iterable하게 만들어 준다.

  ```javascript
  let faces = new Set('😀 😁 😂 🤣 😃 😄 😅')
  iteratorFaces = faces[Symbol.iterator]()
  for (let i of iteratorFaces) {
    console.log(i)
  }
  ```

  - `set.forEach(f)`
  - `set.clear`
  - `set.keys()`
  - `set.values()`
  - `set.entries()`

## Map

Map은 잘 알려진 것처럼, key-value pair로 이루어진 컬렉션이다.

### Map으로 할 수 있는 것

- `new Map`
- `new Map(pairs)`
- `map.has(key)`
- `map.size`
- `mag.get(key)`
- `map.set(key, value)`
- `map.delete(key)`
- `map.clear()`
- `map[Symbol.iterator]()` === `map.entries()`
- `map.forEach(f)`
- `map.keys()`
- `map.values()`

## javascript 가 다른점

아래의 코드를 보자.

```javascript
let messi = new Set()

const 리오넬메시 = { name: '리오넬메시' }
const 라이오넬멧시 = { name: '리오넬메시' }
messi.add(리오넬메시)
messi.add(라이오넬멧시)

console.log(messi.size) //2 ????
```

`리오넬메시`와 `라이오넬멧시`는 내부의 값이 같아 보이기 때문에, set에 한개의 값만 추가 될 것 같지만 사실은 그렇지 않다. 자바스크립트에서는 두개의 값을 다르게 본다. 이유는 자바스크립트가 값을 비교할 때 두가지 다른 방법을 사용하기 때문이다.

- string, number같은 primitive는 값을 비교한다
- array, date, object 등은 reference를 비교한다. (메모리의 같은 위치를 참조하고 있는가?)

```javascript
const 리오넬메시 = { name: '리오넬메시' }
const 라이오넬멧시 = { name: '리오넬메시' }
const 메석대 = 리오넬메시

메석대 === 리오넬메시 // true
메석대 == 라이오넬멧시 // false
리오넬메시 === 라이오넬멧시 //false
```

본질적으로 같은 메모리를 참조하는 값 끼리만 true를 반환하는 것을 볼 수 있다. (두 object를 비교하는 방법은 [여기](https://stackoverflow.com/questions/1068834/object-comparison-in-javascript)를 참조)

다시 `Set`으로 돌아와서, javascript는 저 두 값을 제대로 비교하지 못하기 때문에 set에 두개의 값이 들어가게 된다. 물론 해시코드를 사용하면 가능하지만, javascipt에는 그런거 없다

또 하나 다른 점이라고 한다면, `map`과 `set`에 추가한 순서가 곧 순회하는 순서와 같다는 것이다. 이 역시 다른 언어들과는 다른 점이다.

## WeakMap, WeakSet

- `WeakMap`은 `new` `.has()` `.get()` `.set()` `.delete()` 만 지원한다
- `WeakSet`은 `new` `.has()` `.add()` `.delete()` 만 지원한다.
- `WeakSet`과 `WeakMap`의 key는 반드시 `object`여야 한다.

그렇다. 열거형이 존재하지 않는다. 그 이유는 참조하고 있는 오브젝트가 사라지면 해당 key, value가 사라지는 `WeakMap`, `WeakSet`의 특징 때문이다.

```javascript
let john = { name: 'John' }
// 객체에 접근 가능, 해당 객체는 메모리에서 참조되고 있음.
// 참조를 null로 overwrite
john = null
// 객체는 메모리에서 이제 삭제됨
```

```javascript
let john = { name: 'John' }
let array = [john]
john = null // 참조를 null로 overwrite

// john은 객체안에 살아 있기 때문에 가비지 컬렉팅이 되지 않음.
// 그래서 array[0]으로 접근 가능
array[0]
// { name: "John" }
```

이는 기존 Map, Set에서도 동일하다.

```javascript
let john = { name: 'John' }

let map = new Map()
map.set(john, '윅')

john = null // 참조를 null로 overwrite

// john은 맵안에서 살아있기 때문에
// map.keys() 로 접근 가능
```

그러나 WeakMap, WeakSet은 다르다

```javascript
let john = { name: 'John' }

let weakMap = new WeakMap()
weakMap.set(john, '윅')

john = null // 참조를 null로 overwrite

// john은 메모리에서 사라짐 (가비지 콜렉팅 당함)
```

그럼 도대체 이것은 언제 쓸까? 객체가 사라지면 자동으로 가비지 콜렉팅 해준다는 특성을 활용해, 아래와 같은 것이 가능하다.

```javascript
let john = { name: 'John' }
// map: 유저 => 방문횟수
let visitsCountMap = new Map()
visitsCountMap.set(john, 123)

// john이 사라짐
john = null

// 그러나 Map에서는 계속 남아 있으므로, 따로 처리를 해주어야 함.
// 또한 john은 map에서 key로 사용하고 있으므로 메모리에서도 존재함.
console.log(visitsCountMap.size) // 1
```

그러나 여기서 WeakMap을 사용하면, 자동으로 가비지 콜렉팅이 되므로 Map에 남아있는 key에 대해서 까지 신경쓰지 않아도 된다.
