---
title: '타입스크립트의 구조 타이핑'
tags:
  - typescript
published: true
date: 2021-06-10 22:30:19
description: '얀센 맞고 정신 나가서 하루를 순삭당했습니다'
---

타입스크립트의 타입 체크는 가끔 내가 생각하는 것보다 광범위 해서 생각치도 못한 결과를 만들어 낼 때가 있다. 아래와 같은 코드가 있다고 가정해보자.

```typescript
interface Vector2D {
  x: number
  y: number
}

function calcLength(v: Vector2D) {
  return Math.sqrt(v.x * v.x + v.y * v.y)
}
```

그리고 새로운 interface를 만들고, 이를 함수의 값으로 넣었다.

```typescript
interface Vector2DWithName extends Vector2D {
  name: string
}

const a: Vector2DWithName = { name: 'hi', x: 5, y: 10 }
calcLength(a) // works fine

interface Vector2DName {
  name: string
  x: number
  y: number
}

const b: Vector2DName = { name: 'hello', x: 10, y: 10 }
calcLength(b) // works fine, too.
```

`Vector2DWithName`야 뭐, `extends`로 그 둘 사이에 관계가 어떻게 어떻게 보였다고 치더라도, `Vector2DName`과 `Vector2D` 둘 사이에는 별다른 관계가 선언되거나 한적이 없음에도 정상적으로 작동하는 것을 볼 수 있다.

타입스크립트의 타입시스템은 '구조적으로' 타입이 맞기만 한다면 이를 허용해준다. 여기서 등장한 용어가 바로 [structural typing](https://www.typescriptlang.org/docs/handbook/type-compatibility.html) 이다. (뭐 제대로된 번역을 본적이 없어서 그냥 구조적 타이핑이라고 하겠다.)

일반적으로 다른 언어, C#, Java 등에서는 허용하지 않는 방법이다. 그러나 이 때문에 예기치 못한 문제를 만들어 낼 수 있다.

```javascript
interface Vector3D {
    x: number
    y: number
    z: number
}

function normalize(v: Vector3D) {
    const length = calcLength(v) // z가 고려되지 않음
    return {
        x: v.x / length,
        y: v.y / length,
        z: v.z / length // z의 값이 이상하게 나옴
    }
}

normalize({x:3, y:4, z:5}) // 그러나 에러는 안남
```

일반적으로 함수를 작성할 때, 함수에서 들어오는 인수가 애초에 원하는 대로 의도한 인수만 가지고 있고, 그 외의 값은 안올 것이라고 기대하고, 그게 일반적이다. 그러나 타입스크립트의 타입 시스템에서는 그렇지 않다. 타입스크립트의 타입은 열려있기 때문이다. 이러한 함정(?) 때문에, 타입스크립트를 처음 접하게 되면 아래와 같은 실수를 많이 범하게 된다.

```typescript
function calcLengthV1(v: Vector3D) {
  let length = 0
  for (const axis of Object.keys(v)) {
    const coord = v[axis] // Element implicitly has an 'any' type because expression of type 'string' can't be used to index type 'Vector3D'.
    // No index signature with a parameter of type 'string' was found on type 'Vector3D'.(7053)
    length += Math.abs(coord)
  }
  return length
}
```

내가 선언한 `v`의 키 값들은 x,y,z로 잘되어 있고, 이는 모두 string이라 잘 들어가야 한다. 그리고 값들도 number라서 length에 잘 더할 수 있어야 한다. 그런데 왜 에러가 나지?

> [https://yceffort.kr/2021/05/do-not-use-suppressImplicitAnyIndexErrors](/2021/05/do-not-use-suppressImplicitAnyIndexErrors) 에서 한번 다룬적이 있다.

그러나, 위에서 이야기 한 것처럼 타입스크립트의 타입이 열려 있다는 것을 생각해본다면 타당한 에러다.

```typescript
const v = { x: 1, y: 2, z: 3, name: 'hi, h i~' }
calcLengthV1(v) // name의 값이 NaN이라서 결과가 NaN으로 뜰 수 있다.
```

따라서, 위의 코드는 타입스크립트 상에서 이렇게 바뀌어야 정확하게 값을 낼 수 있다.

```typescript
function calcLengthV2(v: Vector3D) {
  return Math.abs(v.x) + Math.abs(v.y) + Math.abs(v.z)
}
```

이런 구조적 타이핑은 클래스에서도 재밌는 현상(?) 을 만들어 낸다.

```typescript
class MadMonster {
  tan: string
  constructor(tan: string) {
    this.tan = tan
  }
}

const hi1 = new MadMonster('hello') // 원래 내가 의도한 코드
const hi2: MadMonster = { tan: 'hello' } // ?!?!
```

왜 `hi2`가 `MadMonster`로 할당이 가능한걸까? `MadMonster`에는 `tan`이라는 string 속성이 있다. 추가로, `constructor` (`Object.prototype`에서 온) 를 가지고 있는데, 이는 `tan`이라는 인수를 받기 때문에 구조적으로 일치한다. 그렇다. 구조적 타이핑의 결과 인 것이다.

구조적 타이핑이 꼭 이렇게 거지같기만(?) 한 것은 아니다. 테스트 할 때는 유용하게 사용할 수 있다.

```typescript
interface Employee {
  name: string
  id: number
}

function getEmployee(db: DB): Employee[] {
  const rows = db.runQuery('SELECT name, id from EMPLOYEES')
  return rows.map((row) => ({ name: row[0], id: row[1] }))
}
```

DB를 테스트하는 코드를 만든다고 가정했을 때, 구조적 타이핑을 활용하면 아래와 같은 방법으로 테스트 할 수 있다.

```typescript
interface Employee {
  name: string
  id: number
}

interface DB {
  runQuery: (sql: string) => any[]
}

function getEmployee(db: DB): Employee[] {
  const rows = db.runQuery('SELECT name, id from EMPLOYEES')
  return rows.map((row) => ({ name: row[0], id: row[1] }))
}
```

`getEmployee`에는 `runQuery`가 존재하는 DB 어댑터를 넣어주면 된다. 위 코드는 프로덕션/테스트를 가지리 않고 잘 동작할 것이다. 구조적 타이핑을 활용하여, 굳이 실제 DB를 구현하지 않아도 테스트 코드를 짤 수 있게 되었다. 굳이 DB를 mocking할 필요가 없다. 추상화한 `DB` 덕분에, 우리의 로직을 테스트와 프로덕션 상에서 모두 안전하게 작동시킬 수 있다.

## 결론

- 자바스크립트는 덕타이핑 특성을 가지고 있고, 타입스크립트는 구조적 타이핑을 활용하여 이 모델을 구현해 냈다고 볼 수 있다.
- 인터페이스에 할당 되는 값은, 형식적으로 선언되어 있는 속성 이상의 속성을 추가로 가질 수 있다. 즉 열려 있다.
- 클래스 또한 구조적 타이핑을 따르고 있으므로 조심해야 한다.
- 구조적 타이핑은 유닛 테스트 시에 유용하다.
