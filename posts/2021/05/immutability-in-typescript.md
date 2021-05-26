---
title: 'Typescript의 Immutability'
tags:
  - javascript
  - typescript
published: true
date: 2021-05-26 19:19:20
description: '저는 사실 Immutability에 안 좋은 추억이 있습니다'
---

## Table of Contents

## 불변성

Immutability(이하 불변성)이란, 초기에 할당 한 이후에 더이상 상태가 변하지 않는 객체를 의미한다. 프로젝트 내의 모든 객체에 이 불변성을 적용하면, 가독성 향상, 코드에 대한 이해도 증가, 스레드의 안정성 등을 확보할 수 있다.

## 불변성을 논하기에 앞서

소프트웨어 아키텍쳐의 전통적인 객체지향 접근법에서, 모든 클래스 인스턴스는 특정 인스턴스에만 연결된 상태를 가질 수 있다. 상태의 초기화는 클래스 생성자에서 발생하며, 클래스 메소드를 호출 할 때 해당 상태에 대한 변경을 적용할 수 있다. 아무리 이러한 상태 값의 변화가 체계적으로 정리되어 있다고 하더라도, 클래스의 상태가 변할 수 있다는 측면은 클래스에 의존하는 코드의 구조에 크게 영향을 미친다.

이러한 상태 값 변이에 따른 단점을 이해하기에 앞서, 두개의 타입의 함수를 먼저 소개하고자한다.

- synchronous(동기): 현태 실행 컨텍스트에서 즉시 실행되어 리턴한다
- asynchronous(비동기) 현재 실행컨텍스트에서 대기하며, 다른 실행 컨텍스트에서 실행되어 값을 가져온다.

코드 실행환경에 따라, 클로져가 비동기 함수가 끝나는 것을 기다리지 않고 호출하며, 그 함수가 상태값을 바꾼다면, 클로져 내부에서의 함수 호출에 대해 신뢰성을 가질 수가 없다. 동기호출에는 이러한 부수효과가 없다. 그러나, 특정 상태를 기반으로 한 호출은, 동기 함수가 내부에서 상태를 변경해버린다면 모두 무효가 되버린다.

두번째로, 불변함수와 변이함수를 구분해야 한다. 불변함수를 호출하는 것은 부수효과를 만들지 않으며, 결과를 리턴한다는 단하나의 효과(effect) 만 가진다. 그러나 변이 함수는 내부에서 상태를 바꿀 수 있는 가능성이 있고, 이는 부수효과를 불러 이르키게 된다.

세번째로, 객체지향 아키텍쳐에서 개발자는 어떤 클래스에서든 getter와 setter를 정의할 수 있다. `getter`는 단순히 상태값을 리턴하는 행위로, 불변함수로 동작한다. 반면에 `setter`는 상태에 값을 부여하므로 변이를 이르키게 된다.

마지막으로, 이러한 모든 개념을 다 통틀어서, 개발자는 코드의 다양한 위치와 다양한 실행 컨텍스트 (스레드, 콜백 모두)에서 변이되는 상태가 만드는 잠재적인 복잡성을 볼 수 있어야 한다. 개발자들이 미래의 복잡함으로 부터 상태를 보호하는 지침을 따르지 않는다면 코드에 대한 디버깅이나 추론이 복잡한 시스템에서 골칫거리로 작용할 수 있다. 이것은 불변성을 적용하는 근본적인 이유, 프로젝트의 원활한 유지보수를 지원하기 위한 의욕을 떨어 뜨릴 수 있다.

## 자바스크립트의 불변성

자바스크립트는 멀티 패러다임 언어이므로, 개발자에게 함수형 사고를 강요하지 않으면서도 함수형 프로그래밍의 측면을 구현할 수 있다. 언어 자체적으로는 앞서 언급한 불변성을 지원하는데, 이를 위해 몇가지 문자열을 붙여야 한다. 불변성을 적용하기 위해서는, 해당 변수나 객체에 명확하게 표현하는 작업을 거쳐야 한다.

### Primitive, wrapper type

자바스크립트에는 몇가지 원시타입 (`boolean` `number` `bigint` `string` `symbol` `null` `undefined`) 이 있다. 이들 모두 메소드가 없다. 따라서 불변의 방식으로 작동하며, 이는 함수로 이들을 전달하는 것이 부수효과를 만들지 않는 다는 것을 의미한다. 대부분의 자바스크립트 개발자가 5가지 wrapper (object)에 대해 잘 알지 못한다. (`Boolean` `Number` `BigInt` `String` `Symbol`) 이는 언어가 원시타입과 래퍼 객체에 상호교환(interchangeable)을 가능하게 해주기 때문이다.

모든 객체 (원시타입이 아닌것, 함수 포함)는 메소드를 포함하므로 값의 변화가 있을 수 있다. 객체는, 그 정의에 따라 프로토타입을 가지고 있으며, 이러한 프로토타입을 바꾸는 것은 객체의 동작을 바꿀 수도 있다.

### 변수 선언

개발자들은 변수를 선언하기 위해서 `let` `const`를 사용해야 한다. 개인적으로 개발자들에게 `let`의 사용을 자제하도록 하게 하는 편이다. `let`은 스코프 내에서 몇번이고 재할당이 이루어져서 문제가 될 수 있기 때문이다. 경험이 많은 개발자들은, 변수의 재 할당을 여러 함수에 나누고, 이를 별도로 리턴하도록 리팩토링 한다.

### 객체 동결

자바스크립트는 객체를 `얕게` 불변하게 만들어주는 함수를 가지고 있다. `얕게`라는 말에 주목하자. 중첩된 객체에서는 이러한 특성이 적용되지 않는다. `Object.freeze` 함수는 말그대로 객체를 동결시켜 주며, 이를 얕게 불변하게 만들어준다.

```javascript
const obj = {
  a: {
    b: 1,
  },
}
Object.freeze(obj)
obj.a = null // 안됨!
obj.b = true // 안됨!
obj.a.b = 2 // 됨?!!
```

`Object.freeze`는 객체의 런타임 수준에서 직접적으로 제한을 건다. 객체를 동결하면, `writable`과 `configurable`가 false로 바뀐다. [이-글](/2020/10/object-freeze-seal-preventExtensions)을 참고하자. 아무튼, 객체를 정말로 깊게 불변하게 만들기 위해서는, 별도로 써드 파티 라이브러리를 사용하거나 스스로 구현해야 한다.

### 함수 속성

함수 또한 객체라는 점에서, 모든 함수 속성은 잠재적으로 변할 수 있는 가능성이 있다. 객체를 함수에 전달하는 것은 메모리 참조에 의해서 일어나고, 전달된 원시타입은 값에 의해 일어난다고 보자. 흥미롭게도(혹은 귀찮게도) 자바스크립트는 함수의 arguments를 재할당 할 수 있도록 해주는데, 이는 클로져 규칙에 따라 함수 범위 밖에서는 영향을 미치지 않는다.

## 타입스크립트의 불변성

타입스크립트는 불변성과 관련되어 놀라운 기능을 제공한다. compile-time 타입 시스템을 활용하여, 엔드 유저에게 전달되는 코드의 양을 줄이고, 런타임 레벨에 대한 제한을 명시할수도 있게 해준다. 타입스크립트는 불변성을 달성하기 위해, `readonly` 속성의 개념을 도입한다.

### `readonly`

`readonly`는 타입과 인터페이스, 클래스 속성과 생성자에 사용할 수 있다.

- `readonly`: https://www.typescriptlang.org/docs/handbook/typescript-in-5-minutes-func.html#readonly-and-const
- `- readonly`

클래스 레벨에서 상수를 정의하면, 객체 오직 단한번, 객체 생성중에만 할당된 변경 불가능한 속성을 정의할 수 있다.

### `readonly`의 사용

타입스크립트는 다음과 같은 방법으로 얕은 불변성을 제공한다.

- `Readonly<T>`: https://www.typescriptlang.org/docs/handbook/utility-types.html#readonlytype
- `ReadonlyArray<T>`: https://www.typescriptlang.org/docs/handbook/2/objects.html#the-readonlyarray-type
- `ReadonlySet<T>`
- `ReadonlyMap<K, V>`

`Object.freeze<T>`가 결과로 `Readonly<T>`를 리턴한다는 사실을 주목하자. 이는 자바스크립트의 `freeze`와 `readonly`사이의 일종의 연결고리다.

### 깊은 불변성

언급했다시피, `readonly`는 깊은 불변성을 제공하지 않으므로 개발자가 써드파티 라이브러리를 쓰거나 직접 구현해야 한다. [ts-essentials](https://github.com/krzkaczor/ts-essentials)에서 제공하는 [DeepReadonly](https://github.com/krzkaczor/ts-essentials#Deep-wrapper-types)를 사용해보자.

## 불변성을 위한 가이드

프로젝트 내부의 객체에 불변성을 적용하는 것은 매우 중요하므로, 프로젝트의 핵심 아키텍쳐 원리에 통합되어야 한다. 물론 소프트웨어 전문가들이 어느정도까지 이러한 패턴을 적용해야 하는지는 많은 논쟁이 있었지만, 대체로 다음과 같은 타입스크립트 개발을 위한 지침을 추천한다.

- `const`로 변수 선언하기
- 컴파일 타입 불변성 사용
- 클래스 인스턴스의 사용을 함수로만 제한
- `Readonly<T>`를 사용하여 불변의 타입으로 선언
- 불변의 유형에서 적절한 서브타입을 추출하여 변이 타입을 사용하되, 이에 대한 사용을 한정 지을 것
- 얕은 불변성은 깊은 불변성을 효과적으로 적용하기 위하여 모든 기능을 제공할 것 (= 얕은 불변성 만으로 깊은 불변성을 달성할 수 있어야 한다)
- 함수는 불변의 파라미터를 받아야 한다.
- 함수는 불변값을 리턴해야 한다.
- 가장 최선의 함수는 순수함수다.

프로젝트 전체 영역에 불변 타입을 선언하면, 개발자들은 개발시에 불변성을 먼저 염두해둘 수 있으며, 필요한 경우 이러한 불변타입의 도움을 얻어 보다 복잡한 타입을 구성할 수도 있다. 시스템의 대부분의 함수는 변경할 수 없는 구조를 수용하고, 이를 만들어 내야 한다. 아래 코드를 참조하자.

```typescript
type Writable<K extends string | number | symbol, V> = {
  -readonly [P in K]: V
}

type ExtractFromReadonlySet<T> = T extends ReadonlySet<infer R> ? R : never
type ExtractFromReadonlyArray<T> = T extends ReadonlyArray<infer R> ? R : never
type ExtractFromReadonlyMap<T> = T extends ReadonlyMap<infer K, infer V>
  ? [K, V]
  : never
```

```typescript
// 얕은 수준의 불변성 타입을 이용하여, 깊은 불변성을 강제한다.
type User = Readonly<{
  id: string
  groups: ReadonlySet<
    Readonly<{
      id: string
      public: boolean
    }>
  >
}>

// ReadonlySet로 부터 타입을 추출한다.
type ExtractFromReadonlySet<T> = T extends ReadonlySet<infer R> ? R : never

// 함수는 불변의 인자를 받는다
// 함수가 불변 타입의 값을 리턴한다.
// 순수함수
const getUserPublicGroupIds = (user: User): User['groups'] => {
  // 변수는 const로 선언되어야 한다.
  const set = new Set<ExtractFromReadonlySet<User['groups']>>()

  Array.from(user.groups).forEach((group) => {
    if (group.public) {
      set.add(group)
    }
  })

  return set
}
```

## 리팩토링

새로운 프로젝트의 경우엔 상관없지만, 오래된 코드베이스의 리팩토링은 숙련된 개발자에게도 문제가 될 수 있으므로 소프트웨어 전문가들이 항상 프로젝트 전반의 변경을 사전에 계획할 필요가 있다.

타입스크립트 프로젝트에 불변성을 강제하는 작업을 하기 위해, 아래와 같은 목록을 작성해보았다.

- 기존 함수의 반환 값을 불변으로 변경하고, 그에 따른 문제 해결
- 기존 함수의 파라미터를 불변으로 변경하고, 그에 따른 문제 해결
- 얕은 복사, 또는 깊은 복사를 통해 불변의 구조로 변경
- 코드 리팩토링 중에 컴파일러가 잠재적으로 모든 버그를 보여줄 것이라 기대하지 말 것
- 광범위한 테스트에 의존

리턴 타입을 불변하게 만드는 것은 리팩토링의 첫번째 단계다. 함수가 불변한 값으로 리턴하기 위해, 아래와 같은 작업을 수행해야 한다.

- 컴파일러를 만족시키기 위해 얕은 복사를 하거나
- 코드를 다른 방향으로 수정

불변값을 얕은 수준의 불변값으로 변경하는 것은 자바스크립트의 표준 라이브러리에 이미 정의된 방법으로 구현할 수 있다.

- 객체: `Object.assign({}, obj)` `{...obj}`
- 배열: `arr.slice()`, `[...arr]`

리팩토링전

```typescript
type User = {
  id: string
  groupIds: string[]
}

const mutableAppendGroupsToUser = (groupIds: string[], user: User): User => {
  user.groupIds = Array.from(new Set([...user.groupIds, ...groups]))

  return user
}
```

리팩토링후

```typescript
type ReadonlyUser = Readonly<{
  id: string
  groupIds: ReadonlyArray<string>
}>

// 이제 함수는 오직 불변 타입만 받는다.
const immutableAppendGroupsToUser = (
  groupIds: ReadonlyArray<string>,
  user: ReadonlyUser,
): ReadonlyUser => {
  // 더이상 `user.groupIds`를 직접 수정하지 않는다.
  const newGroupIds = Array.from(new Set([...user.groupIds, ...groupIds]))

  // 함수가 완전히 새로운 객체를 리턴한다.
  return Object.assign({}, user, { groupIds: newGroupIds })
}
```

개발자들이 큰 객체에 대한 깊은 복사를 할 때 성능상의 문제를 조심해야 한다. 또한 불변성을 도입하면, 컴파일러의 도움이 있던 없던 간에 이전부터 있었던 버그가 수면위로 떠오르는 일이 나타날 수 있다. 어떤 프로젝트던지, 합리적인 테스트를 거쳐야만 최종 사용자에게 훌륭한 경험을 보장해준다.

> https://levelup.gitconnected.com/the-complete-guide-to-immutability-in-typescript-99154f859fdb
