---
title: '알아두면 유용한 타입스크립트 팁'
tags:
  - typescript
published: true
date: 2022-05-08 04:12:01
description: '"타입"스크립트니까 타입을 잘 할줄 알아야 합니다.'
---

### 제네릭 활용하기

테이블 컴포넌트가 있고, 여기에 props를 할당해서 그린다고 생각해보면, 보통은 이런 코드가 나올 po`것이다.

```tsx
import React from 'react'

interface Props {
  items: Array<{ id: string }>
  renderItem: (item: { id: string }) => React.ReactNode
}

export const Table = (props: Props) => {
  return null
}

export const Component = () => {
  return (
    <Table
      items={[{ id: '1' }]}
      renderItem={(item) => {
        return null
      }}
    />
  )
}
```

하지만 이런 구조는 `{id: string}` 으로 고정되어 있어,여러가지 종류의 props를 그리기에는 무리가 있다. 이럴 때 사용하면 좋은 것이 Generic이다. Generic은 타입, 인터페이스 등에서 외부에서 정의된, 공통의 속성을 사용하고 싶을 때 유용하다.

```tsx
import React from 'react'

interface Props<TItem> {
  items: Array<TItem>
  renderItem: (item: TItem) => React.ReactNode
}

export const Table = <TItem,>(props: Props<TItem>) => {
  return null
}

export const Component = () => {
  return (
    <>
      <Table
        items={[{ id: '1' }]}
        // item이 {id: "1"}로 추론되는 것을 볼 수 있다.
        renderItem={(item) => {
          return null
        }}
      />
      <Table
        items={[{ id: '1', name: 'yceffort' }]}
        // 서로 다른 props가 와도 문제 없다.
        renderItem={(item) => {
          return null
        }}
      />
    </>
  )
}
```

이렇게 props와 interface내에서 사용하는 것 뿐 만 아니라, 타입을 아직 알 수 없는 객체 등을 다룰 때도 유용하다.

```ts
export const getDeepValue = (obj: any, firstKey: string, secondKey: string) => {
  return obj[firstKey][secondKey]
}

const obj = {
  foo: {
    a: true,
    b: 2,
  },
  bar: {
    c: '12',
    d: 18,
  },
}

const value = getDeepValue(obj, 'foo', 'a')

// value any
```

어떠한 객체의 키로 특정한 값을 가져온다고 생각해보자. 객체의 형태를 당장 알 수 없으므로 `any`를 두고, 키는 string으로 두었다. `any`를 쓰는 것은 타입스크립트에서 최대한 자제해야하는 행위다.

이것을 해결하기 위해, 마찬가지로 Generic을 사용할 수 있다.

```typescript
export const getDeepValue = <
  TObj,
  TFirstKeyOfObj extends keyof TObj,
  TSecondKeyOfObj extends keyof TObj[TFirstKeyOfObj],
>(
  obj: TObj,
  firstKey: TFirstKeyOfObj,
  secondKey: TSecondKeyOfObj,
) => {
  return obj[firstKey][secondKey]
}

const obj = {
  foo: {
    a: true,
    b: 2,
  },
  bar: {
    c: '12',
    d: 18,
  },
}

const value = getDeepValue(obj, 'foo', 'a')
// value number
```

`extends`을 활용하면 기존의 제네릭을 상속 받아 또 다른 제네릭을 만들 수 있다. `TFirstKeyOfObj`는 `keyof TObj`, 즉 `TObj`의 키를 상속받아 만들었고, `TSecondKeyOfObj`는 `TObj[TFirstKeyOfObj]`의 키를 상속받아 만든 제네릭이다. 처음보기엔 무언가 복잡해보이지만, 차분하게 읽어보면 별거 없다는 것을 알 수 있다.

### 조건부 타입

타입도 다른 변수들이나 표현과 마찬가지로 조건부로 만들 수 있다.

```typescript
type Animal = {
  name: string
}

type Human = {
  firstName: string
  lastName: string
}

type GetRequiredInformation<TType> = any

export type RequiredInformationForAnimal = GetRequiredInformation<Animal>

export type RequiredInformationForHuman = GetRequiredInformation<Human>
```

`GetRequiredInformation`에서 받은 제네릭 `TTYpe`이 `Animal`인지, `Human`인지 확인하여 새로운 타입을 extends할 수 있는 타입을 만들어보자.

```typescript
type GetRequiredInformation<TType> = TType extends Animal
  ? { age: number }
  : { salary: number }
```

`extends`를 사용하면 단순히 상속하는 것 뿐만 아니라, 마치 조건문으로 사용해서 상속할 수 있는지 여부도 확인할 수 있다. 이에 따라 타입별로 원하는 추가 타입을 선언해 줄 수 있다.

추가로, `GetRequiredInformation`에 `Animal` `Human`외에 다른 것이 오는 것을 막고 싶다면, 아래와 같이 `never`를 사용하면 된다.

```typescript
type GetRequiredInformation<TType> = TType extends Animal
  ? { age: number }
  : TType extends Human
  ? { salary: number }
  : never
```

[과거 글](/2022/03/understanding-typescript-never#왜-never가-필요한가)에서 이야기 했던 것처럼, 그 어떤 것도 사용할 수 없는 불가능한 타입, bottom type을 만들고 싶을 때 `never`를 사용한다.

이러한 방식을 조금더 응용하면, 내가 타입스크립트 컴파일러에 사용할 수 있는 에러도 만들 수 있다.

```typescript
export function deepEqualCompare(a: any, b: any) {
  if (Array.isArray(a) || Array.isArray(b)) {
    throw new Error('배열은 비교할 수 없습니다.')
  }
  return a === b
}
```

```typescript
export function deepEqualCompare<Arg>(a: Arg extends any ? "배열은 비교할 수 없습니다", b: Arg) {
  if (Array.isArray(a) || Array.isArray(b)) {
    throw new Error("배열은 비교할 수 없습니다.")
  }
  return a === b
}

deepEqualCompare([1, 2, 3], [1]) // Argument of type 'number[]' is not assignable to parameter of type '"배열은 비교할 수 없습니다."'.(2345)
```

그러나, 이러한 코드가 동작해버리는 참사가 발생버리기 때문에, `never`를 쓰는 것이 좋다.

```typescript
deepEqualCompare('배열은 비교할 수 없습니다.', '배열은 비교할 수 없습니다.') // ????
// 물론 코드가 잘못된 것은 아니지만, 우리가 원하는 바는 이게 아닐 것이다.
```

### 타입스크립트의 타입을 공부할 때 도움이 되는 것들

- [ts-belt](https://github.com/millsp/ts-toolbelt): 타입스크립트에서 유용하게 사용할 수 있는 다양탄 유틸리티 라이브러리를 제공한다. 찾아보면 별에 별 유틸리티 타입들을 다 제공하는데, 이를 어떻게 만들었을지 상상해 보는 재미가 있다.
- [zod](https://github.com/colinhacks/zod): [joi](https://github.com/sideway/jo)의 타입스크립트 버전이라고 보면된다. 타입스크립트의 스키마를 체크하는데 도와주는 라이브러리다.
- [type-challenges](https://github.com/type-challenges/type-challenges): 알고리즘에 백준이 있다면, 타입스크립트에는 `type-challenge`가 있다. 문제를 하나씩 풀어나가는 재미가 있다. `hard`까지는 그럭저럭 꾸역꾸역할 수 있었는데, `extreme`부터는 약간 그냥 테스트를 위한 테스트 같은 느낌이다. (내가 못풀어서 그런 걸수도 있다.) 실무에서 개발하는 타입스크립트 개발자라면, `medium`까지만 풀어도 충분할 것 같다.
- [TypeScript Error Translator](https://marketplace.visualstudio.com/items?itemName=mattpocock.ts-error-translator): 타입스크립트를 처음 접했을 때 많이 헤매는 것이 잘못된 타입으로 인한 에러인데, 이 에러를 읽기가 처음에는 약간 버거운 경우도 있다. 이러한 불친절한 에러를 사람이 읽기 쉽게 번역해주는 extension이다.
