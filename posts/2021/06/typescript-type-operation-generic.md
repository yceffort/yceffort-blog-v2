---
title: '타입스크립트의 타입과 제네릭 적극 활용하기'
tags:
  - typescript
published: true
date: 2021-06-15 18:23:35
description: 'interface를 더 좋아하지만 type이 더 간지남'
---

소프트웨어 개발 원칙 중의 하나인 [DRY, don't repeat yourself](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) 는 너무나도 유명해서 별로 설명할게 없긴한다. type을 잘 사용하면, 조금 더 효과적으로 소프트웨러를 설계할 수 있다.

## 타입 확장하기

```typescript
interface Person {
  name: string
  age: number
}

// don't
interface PersonWithBirthday {
  name: string
  age: number
  birth: Date
}

// do
interface PersonWithBirth extends {
  birth: Date
}
```

`type`을 사용한다면, `&`으로도 가능하다.

```typescript
type PersonWithBirth = Person & {birth: Date}
```

## 타입 좁히기

이전 예제와 반대의 예제를 들어보자. 이번엔 큰 타입에서 작은 타입의 subset이 필요한 경우다.

```typescript
interface User {
  userId: string
  name: string
  age: number
  email: string
}

interface SelectedUser {
  userId: string
  email: string
}
```

이 역시 중복이 발생하므로, 좋지 못한 방법이다.

```typescript
type SelectedUser = {
  userId: User['userId']
  email: User['email']
}

interface SelectedUser {
  userId: User['userId']
  email: User['email']
}
```

오오 놀랍다. 타입의 값을 마치 객체에서 값을 꺼내온 것마냥 썼다. 이렇게 해두면, `User`의 `userId`가 number로 바뀌게되도, 자동으로 그 타입도 따라가게 될 것이다.

하지만 이 역시도 번거로운 점이 있다. key조차도 나는 똑같이 할 건데, 굳이 키를 다시 선언할 필요가 있을까?

```typescript
type SelectedUser = {
  [k in 'userId' | 'email']: User[k]
}
```

![type-operations1](./images/type-operation1.png)

오옹 신기하다. 하지만 이미 우리는 이것보다 더 편한 방법을 알고 있다.

```typescript
type SelectedUser = Pick<User, 'userId' | 'email'>
```

[Pick](https://www.typescriptlang.org/docs/handbook/utility-types.html#picktype-keys) 을 쓰면 간단하게 해결할 수 있다.

## 공통타입 추출하기

```typescript
interface Save {
  action: 'save'
  body: string
  id: string
}

interface Load {
  action: 'load'
  body: string
  id: string
}

type Action = Save | Load
type ActionType = 'save' | 'load' // Repeat!
```

두 타입은 `action`의 값 외엔 모든게 똑같은데, 여기서 `action`을 또다른 타입으로 추려내기 위해서 `'save' | 'load'`를 썼다. 이것도 마찬가지로, 아래 처럼 바꿀 수 있다.

```typescript
type ActionType = Action['action'] // type ActionType = "save" | "load"
```

## 타입 옵셔널하게 사용하기

객체의 모든 키가 옵셔널 하다면 어떻게 해야할까? 일일히 모두 물음표를 달아야할까?

```typescript
interface Person {
  age?: number
  name?: string
  email?: string
  gender?: string
}
```

그렇지 않다. `Partial<Person>`을 사용하면, 안에 있는 모든 키를 옵셔널하게 바꿔준다. 이는 옵셔널한 값을 받는 상황 (api로 값을 업데이트 한다던지)에 매우 유용하게 쓸 수 있다.

```typescript
function update(options: Partial<Person>) {
  // .. 적당히 값을 받아서 처리한다.
}
```

## 값으로 부터 타입을 추출하기

기본적으로 개발 흐름은 타입을 선언하고 값을 쓰는 형태로 가지만, 반대인 경우가 있을 수 있다. 이 경우에는 아래와 같이 하면 된다.

```typescript
const INIT_VALUES = {
  width: 640,
  height: 480,
  price: 150_000,
  name: 'monitor',
}

type Options = typeof INIT_VALUES
// 위 타입은 아래와 같다.
// type Options = {
//     width: number;
//     height: number;
//     price: number;
//     name: string;
// }
```

한가지 조심해야 할 것은, `typeof`의 사용이다. 아래 두 코드는 엄연히 다르다.

```typescript
const t = typeof INIT_VALUES // "object"
type o = typeof options
// type Options = {
//     width: number;
//     height: number;
//     price: number;
//     name: string;
// }
```

값을 선언한 코드에 `typeof`를 때리면 자바스크립트 런타임의 [typeof](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Operators/typeof)를 실행하게 된다. 값을 가져오고 싶은건지, type을 가져오고 싶은건지 확실히 해야 한다. 조건문 같은 곳에 `typeof`를 둔다면 자바스크립트 런타임의 `typeof`를 실행한다는 것을 명심하자. `type`에 `typeof`를 쓰면, 타입스크립트만 알아듣는다.

이렇게 쓰긴했지만, 어디까지나, 타입을 먼저쓰고 값을 쓰는 경우가 훨씬 더 일반적이고 정확하다.

## 함수의 결과를 타이핑 하기

함수의 결과를 타이핑하고 싶다면, [ReturnType](https://www.typescriptlang.org/docs/handbook/utility-types.html#returntypetype)과 제네릭을 활용하면 된다.

```typescript
function getUserInfo(userId: string) {
  // ...

  return {
    userId,
    name: 'hello',
    age: (Math.random() * 100) / 100,
    email: 'random@email.com',
  }
}

type userInfo = ReturnType<typeof getUserInfo>
// 위와 같다.
// type userInfo = {
//     userId: string;
//     name: string;
//     age: number;
//     email: string;
// }
```

이러한 패턴은 라이브러리에서 함수의 리턴을 emit할 때 많이 쓴다. 주의할 점은, 제네릭에 `getUserInfo`가 아니고 `typeof gerUserInfo`가 들어갔다는 점이다. `ReturnType`는 타입을 제네릭으로 받아야 한다.

```typescript
type t = typeof getUserInfo

// 위와 같다.
// type t = (userId: string) => {
//     userId: string;
//     name: string;
//     age: number;
//     email: string;
// }
```

이로써 함수만 바꾸더라도, 타입까지 자동으로 바뀌어서 [single source of truth](https://ko.wikipedia.org/wiki/%EB%8B%A8%EC%9D%BC_%EC%A7%84%EC%8B%A4_%EA%B3%B5%EA%B8%89%EC%9B%90)를 지킬 수 있었다.

## 제네릭으로 파라미터 제한하기

제네릭 타입은 함수를 위한 타입과 같다. 그리고 함수가 DRY 원칙을 지키기 위한 수단인 것을 감안했을때, 제네릭도 마찬가지로 타입의 DRY를 위한 필수 요소라고 볼 수 있다.

```typescript
interface Name {
  first: string
  last: string
}

type PairProgrammer<T extends Name> = [T, T]

const day1: PairProgrammer<Name> = [
  {last: 'KIM', first: 'YONGCHAN'},
  {last: 'LEE', first: 'JAEYONG'},
]

const day2: PairProgrammer<Name> = [
  {last: 'KIM'}, // Property 'first' is missing in type '{ last: string; }' but required in type 'Name'.
  {last: 'LEE'}, // Property 'first' is missing in type '{ last: string; }' but required in type 'Name'.
]
```

> 한가지 조금 개인적으로 아쉬운것은, 제네릭 파라미터를 생략할 수 없다는 점이다. `PairProgrammer<Name>` 대신 `PairProgrammer`를 쓸 수 없다는 점이다.

## Pick 구현해보기

이렇게 `extends` 키워드까지 활용한다면, `Pick`과 동일한 타입을 추출하는 나만의 `Pick`을 만들 수 있다.

```typescript
type MyPick<T, K extends keyof T> = {
  [k in K]: T[k]
}
```
