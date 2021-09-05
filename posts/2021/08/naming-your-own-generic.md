---
title: '타입스크립트의 제네릭은 적절한 네이밍과 함께 사용하자'
tags:
  - typescript
  - javascript
published: true
date: 2021-08-27 23:27:41
description: '무지성 T, U, K 멈춰!'
---

타입스크립트의 제네릭은 언어가 제공하는 강력한 기능 중 하나다.제네릭을 사용하면, 타입스크립트에서 매우 유연하고 동적인 타입 생성을 가능하게 한다. 특히, 타입스크립트 최신버전에 들어서 string 리터럴 타입과 재귀 조건 타입이 생겨나면서, 재밌는 작업을 수행할 수 있다.

## (시작에 앞서) string literal type과 조건 타입

```typescript
type OnString = `on${string}`
const onClick: OnString = 'onClick'
// Type '"handleClick"' is not assignable to type '`on${string}`'
const handleClick: OnString = 'handleClick'
```

`infer` 키워드를 사용하면 더 재밌는 것도 할 수 있다. `infer`는 조건부 타입으로, `extends` 키워드 오른쪽에 사용할 수 있다.

```typescript
type Unpack<A> = A extends Array<infer E> ? E : A

type Test = Unpack<Apple[]>
// Apple
type Test = Unpack<Apple>
// Apple
```

만약 배열로 판단되면 배열에서 그 타입만을, 아니면 그 타입을 그대로 리턴하도록 했다.

아래 예제도 살펴보자.

```typescript
type ToCamel<S extends string> = S extends `${infer Head}_${infer Tail}`
  ? `${Head}${Capitalize<ToCamel<Tail>>}`
  : S

type T0 = ToCamel<'foo'> // "foo"
type T1 = ToCamel<'foo_bar'> // "fooBar"
type T2 = ToCamel<'foo_bar_baz'> // "fooBarBaz"
```

넘어온 제네릭을 재귀적으로 살펴보면서, `_`스타일의 snake case를 camelCase로 변경하였다.

## 제네릭과 String literal type, 그리고 조건에 따른 타입을 활용한 복잡한 예제

```typescript
type RouteParameters<T> = T extends `${string}/:${infer U}/${infer R}`
  ? { [P in U | keyof RouteParameters<`/${R}`>]: string }
  : T extends `${string}/:${infer U}`
  ? { [P in U]: string }
  : {}

type X = RouteParameters<'/api/:hello/:javascript/typescript/:world'>
// type X = {
//     hello: string;
//     javascript: string;
//     world: string;
// }
```

제네릭 타입을 선언하고, 또 그 안에서 제네릭타입을 선언했다. 이 작업은 `<>` 사이에서 이루어졌다. 그리고 이를 재귀적으로 처리함으로써, `:parameter`들을 객체 형태의 타입으로 처리할 수 있었다.

## 어려우니까, 처음으로 돌아와보자.

일단 앞선 예는 복잡하니, 다시 기초로 돌아와보자.

type에 제네릭을 넘겨주려면 우리는 보통 아래와 같이 처리한다.

```typescript
type Foo<T extends string> = ...
```

그리고 이 제네릭 타입은 기본값을 가질 수도 있다.

```typescript
type Foo<T extends string = "hello"> = ...
```

기본값을 사용하게 되면, 해당 제네릭의 사용처를 제한하는 것이기 때문에 순서가 중요해진다. 이는 자바스크립트의 함수와 비슷하다. 제네릭도 일종의 자바스크립트 함수의 인수의 성질과 비슷하다. 따라서 우리는 제네릭도 적절하게 네이밍을 해주는 것이 중요하다.

## 제네릭 타입 파라미터에 이름을 지어주기의 중요성

대부분의 제네릭 타입은 `T`로 시작하는 네이밍을 지어준다. 보통, `T`를 쓰게 되면, 그 이후에는 `U` `V` `W`를 쓰거나, `key`의 약자라는 의미로 `K`를 쓰곤 한다.

거의 모든 프로그래밍언어와 마찬가지로, 제네릭이라는 컨셉은 오래전 부터 존재해 왔다. 이런 제네릭의 시작은 `Ada` `ML`과 같은 70년대 언어에서 그 기원을 찾아볼 수 있다.

뭐, 그 때부터 `T`를 쓰는 것이 시작되었는지 어쨌는지 모르겠지만 어쨌거나 대부분이 제네릭을 선언할 때 `T`를 사용한다는 사실엔 변함이 없고, 우리또한 모두 그것에 익숙하다.

그러나 한개 일 때는 모르겠지만, 두 개부터는 조금씩 이해가 안된다. `Pick<K, U>`를 예를 들어보자. `K` `U`만을 봐서는 이게 무엇을 의미하는지 알 수가 없다. 아무도 저 두 알파벳만 봐서는, `U`가 객체타입이고, `K`가 그 `U`에 있는 키 중 하나는 사실을 알 수 없다.

따라서 우리는 공식 문서처럼, 아래와 같이 사용한다면 훨씬 이해하기 편하다.

```typescript
type Pick<Obj, Keys> = ...
```

https://www.typescriptlang.org/docs/handbook/utility-types.html#picktype-keys

## 제네릭을 네이밍하는 방법

타입은 일종의 문서화라고 볼수 있고, 타입 파라미터는, 일반적인 함수와 마찬가지로 이름을 통해서 부를 수 있다. 일반 함수와 마찬가지로, 제네릭 네이밍에 대한 가이드를 살펴보자.

1. 모든 타입 파라미터는, 타입과 마찬가지로 대문자로 시작한다.
2. 제네릭이 사용법이 완전히 명확하다면, 한단어를 사용한다. `RouteParameters`의 경우에는 `route`를 받는 것이 명확할 것이다.
3. 왠만하면 `T`를 쓰지말자. (이는 너무 제네릭하다) `route`와 마찬가지로 명확하게 나타낼 수 있는 단어로 사용하자.
4. 한 글자, 또는 짧은 단어, 그리고 약어를 사용해야 하는 경우는 거의 없다.
5. 빌트인 타입과 구별하기 위해서 prefix를 사용하자.
6. prefix를 제네릭네이밍에 사용하면 더 용도를 뚜렷이 구분할 수 있게 해준다. `Obj`보다는, `URLObj`가 낫다.
7. `infer`의 경우에도 제네릭 타입과 마찬가지의 룰이 적용된다.

위 규칙을 잘 염두해두고, `RouteParameters`를 정확한 제네릭 네이밍과 함께 다시 써보자.

```typescript
type RouteParameters<Route> =
  Route extends `${string}/:${infer Param}/${infer Rest}`
    ? { [Entry in Param | keyof RouteParameters<`/${Rest}`>]: string }
    : Route extends `${string}/:${infer Param}`
    ? { [Entry in Param]: string }
    : {}
```

확실히 이전보다 읽기가 편해졌다. (물론, 저 타입이 복잡하지 않다는 건 아니다 😑)

제네릭을 사용할때, `T` `K` `U`와 같은 약어보다는 적절한 네이밍을 사용한다면, 보다 다른 개발자들과 프로그래밍을 하는데 수월해질 것이다.
