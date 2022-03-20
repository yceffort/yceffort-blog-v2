---
title: '타입스크립트의 Omit은 어떻게 동작할까? Exclude, Pick 부터 알아보기'
tags:
  - typescript
published: true
date: 2022-03-16 22:08:36
description: '헬퍼 타입도 잘 알고 써야 도움이 된다'
---

## Table of Contents

## exclude

[exclude](https://www.typescriptlang.org/docs/handbook/utility-types.html#excludeuniontype-excludedmembers)는 여러개의 타입이 함께 존재하는 유니언 타입에서 특정 타입을 제거하는 유틸리티 타입이다. `exclude`로 제거할 수 있는 것은 하나의 타입 부터 유니언 까지 가능하다.

```typescript
type T0 = Exclude<'a' | 'b' | 'c', 'a'>
// type T0 = "b" | "c"
type T1 = Exclude<'a' | 'b' | 'c', 'a' | 'b'>
// type T1 = "c"
type T2 = Exclude<string | number | (() => void), Function>
// type T2 = string | number
```

[exclude의 동작방식](https://github.com/microsoft/TypeScript/blob/546a87fa31086d3323ba4843a634863debb75781/lib/lib.es5.d.ts#L1503-L1506)을 보면 다음과 같이 확인할 수 있다.

```typescript
/**
 * Exclude from T those types that are assignable to U
 */
type Exclude<T, U> = T extends U ? never : T
```

### extends

제네릭에서 사용되는 `T extends U`라는 키워드는 **T가 U라는 타입인지** 를 의미한다.

즉, 위 예시를 해석하면 다음과 같다.

> `T`가 `U`의 타입 이라면, `never` (빈 타입)을, 그렇지 않다면 `T` 그자체, 즉 원래대로 돌려준다

## pick

[pick](https://www.typescriptlang.org/docs/handbook/utility-types.html#picktype-keys)은 객체 타입에서, 넘겨받은 키에 해당하는 키만 리턴하는 새로운 객체 타입을 만들어준다.

```typescript
interface Todo {
  title: string
  description: string
  completed: boolean
}

type TodoPreview = Pick<Todo, 'title' | 'completed'>

const todo: TodoPreview = {
  title: 'Clean room',
  completed: false,
}
```

pick이 작동하기 위해서는, 먼저 객체에서 키를 뽑아서 해당 키를 제외해야 하므로, 객체 타입에서 키를 뽑는 법 부터 알아야 한다.

```typescript
keyof Todo // "title" | "description" | "completed" | "createdAt"
```

그 다음, 이 키에 해당 하는 객체 타입의 값만 뽑아 오면 될 것이다.

```typescript
type Pick<T, Key extends keyof T> = {
  [NewKey in key]: T[key]
}
```

작동방식을 확인하면 거의 유사하다는 것을 알 수 있다.

> [타입스크립트 원본 코드 확인해보기](https://github.com/microsoft/TypeScript/blob/546a87fa31086d3323ba4843a634863debb75781/lib/lib.es5.d.ts#L1489-L1494)

## Omit

[omit](https://www.typescriptlang.org/docs/handbook/utility-types.html#omittype-keys) 은 객체 타입 (interface 등)에서 특정 키를 기준으로 생략하여 타입을 내려주는 유틸리티 타입이다.

```typescript
interface Todo {
  title: string
  description: string
  completed: boolean
  createdAt: number
}

// description을 제외
type TodoPreview = Omit<Todo, 'description'>

const todo: TodoPreview = {
  title: 'Clean room',
  completed: false,
  createdAt: 1615544252770,
}
```

마찬가지로 키를 먼저 뽑아온다.

```typescript
type TodoKeys = keyof Todo // "title" | "description" | "completed" | "createdAt"
```

그리고 이번에는 해당하는 값을 가져오는 것이 아니고, 제거를 해야한다. 여기서 부터 조금씩 복잡해지는데, 하나씩 해보자.

먼저 앞서 사용했던 `Pick`과 `Exclude`를 활용하여, `TODO`에서 `title`만 제거해보자.

```typescript
type TodoWithoutTitle = Pick<Todo, Exclude<keyof Todo, 'title'>>
// type TodoWithoutTitle = {
//     description: string;
//     completed: boolean;
//     createdAt: number;
// }
```

이를 깔끔하게 제네릭으로 정리하면 다음과 같다.

```typescript
type Omit<T, K> = Pick<T, Exclude<keyof T, K>>
```

객체의 키로 `string`, `number`, `symbol`만 가능하기 때문에, 조금 아래와 같이 추가할 수도 있다.

```typescript
type Omit<T, K extends keyof string | number | symbol> = Pick<
  T,
  Exclude<keyof T, K>
>
```

> https://github.com/microsoft/TypeScript/blob/546a87fa31086d3323ba4843a634863debb75781/lib/lib.es5.d.ts#L1513-L1516
> 뭔가 저건 과하다고 생각한건지 `any`로 퉁쳤다.

### Omit 과정 다시한번 살펴보기

```typescript
interface Todo {
  title: string
  description: string
  completed: boolean
  createdAt: number
}

type TodoWithoutTitle = Omit<Todo, 'title'>

type TodoWithoutTitle = Pick<Todo, Exclude<keyof Todo, 'title'>>

type TodoWithoutTitle = Pick<
  Todo,
  Exclude<'title' | ' description' | 'completed' | 'createdAt', 'title'>
>

type TodoWithoutTitle = Pick<
  Todo,
  | ('title' extends 'title' ? never : 'title')
  | ('description' extends 'title' ? never : 'description')
  | ('completed' extends 'title' ? never : 'completed')
  | ('createdAt' extends 'title' ? never : 'createdAt')
>

type TodoWithoutTitle = Pick<
  Todo,
  never | 'description' | 'completed' | 'createdAt'
>

type TodoWithoutTitle = {
  [Key in 'description' | 'completed' | 'createdAt']: User[Key]
}

type TodoWithoutTitle = {
  description: Todo['description']
  completed: Todo['completed']
  createdAt: Todo['createdAt']
}

type TodoWithoutTitle = {
  description: string
  completed: boolean
  createdAt: number
}
```
