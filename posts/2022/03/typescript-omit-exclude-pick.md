---
title: '타입스크립트에서 omit, exclude, pick이 동작하는 방식'
tags:
  - typescript
published: true
date: 2022-03-16 22:08:36
description: '타입스크립트 시리즈는 일단 여기까지'
---


## Table of Contents

### exclude

[exclude](https://www.typescriptlang.org/docs/handbook/utility-types.html#excludeuniontype-excludedmembers)는 여러개의 타입이 함께 존재하는 유니언 타입에서 특정 타입을 제거하는 유틸리티 타입이다. `exclude`로 제거할 수 있는 것은 하나의 타입 부터 유니언 까지 가능하다.

```typescript
type T0 = Exclude<"a" | "b" | "c", "a">;
// type T0 = "b" | "c"
type T1 = Exclude<"a" | "b" | "c", "a" | "b">;
// type T1 = "c"
type T2 = Exclude<string | number | (() => void), Function>;
// type T2 = string | number
```

### pick

### Omit

[omit](https://www.typescriptlang.org/docs/handbook/utility-types.html#omittype-keys) 은 객체 타입 (interface 등)에서 특정 키를 기준으로 생략하여 타입을 내려주는 유틸리티 타입이다.

```typescript
interface Todo {
  title: string;
  description: string;
  completed: boolean;
  createdAt: number;
}

// description을 제외
type TodoPreview = Omit<Todo, "description">;
 
const todo: TodoPreview = {
  title: "Clean room",
  completed: false,
  createdAt: 1615544252770,
};
```

omit이 작동하는 방법에 대해서 이해하기 위해서는, 먼저 객체에서 키를 뽑아서 해당 키를 제외해야 하므로, 객체 타입에서 키를 뽑는 법 부터 알아야 한다.

```typescript
type TodoKeys = keyof Todo // "title" | "description" | "completed" | "createdAt"
```