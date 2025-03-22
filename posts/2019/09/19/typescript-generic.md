---
title: 타입스크립트 제네릭
date: 2019-09-20 12:10:14
published: true
tags:
  - typescript
description:
  '## 제네릭이란 제네릭은 클래스 내부에서 사용하는 데이터의 타입을 외부에서 지정하는 것을 의미한다. 어떤 타입의
  데이터를 쓸지를, 클래스 선언부가 아니라 외부에서 결정하는 것이다. 일단 자바 코드로 한번 살펴보자.  ```java class
  Person<T>{     public T name; }  Person<String> p1 = new Person<...'
category: typescript
slug: /2019/09/19/typescript-generic/
template: post
---

## 제네릭이란

제네릭은 클래스 내부에서 사용하는 데이터의 타입을 외부에서 지정하는 것을 의미한다. 어떤 타입의 데이터를 쓸지를, 클래스 선언부가 아니라 외부에서 결정하는 것이다. 일단 자바 코드로 한번 살펴보자.

```java
class Person<T>{
    public T name;
}

Person<String> p1 = new Person<String>();
Person<StringBuilder> p1 = new Person<StringBuilder>();
```

`T`라는 데이터 타입은 존재하지 않는다. `T`는 name의 타입으로, 아래 처럼 Person을 사용하는 곳에서 정해진다. 따라서 `string`이 될수도, `stringbuilder`가 될수도 있는 것이다.

하지만 자바스크립트에서는 제네릭을 쓸일이 없다. 타입이 없기 때문에, 타입에 맞지 않는 코딩을 한다면 런타임에서 에러가 발생한다. 하지만 타입스크립트는 정적타입 언어이기 때문에 제네릭이 필요하게 되었다.

### any를 그냥 쓰면 안되나?

아래 코드를 살펴보자.

```typescript
class School {
  private students: any[] = []

  constructor() {}

  go(student: any): void {
    this.students.push(student)
  }

  bye(): void {
    this.students.pop()
  }
}
```

```typescript
const school = new School()
stack.push('라이오넬 멧시')
stack.push(10)
stack.pop().substring(0)
stack.pop().substring(0) // 에러
```

`string`에 이어서 `number`도 일일이 대응하기 위해서는 `any`를 쓰거나, 상속을 받아야 할 것이다.

### typescript 문법

```typescript
class School<T> {
  private students: T[] = []

  constructor() {}

  go(student: T): void {
    this.students.push(student)
  }

  bye(): T {
    return this.students.pop()
  }
}
```

`<T>`는 제네릭을 의미하며, 그안에 타입으로 사용될 `T`를 넣었다. 다른 문자도 되지만, 대게는 `T`를 쓰고 `Type Variables`라고 한다.

```typescript
const numberSchool = new School<number>()
const stringSchool = new School<string>()
const stringSchool = new School<boolean>()
```

이제 각각의 타입이 선언되어 사용될 수 가 있다.

### 함수에 써보기

다양한 타입의 array를 받아서 그 array의 첫번째를 리턴하는 함수를 만든다고 가정해보자. any를 사용한다면

```typescript
function returnFirstItem(items: any[]): any {
  return items[0]
}
```

하지만 제네릭을 쓴다면

```typescript
function returnFirstItem<T>(items: T[]): T {
  return items[0]
}

returnFirstItem < number > [0, 1, 2, 3]
```

이 된다.

### 여러개 Generic

```typescript
function multipleGeneric<T, U>(a1: T, a2: U): [T, U] {
  return [a1, a2]
}

multipleGeneric<string, boolean>('true', true)
```

### rest에서 제네릭

```typescript
interface XYZ {
  x: any
  y: any
  z: any
}

function dropXYZ<T extends XYZ>(obj: T) {
  let {x, y, z, ...rest} = obj
  return rest
}
```

객체에서 x, y, z를 빼다가 나머지를 리턴하는 함수이다. 객체에서 x, y, z가 없다면 컴파일 단계에서 에러가 날 것이고, x, y, z 가 있다면 어떤 타입이든 상관없이 x, y, z를 제거하고 리턴해줄 것이다.

만약 x, y, z가 없는 리턴타입까지 정확하게 명사히고 싶다면 이런 짓도 가능하다.

```typescript
interface XYZ {
  x: any
  y: any
  z: any
}

// Pick<T, a>는 T에서 a만 받는 다는 것이다
// Exclude<keyof T, keyof XYZ>는 앞에 타입에서 뒤에 있는 타입을 제외해준다.
type DropXYZ<T> = Pick<T, Exclude<keyof T, keyof XYZ>>

function dropXYZ<T extends XYZ>(obj: T): DropXYZ<T> {
  let {x, y, z, ...rest} = obj
  return rest
}
```

conditional types에 대해서도 알아봐야 겠다.
