---
title: 'Array vs ArrayLike, Promise vs PromiseLike'
tags:
  - typescript
published: true
date: 2021-11-05 00:55:08
description: '이걸 유사 배열이?'
---

타입스크립트에는 `ArrayLike`라는게 존재한다. `Array`는 일반적인 배열을 의미하는데, `ArrayLike`는 무엇일까? 이를 알아보기 위해 `lib.es5.d.ts`에 가서 각각의 스펙을 살펴보자.

## Array

### `ArrayLike<T>` 

```typescript
interface ArrayLike<T> {
    readonly length: number;
    readonly [n: number]: T;
}
```

### `Array<T>`

```typescript
interface Array<T> {
    /**
     * Returns the value of the first element in the array where predicate is true, and undefined
     * otherwise.
     * @param predicate find calls predicate once for each element of the array, in ascending
     * order, until it finds one where predicate returns true. If such an element is found, find
     * immediately returns that element value. Otherwise, find returns undefined.
     * @param thisArg If provided, it will be used as the this value for each invocation of
     * predicate. If it is not provided, undefined is used instead.
     */
    find<S extends T>(predicate: (this: void, value: T, index: number, obj: T[]) => value is S, thisArg?: any): S | undefined;
    find(predicate: (value: T, index: number, obj: T[]) => unknown, thisArg?: any): T | undefined;

    /**
     * Returns the index of the first element in the array where predicate is true, and -1
     * otherwise.
     * @param predicate find calls predicate once for each element of the array, in ascending
     * order, until it finds one where predicate returns true. If such an element is found,
     * findIndex immediately returns that element index. Otherwise, findIndex returns -1.
     * @param thisArg If provided, it will be used as the this value for each invocation of
     * predicate. If it is not provided, undefined is used instead.
     */
    findIndex(predicate: (value: T, index: number, obj: T[]) => unknown, thisArg?: any): number;

    /**
     * Changes all array elements from `start` to `end` index to a static `value` and returns the modified array
     * @param value value to fill array section with
     * @param start index to start filling the array at. If start is negative, it is treated as
     * length+start where length is the length of the array.
     * @param end index to stop filling the array at. If end is negative, it is treated as
     * length+end.
     */
    fill(value: T, start?: number, end?: number): this;

    /**
     * Returns the this object after copying a section of the array identified by start and end
     * to the same array starting at position target
     * @param target If target is negative, it is treated as length+target where length is the
     * length of the array.
     * @param start If start is negative, it is treated as length+start. If end is negative, it
     * is treated as length+end.
     * @param end If not specified, length of the this object is used as its default value.
     */
    copyWithin(target: number, start: number, end?: number): this;
}
```

`Array`는 딱봐도 우리가 일반적으로 아는 배열에 들어가는 메소드들이 정의되어 있지만, `ArrayLike`는 그렇지 않다. `length`와 index로만 접근할 수 있도록 구현되어 있다. 이는 바로 우리가 잘 알고 있는 유사 배열 객체다. 배열 처럼 순회할 수 있지만, 그 뿐인 유사 배열 객체. 대표적으로는 

- https://developer.mozilla.org/en-US/docs/Web/API/HTMLCollection
- https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions/arguments

가 있다.

## Promise

그렇다면 이번에는 `Promise`를 살펴보자.

### `Promise<T>`

```typescript
interface Promise<T> {
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): Promise<T>
}
```

### `ArrayLike<T>` 

