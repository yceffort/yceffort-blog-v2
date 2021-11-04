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

### `Promise<T>` (lib.2018.promise.d.ts)

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

### `Promise<T>` (lib.es5.d.ts)

```typescript
interface Promise<T> {
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): Promise<TResult1 | TResult2>;

    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): Promise<T | TResult>;
}
```

### `PromiseLike<T>` 

```typescript
interface PromiseLike<T> {
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): PromiseLike<TResult1 | TResult2>;
}
```

`Promise<T>`에는 `finally`만 있고, `PromiseLike<T>`에는 `then` 밖에 없다. 🤔 이 둘의 차이를 먼저 알 필요가 있다.

### `then` vs `finally`

- `finally`: promise가 처리되면 충족되거나 (resolve) 거부되거나 (reject) 상관없이 실행하는 콜백함수다. Promise의 성공적으로 수행되었는지, 거절되었는지에 관계없이 Promise가 처리된 후에 무조건 한번은 실행되는 코드다.
- `then`: 은 우리가 잘 아는 것처럼 Promise를 리턴하고 두개의 콜백함수를 받는다. 하나는 충족되었을 때 (`resolve`) 그리고 거부되었을 때 (`reject`)를 위한 콜백 함수다.

```javascript
p.then(onFulfilled, onRejected);

p.then(function(value) {
  // 이행
}, function(reason) {
  // 거부
});
```

그리고 또한가지는 `finally`는 Promise 체이닝에서 결과를 받을 수 없다는 것이다. 

```javascript
const result = new Promise((resolve, reject) => resolve(10))
  .then(x => {
    console.log(x); // 10
    return x + 1;
  })
  .finally(x => {
    console.log(x); // undefined
    return x + 2;
  });
// then에서 리턴했던 11을 resolve 한다.
result // Promise {<fulfilled>: 11}
```

또다른 차이는 에러핸들링과 Promise chaining이다. 만약 promise chaining에서 에러처리를 미루고 다른 어딘가에서 처리하고 싶다면, `finally`를 사용하면 된다.

```javascript
new Promise((resolve, reject) => reject(0))
  .catch(x => {
    console.log(x); // 0
    throw x;
  })
  .then(x => {
    console.log(x); // Will not run
  })
  .finally(() => {
    console.log('clean up'); // 'clean up'
  });
// Uncaught (in promise) 0
// try catch 로 잡으면 잡힌다!
```

끝으로 `finally`는 es2018에서 나온 메소드 이기 때문에 `lib.es2018.promise.d.ts`에 존재한다. https://2ality.com/2017/07/promise-prototype-finally.html

아무튼 다시 돌아가서, catch가 없는 `PromiseLike`는 왜 존재하는 것일까? 🤔 Promise가 정식 스펙이 되기 전, Promise를 구현하기 위한 다양한 라이브러리가 존재했다.

- https://promisesaplus.com/
- http://bluebirdjs.com/docs/getting-started.html

이들은 표준이전에 태어나 `catch` 구문없이 promise를 처리하고 있었고, 타입스크립트는 
이를 지원하기 위해서 `PromiseLike`를 만든 것이었다.

따라서 `Promise` 뿐만 아니라 좀더 광의의 `Promise` (표준 이전에 만들어진 라이브러리로 만들어진 `Promise`)를 처리하기 위해서 `PromiseLike` 타입을 추가하게 된 것이다.