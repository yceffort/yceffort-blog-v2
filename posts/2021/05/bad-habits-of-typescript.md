---
title: '타입스크립트에서 조심해야할 습관'
tags:
  - javascript
  - typescript
published: true
date: 2021-05-25 20:57:49
description: 'M1 맥북 프로 너무 좋네여'
---

## Table of Contents

## any

타입스크립트에서 얼마나 타입을 잘 지키는지는, `any`를 얼마나 사용하느냐에 달려 있다고 봐도 될 것 같다. 그러나 불가피하게 `any`가 사용되는 경우도 있다. `JSON.parse`가 그 중 하나다. 이 메소드는, 리턴타입을 추론 할 수 없기 때문에 `any`로 리턴을 한다.

```typescript
JSON.parse('{hello: world}')
// (method) JSON.parse(text: string, reviver?: ((this: any, key: string, value: any) => any) | undefined): any
```

`any`의 유혹은 강력하다. 그냥 뭐든지 넘겨주기 때문이다.

```typescript
function testNumber(num: number) {
  return num + 1
}

const num: any = 'eleven' // any 라서 number만 넘길 수 있었는데 무시되었다.
testNumber(num)
```

따라서 `any`의 사용은 최소화 하는 것이 좋다. 타입을 아직 알 수 없을 때는, `any`대신 `unknown`을 쓰는게 좋다.

## Type Assertion

`type assertion`은 엄밀히 말해서 `cast`와는 다르다. 타입스크립트에서 `type assertion`이란 `x as number`와 같은 형태를 의미한다. 타입이 있는 다른언어 (C, Java등) 에서 `cast`는 런타임에 영향을 미친다. 예를 들어, `(int)f`가 있다면 float이 런타임 중에 int로 바뀌게 된다. [그러나 타입스크립트는 런타임시에 타입을 제거하기 때문에](http://neugierig.org/software/blog/2016/04/typescript-types.html) 실제로 아무런 영향을 미치지 않는다.

```typescript
const strOrNum = Math.random() ? '42' : 42
const num = strOrNum as number
```

을 컴파일하면 별 차이가 없다.

```javascript
'use strict'
const strOrNum = Math.random() ? '42' : 42
const num = strOrNum
```

따라서 `type assertion`이란 타입으로 캐스팅을 하는게 아니고, 그냥 그게 이 타입이라고 주장하는 것이다. (그래서 `assertion`이기도 하구)

```typescript
function testNumber(num: number) {
  return num + 1
}

const num = Math.random() ? 'eleven' : 11
testNumber(num as number) // 반반쯤의 확률로 에러가 난다.
```

이러한 `as`의 사용은 api 호출에서도 자주 보인다.

```typescript
const response = await fetch('/api/user')
const result = (await response.json()) as UserInterface
```

이러한 경우에는, [타입 가드](https://basarat.gitbook.io/typescript/type-system/typeguard)를 써서 막는 것이 좋다.

```typescript
function isUser(data: unknown): data is UserInterface {
  return data && typeof data === 'objet' && 'name' in data // ...
}
const response = await fetch('/api/user')
const result = await response.json()

if (!isUser(result)) {
  throw new Error(`${result}는 UserInterface가 아니예욧`)
}
```

조금더 빡세게(?) 타입가드를 하고 싶다면, [Zod](https://github.com/colinhacks/zod)와 같은 도구를 사용해보는 것도 좋다. 혹은 [typescript-json-schema](https://github.com/YousefED/typescript-json-schema)로 JSON schema에서 타입을 뽑아낸 다음에 검사하는 방법도 있을 수 있다.

## 객체와 배열 lookup

타입스크립트는 배열을 참조할 때 별다른 처리를 하지 않기 때문에, 에러가 날 수 있다.

```typescript
const l = [1, 2, 3]
const item = l[3]
item + 1 // error 지만, 컴파일시에는 모른다.
```

```typescript
const user: { [key: string]: string } = { name: 'kyc' }
user.age + 1 // error 지만, 컴파일 시에는 모른다.
```

왜 타입스크립트가 가만 뒀을까? 아마도 이러한 상황은 매우 자주 있는 일이고, 이것이 유효한지 체크하는 것이 꽤 어려운일이어서 그런걸수도 있다. 근데, 이를 체크하는 방법이 있다. 바로 [noUncheckedIndexedAccess](https://www.typescriptlang.org/tsconfig#noUncheckedIndexedAccess)다.

```typescript
const l = [1, 2, 3]
const item1 = l[3]
const item2 = l[2]
item1 + 1 // Object is possibly 'undefined'.(2532)
item2 + 1 // 근데 문제는 이거까지 에러가 난다는 거다.

l.map((item) => item + 1) // 이런건 괜찮다.
```

[확인해보기](https://www.typescriptlang.org/play?noUncheckedIndexedAccess=true#code/MYewdgzgLgBANjAvDA2gRgDQwExYMwC6AsAFCiSwCWUApgLZpLwqGnnQzX3ZNwrbESXBjADUMRqWE9xaIA)

위에서 보이는 것 처럼, `noUncheckedIndexedAccess`는 적당히 경고를 날려주는 장점도 있지만, 그다지 똑똑하지 않다는 단점도 있다.

결론적으로, 객체나 리스트를 lookup 할때는 undefined가 있을 가능성에 대해서 염두해 두어야 한다.

## 부정확한 타입 정의

자바스크립트의 라이브러리에 타입선언을 집어넣는 것은 일종의 거대한 `type assertion`이다. 그들의 라이브러리가 이렇게 정적으로 모델링 했다고 주장하는 거지만, 이를 보장하는 것은 아무것도 없다. (물론 이는 라이브러리가 타입스크립트로 작성되지 않았다는 것에 한해서다. 타입스크립트로 작성되지 않고, 타입만 있다면 이런일이 발생할 수 있다.)

예를 들면 2년째 고쳐지지 않는 [잘못된 타이핑](https://github.com/alex3165/react-mapbox-gl/issues/776) 이라던가...

이런 문제를 해결하는 가장 좋은 방법은, 직접 버그를 고치는 것이다. 직접 [DefinitelyTyped](https://github.com/DefinitelyTyped/DefinitelyTyped)에 쳐들어가서 고치면 된다. 이게 좀 부담되는 나와 같은 ISTJ 들에겐 [augmentation](https://www.typescriptlang.org/docs/handbook/declaration-merging.html) 이나, 최악의 옵션으로 type assertion을 활용하는 방법도 있다.

물론, 몇몇 함수는 이렇게 타입을 선언하기가 굉장히 어렵다는 것도 이해해줘야 한다. [String.prototype.replace](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/replace#specifying_a_function_as_a_parameter)의 파라미터를 잠깐 보자. [Object.assign](https://github.com/microsoft/TypeScript/pull/28553#issuecomment-440004598)의 경우엔, 피치못할 사정으로 인해 잘못된 타이핑이 된 경우도 있었다.

## variance and arrays

[typescript github의 이 이슈](https://github.com/microsoft/TypeScript/issues/9825#issuecomment-234115900)를 살펴보자.

```typescript
function addDogOrCat(arr: Animal[]) {
  arr.push(Math.random() > 0.5 ? new Dog() : new Cat())
}

const z: Cat[] = [new Cat()]
addDogOrCat(z) // 개가 들어갈 수도 있다.
```

이러한 짓을 방지하려면 어떻게 해야할까? `readonly`를 쓰는 방법이 있을 수 있다.

```typescript
function addDogOrCat(arr: readonly Animal[]) {
  arr.push(Math.random() > 0.5 ? new Dog() : new Cat())
  //  Property 'push' does not exist on type 'readonly Animal[]'.
}
```

아니면, `push` 대신 이렇게 하거나.

```typescript
function dogOrCat(): Animal {
  return Math.random() > 0.5 ? new Dog() : new Cat()
}

const z: Cat[] = [new Cat(), dogOrCat()]
```

관련해서 읽어볼만 한 좋은 글: https://iamssen.medium.com/typescript-%EC%97%90%EC%84%9C%EC%9D%98-%EA%B3%B5%EB%B3%80%EC%84%B1%EA%B3%BC-%EB%B0%98%EA%B3%B5%EB%B3%80%EC%84%B1-strictfunctiontypes-a82400e67f2

## 함수 호출에 따른 부작용

```javascript
interface UserInterface {
  name: string
  age?: number
}

function userProcessor(user: UserInterface, processor: (user: UserInterface) => void) {
  if (user.age) {
    processor(user)
    document.body.innerHTML = `${user.age + 1}`
  }
}
```

만약 `processor`에서 이런짓을 하면 어떻게 될까?

```typescript
userProcessor({ name: 'kyc', age: 15 }, (u) => delete u.age)
// 타입 체크는 성공하지만, 에러가 날거다.
```

물론 만일을 위해서 `if`처리가 추가되었지만, `processor`에서 이를 무효화 했다. 타입스크립트는 저 함수에서 무슨짓을 할지 모르기 때문에, 어찌보면 당연한 것이다.

자바스크립트에서 파라미터를 조작하는 일은 흔치 않고 또한 안티패턴이기 때문에, 타입스크립트에서는 이러한 동작을 허용해 뒀을 것이다.

이를 수정해두는 방법은, 역시 `readonly`를 사용하는 것이 있다.

```typescript
function userProcessor(
  user: UserInterface,
  processor: (user: Readonly<UserInterface>) => void,
) {
  if (user.age) {
    processor(user)
    document.body.innerHTML = `${user.age + 1}`
  }
}

userProcessor({ name: 'kyc', age: 15 }, (u) => delete u.age)
// The operand of a 'delete' operator cannot be a read-only property.
```

> Readonly는 얕은 비교만 하기 때문에, [ts-essentials](https://github.com/krzkaczor/ts-essentials)를 사용해야 깊은 비교를 할 수 있다.

당연하게도, 가장 좋은 방법은 객체의 처리를 객체가 정의된 이후에 하는 것이다.

```typescript
function processFact(
  user: UserInterface,
  processor: (user: UserInterface) => void,
) {
  const { age } = user
  if (age) {
    processor(user)
    document.body.innerHTML = `${age + 1}` // safe
  }
}
```

## Further reading

- https://frenchy64.github.io/2018/04/07/unsoundness-in-untyped-types.html
- https://www.typescriptlang.org/docs/handbook/type-compatibility.html#a-note-on-soundness
- https://www.typescriptlang.org/play?strictFunctionTypes=false&q=209#example/soundness
