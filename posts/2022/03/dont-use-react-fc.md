---
title: 'React.FC를 사용하지 않는 이유'
tags:
  - typescript
  - react
published: true
date: 2022-03-25 16:11:06
description: 'React.FC가 잘못됐다는 이야기는 아닙니다'
---

## Table of Contents

이따금씩 다른 사람들이 만들어둔 컴포넌트 코드를 보면, 함수형 컴포넌트에 `React.FC<>`를 달아두어서 함수를 타이핑 한 것을 종종 볼 수 있었다.

그러나 나는 그러한 방식을 썩 선호하지는 않는다. 그 이유는 다음과 같다.

## `React.FC<>`란 무엇인가?

리액트에서는 크게 두가지 방법으로 컴포넌트를 정의할 수 있다.

1. `Component`를 extending하는 클래스 컴포넌트
2. `JSX`를 리턴하는 함수형 컴포넌트

일단 리액트는 타입스크립트로 작성되있지 않기 때문에, 리액트 커뮤니티에서는 `@types/react`패키지를 제공하여 리액트에 대한 타이핑을 지원하고 있다. 여기에는 `FC`라고 하는 제네릭 타입이 있는데, 이를 활용하면 함수형 컴포넌트를 아래와 같이 타이핑 할 수 있게 도와준다.

```typescript
import { FC } from 'react'

type GreetingProps = {
  name: string
}

const Greeting: FC<GreetingProps> = ({ name }) => {
  return <h1>Hello {name}</h1>
}
```

그리고, 이 FC는 다음과 같은 구조로 되어 있다.

```typescript
type FC<P = {}> = FunctionComponent<P>

interface FunctionComponent<P = {}> {
  (props: PropsWithChildren<P>, context?: any): ReactElement<any, any> | null
  propTypes?: WeakValidationMap<P> | undefined
  contextTypes?: ValidationMap<any> | undefined
  defaultProps?: Partial<P> | undefined
  displayName?: string | undefined
}
```

> [github 소스 코드 보기](https://github.com/DefinitelyTyped/DefinitelyTyped/blob/0beca137d8552f645064b8a622a6e153864c66ee/types/react/index.d.ts#L548-L556)

## 함수를 타이핑 하지만, 인수를 타이핑 하지는 않는다.

`React.FC`는 함수를 타이핑해준다. 이름에서 할 수 있는 것처럼. 함수 타이핑은 일반적인 기명 함수에 적용하기 매우 어렵다. 만약 아래와 같은 코드에 함수 타이핑을 적용해본다고 가정해보자.

```typescript
function Greeting({ name }) {
  return <h1>Hello {name}</h1>
}
```

먼저 쉽게할 수 있는 방법 중 하나는, 익명 함수를 변수에 할당하여 타이핑 하는 것이다.

```typescript
const Greeting: FC<GreetingProps> = function ({ name }) {
  return <h1>Hello {name}</h1>
}
```

혹은 화살표 함수를 쓸 수도 있겠다.

```typescript
const Greeting: FC<{ name: string }> = ({ name }) => {
  return <h1>Hello {name}</h1>
}
```

그러나 우리가 일반적으로 쓰는 기명 함수 방식에서는 이러한 타이핑을 사용할 수 없다. 만약 함수 타이핑을 사용하지 않는다면, 함수를 기명이건 익명이건 어떤 방식으로 사용해도 문제가 없다.

```typescript
function Greeting({ name }: GreetingProps) {
  return <h1>Hello {name}</h1>
}
```

## `React.FC<>`는 항상 children을 가질수 있다.

`React.FC<>`로 타이핑 하는 것은 컴포넌트에 children이 있을 수 있다는 것을 의미한다.

```typescript
export const Greeting: FC<GreetingProps> = ({ name }) => {
  return <h1>Hello {name}</h1>
}

const App = () => (
  <>
    <Greeting name="Stefan">
      <span>{"I can set this element but it doesn't do anything"}</span>
    </Greeting>
  </>
)
```

`Greeting`에는 딱히 `children`을 렌더링하거나 처리하는 코드가 없음에도 위 코드는 정상적으로 처리되는 것을 볼수 있다.

대신, 일반적인 방법으로 한다면 아래코드는 다음과 같은 결과가 나온다.

```typescript
function Greeting({ name }: {name: string}) {
  return <h1>Hello {name}</h1>
}
const App = () => <>
  // Property 'children' does not exist on type 'IntrinsicAttributes & { name: string; }'.ts(2322)
  <Greeting name="Stefan">
    <span>{"I can set this element but it doesn't do anything"}</span>
  </Greeting>
</>
```

최소한 컴포넌트에 children의 존재가 가능한지 여부를 확인하는 것은 도움이 될 수 있다. 만약 컴포넌트에 children이 존재할 수도 있다는 것을 알리기 위해서는, `PropsWithChildren`을 사용하는 것이 좋다.

```typescript
type PropsWithChildren<P> = P & { children?: ReactNode | undefined }
```

[https://github.com/DefinitelyTyped/DefinitelyTyped/blob/0beca137d8552f645064b8a622a6e153864c66ee/types/react/index.d.ts#L830](https://github.com/DefinitelyTyped/DefinitelyTyped/blob/0beca137d8552f645064b8a622a6e153864c66ee/types/react/index.d.ts#L830)

```typescript
function Card({ title, children }: PropsWithChildren<{ title: string }>) {
  return (
    <>
      <h1>{title}</h1>
      {children}
    </>
  )
}
```

## `React.FC<>`는 defaultProps를 쓰지 못하게 만든다.

`defaultProps`는 클래스 기반 컴포넌트의 유물로, props에 기본값을 세팅할 수 있도록 도와준다. 함수형 컴포넌트에서는, 자바스크립트의 기본적인 기능을 활용하면 기본값을 제공할 수 있다.

```typescript
function LoginMsg({ name = 'Guest' }: LoginMsgProps) {
  return <p>Logged in as {name}</p>
}
```

타입스크립트 3.1 버전 이후로, `defaultProps`를 이해하는 메커니즘이 추가되었으며, 이는 사용자가 세팅한 값을 기반으로 기본값이 설정된다. 그러나 `React.FC`는 `defaultProps`에 대한 타이핑 하기 때문에 이러한 기본값에 대한 연결고리를 끊어버리게 된다. 아래 코드를 살펴보자.

```typescript
type GreetingProps = {
  name: string
}

export const Greeting: FC<GreetingProps> = ({ name }) => {
  return <h1>Hello {name}</h1>
}
음
Greeting.defaultProps = {
  name: 'World',
}

const App = () => (
  <>
    {/* name에 world가 들어오지 않음 💥*/}
    <Greeting />
  </>
)
```

하지만, 일반적인 함수 방식이라면 `defaultProps`는 여전히 유효하다.

```typescript
export const Greeting = ({ name }: GreetingProps) => {
  return <h1>Hello {name}</h1>
}

Greeting.defaultProps = {
  name: 'World',
}

const App = () => (
  <>
    {/* Yes! ✅ */}
    <Greeting />
  </>
)
```

## Stateless Function Component의 과거

예전에는 모두가 함수형 컴포넌트를 stateless function component (무상태 함수형 컴포넌트)라고 불렀었다.

```typescript
/**
 * @deprecated as of recent React versions, function components can no
 * longer be considered 'stateless'. Please use `FunctionComponent` instead.
 *
 * @see [React Hooks](https://reactjs.org/docs/hooks-intro.html)
 */
```

[https://github.com/DefinitelyTyped/DefinitelyTyped/blob/0beca137d8552f645064b8a622a6e153864c66ee/types/react/index.d.ts#L532-L548](https://github.com/DefinitelyTyped/DefinitelyTyped/blob/0beca137d8552f645064b8a622a6e153864c66ee/types/react/index.d.ts#L532-L548)

훅이 소개된 이후로, 함수형 컴포넌트에는 많은 상태가 들어오기 시작했고 이제는 더이상 stateless하게 취급하지 않는다. 위 코드에서 볼 수 있는 것 처럼, `SFC`는 `FC`가 되었다. 또 훗날 `FC`가 무엇으로 바뀔 수 있을지도 모를일이다. 그러나 단순히 인수 (props)를 타이핑 하는 것은 이후에 함수의 타입이 바뀌더라도 안전하게 처리할 수 있다.

## Summary

`React.FC`를 쓰는 것이 꼭 나쁜 것 만은 아니다. 여전히 이것을 사용하는게 좋은 경우도 있을 것이고, 그렇다고 이를 억지로 고칠 필요도 없을 수 있다. 그러나 props를 타이핑 하는 것이 조금더 자바스크립트의 느낌과 비슷하고, 다양한 경우의 수로 부터 조금더 안전해 질 수는 있다.
