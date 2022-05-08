---
title: '알아두면 유용한 타입스크립트 팁'
tags:
    - typescript
published: true
date: 2022-05-08 04:12:01
description: ''
---

### 제네릭 활용하기

테이블 컴포넌트가 있고, 여기에 props를 할당해서 그린다고 생각해보면, 보통은 이런 코드가 나올 것이다.

```typescript
import React from 'react'

interface Props {
  items: Array<{id: string}>
  renderItem: (item: {id: string}) => React.ReactNode
}

export const Table = (props: Props) => {
  return null
}

export const Component = () => {
  return (
    <Table items={[{id:"1"}]}
    renderItem={(item) => {
      return null;
    }}
    />
  )
}
```

하지만 이런 구조는 `{id: string}` 으로 고정되어 있어,여러가지 종류의 props를 그리기에는 무리가 있다. 이럴 때 사용하면 좋은 것이 Generic이다. Generic은 타입, 인터페이스 등에서 외부에서 정의된, 공통의 속성을 사용하고 싶을 때 유용하다.

```typescript
import React from 'react'

interface Props<TItem> {
  items: Array<TItem>
  renderItem: (item: TItem) => React.ReactNode
}

export const Table = <TItem, >(props: Props<TItem>) => {  
  return null
}

export const Component = () => {
  return (
    <Table items={[{id:"1"}]}
    // item이 {id: "1"}로 추론되는 것을 볼 수 있다.
    renderItem={(item) => {
      return null;
    }}
    />
     <Table items={[{id:"1", name:"yceffort"}]}
    // 서로 다른 props가 와도 문제 없다.
    renderItem={(item) => {
      return null;
    }}
    />
  )
}
```