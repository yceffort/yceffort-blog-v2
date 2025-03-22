---

title: 자바스크립트 자료 구조
tags:
  - javascript
  - typescript
published: true
date: 2020-06-29 07:42:01
description: "`toc tight: true, from-heading: 1 to-heading: 4 ` 타입스크립트로
구현해보는 일반적인 자료구조 ## Stack - push와 pop으로 구성된 stack - LIFO ```javascript
export default class Stack<T> { private stack: T[] construc..."
category: javascript
slug: /2020/06/javascript-data-structure/
template: post
---

## Table of Contents

타입스크립트로 구현해보는 일반적인 자료구조

## Stack

- push와 pop으로 구성된 stack
- LIFO

```javascript
export default class Stack<T> {
  private stack: T[]

  constructor() {
    this.stack = []
  }

  push(value: T) {
    this.stack.push(value)
  }

  pop(): T | undefined {
    return this.stack.pop()
  }

  size(): number {
    return this.stack.length
  }
}
```

## Queue

- 데이터 삽입과 삭제가 서로 반대쪽에서 일어나는 자료구조
- FIFO

```typescript
export default class Queue<T> {
  private queue: T[]

  constructor() {
    this.queue = []
  }

  dequeue(): T | undefined {
    return this.queue.shift()
  }

  enqueue(value: T) {
    this.queue.push(value)
    return this
  }

  size() {
    return this.queue.length
  }
}
```

## 우선순위 큐

- 각 원소들이 우선순위를 가지고 있는 큐
- 큐에서 무작정 `pop`이나 `shift`하는 것이 아니라, 우선순위가 가장 높은 것이 나오는 형태

```typescript
export type PQItem<T> = {priority: number; data: T}

export default class PriorityQueue<T> {
  private queue: PQItem<T>[]

  constructor() {
    this.queue = []
  }

  enqueue(value: PQItem<T>) {
    this.queue.push(value)
  }

  dequeue(): PQItem<T> | undefined {
    let entry = 0

    this.queue.forEach((_, i) => {
      const nextIndex = i + 1

      if (!this.queue[nextIndex]) {
        return undefined
      }

      if (this.queue[entry].priority > this.queue[nextIndex].priority) {
        entry = nextIndex
      }
    })

    const [dequeuedItem] = this.queue.splice(entry, 1)

    return dequeuedItem
  }
}
```

## 연결 리스트

```typescript
export class Node<T> {
  data: T
  next: Node<T> | null

  constructor(data: T) {
    this.data = data
    this.next = null
  }
}

export default class LinkedList<T> {
  // TODO
```

## 해쉬테이블

## 이진 트리
