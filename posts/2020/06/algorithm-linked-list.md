---
title: 알고리즘 - 연결 리스트
tags:
  - algorithm
  - javascript
  - python
published: true
date: 2020-06-19 04:34:32
description:
  '## 연결리스트 연결리스트, Linked List 는 각 노드들이 한 줄로 연결되어 있는 방식으로 각 노드는 데이터와
  포인터 (다음 노드의 정보)를 가지고 있다. 연결리스트는 일반적인 배열과 다르게 삽입과 삭제가 `O(1)`에 가능하다는 장점이 있다. 하지만
  특정 n번 째 정보를 찾는 데에는 `O(n)`시간이 걸린다는 단점도 있다.  ![단일 연결 리스트...'
category: algorithm
slug: /2020/06/algorithm-linked-list/
template: post
---

## 연결리스트

연결리스트, Linked List 는 각 노드들이 한 줄로 연결되어 있는 방식으로 각 노드는 데이터와 포인터 (다음 노드의 정보)를 가지고 있다. 연결리스트는 일반적인 배열과 다르게 삽입과 삭제가 `O(1)`에 가능하다는 장점이 있다. 하지만 특정 n번 째 정보를 찾는 데에는 `O(n)`시간이 걸린다는 단점도 있다.

![단일 연결 리스트](https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Single_linked_list.png/800px-Single_linked_list.png)

![이중 연결 리스트](https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Doubly_linked_list.png/800px-Doubly_linked_list.png)

이중 연결리스트는 이 포인터에 앞, 뒤 정보가 모두 담겨 있다.

![원형 연결 리스트](https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Circurlar_linked_list.png/800px-Circurlar_linked_list.png)

마지막 노드의 포인터가 첫 번째 노드를 가르킨다.

## 구현

먼저 가장 최초에 있는 노드를 `header`라고 부르고, 맨 마지막에 있는 노드를 `tail`이라고 부른다. 또한 일반적인 배열과는 다르게, 배열의 시작을 1로 하고, 0번째 노드는 비어있는 노드로 가정한다. 이는 구현에 있어 조금 더 편하게 하기 위함이다.

그리고 구현해볼 메소드는 다음과 같다.

- print: 리스트를 볼 수 있도록 프린트 한다.
- getAt(n): n번째 노드를 꺼낸다.
- insertAt(n, node): n번째에 `node`를 삽입한다.
- insertAfter(prev, new): `prev`이후에 `new`를 삽입한다.
- popAt(n): n번째 노드를 삭제한다.
- popAfter(prev): `prev` 이후 노드를 삭제한다.
- size: 전체 노드 개수를 계산한다.
- traverse: 전체 노드를 배열로 리턴한다.

### python

#### Node

```python
class Node:
  def __init__(self, item):
    self.data = item # 데이터 정보
    self.next = None # 다음 노드 정보
```

#### Linked List

```python
class LinkedList:
  def __init__(self):
    self.size = 0
    self.head = Node(None)
    self.tail = None

  def traverse(self):
    result = []
    curr = self.head
    while curr.next:
      curr = curr.next
      result.append(curr.data)
    return result

  def getAt(self, n):
    if n < 0 or n > self.size:
      return None

    i = 0
    curr = self.head
    while i < n:
      curr = curr.next
      i += 1

    return curr

  def insertAfter(self, prev, new):
    # 새로운 노드의 다음 노드는 이전 노드의 다음 정보
    new.next = prev.next
    # prev가 tail일 경우
    if prev.next is None:
      self.tail = new
    # 이제 이전 노드의 다음 노드는 사로운 노드
    prev.next = new
    self.size += 1


  def insertAt(self, n, new):
    if n < 1 or n > self.size + 1:
      return False

    # tail에 들어갈 경우
    if n != 1 and n == self.size + 1:
      prev = self.tail
    else:
      prev = self.getAt(n - 1)

    return self.insertAfter(prev, new)

  def popAfter(self, prev):
    popData = prev.next.data
    ## tail 을 제거하려고 할때
    if prev.next.next is None:
      self.tail = prev
      prev.next = None
    else:
      prev.next = prev.next.next

    self.size -= 1

    return popData

  def popAt(self, n):
    if n < 1 or n > self.size:
      raise IndexError

    # head일 경우
    if n == 1:
      prev = self.head
    else:
      prev = self.getAt(n - 1)

    return self.popAfter(prev)

  def print(self):
    if self.size == 1:
      print("this linked list is empty")
    else:
      print(self.traverse())
```
