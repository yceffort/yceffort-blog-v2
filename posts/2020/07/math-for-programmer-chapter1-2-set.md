---
title: 프로그래머 기초 수학 1-2 - 집합
tags:
  - programming
  - math
published: true
mathjax: true
date: 2020-07-17 07:28:25
description: '집합'
category: programming
slug: /2020/07/math-for-programmer-chapter1-2-set/
template: post
---

## Table of Contents

## 집합

개별적인 개체들의 모임을 집합이라고 하며, 집합을 이루는 개체를 원소라고 한다. 집합을 이루는 원소는 `{ }` 중괄호 로 나타내며, 원소나열법과 조건 제시법으로 표시한다. 순서는 상관없지만, 중복은 하지 않는다.

$$
A = \{ 1, 5, 7, 3, 9 \}
$$

$$
B = \{ 2, 4, 6, \ldots, 100 \}
$$

조건제시법은 집합의 원소들에 공통되는 조건법을 기술하는 방법이다.

$$
C = \{ x | 1 \leq x \leq 100 \}
$$

어떤 원소가 집합에 속하는지는 $\in$ $\notin$ 으로 표시한다. 어떤 집합의 원소의 개수는 $| A |$ 로 나타낸다.

어떤 집합 A의 모든 원소가 집합 B에 속해 있을 경우 부분집합이라고 하고 $A \subset B$ 라고 쓰며, 이 경우에는 A가 B에 포함된다고 한다. 부분집합이 아닐 경우엔 $A \not\subset B$ 으로 표시한다.

원소의 개수가 유한할 경우 유한집합, 무한할 경우 무한집합으로 부른다. 아무런 원소도 없을 경우에는 공집합이라고 하며, 기호로는 $\emptyset$ 으로 표시한다.

만약 $A \subset B$ $B \subset A$ 가 동시에 성립한다면 두 집합의 원소는 완전히 동일한 것으로, `상동`이라고 한다.한편 $A \subset B$이지만 $A \neq B$ 인 경우, B에는 A의 원소가 아닌것도 포함되어 있다는 뜻이므로, 진부분집합이라고 한다.

벤다이어 그램을 이용하면 이해하기가 더 쉽다.

![venn](./images/venn1.png)

집합 A와 B를 한 데 모은 집합르 합집합이라고 하며, $A \cup B$ 이라고 한다. 합집합을 조건제시법으로 하면 아래와 같다.

$$
A \cup B = \{ x | (x \in A) \lor (x \in B) \}
$$

반대로 공통된 요소만 골라냈다면 교집합이라고 하며 $A \cap B$라고 나타낸다.

$$
A \cap B = \{ x | (x \in A) \land (x \in B) \}
$$

만약 교집합이 $\emptyset$ 인 경우, 두집합 간에 공통된 원소가 하나도 없는 경우는 서로소라고 한다.

때로는 어떤 집합을 제외한 나머지 모든 것을 나타내야할 수도 있다. 기본전제가 되는 집합을 $U$ 전체 집합이라고 한다. 그리고 $U$에서 $A$를 제외 한 것을 $A$ 의 여집합이라고 하며, $A^c$ 로 나타낸다.

$$
A^c = \{ x | (x \notin A) \land (x \notin U) \}
$$

합집합 $A \cup B$ 의 크기를 구할 때는 주의해야한다.

$$
|A \cup B| = |A| + |B| - | A \cap B |
$$

이것은 아래의 식과 동일하다.

$$
A - B = \{ x | (x \in A) \cap (x \notin B) \} = A \cap B^c
$$

그리고 이러한 집합 연산에도 드모르간의 법칙이 성립한다.

$$
( A \cup B )^c = \{ x | \lnot(x \in A \lor x \in B) \} = \{ x | (x \notin A) \land (x \notin B) \} = A^c \cap B^c
( A \cap B )^c = \{ x | \land (x \in A \land x \in B) \} = \{ x | (x \notin A) \lor (x \notin B) \} = A^c \cup B^c
$$

집합 연산에서 교환, 결합, 분배 법칙은 어떨까? 먼저 교환법칙을 살펴보자. 합집합과 교집합은 앞뒤가 바뀌어도 성립하지만, 차집합은 그렇지 못하다.

$$
B \cup A = \{ x | x \in B ) \lor (x \in A) \} = A \cup B
B \cap A = \{ x | x \in B ) \land (x \in A) \} = A \cap B
B-A = B \cap A^c \not = A \cap B^c = A-B
$$

결합법칙 역시 합집합과, 교집합의 경우애는 성립하지만, 차집합은 성립하지 않는다.

$$
(A \cup B) \cup C = A \cup (B \cup C)
(A \cap B) \cap C = A \cap (B \cap C)
(A - B) - C = (A \cap B^c) - C = A \cap B^c \cap C^c
A - (B - C) = A - (B \cap C^c) = A \cap (B \cap C^c)^c = A \cap (B^c \cup C)
$$

분배 법칙 또한 합집합 교집합에서는 성립한다는 것을 알 수 있다.

$$
A \cup (B \cap C) = (A \cup B) \cap (A \cup C)
A \cap (B \cup C) = (A \cap B) \cup (A \cap C)
$$
