---
title: 프로그래머 기초 수학 1-1 - 명제와 논리연산
tags:
  - programming
  - math
published: true
mathjax: true
date: 2020-07-16 06:53:30
description:
  "```toc from-heading: 2 to-heading: 3 ``` ## 명제  명제란 참인지 거짓인지 판별할 수
  있는 문장이나 수식을 말한다.  - 달은 지구의 위성이다 (참) - 고래는 어류다 (거짓) - $7 \\times 8 = 56$ (참) -
  $x^2 - 2x - 1 = 0$  ($x$ 값이 정해지지 않아 알수 없다. 이는 명제다 가이다.)..."
category: programming
slug: /2020/07/math-for-programmer-chapter1-1-logical-operation/
template: post
---

## Table of Contents

## 명제

명제란 참인지 거짓인지 판별할 수 있는 문장이나 수식을 말한다.

- 달은 지구의 위성이다 (참)
- 고래는 어류다 (거짓)
- $7 \times 8 = 56$ (참)
- $x^2 - 2x - 1 = 0$ ($x$ 값이 정해지지 않아 알수 없다. 이는 명제다 가이다.)

## 논리연산

이러한 명제에도 기본적인 연산이 존재한다. 명제는 진리값을 다루므로 그에 대한 연산은 논리적인 성질을 띄고, 이러한 논리연산을 명제에 적용하면 그 결과 새로운 명제가 만들어진다.

| 논리연산           | 기호 | 뜻                                |
| ------------------ | ---- | --------------------------------- |
| 논리합(OR)         | ∨    | 적어도 하나 이상의 명제가 참인가  |
| 논리곱(AND)        | ∧    | 주어진 모든 명제가 참인가?        |
| 부정(NOT)          | ¬    | 원래 명제의 참과 거짓을 뒤바꾼다. |
| 배타적 논리합(XOR) | ⊕    | 둘중하나만 참인가?                |

논리연산의 결과를 표 형태로 쉽게 나타낸 것을 진리표라고 한다.

| $p$ | $q$ | $p \lor q$ |
| --- | --- | ---------- |
| $T$ | $T$ | $T$        |
| $T$ | $F$ | $T$        |
| $F$ | $T$ | $T$        |
| $F$ | $F$ | $F$        |

| $p$ | $q$ | $p \land q$ |
| --- | --- | ----------- |
| $T$ | $T$ | $T$         |
| $T$ | $F$ | $F$         |
| $F$ | $T$ | $F$         |
| $F$ | $F$ | $F$         |

| $p$ | $\lnot q$ |
| --- | --------- |
| $T$ | $F$       |
| $F$ | $T$       |

$p \lor \lnot p$와 같이 항상 참인 명제를 항진명제라고 하고, $p \land \lnot p$ 처럼 항상 거짓인 명제는 모순명제라고 한다.

만약 명제를 연산한 결과에 부정연산을 하게 되면 어떻게 될까?

$$
\lnot (p \lor q)
$$

위 연산은 `p나 q 둘중 하나라도 참이면` 의 결과가 참이니, OR을 부정했다면 그 반대로 `p와 q 둘중 어떤 것도 참이 아니라면` 연산의 결과가 될 것이다.

$$
\lnot p \land \lnot q
$$

이 처럼 같은 진리값을 갖는 두가지 명제를 동치라고 하고, 기호로는 $\equiv$ 라고 한다.

위에서 언급했던 식은 이렇게 쓸 수 있다.

$$
\lnot (p \lor q) \equiv \lnot p \land \lnot q
$$

$$
\lnot (p \land q) \equiv \lnot p \lor \lnot q
$$

이러한 동치관계는 [드모르간의 법칙](https://ko.wikipedia.org/wiki/%EB%93%9C_%EB%AA%A8%EB%A5%B4%EA%B0%84%EC%9D%98_%EB%B2%95%EC%B9%99) 이라고 한다.

> 논리합은 논리곱과 부정기호로, 논리곱은 논리합과 부정기호로 표현할 수 있음을 가리키는 법칙이다. (킹무위키)

```
not(A or B)=(not A) and (not B)
not(A and B)=(not A) or (not B)
```

논리합은 덧셈과, 논리곱은 곱셈과 유사한면이 있고, 진리값 T, F는 각각 1, 0 의 성질을 갖는다. 그러나 진리값이 숫자는 아니므로 엄연히 차이가 있다.

$$
p \lor F \equiv p
\\
p \land F \equiv p
$$

숫자와는 다르게 자기 자신의 논리합과 논리곱은 자신으로 돌아온다.

$$
p \lor p \equiv p
\\
p \land p \equiv p
$$

교환법칙도 적용된다.

$$
p \lor q \equiv q \lor p
\\
p \land q \equiv q \land p
$$

결합법칙 역시 모두에게 동일하게 적용된다.

$$
(p \lor q) \lor r \equiv p \lor (q \lor r)
\\
(p \land q) \land r \equiv p \land (q \land r)
$$

분배법칙은 수학과 약간 다르다.

$$
p \lor (q \land r) \equiv (p \lor q) \land (p \lor r)
\\
p \land (q \lor r) \equiv (p \land q) \lor (p \land r)
$$

## 배타적 논리합 $\oplus$

이 연산은 둘 중 어느 한쪽만 참일 때만 참이다. 순서는 상관없다. 따라서 교환법칙이 적용된다. 그리고 마찬가지로 결합법칙도 적용된다.

그리고 배타적 논리합은 아래와 같이 특이현 결과를 낳기도 한다.

$$
p \oplus T \equiv \lnot p
\\
p \oplus F \equiv \lnot p
\\
p \oplus p \equiv F
$$

그리고 어떤 명제에서 다른 명제를 두번 연달아 XOR 하면 다시 원래의 연산으로 돌아온다.

$$
(p \oplus q) \oplus q \equiv p
$$

이를 증명해보자.

$$
(p \oplus q) \oplus q
\\
p \oplus (q \oplus q)
\\
p \oplus F \equiv p
$$

이게 코드에서 무슨 소용이 있냐고?

다음과 같은 조건문이 있다고 가정해보자.

```javascript
var question = (!cond1 || cond2) && !(cond1 && cond2)
```

이는 조건문을 파악하기 어렵고 까다롭게 만든다. 논리연산으로 잘 만들어보자.

$$
(\lnot p \lor q) \land (p \land q)
$$

이를 드모르간의 법칙과 분배법칙을 활용해보자.

$$
(\lnot p \lor q) \land \lnot(p \land q)
\\
(\lnot p \lor q) \land (\lnot p \lor \lnot q)
\\
\lnot p \lor (q \land \lnot q)
\\
\lnot p \lor F
\\
\lnot p
$$

와! 결국 위 코드는 이것과 같았다.

```javascript
var question = !cond1
```
