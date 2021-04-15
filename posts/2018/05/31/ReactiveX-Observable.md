---
title: ReactiveX) Observable
date: 2018-05-31 09:01:45
published: true
tags:
  - programming
description:
  보통 일반적인 프로그램의 경우에는, 하나씩 작성된 순서에 따라 로직이 실행되고, 완료되면 또다른 로직이 실행되는 등의
  순서가 있음을 알수 있다. 그러나 이와 달리 ReactiveX는 "Observer"에 의해 임의의 순서에 따라 병렬적으로 실행되고 나중에
  결과나 나온다.  즉 메서드를 호출하는 것이 아니라, Observable안의 데이터를 조회하고, 변환...
category: programming
slug: /2018/05/31/ReactiveX-Observable/
template: post
---

보통 일반적인 프로그램의 경우에는, 하나씩 작성된 순서에 따라 로직이 실행되고, 완료되면 또다른 로직이 실행되는 등의 순서가 있음을 알수 있다.

그러나 이와 달리 ReactiveX는 "Observer"에 의해 임의의 순서에 따라 병렬적으로 실행되고 나중에 결과나 나온다.

즉 메서드를 호출하는 것이 아니라, Observable안의 데이터를 조회하고, 변환 하는 등의 프로세스를 정의한 후, Observable이 이벤트를 발생시키면 옵저버의 관찰자가 그 순간을 감지하고 준비된 연산을 실행하고 결과를 리턴한다. 그래서 Observable을 Subscribe한다는 표현을 쓴다.

일반적으로 Observable을 구현하기 위해서 다음과 같은 절차를 거친다.

1. 비동기 호출로 결과를 리턴받고, 필요한 동작을 처리하는 메서드를 정의한다.
2. Observable로 비동기 호출을 정의한다.
3. Subscribe를 통해 옵저버를 Observable에 연결 시킨다.
4. 메서드 호출로 결과가 리턴 될 때마다, 옵저버의 메서더는 리턴 값 또는 항목을 사용해서 연산을 한다.

`onNext` Observable은 새로운 항목을 push 할 때 마다 이 메서드를 호출한다. 이 메서드는 Observable 이 배출하는 항목을 parameter 로 전달 받는다.

`oNError` 기대하는 데이터가 생성되지 않았거나, 오류가 발생할 경우 호출된다. 이 경우 `onNext`, `onCompleted`는 호출되지 않는다.

`onCompleted` 마지막 `onNext`가 호출된 뒤에 호출된다.

더이상 구독을 하지 않으려면 `unSubscirbe`를 호출한다.

Observable이 연속된 항목을 push하는 방법에는 두가지가 있는데 바로 hot과 cold다. hot은 생성되자마자 push하고, cold는 옵저버가 구독할때까지 배출하지 않는다.

이러한 Observable은 연산자와 연결하여 사용할 수 있는데, 이런 연산자들은 리턴된 observable을 변경하여 제공할 수 있다.
