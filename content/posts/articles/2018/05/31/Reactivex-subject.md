---
title: ReactiveX) Subject
date: 2018-05-31 09:43:40
published: true
tags:
  - programming
description: subject는 옵저버나 observable처러 행동하는 일부 ReactiveX구현체에서만 사용가능한 일종의 프록시다.
  subject는 옵저버이기 때문에 하나이상의 observable을 구독할 수 있으며, 동시에 observable 이기도 하기 때문에 항목을
  하나하나 거치면서 다시 push하고 새로운 항목을 push할 수 있다. 총 4종류의 subject...
category: programming
slug: /2018/05/31/Reactivex-subject/
template: post
---
subject는 옵저버나 observable처러 행동하는 일부 ReactiveX구현체에서만 사용가능한 일종의 프록시다. subject는 옵저버이기 때문에 하나이상의 observable을 구독할 수 있으며, 동시에 observable 이기도 하기 때문에 항목을 하나하나 거치면서 다시 push하고 새로운 항목을 push할 수 있다.

총 4종류의 subject가 있다.

### AsyncSubject
![](http://reactivex.io/documentation/operators../../../images/S.AsyncSubject.png)

Observable이 마지막으로 push한 값만 push 하고, 원 Observable의 동작이 다 끝나면 동작한다. 아무값도 push되지 않으면, 이 subject 역시 배출하지 않는다. 또한 맨 마지막 값 바로 뒤에 오는 옵저버에도 값을 전달하는데, 만약 오류에 의해 종류될 경우 이 오류를 그냥 전달한다.

### BehaviorSubject
![](http://reactivex.io/documentation/operators../../../images/S.BehaviorSubject.png)
옵저버가 BehaviorSubject를 구독하면, 옵저버는 Observable 이 가장 최근에 발행한 항목 (또는 값이 없을 경우 맨처음 값이나 기본값) 을 push하며, 이후 Observable 이 push한 값을 push한다.

### PublishSubject
![](http://reactivex.io/documentation/operators../../../images/S.PublishSubject.png)
Subscribe 이후에 push한 항목에 대해서 모두 Observable에게 배출한다. 하지만 이 때문에 subject가 생성되는 시점과 구독하는 시점사이에 빈 공간이 생긴다는 단점이 있다. 따라서 모든 push하는 모든 항목을 받기 위해서는 cold observable을 생성하거나, 아래의 subject를 사용해야 한다.

### RelaySubject
![](http://reactivex.io/documentation/operators../../../images/S.ReplaySubject.png)
PublishSubject 와 다르게, 구독한 시점에 상관없이 Observable이 push한 모든 항목을 받는다. 다만 재생버퍼의 크기가 특정이상으로 증가할 경우에는, 처음 배출후 지정한 시간이 경과하면 오래된 항목을 제거한다. 또한 onNext사용을 주의해야한다. (순서의 모호함이 있기 때문)
