---
title: 'nodejs의 메모리 제한'
tags:
  - javascript
  - nodejs
published: true
date: 2021-12-13 19:21:45
description: ''
---

## V8 가비지 콜렉션

힙은 메모리 할당이 필요한 곳이고, 이는 여러 `generational regions`로 나뉜다. 이 `region`들은 단순히 `generations`이라고 불리우고, 이 객체들은 라이프 사이클 동안 같은 세대 (generation)을 공유한다.

여기에는 `young generation`과 `old generation`이 있다. 그리고 `young generation`의 `young objects`는 또다시 `nursery`(유아)와 `intermediate`(중간) 세대로 나뉜다. 이 객체들이 가비지 컬렉션에서 살아남게 되면, `older generation`에 합류하게 된다.

![generation](https://v8.dev/_img/trash-talk/02.svg)

이 `generation` 가설의 기본 원리는 대부분의 객체가 older로 넘어가기전에 죽는다 (가비지 콜렉팅 당한다)는 것이다.V8 가비지 컬렉터는 이러한 기본적인 가정을 기반으로 설계 되어 있으며, 여기에서 살아남은 객체만 승격하게 된다. 객체는 살아남으면서 다음 영역으로 복사되고, 그리고 결국엔 `old generation`이 되는 것이다.

node에서 메모리가 소비되는 영역은 크게 세군데로 볼 수 있다.

- code
- call stack: 숫자, 문자열, boolean 과 같은 primitive values 또는 함수
- heap memory

우리는 여기에서 힙 메모리를 중점적으로 볼 것이다.

가비지 콜렉터에 대해 간단히 알아봤으니, 힙에 메모리를 할당해보자.

```javascript
function allocateMemory(size) {
  // Simulate allocation of bytes
  const numbers = size / 8;
  const arr = [];
  arr.length = numbers;
  for (let i = 0; i < numbers; i++) {
    arr[i] = i;
  }
  return arr;
}
```

지역 변수는 함수 호출이 call stack에서 끝나는 즉시 `young generation`에 있다가 사라지게 된다. 숫자와 같은 기본형 변수들은 힙에 도달하지 못하고 대신 호출 스택에서 할당된다. `arr`의 경우 힙에 들어가서 가비지 콜렉션에서 살아남을 수 있다.
