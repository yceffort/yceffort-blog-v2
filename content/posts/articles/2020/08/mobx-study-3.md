---
title: MobX를 공부하자 (3) - 기본 개념과 원칙
tags:
  - javascript
  - MobX
published: true
date: 2020-08-25 20:07:31
description: 'MobX 1페이지 요약에 대한 간단한 번역'
category: MobX
template: post
---

[원본](https://mobx.js.org/intro/concepts.html)

# 개념과 원칙

## Table of Contents

## 개념

### 1. 상태 (State)

상태란 애플리케이션에서 파생되는 값이다. 일반적으로, 할일 목록 같은 도메인별 상태와 현재 선택된 엘리먼트를 나타내는 뷰 상태가 있다.

### 2. 파생 (Derivations)

더이상의 추가적인 상호작용없이, 상태에서 파생되어지는 값을 모두 파생 (Derivations) 이라고 한다.

> 최종적인 interaction 끝에 만들어진 값을 derivations 이라고 하는 것 같습니다.

여기서 Derivations은 다양한 것이 될 수 있다.

- 유저 인터페이스
- 남은 할일 숫자와 같이 파생되어진 데이터
- 서버에서 전송해온 백엔드 데이터

MobX는 derivations을 두종류로 구분한다.

- Computed values(계산된 값): 순수함수를 활용하여 현재 observable을 하고 있는 값들로 부터 계산되는 값
- Reaction: 상태가 바뀌면 자동으로 일어나야 하는 부수 효과. 이것은 명령형 프로그래밍과 반응형 프로그래밍 사이의 가교로서 필요하다. 이해를 더 쉽게 하기 위해서, I/O를 달성하기 위한 도구로써도 필요하다.

MobX를 처음 사용하는 초심자들은, 리액션을 너무 자주 사용하는 경향이 있다. 중요한 것은 바로 이것이다: **현재 상태를 기반으로 값을 계산하고 싶다면 `computed`를 활용하라.**

스프레드시트와 유사하게, 스프레드시트에서의 수식은 값을 계산해서 파생된 값이다. 그러나 사용자로서, 그러한 파생을 볼 수 있으려면 GUI의 일부를 다시페인팅 하는등의 리액션이 필요하다.

### 3. 액션 (Actions)

액션은 상태를 변화시키는 모든 코드 조각을 의미한다. 사용자 이벤트, 백엔드 데이터, 예약된 이벤트 등 액션은 스프레드시트 셀에 새 값을 입력하는 동작과 같다.

액션은 MobX에 명시적으로 정의되어있어서, 코드를 보다 명확하게 구성할 수 있다. MobX가 엄격모드로 동작될경우, MobX는 어떤 상태도 외부 액션으로 수정할 수 없도록 강제될 것이다.

## 원칙

MobX는 액션이 상태를 변경하는 단방향 데이터 흐름을 지원하고, 이에 영향을 받는 모든 뷰를 업데이트 한다.

![MobX Principles](https://mobx.js.org/assets/action-state-view.png)

- 모든 파생은 상태가 변경될때 자동으로 한번에 업데이트 된다. 따라서 중간에 값을 관찰하는 것은 불가능하다.
- 모든 파생은 기본적으로 동기로 업데이트 된다. 예를 들어, 액션을 통해 상태를 변경한 후, computed value를 직접 검사할 수 있다는 것을 의미한다.
- computed values는 게으르게 업데이트 된다. 현재 사용되지 않은 computed value는 부수효과에 필요할 때 까지 업데이트 되지 않는다. 만약 뷰에서 더이상 사용되지 않으면, 자동으로 가비지 콜렉팅 된다.
- 모든 computed values는 순수해야 한다. 이러한 값들이 상태를 바꾸면 안된다.

## 코드 예제

아래 예제는 위에서 언급한 기본 개념과 원칙을 묘사하고 있다.

```jsx
import { observable, autorun } from 'mobx'

var todoStore = observable({
  /* 관찰의 대상이되는 state */
  todos: [],

  /* 관찰의 대상에서 파생된 값 */
  get completedCount() {
    return this.todos.filter((todo) => todo.completed).length
  },
})

/* 상태값을 관찰하는 함수 */
autorun(function () {
  console.log(
    'Completed %d of %d items',
    todoStore.completedCount,
    todoStore.todos.length,
  )
})

/* 상태값을 수정하는 액션 */
todoStore.todos[0] = {
  title: 'Take a walk',
  completed: false,
}
// -> 동기적으로 콘솔에 로그를 찍는다. 'Completed 0 of 1 items'

todoStore.todos[0].completed = true
// -> 동기적으로 콘솔에 로그를 찍는다. 'Completed 1 of 1 items'
```
