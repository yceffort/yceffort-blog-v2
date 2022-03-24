---
title: '리액트 훅을 사용할 때 조심해야 할 것'
tags:
  - typescript
  - javascript
  - react
published: true
date: 2022-03-24 10:44:01
description: 'deps에 primitive 값만 사용하기, 훅을 컴포넌트에서 제거하기'
---

## Table of Contents

## Introduction

리액트에서 훅이 등장한 2018년 이래로, 리액트 커뮤니티에서는 함수형 컴포넌트 사용에 많은 탄력을 받았다. 훅을 사용하여, 함수형 컴포넌트의 상태 로직 (stateful logic)과 렌더링 로직을 매우 손쉽게 분리할 수 있게 되었다.

이 후 수년간 리액트에서 훅을 사용해오면서, 훅이 항상 편리함만을 제공해주는 것은 아니다. 모든 코드가 그렇지만 당연히 여기에도 위험성이 존재하며, 훅도 마찬가지다.

## 클로져

객체/함수지향 프로그래밍에 대해 잘못 알려진 사실 중 하나는, 객체 지향은 stateful하고, 함수지향은 stateless 하다는 것이다. 그리고 이 논쟁에 뒤따르는 사실 중 하나는, 상태는 보통 나쁜 것으로 치부되기 때문에 상태를 피하고, 더 나아가 객체 지향을 피해야 한다는 사실로 이어진다. 이 말 중 일부는 옳지만, 대부분의 진실이 그렇듯, 잘못된 사실도 있다.

`state`, 즉 상태란 무엇인가? 컴퓨터에서는 '계산을 한 값을 보관해두는 것'이라고 불리우는, 주로 메모리에 들어가 있는 값을 의미한다. 변수에 무언가를 저장할 때 마다 그 변수에 주어진 라이프 타임 동안 상태를 유지하게 된다. 그리고 프로그래밍 패러다임의 유일한 차이는, 이 변수를 얼마나 오래 보관해두느냐, 그리고 이 결정이 어떤 트레이드 오프를 가지고 오느냐 정도로 볼 수 있다.

아래 동일한 작업을 하는 함수형 코드와 객체지향 코드를 살펴보자.

```javascript
class Hello {
  i = 0
  inc() {
    return this.i++
  }
  toString() {
    return String(this.i)
  }
}
const h = new Hello()
console.log(h.inc()) // 1
console.log(h.inc()) // 2
console.log(h.toString()) // "2"
```

```javascript
function Hello() {
  let i = 0
  return {
    inc: () => i++,
    toString: () => String(i),
  }
}
const h = Hello()
console.log(h.inc()) // 1
console.log(h.inc()) // 2
console.log(h.toString()) // "2"
```

여기에서 메모리를 유지하는 메커니즘 (`i`) 은 많은 공통점을 가지고 있다. 클래스는 객체의 인스턴스를 참조하는 `this`를 사용하는 방식으로, 함수형은 범위내 모든 변수를 기억하는 클로져를 활용하는 방식으로 이 기능을 구현했다.

클로져는 함수를 `stateful`하게 만들어 주기 때문에 매우 중요한 개념이라 볼 수 있다. 그러나 클로져의 한가지 중요한 문제점이라고 한다면, 메모리 누수가 쉽게 일어난다는 점이다. 함수가 스코프를 넘어서도 살아있을 수 있기 때문에, 가비지 콜렉터가 이를 수집할 수가 없게 된다. 위 예제에서는, `inc`가 존재하는한, `i`는 가비지 콜렉팅 당하지 않을 것이다.

클로져에서 또한가지 조심해야 할 것은, 명시적 의존성을 암묵적인 의존성으로 바꿔버린다는 것이다. 함수에 인수를 넘겨주면, 그 함수의 의존성은 명시적이라고 볼 수 있지만, 프로그램이 이 클로저가 무엇에 의존성을 가지고 있는지는 알 수 없게 된다. 즉, 클로저가 메모리에 보관하는 값은 호출에 따라서 변화할수도, 그 결과에 따라 다른 값을 만들어 버릴 수도 있다.

## 클로져와 훅

```javascript
function User({ user }) {
  useEffect(() => {
    console.log(user.name)
  }, []) // exhaustive-deps

  return <span>{user.name}</span>
}
```

훅에 있는 개념 중 하나는, 의존성에 변화가 있을 때마다 (`dependencies`) 부수효과가 발생한다는 것이다. 예를 들어, `useEffect`는, 엑셀 시트처럼, 부수효과에 필요한 입력값이 달라지는 경우에만 실행된다. `useMemo` `useCallback`도 마찬가지다.

훅은 예제의 `user` 처럼, 해당 스코프에서 정보를 보고 유지할 수 있기 때문에 클로져의 이점을 누릴 수 있다. 그러나, 이러한 종속성이 암묵적이어서, 이 사이드 이펙트가 언제 실행되어야 하는지 알 수 없다.

클로져는 훅 api에 일련의 `dependencies`가 필요한 이유다. 이 결정은 프로그래머가 이러한 암묵적인 의존성을 명확히 하는 책임을 지도록 강요하고, 따라서 일종의 '휴먼 컴파일러'로서 기능한다. dependencies를 선언하는 것은 수동으로 하는 보일러 플레이트 작업이며, C 메모리 관리와 같이 오류가 발생하기 십상이다.

이 문제를 해결하기 위한 리액트의 해결책은 `eslint` 이지만, 리액트 훅을 커스텀 훅으로 구성하면 문제가 또 발생한다.

이 문제를 완전히 피할 수 있는 방법은, 훅을 컴포넌트 외부로 이동시키는 것이다. 이렇게 한다면, 의존관계로 사용할 수 있는 인수를 강제적으로 건내받을 수 있게 된다.

```javascript
const createEffect =
  (fn) =>
  (...args) =>
    useEffect(() => fn(...args), args)
const useDebugUser = createEffect((user) => {
  console.log(user.name)
})

function User({ user }) {
  useDebugUser(user)

  return <span>{user.name}</span>
}
```

훅을 클로져 외부로 이동시키면, dependencies를 수동으로 추적하거나 subscription이 부족해지는 문제 (`dependencies`가 모자른)에 직면할 필요가 없다. 그러나 여전히 리액트와 자바스크립트가 두 종속성이 같은지를 판단하는 문제에 대해서는 여전히 취약하다.

## 동일성과 메모리

동일성이라고 한다면, 변하지 않은 것 이라고 이해하 면 쉽다. 예를 들어, 3은 언제나 3이다. 자바스크립트에서는, 이러한 동등 비교를 실행할 수 있는 여러가지 방법이 있다. 예를 들어, `==` `===` `Object.is`는 완전히 다른 방식이며, 각 다른 결과를 얻을 수 있다. `Object.is`의 경우, 연산한 값이 같은지 확인한다.

- 두 개가 `undefined` 인지
- 두 개가 `null`인지
- 두 개가 `true` `false` 인지
- 두 개가 `+0` 인지
- 두 개가 `-0` 인지
- 두 개가 `NaN`인지
- 혹은 0, `NaN`이 아니며 같은 값을 가지고 있는지
- 문자열의 경우, 크기와 구성하고 있는 글자가 같은 순서인지
- 나머지 non-primitive의 경우, 이들은 mutate하기 때문에, 메모리 참조가 같은지 비교한다. 이는 일반적인 개발자의 직관과 다르다. `Object.is([], [])`는 두 배열 객체의 메모리 포인터가 다르므로 `false` 가 나온다.

> [MDN 문서](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Object/is#%EC%84%A4%EB%AA%85) 참고

## 훅과 동일 비교

훅은 dependencies를 비교할 때 `Object.is`를 사용한다. 따라서 의존성을 비교했을 때, 이 둘이 다를 때만 실행 된다. 여기서 같다는 것은 `Object.is`를 사용하여 비교한다.

```javascript
const User({ user }) {
  useEffect(() => {
    console.log(`hello ${user.name}`)
  }, [user]) // eslint barked, so we added this dependency

  return <span>{user.name}</span>
}
```

위 컴포넌트에서, `useEffect`는 얼마나 실행될까? 알 수 없다. `user`가 달라지는 횟수 만큼 실행될 것이다. `user`가 메모리에 어떻게 할당되었는지 모른다면, 이 객체가 어떻게 동일 비교를 할 수 있는지 알 수 없다. 즉 이 코드는 동작할 수 있지만, 올바르지 않으며 상위컴포넌트에서 변경이 일어나면 완전히 망가질 수도 있다.

```javascript
function App1() {
  const user = { name: 'paco' }

  return <User user={user} />
}

const user = { name: 'paco' }
function App2() {
  return <User user={user} />
}
```

위 예제에서, 우리는 훅의 미묘한 부분을 알 수 있다.

`App1`은 매번 새로운 객체를 할당한다. 이 객체는 개발자가 볼 때는 항상 동일하지만, `Object.is`가 볼 때는 그렇지 않다. 즉, 이 컴포넌트는 렌더링 할 때마다 `useEffect`가 실행될 것이다.

`App2`는 항상 같은 객체 포인터를 참조한다. 즉, 렌더링 횟수에 관계없이 사이드 이펙트가 실행되는 것은 한번 뿐이다.

실제 코드는 이보다 훨씬 더 복잡하기 때문에, 개발자는 객체가 언제, 얼마나 할당되는지 이해하기 쉽지 않다.

이번엔 실제 프로덕션에서 사용될 법한 코드를 살펴보자.

```javascript
function App({ options, teamId }) {
  const [user, setUser] = useState(null)
  const params = { ...options, teamId }

  useEffect(() => {
    fetch(`/teams/${params.teamId}/user`)
      .then((response) => response.json)
      .then((user) => {
        setUser(user)
      })
  }, [params])

  return <User user={user} params={params} />
}
```

위 예제는 동일한 요청을 반복적으로 시도할 것이다. 객체를 재구성하게 되면, 렌더링 시마다 새로운 객체를 할당하므로 `useEffect`의 dependencies로 사용하기에는 부적절하다.

## 결론

훅은 다른 기술과 마찬가지로, 새로운 기술로서의 일종의 과대 광고 효과를 어느정도 누렸다고도 볼 수 있다. 많은 개발자가 상태 관리 솔루션 대신, 상태 저장 로직을 구현하기 위해 훅을 채택했다. API는 쉬워 보이지만, 그 내부 동작은 복잡하기 때문에 부정확할 위험이 높아진다.

대부분의 버그는 컴포넌트에서 훅을 떼고, 유일한 의존관계로 primitive 를 사용하여 해결할 수 있다. 타입스크립트를 사용하는 경우, 자체 훅을 만들어 엄격하게 입력하고 관리할 수 있다. 이를 통해 개발자들이 훅이 가진 한계를 이해하는데 큰 도움이 될 수 있다.

```typescript
type Primitive = boolean | number | string | bigint | null | undefined
type Callback = (...args: Primitive[]) => void
type UnsafeCallback = (...args: any[]) => void

const createEffect =
  (fn: Callback): Callback =>
  (...args) => {
    useEffect(() => fn(...args), args)
  }

const createUnsafeEffect =
  (fn: UnsafeCallback): UnsafeCallback =>
  (...args) => {
    useEffect(() => fn(...args), args)
  }
```
