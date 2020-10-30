---
title: 'JSX에서 && 대신에 3항 연산자를 더 선호하는 이유'
tags:
  - javascript
  - react
published: true
date: 2020-10-30 23:51:20
description: '사실 그냥 (몇 가지 합리적인 이유가 있는) 개인적인 취향임'
---

다른 사람들이 쓴 리액트 코드를 볼 때 마다 몇가지 눈여겨 보는게 있는데, 그 중 하나가 jsx안의 조건절이다. 이 사람은 `&&`을 선호나는지, 아니면 3항연산자를 선호하는지 살펴본다. 근데 보통은 간단해서 그런지 `&&`를 쓰는 경우가 더 많은 것 같다. 예전에는 이 것과 관련해서 코드 리뷰를 올려볼까도 헀는데, 멀쩡히 작동하는 코드인데 괜히 시비 거는 것 같아서, 고칠 코드도 많아서, 그리고 결정적으로 소심해서 그냥 그냥 넘어가고는 했다. 그렇다. 이 글은 그냥 소심한 반항인 것이다.

```jsx
export default function StudentsList({ students }) {
  return (
    <ul>
      {students.map(({ id, name, score }) => (
        <li key={id}>
          {name} {score}
        </li>
      ))}
    </ul>
  )
}
```

여기에 이제 `&&`로 다음과 같은 조건을 추가했다고 생각해보자.

```jsx
export default function StudentsList({ students }) {
  return (
    <ul>
      {students.length &&
        students.map(({ id, name, score }) => (
          <li key={id}>
            {name} {score}
          </li>
        ))}
    </ul>
  )
}
```

만약 `students`에 `[]` 빈 배열이 온다면 렌더링이 안되어야 할 것 같지만, 실제로는 `0`이 프린트 된다.

이 버그는 예전 회사에서도 봤던 버그고, 심지어 페이팔에서도 이러한 버그가 있었다고 한다.

![paypal-bug](https://kentcdodds.com/static/330366840b58941a34169d30db87884b/e3189/no-contacts.png)

자바스크립트에서 `0`은 `falsy`한 값으로 취급되기 때문에, `&&` 우측에 있는 값은 계산하지 않는다. 근데 어디까지나 `falsy`한 값인 거지, `null`이나 `undefined`처럼 렌더링을 안하는 값은 아니기 떄문에 (순수하게 빈값으로 계산되지는 않으므로) 0이 나오게 된다.

![terrible](https://upload.inven.co.kr/upload/2014/12/23/bbs/i3324974586.jpg)

해결책은, `students.length === 0 && ...`을 쓰거나, 삼항연산자로 바꾸면 된다.

```jsx
export default function StudentsList({ students }) {
  return (
    <ul>
      {students.length
        ? students.map(({ id, name, score }) => (
            <li key={id}>
              {name} {score}
            </li>
          ))
        : null}
    </ul>
  )
}
```

아래와 같은 코드는 어떤가?

```jsx
// students가 undefined로 넘어왔다면?
export default function StudentsList({ students }) {
  return (
    students && (
      <ul>
        {students.map(({ id, name, score }) => (
          <li key={id}>
            {name} {score}
          </li>
        ))}
      </ul>
    )
  )
}
```

다음과 같은 에러가 날것이다.

```bash
Nothing was returned from render. This usually means a return statement is missing. Or, to render nothing, return null.
```

(대충 또 끔찍하다는 이미지)

이 경우에는 `undefined && ...` 이 되기 때문에, `undefined`가 리턴되고, 위와 같은 에러가 뜨게 된다.

```javascript
0 && true // 0
true && 0 // 0
false && true // false
true && '' // ''
[] && true // true
```

이 처럼 `&&`는 자주 쓰는 `if`문 과 동작이 미묘하게 다르다. 그래서 이게 자바스크립트 퀴즈에 종종 나오곤한다.

```javascript
const notifications = 1

console.log(
  `You have ${notifications} notification${notifications !== 1 && 's'}`,
)
```

> 출처: https://quiz.typeofnan.dev/short-circuit-notifications/ `&&`은 정식 명칭으로 `Short-circuit evaluation`이다.

지금까지 말한 예제와는 조금 다르지만, 이것도 마찬가지로 의도 대로 동작하지 않는다.

```bash
You have 1 notificationfalse
```

## 그래서

명백하게 조건에 따라서 jsx에서 렌더링을 하고 싶지 않다면 삼항연산자를 썼으면 좋겠다.

- 우리가 이해하고 있는 `if`문 동작과 일치한다. 
- 불필요한 버그나, 의도치 않은 동작 만들지도 않는다. 
- jsx에 `null`이 명시적으로 표시되어 있으면 다른 개발자들도 이 부분에서 렌더링이 안되는 경우가 있다는 것을 명시적으로 알 수 있고 이를 유념할 수 있다.
- 삼항연산자가 더 이쁘다. 👀

