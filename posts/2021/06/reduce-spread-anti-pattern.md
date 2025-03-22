---
title: 'reduce에 spread 를 쓰면 안되는 이유'
tags:
  - javascript
published: true
date: 2021-06-22 17:35:37
description: '솔직히 뭔가 멋있어서 많이 쓰긴 함'
---

자바스크립트 프로젝트를 개발하다보면, 배열을 하나의 객체로 묶어야될 필요성이 생길 때가 있다. 아래 예제를 들어보자.

```javascript
const items = [
  {name: 'obama', age: 10, country: 'USA'},
  {name: 'trump', age: 14, country: 'USA'},
  {name: 'moon', age: 37, country: 'KOREA'},
  {name: 'clinton', age: 64, country: 'USA'},
  {name: 'bush', age: 49, country: 'USA'},
]
```

이렇게 배열로 되어있는 것을 이름을 키로 하는 객체로 바꾼다고 한다면, 아마 다들 [reduce](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Array/Reduce)를 생각할 것이다.

```javascript
const result = items.reduce((prev, item) => {
  prev[item.name] = {age: item.age, country: item.country}
  return prev
}, {})

// {
//   obama: { age: 10, country: 'USA' },
//   trump: { age: 14, country: 'USA' },
//   moon: { age: 37, country: 'KOREA' },
//   clinton: { age: 64, country: 'USA' },
//   bush: { age: 49, country: 'USA' }
// }
```

혹은 for loop를 활용하여 똑같이 할 수 있다.

```javascript
const result2 = {}
for (let item of items) {
  result2[item] = {age: item.age, country: item.country}
}
```

둘 중에 뭐가 더 읽기 좋은 코드냐고 묻는다면, 당연히 후자일 것이다. 내가 아무리 `reduce`를 좋아한다고 해도 그건 반박 불가능한 사실이다. 그러나 요즘 리액트 커뮤니티의 성장으로 인해 함수형 프로그래밍 스타일로 코드를 작성하는 일이 잦아짐에 따라서 reduce를 쓰는일이 많아지고 있다.

그러나 문제는 아래의 스타일이다.

```javascript
const result3 = items.reduce(
  (prev, item) => ({
    ...prev,
    [item.name]: {age: item.age, country: item.country},
  }),
  {},
)
```

> 이제부터 이런 코드를 `reduce...spread`라고 부르겠다.

이 spread operator는 [Object.assign](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Object/assign)과 유사하게 동작하는데, 소스 객체의 속성을 순환하면서 각 속성을 타켓의 객체에 하나씩 복사해 넣는다. 위 코드의 경우에는, 새로운 객체를 계속해서 복사하는 것이다. 위 코드의 문제는 무엇일까?

위코드의 동작을 살펴본다면, 대충 아래와 같을 것이다.

```bash
# 첫번째
result = {}

{...result, 'obama': {age: 10, country: 'USA'}}

# 두번째
result = {'obama': {age: 10, country: 'USA'}}

{...result, 'trump': {age: 14, country: 'USA'}}

# 세번째

result = {'obama': {age: 10, country: 'USA'}, 'trump': {age: 14, country: 'USA'}}

{...result, 'moon': {age: 37, country: 'KOREA'}}

##....
```

> 의사 코드이기 때문에 대충 보면 된다.

루프가 반복될때마다, 모든 루프에서 정확히 n회 실행되는 것은 아니지만, 이렇게 될 경우 $$O(n^2)$$라 볼 수 있다.

바벨이나 타입스크립트 트랜스파일러를 사용하면 해당 코드를 어떻게 변환할까?

그전에 먼저, [tc39에 나와있는 객체 전개 연산자의 스펙](https://github.com/tc39/proposal-object-rest-spread/blob/master/Spread.md)을 살펴보자.

```javascript
let aClone = {...a}
let aClone = Object.assign({}, a)
```

위 구문도 거의 동일한 작업을 수행하므로, 전개 연산자와 비슷한 시간 복잡성을 가질 것이다. 즉, 새 객체를 생성한다음, 나머지 객체의 키를 새객체에 반복해서 복사하는 것이다.

- [babel transpile](https://babeljs.io/repl#?browsers=defaults%2C%20not%20ie%2011%2C%20not%20ie_mob%2011&build=&builtIns=false&corejs=false&spec=true&loose=true&code_lz=MYewdgzgLgBAllApgWwjAvDA2gKBjAbzAENlEAuGAchACNTiqAaGYgcwpgEYAGF0AK5goAJwCelKgFUAygEEqAXyZ5CJMpNEDkAB2asOlLgBZ-IIaInVZC5aqKlOVZCHD72nAMwB2MxfGSANIA8gBKAKK2KvgOGtTAADZwwm4sHpQAbKYwgsIB1vJKLPbqTrQCEAAW7oYwxgCcfnlW0oXKMDgAujg4oJCwIogQAglQnhjwSKgAdIMAJgLAiAAUyzqDAG4sCCgAlBgAfDDLBKrT5-uIW9g7yNOlnZQE6ZMo0x5NlpS307mWijhFLsWAQgUA&debug=false&forceAllTransforms=true&shippedProposals=false&circleciRepo=&evaluate=true&fileSize=false&timeTravel=false&sourceType=script&lineWrap=true&presets=env%2Creact&prettier=false&targets=&version=7.14.3&externalPlugins=)
- [typescript](https://www.typescriptlang.org/play?#code/MYewdgzgLgBAllApgWwjAvDA2gWAFAwwDeYAhsogFwwDkIARuaTQDQykDmVMAjAAxtQAVzBQATgE9qNAKoBlAII0Avi3yES5bjXFDkAB1bsu1HgBZBIEeKm15S1euJkK05CHBHO3AMwB2S2tJaQBpAHkAJQBRBzUCZy1pYAAbOFFPNm9qADYLGGFRYLtFFTYnTVdaeiEIAAsvExgzAE5AwttZEtUYfABdfHxQSFgxRAghZKgfDHgkVAA6UYATIWBEAAp1-VGANzYEFABKDAA+GHWiJ3nr7cQ97APkeZdEXuoiLNmUee82m2pHvMCjZlPhlIc2ERwUA)

따라서 reduce에서 전개연산자로 합성해서 리턴하는 것은 굉장히 느린 코드라고 볼 수 있다.

여담으로, 이와 관련된 키보드 배틀(?) 이 트위터에서 열린 적이 있다.

https://twitter.com/fildon_dev/status/1396252890721918979

## 문제의 코드

```javascript
const users = [
  {id: 1, name: 'Tony Stark', active: false},
  {id: 2, name: 'Bruce Banner', active: true},
  {id: 3, name: 'Natasha Romanoff', active: false},
  {id: 4, name: 'Chris Evans', active: true},
  {id: 5, name: 'Chris Hemsworth', active: false},
  {id: 6, name: 'Clark Gregg', active: false},
]

// good?
const result1 = users.reduce((acc, curr) => {
  if (curr.active) {
    return acc
  }
  return {...acc, [curr.id]: curr.name}
})

// bad?
const result2 = users
  .filter((user) => !user.active)
  .map((user = (user) => ({[user.id]: user.name})))
  .reduce(Object.assign, {})
```

그래서 뭐가 더 성능이 좋고 빠를까? 저 두개가 최선의 방법일까? 만약에 나라면,,,

🤔
