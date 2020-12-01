---
title: '왜 moment 는 deprecated 되었을까'
tags:
  - javascript
published: true
date: 2020-12-01 23:43:45
description: '👋👋'
---

Datetime을 다루는 것은 분명 쉬운 일은 아니다. 한참 vanilla 자바스크립트에 취해 있을 때, least library challenge(?)의 일환으로 datetime을 내재화 해서 관리하곤 했지만 이는 분명 어려운 일이었다. 그 때 마다 결국 가장 익숙한 moment로 돌아와서 하곤 했는데, 언젠가 bundle 분석을 한 뒤로 moment에 대한 사용을 조금 꺼리기 시작했다. moment는 내가 쓰는 기능 대비 정말 큰 용량을 차지하고 있었다. (timezone을 제외 하더라도) 분명 내가 moment의 좋은 기능을 쉽게 사용하는 면도 있었지만, 조금이라도 더 빠른 애플리케이션을 사용하기 위해서는 moment 외에 다른 대안이 필요했다.

그러던 중, [moment 가 deprecated 된다는 소식을 알려왔다.](https://momentjs.com/docs/)

> ... The modern web looks much different these days. Moment has evolved somewhat over the years, but it has essentially the same design as it did when it was created in 2011. Given how many projects depend on it, we choose to prioritize stability over new features.

조금 많이 뒷북이긴 하지만, 왜 moment가 역사 속으로 사라졌는지 몇가지 이유를 짚고 넘어가고자 한다.

## 1. 느리다

단도직입적으로 그래프를 통해서 알수 있다. 다른 datetime library 대비 속도가 많이 느렸다.

![speed1](https://raygun.com/blog/wp-content/uploads/2017/09/image4-2.png)

![speed2](https://raygun.com/blog/wp-content/uploads/2017/09/image3.png)

![speed3](https://raygun.com/blog/wp-content/uploads/2017/09/image1.png)

출처: https://raygun.com/blog/moment-js-vs-date-fns/

뭐 여러가지 이유가 있겠지만, regex를 주로 쓰는 moment에 대비 다른 라이브러리들은 `Z`로 끝나면 `new Date(string)`을 쓴다던지, 혹은 느린 regex 대신에 자체적으로 개발한(?) `if` 와 `charAt` 등을 쓴다던지 다양한 노력들을 하고 있었다. regex를 파싱해서 이해하는 작업은 확실히 느리다.

## 2. 무겁다

![size-of-datetime-libraries](./images/size-of-datetime-libraries.png)

출처: https://inventi.studio/en/blog/why-you-shouldnt-use-moment-js

https://github.com/jmblog/how-to-optimize-momentjs-with-webpack

기본적으로 momentjs는 232kb, (gzip시 66kb) 이며, webpack으로 locale을 제거할 경우 사이즈는 68kb (gzip시 23kb) 까지 떨어진다. 그리고 더이상의 tree shaking은 불가능하다. js-joda가 제법 크긴 하지만 기간과 타임존까지 기본으로 제공하는 라이브러리라는 것을 알아둬야 한다. 그리고 나머지 라이브러리들은 트리쉐이킹이 가능하다.

## 3.mutable이다.

이는 moment 공식 가이드에서도 언급한 문제다.

> As an example, consider that Moment objects are mutable. This is a common source of complaints about Moment. We address it in our usage guidance but it still comes as a surprise to most new users. Changing Moment to be immutable would be a breaking change for every one of the projects that use it. Creating a "Moment v3" that was immutable would be a tremendous undertaking and would make Moment a different library entirely. Since this has already been accomplished in other libraries, we feel that it is more important to retain the mutable API.

https://inventi.studio/en/blog/why-you-shouldnt-use-moment-js

```javascript
const startedAt = moment()
const endedAt = startedAt.add(1, 'year')

console.log(startedAt) // > 2020-02-09T13:39:07+01:00
console.log(endedAt) // > 2020-02-09T13:39:07+01:00
```

`moment`를 조작하는 모든 method 들은 리턴 값과 참조값 모두를 바꿔 버리기 때문에, 에러를 만들 소지가 높다.

## 4. 디버깅이 어렵다.

`moment`안에 파라미터를 넣는 것은 좋은 아이디어이긴하지만, 그 안에 따라서 동작이 매우 일관적이지 못하다. 예를 들어, moment 안에 잘못된 값을 넣었을 경우 에러가 나는게 아니라 그냥 현재 시간이 나와버릴 수도 있다.

```javascript
moment().format() // > 2019-02-08T17:07:22+01:00
moment(undefined).format() // > 2019-02-08T17:07:22+01:00
moment(null).format() // > Invalid date
moment({}).format() // > 2019-02-08T17:07:22+01:00
moment('').format() // > Invalid date
moment([]).format() // > 2019-02-08T17:07:22+01:00
moment(NaN).format() // > Invalid date
moment(0).format() // > 1970-01-01T01:00:00+01
```

요약하자면, `undefined`는 가능하지만, `null` `''`, `NaN`은 안된다.

## 결국

lighthouse에서도 에러가 뜨고

![lighthouse warning](https://pbs.twimg.com/media/EhM0XE3UwAA2Co5?format=jpg&name=medium)

https://twitter.com/addyosmani/status/1304676118822174721

moment를 최적화 하는 방법까지도 알려지기 시작했다.

https://github.com/GoogleChromeLabs/webpack-libs-optimizations#moment

## 대안

사이즈가 중요한 프론트엔드의 경우 `date-fns`나 `day.js`가 좋다. 그 외의 경우에는 기능이 가장 리치한 `js-joda`를 사용하는 것이 좋다.

|          | size   | size(gzip) | speed(to) | tree-shaking | immutable | throw error | timezone |
| -------- | ------ | ---------- | --------- | ------------ | --------- | ----------- | -------- |
| moment   | 232/68 | 66/26      | 16.527    | X            | X         | X           | O        |
| day.js   | 6      | 3          | 9.129     | X            | O         | X           | X        |
| luxon    | 64     | 18         | 15.406    | X            | O         | O           | O        |
| js-joda  | 208    | 39         | 11.397    | X            | O         | O           | O        |
| date-fns | 30     | 7          | 5.175     | O            | O         | X           | X        |
| native   |        |            | 1.297     |              | X         | X           | X        |

출처: https://inventi.studio/en/blog/why-you-shouldnt-use-moment-js#fnref2

그럼에도, 저렇게 큰 라이브러리 자체를 deprecated 시킬 수 있다는 것 만으로도 자바스크립트 생태계가 건강하게 나아가고 있다는 방증인 것 같다. 여전히, 많은 수의 프로젝트가 moment에 의존하고 또 그 편리함에 많은 도움을 얻었다. moment가 이야기 한 `modern web looks much different these days` 처럼, 이제는 다른 라이브러리를 쓸 때가 왔다. 그리고 나 또한, 오래되고 낡은 코드를 과감하게 deprecated 시킬 용기가 필요하다.
