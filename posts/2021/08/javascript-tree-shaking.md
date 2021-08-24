---
title: '트리쉐이킹으로 자바스크립트 사이즈 줄이기'
tags:
  - web
  - javascript
  - browser
published: true
date: 2021-08-24 12:47:40
description: '트리쉐이킹은 직접 해드세요 제발'
---

## Table of Contents

## Introduction

오늘날의 웹 애플리케이션의 크기는 꽈거에 비해 꽤 커졌다. 특히, 자바스크립트의 비중이 그렇다. [http archive](https://httparchive.org/reports/state-of-javascript#bytesJs)의 자료를 보면, 자바스크립트 크기의 중위값을 본다면 데스크톱은 476.4KB, 모바일의 경우 439.0kb 정도로 무시할 수 있는 수준이 아니다. 그리고 이는 단순히 transfer 기준인 걸 알아야 한다. 일반적으로 네트워크를 오갈 때는 압축된 번들이 온다는 것을 고려해 봤을 때, 실제 압축이 해제된 크기를 본다면 더 클 것이다. 리소스를 처리할 때는 압축이 되지 않은 파일을 기준으로 하기 때문에 우리는 이점을 잘 기억해둬야 한다. 압축된 300kb의 자바스크립트 번들은, 압축 해제시 약 900kb 정도가 될 것이고, 이는 파서와 컴파일러에 900kb 만큼의 부담이 갈 것이다.

그리고 자바스크립트는 처리하는데 많은 비용이 드는 리소스다. 다운로드 후 비교적 가벼운 디코딩 시간만 소요되는 이미지와는 다르게, 자바스크립트는 파싱도 해야하고, 컴파일도 해야하고, 그리고 마지막으로 실행도 되어야 한다. 즉, 다른 리소스에 비해 자바스크립트는 _비싼_ 리소스다. [자바스크립트 엔진의 효율성을 개선하기 위한 작업](https://v8.dev/blog/background-compilation)이 지속적으로 이뤄지고 있지만 자바스크립트의 성능 향상 작업은 어디까지나 개발자의 몫이다.

이를 위해 자바스크립트의 성능을 향상시키는 다양한 기술들이 있다. [Code Splitting](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking#:~:text=improve%20JavaScript%20performance.-,Code%20splitting,-%2C%20is%20one%20such)과 같이 애플리케이션 자바스크립트를 청크로 분할하고, 이러한 청크를 필요한 애플리케이션 경로에만 제공하여 성능을 향상시킬수도 있다. 이 기술도 제법 괜찮지만, 자바스크립트가 많이 사용되는 애플리케이션의 일반적인 문제, 사용하지 않는 코드가 포함될 수도 있다. 이 문제를 해결하기 위한 것이 바로 트리쉐이킹이다.

## Tree shaking?

[Tree shaking](https://en.wikipedia.org/wiki/Tree_shaking)은 사용되지 않는 코드를 제거하는 기법을 의미한다. 이 용어는 [Rollup](https://github.com/rollup/rollup#tree-shaking) 덕분에 유명세를 타긴했지만, 사용되지 않는 코드를 제거한다는 개념은 원래도 존재하고 있었다. 그리고 이 개념은 [Webpack](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking#:~:text=found%20purchase%20in-,webpack,-%2C%20which%20is%20demonstrated)에서도 소개되었다.

트리쉐이킹이라는 용어는 애플리케이션을 일종의 나무와 같은 구조로 보는 대에서 유래되었다. 트리의 각 노드는 앱에 고유한 기능을 제공하는 종속성을 나타낸다. 최신 애플리케이션에서는, 다음과 같은 `import`를 활용하여 이러한 디펜던시 (종속성)을 가져온다.

```javascript
// Import all the array utilities!
import arrayUtils from 'array-utils'
```

애플리케이션 초기단계에서는 이러한 디펜던시가 상대적으로 적을 수도 있다. 그리고 처음에는 `import`했을 때 모든 디펜던시를 사용했을 수도 있다. 그러나 애플리케이션이 점점 커질 수록 이 디펜던시도 같이 커지게 된다. 그리고 시간이 지날 수록 사용하지 않는 디펜던시가 제거되지 않는 경우도 생기게 된다. 이러한 문제를 해결하기 위해, 트리쉐이킹은 static import 문을 사용하여 ES6 모듈의 특정 부분만을 가져오는 방법을 사용한다.

```javascript
// Import only some of the utilities!
import { unique, implode, explode } from 'array-utils'
```

위 `import`문과 차이점이라고 한다면, 모든 것을 `import` 하는 이전 코드와는 다르게 여기에서는 딱 필요한 method들만 `import`했다는 것이다. 이 코드가 dev build에서는 어차피 모든 모듈을 가져오기 때문에 실질적으로 변화가 일어나지 않는다. 그러나 프로덕션 빌드에서는 명시적으로 가져오지 않는 es6모듈에 대해 `export`를 `shake` 하도록 웹팩을 구성하여 프로덕션 빌드를 더 작게 만들 수 있다.

### 트리 쉐이킹을 할 수 있는지 확인해보기

[이 예제 저장소](https://github.com/malchata/webpack-tree-shaking-example)로 웹팩에서 어떻게 트리쉐이킹이 일어나는지 확인 해보고자 한다. 이 애플리케이션은 간단한 데이터베이스에서 검색을 하는 기능을 제공하고 있다. 쿼리를 입력하면, 제품목록이 뜬다.

그리고 이 애플리케이션의 자바스크립트 코드는 벤더 코드 (`Preact`, `Emotion`)와 애플리케이션 코드 번들 (청크)로 나눠져 있다.

```bash
                 Asset        Size  Chunks             Chunk Names
js/vendors.a3722bf0.js    37.1 KiB       0  [emitted]  vendors
js/main.951b863a.js       20.8 KiB       1  [emitted]  main
```

번들크기가 21.1kb로 큰 편은 아니다. 그러나 이는 트리쉐이킹이 되지 않았다는 점을 알아야 한다.

애플리케이션 코드의 `FilterablePedalList` 컴포넌트에서 아래와 같은 코드를 확인할 수 있다.

```javascript
import * as utils from '../../utils/utils'
```

아마 이런 코드를 어디에선가 본적이 있을 것이다. 이 같이 `import`하는 것은 주의할 필요가 있다. 이 뜻은 `../../utils/utils`에 있는 것을 모두 `utils` 네임스페이스에 저장하라는 뜻이다. 여기서 중요한 것은 저 모듈에 얼마나 많은 모듈이 있는가이다.

확인해보니 약 1300줄의 코드가 있는 것을 확인해 볼 수 있다. 뭐 물론, 그렇다고 이게 꼭 잘못되었다고만 할 순 없다. 저기에 있는 모든 모듈을 쓰고 있다면 불가피한 선택이었을 수도 있다. 그러나 실제로 저 컴포넌트에서 쓰고 있는 `utils` 모듈은 고작 3개 뿐이다.

물론, 이 예시가 극단적인 케이스이긴 하다. 그러나 이러한 가상 시나리오가 실제 애플리케이션 코드에서 발견할 수 있는 최적화 사례와 유사하다는 사실은 분명하다. 이제 이를 트리쉐이킹 하기 위해서는 어떻게 해야할까?

### 바벨이 es6 모듈을 commonjs module로 변환하지 않도록 하기

Babel은 대부분의 웹 애플리케이션이 필요로 하는 필수 도구다. 그러나 아쉽게도, 트리쉐이킹과 같은 간단한 작업도 이 babel 때문에 어려워 지는 경우가 발생한다. 만약 [babel-preset-env](https://developers.google.com/web/fundamentals/performance/optimizing-javascript/tree-shaking#:~:text=If%20you%27re%20using-,babel-preset-env,-%2C%20one%20thing%20it)를 사용한다면, 이 모듈이 es6를 자동으로 commonjs 변환해준다. 즉, `import`를 `require`로 바꿔주기도 한다. 이는 훌륭한 기능이지만, 트리 쉐이킹 관점에서는 그렇지 못하다.

트리쉐이킹 관점에서 commonjs의 문제점은 웹팩이 어떤 모듈이 사용중인지 아닌지를 판단하여 제거하기가 어렵다는 것이다. 이를 위해 `.babelrc`에서 `commonjs`로 변환하지 못하도록 설정을 추가해 줘야 한다.

```javascript
{
  "presets": [
    ["env", {
      "modules": false
    }]
  ]
}
```

`"modules": false`를 지정하면, babel이 우리가 원하는 대로 동작하게 되어 디펜던시를 분석하고 사용되지 않는 디펜던시를 제거할 수 있다. 또한 웹팩은 코드를 광범위하게 호환되는 형식으로 변환하므로, 이 프로세스는 호환성 문제를 일으키지 않는다.

### 사이드 이펙트 (부수효과)를 염두해두기

이렇게 트리쉐이킹을 할 때 고려해야할 또다른 측면은 프로젝트의 모듈이 부수효과를 일으키지는 지 여부다.

```javascript
let fruits = ['apple', 'orange', 'pear']

console.log(fruits) // (3) ["apple", "orange", "pear"]

const addFruit = function (fruit) {
  fruits.push(fruit)
}

addFruit('kiwi')

console.log(fruits) // (4) ["apple", "orange", "pear", "kiwi"]
```

`addFruit`은 `fruit` 배열의 마지막에 원소를 추가하지만, 이는 `addFruit`의 범위를 벗어나는 일을 하고 있다.

이러한 부수효과는 es6 모듈에서도 똑같이 적용되며, 이 또한 트리쉐이킹 관점에서 문제가 될 수 있다. 예측 가능한 입력값을 받고, 예측가능한 출력을 내뱉는 모듈을 트리쉐이킹 할 경우, 이를 사용하지 않을 때 트리쉐이킹을 하면 안전하게 처리할 수 있다.

웹팩의 경우 `package.json`에 `sideEffects: false`로 지정하여 패키지와 패키지 사이에 부수효과가 없음을 암시할 수 있다.

```json
{
  "name": "webpack-tree-shaking-example",
  "version": "1.0.0",
  "sideEffects": false
}
```

혹은, 특정 파일에 대해서만 부수효과가 없다고 지정할 수 있다.

```json
{
  "name": "webpack-tree-shaking-example",
  "version": "1.0.0",
  "sideEffects": ["./src/utils/utils.js"]
}
```

후자의 예제의 경우, 여기에서 지정된 파일은 부수효과가 없는 것으로 가정한다. `package.json`에 추가하고 싶지 않으면, [`module.rules`를 활용하여 설정할 수 있다.](https://github.com/webpack/webpack/issues/6065#issuecomment-351060570)

### 필요한 것반 import 하기

`Babel`의 설정을 es6로 유지하도록 변경했지만, 모듈에서 필요한 함수만 가져오도록 수정해야 한다.

```javascript
import { simpleSort } from '../../utils/utils'
```

이 구문은, `../../utils/utils`에서 `simpleSort`만 가져오도록 지정한다. 전체 유틸리티 모듈이 아닌, 하나의 함수만 가져오므로 기존의 `utils.simpleSort`를 모두 `simpleSort`로 수정해야 한다.

이제 번들 크기를 다시 확인해보자.

```bash
                 Asset        Size  Chunks             Chunk Names
js/vendors.a3722bf0.js    37.1 KiB       0  [emitted]  vendors
   js/main.951b863a.js    20.8 KiB       1  [emitted]  main
```

```bash
                 Asset        Size  Chunks             Chunk Names
js/vendors.b007c500.js    36.9 KiB       0  [emitted]  vendors
   js/main.2b536ea2.js    8.45 KiB       1  [emitted]  main
```

두개 번들 크기 모두 줄었지만, 여기서 가장 큰 해택을 본 것은 `main` 쪽이다. 실제로 사용하지 않는 부분을 제거하여 약 60%의 코드를 날려버릴 수 있었다. 이렇게 하면 스크립트가 다운로드 하는데 걸리는 시간 뿐만 아니라 처리하는데 걸리는 시간도 줄일 수 있다.

### 무엇을 해야할지 감이 오지 않을 때
