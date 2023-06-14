---
title: '3부) 왜 esmodule 이어야 하는가?'
tags:
  - nodejs
published: true
date: 2023-06-02 14:23:28
description: '2부는 어디갔냐구요? 내맘입니다.'
---

# Table of Contents

## 서론

지금은 좀 지나간 이야기 이지만, esmodule이 표준으로 정착하면서 자바스크리트 개발자 사이에는 많은 갑론을박이 오간 주제가 있는데, 바로 npm 라이브러리가 이제 esmodule 만을 지원해야 하는가 였다. commonjs는 여러 가지로 브라우저 중심의 생태계에서 어울리지 않기 때문에, 이제 esmodule이 그자리를 대신해야 한다는 주장이 있었다. 물론 nodejs는 이후 버전업에도 [commonjs가 기본값을 유지하도록 만들어졌지만](https://yceffort.kr/2023/05/what-is-commonjs#nodejs-%EB%8A%94-%EC%96%B8%EC%A0%9C-commonjs%EB%A5%BC-%EC%82%AC%EC%9A%A9%ED%95%A0%EA%B9%8C), 이러한 방향성이 잘못되었다고 이야기하는 사람들도 있다. 그렇다면 왜 이렇게 `commonjs`는 미움(?) 을 받고 있는지, 왜 `esmodule`로 통일되는 미래를 꿈꾸고 있는지 살펴보자.

## 왜 esmodule로의 통일을 꿈꾸는가?

### dual package hazard

먼저 nodejs는 commonjs와 esmodule을 동시에 지원하기 위해 조건부 exports라고 하는 새로운 기능을 내놓았다. 자세한 내용은 [문서](https://nodejs.org/api/packages.html#conditional-exports)를 확인해보자. 요약하자면 다음과 같다.

```json
{
  "exports": {
    "import": "./index-module.js",
    "require": "./index-require.cjs"
  },
  "type": "module"
}
```

만약 `package.json`이 위와 같이 선언되어 있다면 `require('something')`을 하는 곳은 `exports.require`를, `import 'something'`을 하는 곳은 `exports.import`를 사용하게 된다. 이로써 사용하는 쪽의 모듈이 `esmodule`이든 `commonjs`든 상관없이 안정적으로 사용할 수 있게 되는 것이다.

그러나 이는 문제가 있다. 바로 이 `./index-module.js`와 `./index-require.js`가 동일하지 않다는 것이다. 예를 들어 `class` 생성자를 export 하는 패키지가 있다고 가정해보자. 이 패키지가 내보내는 것은 하나의 `class`이지만, 만약 `instanceof`로 `require`와 `import`를 비교하면 각각 다른 곳에 존재하는 생성자이므로 코드 상으로는 동일할지라도 `false`가 반환될 것이다. 또 객체일 경우, `require`의 객체에 값을 추가한다고 해서 `esmodule` 객체에는 또 추가가되지 않을 것이다. 엄연히 두 객체는 다르기 때문이다. 그러나 이는 사용하는 측면에서는 하나의 패키지에서 일어나는 일이기 때문에 혼란을 야기할 수 있다.

물론 대부분의 애플리케이션의 경우에는 하나의 모듈 시스템만 사용하기 때문에 발생하지 않을 것이라 생각할 수도 있다. 그러나 `a`라는 패키지가 `dual package`를 `export`하는데, `b`라는 패키지는 `commonjs`이고, `b`에서 `a`를 참조하여 `a.require`를 사용하게 된다면 이러한 점들이 문제가 될 수 있다.

이러한 문제는 [`esmodule`과 `commonjs`가 호환되지 않고](https://yceffort.kr/2020/08/commonjs-esmodules) `nodejs`가 두 패키지를 동시에 지원하기 때문에 결국 계속해서 안고 가야하는 문제다.

### 라이브러리 제작자들은 항상 commonjs와 esmodule 두개를 모두 고려해야 한다.

dual package hazard를 무시하고, 모든 모듈 시스템을 지원하기 위해 `subpath`를 사용하기로 했다고 가정해보자. 그러나 두 모듈은 태생 부터 다른 코드이고, 번들링 시에도 복잡하고 까다로운 설정을 거쳐야 하기 때문에 굉장히 귀찮다. 사실 대부분의 경우에는 toss/slash 와 같이 번들링에 모든 것 맡겨버리고, 이후 동작에 대해서는 잘되는지 설치해서 확인하는 정도일 것이다. 만약 잘되는지 확인한다고 가정하더라도, 이 작업은 2배의 복잡성을 지니게 된다. 또한 AST 생성 과정자체도 매우 다르기 때문에 정적 분석하는 것도 쉽지 않다.

### 사이즈의 크기가 두배

dual exports 는 결국 같은 코드를 다른 두 모듈 시스템으로 빌드하여 배포하기 때문에, 사이즈도 2배가 된다. 물론 이는 서비스되는 애플리케이션의 프레임워크 때문에 적절히 트리 쉐이킹이 되어 엔드 유저의 크기에는 영향을 미치지 않는다 하더라도, 여전히 문제가 된다.

제아무리 `dependencies`가 가볍다 하더라도 `node_modules`의 크기가 크다는 사실은 자바스크립트 개발자라면 누구나 알 것이다. `node_modules`의 크기 문제는 예전부터 지적되던 문제로, dual exports 시에는 이 문제를 부채질 하게 된다. `node_modules`가 커지면 CI, CD, 그리고 서버리스 환경에 큰 부담이 된다. 사용하지 도 않은 50%의 코드 때문에 설치 속도가 느려지고, 디스크 공간을 차지하며, 클라우드 환경이 대세가 되는 요즘 부담으로 자리잡게 될 것이다. 장담컨데 이러한 문제가 nodejs 서버리스 서비스가 자리잡는데 악영향을 미친다고 생각한다. 서비스 자체에는 영향이 없다하더라도, 결국 개발을 둘러싼 모든 환경에 더큰 비용이라는 악영향을 미치게 될 것이다.

### 대부분이 사용하지 않는 `require`

최근 까지 개발을 계속해서 해온 개발자라면, 마지막으로 `require`를 쓴 경험이 희미한 개발자가 대부분일 것이다. MZ한 요즘 개발자라면, `require` 존재 자체를 모를 수도 있을 것이다. 만약 내가 작성한 코드가 `require`를 사용해서 번들 된다면, 코드를 이해하고 번들링이 동작한 맥락을 이해하는 것이 더욱 어려워 질 것이다. 그리고 사실 대부분의 사람들은 `import`로 작성된 코드가 `commonjs`로 변환되는 것에 대해 관심을 가지고 있지도 않다. 대부분의 개발자들은 아무튼 `workaround`가 있으면 그만이고, 어떤식으로든 되던지 상관하지 않기도 하다.

```javascript
webpack: (config) => {
    config.module.rules.push({
      test: /\.m?js$/,
      type: 'javascript/auto',
      resolve: {
        fullySpecified: false,
      },
    });
    return config;
  },
```

> `.mjs`를 지원하지 않는 시스템에서 `.mjs`를 지원하기 위해 추가해야하는 웹팩 설정. webpack@4 기반 프로젝트 (react-scripts@5 미만 사용자라면 다 경험해보았을 것이다)

이러한 상황에서 과연 `commonjs`의 존재의미는 무엇인가? 결국 레거시로 분류된 모듈 시스템을 지원하기 위한 허들에 지나지 않게 된다.

## esmodule을 기본으로 지원하기 위한 험난한 과정

열정적인 라이브러리 개발자들은 이러한 문제점을 알고 있지만, esmodule을 기본으로 지원하기 시작한 것은 표준이 나온 시기에 비하면 비교적 최근이다. 그리고 이름만들어도 널리 알려진 라이브러리들의 경우에는 여전히 esmodule 을 지원하기 시작했다.

- typescript: 4.7 버전이 릴리즈 되고 나서야 비로소 `compilerOptions.module: "node16"`을 통해 esmodule을 지원하기 시작했다. https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-7.html
- nextjs: 12 버전에 들어서부터 esmodule을 비로소 지원했다. https://nextjs.org/blog/next-12#es-modules-support-and-url-imports 그전까지는 `.mjs`를 `import`하면 에러가 발생했다.
- jest: jest의 경우 여전히 esmodule을 지원하고 있지 못하며, 최근에 들어서야 비로 실험기능으로 지원하기 시작했다. https://jestjs.io/docs/ecmascript-modules

esmodule 이 제안되고 nodejs 에서 채택되었음에도 지금까지 커뮤니티가 미적지근한 것은, 아무래도 두 모듈 시스템이 호환되지 않는 다는 사실때문일 것이다. 이 두 모듈 시스템을 원활하게 지원하는 것은 단순히 라이브러리 개발자 뿐만 아니라, 타입스크립트나 nextjs, jest와 같이 수많은 사용자를 보유하고 있는 프레임워크에도 부담스러운 작업이라는 사실은 방증한다.

## 결국 esmodule 로 가야하지 않을까?

commonjs를 보고 있노라니, 몇 년전 웹 생태계에서 어도비 플래쉬가 사라져 가던 것이 생각나는 것 같다. 그 때만 하더라도 대부분의 웹 개발자들은 플래쉬가 사라지는 미래를 생각하지 못했다. 대부분의 홈페이지에서는 플래쉬 설치를 요구 했고, 플래쉬로 만들어진 홈페이지가 사방에 존재했고, 플래쉬 개발자들이 각광받던 시대였다. 그 때 당시에도 플래쉬 개발자들은 "너무 많은 곳에서 사용되고 있기 때문에 절대 사라지지 않을 것" 이라고 믿고 있었다. 그러나 스티브 잡스라는 강력한 인물의 드라이브와 웹 표준의 등장으로 인해 이제 웹 어디에서도 볼 수 없는 코드가 되어 버렸다.

그렇다면 결국 nodejs가 과감하게 commonjs의 손을 떼 버리고 어느날 부터 esmodule 만을 지원해야만 commonjs가 사라지는 일이 가능해질까? 사실 이러한 미래는 nodejs 팀에 스티브 잡스같이 미친 리더가 있지 않은 이상, nodejs 보다는 npm 생태계에 달려있지 않을까 싶다. npm 생태계에서 점차 commonjs를 지원하고 esmodule을 우선시 하거나, 일부 열정적인 개발자들이 나서서 commonjs의 중단을 시작해야 할 것이다. 그렇다면 nodejs 개발 팀도 생각을 고치게 될 것이다.

내부에서 라이브러리를 개발하면서, dual exports 전략으로 패키지를 배포하고 있지만 esmodule로 그냥 폭력적으로 다 넘어갔으면 하는 생각을 몇번씩이고 했다. 그 때마다 `react-scripts@3`으로 작성된 레거시 애플리케이션을 보면서 마음을 다스렸지만, 아직까지도 자바스크립트 개발 환경 여기저기에서 commonjs와 esmodule로 인한 비용을 계속해서 지불하고 있는 것이 사실이다. 언젠간 esmodule이 정식 표준으로 자리잡고, commonjs가 nodejs 문서에서 deprecated가 되는 날이 오지 않을까? @types 패키지 생태계의 미래를 상상해보면서 같이 공상을 해본다.
