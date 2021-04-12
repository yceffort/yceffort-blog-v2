---
title: '더 빠른 웹 애플리케이션을 위한 모던 자바스크립트'
tags:
  - javascript
  - browser
published: true
date: 2020-12-15 20:02:35
description: '아니 그래서 IE 11 언제 없앨 건데요'
---

오늘날 90%가 넘는 브라우저가 모던 자바스크립트를 실행할 수 있음에도 불구하고, 레거시 자바스크립트는 오늘날 웹 성능 문제에 큰 원인 중 하나로 남아 있다. ES2017 문법을 사용하여 웹 페이지 또는 패키지를 작성하고 퍼블리싱 하면 성능을 매우 향상 시킬 수 있다.

## 모던 자바스크립트란 무엇일까

모던 자바스크립트는 특정 ECMAScript 버전으로 작성된 코드를 말하는게 아니고, 모던 브라우저에서 지원하는 문법으로 이루어진 것을 의미한다. 크롬, 엣지, 파이어폭스, 사파리와 같은 모던 웹 브라우저는 브라우저 시장의 90% 이상을 차지하고 있으며, 이 렌더링 엔진에 의존하는 다른 브라우저가 5% 쯤 된다. 따라서 글로벌 웹 트래픽의 95%가 지난 10년간 가장 널리 사용되는 자바스크립트 언어 기능을 지원하는 브라우저에서 비롯된다.

> 아마도 엣지 레거시와 구 IE를 제거한 수치로 보면 될 것 같다. 우리나라에서는 약 93% 정도 된다.

여기서 말하는 문법은 이정도다.

- 클래스 (ES2015)
- 화살표 함수 (ES2015)
- 제네레이터 (ES2015)
- 블록 스코프 (ES2015)
- 디스트럭쳐링 (ES2015)
- 전개 연산자 (ES2015)
- 객체 축약 (ES2015)
- async await (ES2017)

이 후에 나온 기능, 예를 들어 ES2020, ES2021에 나온 기능들은 브라우저의 70% 정도가 지원한다. 여전히 70%면 많은 수치이지만, 그 기능에 온전히 기대는 것은 안전하지 않다. 따라서 모던 자바스크립트를 타겟으로 한다면, 가장 널리 알려져 있으며 지원하는 브라우저가 많은 ES2017 정도가 적당해 보인다. 다른 말로 하자면,  ES2017이 오늘날의 가장 모던한 문법인 것이다.

> 물론 이 글은 2020년 12월 15일에 작성되어 있습니다. 시간이 흐르면 더 달라지겠죠?

https://dev.to/garylchew/bringing-modern-javascript-to-libraries-432c

## 레거시 자바스크립트

레거시 자바스크립트란 위에서 언급한 새로운 기능을 사용하지 않은 코드라 볼 수 있다. 대부분의 개발자는 모던한 문법으로 작성하지만, 브라우저의 지원을 늘리기 위하여 레거시 구문으로 컴파일한다🤪. 레거시 구문으로 컴파일하면 브라우저 지원 범위가 증가하지만, 그 효과는 우리가 생각하는 것보다 작다. 레거시 자바스크립트로 95% 정도의 수치를 98% 로 늘릴 수 는 있지만, 그 비용은 엄청나다.

> 나머지 2%는 no javascript 가 아닐까 싶습니다

- 레거시 자바스크립트는 모던 자바스크립트와 비교 했을 때 일반적으로 20% 정도 용량이 더 크고 느리다. 만약 여기에서 잘못 구성한다면 그 차이가 더 커질 수 있다.
- 별도로 설치하는 자바스크립트 라이브러리의 경우 일반적인 자바스크립트 코드의 90% 정도를 차지 한다. 이 코드에 폴리필 등이 추가 된다면 더 많은 자바스크립트 오버헤드가 발생할 수 있다.

## npm에서 모던 자바스크립트

[최근에 nodejs는 패키지의 엔트리 포인트를 지정할 수 있게 해주었다.](https://nodejs.org/api/packages.html#packages_package_entry_points)

```json
{
  "exports": "./index.js"
}
```

`exports` 필드에서 참조하는 모듈은 최소 ES2019를 지원하는 12.8의 노드 버전을 의미한다. 따라서 exports를 참조하는 모듈은 모던 자바스크립트로 작성할 수 있음을 의미한다. 따라서 모던 코드만 있는 패키지를 배포하고, 트랜스파일링은 사용하는 사람에게 맡기고 싶다면 위처럼 `exports` 필드만 사용하면 된다.

`exports`와 함께 `main`을 사용한다면, 레거시 브라우저를 위한 ES5와 CommonJS를 제공할 수 있다. 일종의 modern 한 코드의 레거시 폴백이다.

```json
{
  "name": "foo",
  "exports": "./modern.js",
  "main": "./legacy.cjs"
}
```

CommonJS로 작성된 fallback에, `module`을 추가한다면 유사하게 legacy fallback 번들로 동작하지만, 자바스크립트 모듈 문법인 `import`와 `export`도 사용 가능하다.

```json
{
  "name": "foo",
  "exports": "./modern.js",
  "main": "./legacy.cjs",
  "module": "./module.js"
}
```

이렇게 하는 이유는, 웹팩과 롤업과 같은 번들러에서 트리쉐이킹을 할 수 있도록 하기 위해서다. 이 코드는 `import` `export` 이외에 모던 코드를 사용하지 않는 레거시 번들 이므로, 레거시 코드를 지원함과 동시에 번들링에 최적화 시킬 수 있다는 장점을 가지고 있다.

## 애플리케이션의 모던 자바스크립트

써드파티 디펜던시는 프로덕션 웹 애플리케이션의 대부분을 차지한다. npm 의존성은 역사적으로 레거시 ES5로 퍼블리싱 되어왔지만, 이에 의존하는 것은 더이상 안전하지 못한 가정이며, 애플리케이션에서 브라우저 지원을 해칠 수도 있는 위험한 행동이다.

모던 자바스크립트로 이동하는 npm 패키지가 점차 많아지면서 이를 다룰 수 있는 도구를 설치하는 것이 중요 해졌다. 여거라지 방법이 있지만, 일반적으로 좋은 아이디어는 의존성을 내가 작성하는 코드의 문법 수준과 맞춰 트랜스파일 하는 것이다.

## 웹팩

웹팩5에서 부터, 번들 및 모듈에 대한 코드를 생성할 때 마다 사용할 문법을 설정할 수 있다. 이는 코드나 의존성을 트랜스파일링 하는 것이 아니고, 오직 webpack에 의해 생성된 코드를 붙일 때 영향을 미친다. 브라우저 지원 타겟을 설저하기 위해서는 `browserlist`설정을 추가하거나, 혹은 웹팩 설정에 바로 넣어두면 된다.

```javascript
module.exports = {
  target: ['web', 'es2017'],
};
```

또한 웹팩에서 모던 ES 모듈 환경을 타겟으로 한다면, 불필요한 wrapper 함수를 생략하여 번들 크기를 최적화하는 설정을 할 수 다.

```javascript
module.exports = {
  target: ['web', 'es2017'],
  output: {
    module: true,
  },
  experiments: {
    outputModule: true,
  },
};
```

이외에도 웹팩을 활용하여 모던 자바스크리븥 문법을 활용하면서 동시에 레거시 브라우저도 지원할 수 있도록 도와주는 많은 도구들이 존재한다.

## Optimize Plugin

[Optimize Plugin](https://github.com/developit/optimize-plugin)은 개별 자바스크립트 파일을 레거시 자바스크립트로 만드는 대신에, 최종적으로 만들어진 모던 번들 코드를 레거시 자바스크립트로 변환시켜주는 도구다. 웹팩 설정을 통해 모든 것이 모던 자바스크립트라고 가정하기 때문에, 특별한 처리가 필요하지 않다. 또한 이는 개별 모듈이 아닌 번들 레벨에서 동작하므로 애플리케이션의 코드와 디펜던시를 동등하게 처리한다. 이는 최종 번들링된 파일을 다시 레거시 문법으로 만드는 것이기 때문에, 전통적인 방식보다 빠를 수 있다. 이러한 두개의 모듈은 [module/nomodule pattern](https://web.dev/serve-modern-code-to-modern-browsers/)에서 사용된다.

```javascript
// webpack.config.js
const OptimizePlugin = require('optimize-plugin');

module.exports = {
  // ...
  plugins: [new OptimizePlugin()],
};
```

`Optimize Plugin`은 모던 코드와 레거시 코드를 따로 번들링 하는 기존 방식보다 더 빠르고 효율적이다. 또한 `Babel`처리도 가능하며, `Terser`를 활용하여 각각의 번들 크기를 줄일 수도 있다. 마지막으로, 레거시 번들에 필요한 폴리필을 따로 관리하기 때문에, 모던 브라우저에서 이를 로딩하지 않도록 도와준다.

https://storage.googleapis.com/web-dev-assets/fast-publish-modern-javascript/transpile-before-after.webm
  ## BabelEsmPlugin

  웹팩 플러그인 중 하나인 [BabelEsmPlugin](https://github.com/prateekbh/babel-esm-plugin)는 `@babel/preset-env` 와 함께 사용할 수 있으며, 현재 가지고 있는 번들을 모던 브라우저에 서비스 할 수 있도록 트랜스파일링을 최소화 해준다. 이는 Next.js나 preact cli에서도 사용하는 가장 유명한 module/nomodule 솔루션이다.

  ```javascript
  // webpack.config.js
const BabelEsmPlugin = require('babel-esm-plugin');

module.exports = {
  //...
  module: {
    rules: [
      // your existing babel-loader configuration:
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env'],
          },
        },
      },
    ],
  },
  plugins: [new BabelEsmPlugin()],
};
```

`BabelEsmPlugin`는 애플리케이션에서 크게 분리된 두가지 빌드를 실행하기 때문에 다양한 웹팩 설정을 지원한다. 두 번 컴파일 하는 것은 대규모 애플리케이션에 약간의 추가 시간이 걸릴 수 있지만, 이는 BabelEsmPlugin이 기존의 웹팩 설정에 원활한 통합 등을 도와주며, 편의성도 제공한다.

## node_modules을 트랜스파일 하기 위한 babel-loader 설정

위에서 언급한 두개의 플러그인 대신 `babel-loader`를 사용하고 있다면, npm 모듈을 모던 자바스크립트로 사용하기 위한 중요한 단계가 남아 있다. 두개의 개별적인 바벨 로더를 구성해서 정의한다면, node_modules에서 발견되는 최신 언어 스펙을 ES2017로 자동으로 컴파일 하는 동시에, 프로젝트에 구성된 babel 플러그인과 사전 설정으로 자신의 코드를 1차적으로 변환할 수 있다. 이는 module/nomodule 설정의 번들을 생성하지는 않지만, 오래된 브라우저 지원을 깨트리지 않고 모던 자바스크립트가 포함된 npm 패지키를 설치하고 사용하는 것을 가능하게 한다. 

[webpack-plugin-modern-npm](https://www.npmjs.com/package/webpack-plugin-modern-npm) 을 사용하면, npm 디펜던시에 `exports`가 있는 코드들을 컴파일한다.

```javascript
// webpack.config.js
const ModernNpmPlugin = require('webpack-plugin-modern-npm');

module.exports = {
  plugins: [
    // auto-transpile modern stuff found in node_modules
    new ModernNpmPlugin(),
  ],
};
```

대신에 이 기능을 수동으로 설정할 수 도 있다.

```javascript
// webpack.config.js
module.exports = {
  module: {
    rules: [
      // Transpile for your own first-party code:
      {
        test: /\.js$/i,
        loader: 'babel-loader',
        exclude: /node_modules/,
      },
      // Transpile modern dependencies:
      {
        test: /\.js$/i,
        include(file) {
          let dir = file.match(/^.*[/\\]node_modules[/\\](@.*?[/\\])?.*?[/\\]/);
          try {
            return dir && !!require(dir[0] + 'package.json').exports;
          } catch (e) {}
        },
        use: {
          loader: 'babel-loader',
          options: {
            babelrc: false,
            configFile: false,
            presets: ['@babel/preset-env'],
          },
        },
      },
    ],
  },
};
```

이 방법을 사용할 때는 minifier가 이러한 모던 코드를 지원할 수 있도록 해야 한다. `Terser` `Uglify-es`에 `{ecma: 1027}`를 지정할 수 있는 옵션이 있으며, 경우에 따라 압축하거나 포맷하는 와중에 `ES2017`구문을 생성하기도 한다. 

## 추가적인 빌드 툴

- https://parceljs.org/
- https://www.snowpack.dev/
- https://github.com/vitejs/vite
- https://github.com/preactjs/wmr

## 결론

ES2017 이 요즘 흔히 말하는 모던 자바스크립트에 가장 근접한 스펙이며, npm, babel, rollup 등을 빌드 시스템에 사용하여 패키지와 문법을 모던 자바스크립트에서 동작할 수 있도록 설정할 수 있다. 

출처: https://web.dev/publish-modern-javascript/