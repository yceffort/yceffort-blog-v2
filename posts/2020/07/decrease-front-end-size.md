---
title: 프론트엔드 사이즈 줄이기
tags:
  - javascript
  - web
published: true
date: 2020-07-01 09:45:10
description: "[이
  글](https://developers.google.com/web/fundamentals/performance/webpack/decreas\
  e-frontend-size)을 대충 번역했습니다.  ```toc tight: true, from-heading: 2 to-heading:
  4 ```  ## webpack 4버전 이상의 경우 프로덕션 모드를 사용하..."
category: javascript
slug: /2020/07/decrease-front-end-size/
template: post
---

[이 글](https://developers.google.com/web/fundamentals/performance/webpack/decrease-frontend-size)을 대충 번역했습니다.

## Table of Contents

## webpack 4버전 이상의 경우 프로덕션 모드를 사용하기

webpack 4버전 부터 [mode](https://webpack.js.org/concepts/mode/)라고 하는 플래그가 추가되었다. `development` `production`을 설정해두는데, 이는 webpack에 현재 어떤 버전으로 빌드하려고 하는지 알려준다.

```json
// webpack.config.js
module.exports = {
  mode: 'production',
};
```

`production`모드는 프로덕션 환경에서 실제 앱을 빌드할 때 사용한다. 이 모드를 사용하면 웹팩은 코드 최소화, dev 라이브러리 삭제, [등등](https://medium.com/webpack/webpack-4-mode-and-optimization-5423a6bc597a)의 최적화를 진행하게 된다.

## minification 을 켜두기

minification이란 코드에서 띄어쓰기를 제거하거나, 변수명을 짧게 하는 등으로 코드의 양 자체를 줄이는 것이다.

```javascript
function map(array, iteratee) {
  let index = -1
  const length = array == null ? 0 : array.length
  const result = new Array(length)

  while (++index < length) {
    result[index] = iteratee(array[index], index, array)
  }
  return result
}
```

```javascript
// minified
function map(n, r) {
  let t = -1
  for (const a = null == n ? 0 : n.length, l = Array(a); ++t < a; )
    l[t] = r(n[t], t, n)
  return l
}
```

### 번들 수준의 minification

번들 수준의 minification은 컴파일 이후에 전체 번들을 압축하는 것이다.

```javascript
// 1. 코드가 이렇게 있다
// comments.js
import './comments.css'
export function render(data, target) {
  console.log('Rendered!')
}
```

```javascript
// 2. 웹팩이 대충 이런 모습으로 컴파일 한다.
```

```javascript
// bundle.js (part of)
'use strict'
Object.defineProperty(__webpack_exports__, '__esModule', { value: true })
/* harmony export (immutable) */ __webpack_exports__['render'] = render
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__comments_css__ =
  __webpack_require__(1)
/* harmony import */ var __WEBPACK_IMPORTED_MODULE_0__comments_css_js___default =
  __webpack_require__.n(__WEBPACK_IMPORTED_MODULE_0__comments_css__)

function render(data, target) {
  console.log('Rendered!')
}
```

```javascript
// 3. 압축한다.
// minified bundle.js (part of)
'use strict'
function t(e, n) {
  console.log('Rendered!')
}
Object.defineProperty(n, '__esModule', { value: !0 }), (n.render = t)
var o = r(1)
r.n(o)
```

- webpack4에서는 번들 수준 최소화가 프로덕션 모드 또는 명시되지 않은 모드에서 자동으로 진행된다.내부적으로는 [Uglify minifier](https://github.com/mishoo/UglifyJS2)를 사용한다. 만약 최소화를 하고 싶지 않다면, development 모드를 키거나 `optimization.minimize`에 false를 주면 된다.
- webpack3에서는 [Uglify minifier](https://github.com/mishoo/UglifyJS2)를 직접 사용해야 한다. 해당 플러그인은 webpack에서 자동으로 딸려 오므로, 설정에 아래 코드를 추가하면 된다.

```javascript
// webpack.config.js
const webpack = require('webpack')

module.exports = {
  plugins: [new webpack.optimize.UglifyJsPlugin()],
}
```

### loader-specific 옵션

코드를 줄이는 두번째 방법은 loader-specific 옵션을 사용하는 것이다. [loader](https://webpack.js.org/concepts/loaders/) 이 옵션을 사용하면, minifier가 줄이지 못하는 코드를 줄여줄 수 있다. 만약 css를 위해 `css-loader`를 사용하고 있다면, 파일은 아래와 같이 문자열로 컴파일 된다.

```css
/* comments.css */
.comment {
  color: black;
}
```

```javascript
// minified bundle.js (part of)
;(exports = module.exports = __webpack_require__(1)()),
  exports.push([module.i, '.comment {\r\n  color: black;\r\n}', ''])
```

minifier는 코드가 문자열이기 때문에 더이상 최소화 할 수 없다. 이를 최소화 하기 위해서는, 아래와 같이 옵션을 추가하면 된다.

```javascript
// webpack.config.js
module.exports = {
  module: {
    rules: [
      {
        test: /\.css$/,
        use: [
          'style-loader',
          { loader: 'css-loader', options: { minimize: true } },
        ],
      },
    ],
  },
}
```

### 참고할 만한 것들

- [Uglify JsPlugin docs](https://github.com/webpack-contrib/uglifyjs-webpack-plugin)
- 다른 최소화 라이브러리 [Babel Minify](https://github.com/webpack-contrib/babel-minify-webpack-plugin) [Google Closure Compiler](https://github.com/roman01la/webpack-closure-compiler)

## NODE_ENV=production 명시하기

> webpack4에서 production 모드를 사용하고 있다면, 이미 해당 옵션은 켜져 있을 것입니다. 아래의 팁은 webpack3와 관련된 것입니다.

`NODE_ENV`의 값을 `production`로 해두면, 코드의 크기를 줄일 수 있다.

라이브러리들은 환경 변수인 `NODE_ENV`값을 감지하여 어떻게 동작할지를 판단한다. 예를 들어, vue.js의 경우 production으로 값이 주어져 있지 않다면, 아래와 같은 메시지를 보여준다.

```javascript
// vue/dist/vue.runtime.esm.js
// …
if (process.env.NODE_ENV !== 'production') {
  warn('props must be strings when using array syntax.')
}
// …
```

리액트도 비슷하게 동작한다. development에서 빌드시 다음과 같은 경고문을 낼 수 있다.

```javascript
// react/index.js
if (process.env.NODE_ENV === 'production') {
  module.exports = require('./cjs/react.production.min.js')
} else {
  module.exports = require('./cjs/react.development.js')
}

// react/cjs/react.development.js
// …
warning$3(
  componentClass.getDefaultProps.isReactClassApproved,
  'getDefaultProps is only used on classic React.createClass ' +
    'definitions. Use a static property named `defaultProps` instead.',
)
// …
```

이런 체크는 production에서는 불필요하여 코드의 사이즈를 늘릴 뿐이다. webpack4에서는 아래와 같이 하면된다.

```javascript
module.exports = {
  optimization: {
    nodeEnv: 'production',
    minimize: true,

```

이 코드는 `process.env.NODE_ENV`를 모두 `production`으로 바꿔버리는 효과를 가지고 있다. 또한 minifer는 `process.env.NODE_ENV !== 'production'` 코드를 모두 날려버린다. 어쨌든 false라 절대 탈 수 없는 코드 이기 때문이다.

## ES Module 사용하기

프론트엔드 사이즈를 줄이는 또한가지 방법은 [ES modules](https://ponyfoo.com/articles/es6-modules-in-depth)을 사용하는 것이다. ES modules을 사용해야 웹팩에서 트리쉐이킹을 할 수 있다. 트리쉐이킹이란 번들러가 전체 디펜던시 트리를 싹 뒤져서, 사용하지 않는 부분을 삭제해 버리는 것이다. 따라서 ESModule syntax를 사용해야 트리쉐이킹이 가능하다.

```javascript
// comments.js
export const render = () => {
  return 'Rendered!'
}
export const commentRestEndpoint = '/rest/comments'

// index.js
import { render } from './comments.js'
render()
```

웹팩이 `commentRestEndpoint`는 안쓰는 것으로 판단해 따로 export하지 않는다.

```javascript
// bundle.js (part that corresponds to comments.js)
;(function (module, __webpack_exports__, __webpack_require__) {
  'use strict'
  const render = () => {
    return 'Rendered!'
  }
  /* harmony export (immutable) */ __webpack_exports__['a'] = render

  const commentRestEndpoint = '/rest/comments'
  /* unused harmony export commentRestEndpoint */
})
```

그리고 사용하지 않는 코드를 minifier가 날려버린다.

```javascript
// bundle.js (part that corresponds to comments.js)
;(function (n, e) {
  'use strict'
  var r = function () {
    return 'Rendered!'
  }
  e.b = r
})
```

> 웹팩에서 minifier가 없다면 트리쉐이킹이 동작하지 않습니다. 사용하지 않는 코드를 export하지 않는 것 (트리쉐이킹)과 사용하지 않는 코드를 지우는 것(minifier)은 한쌍이기 때문입니다.

> ESModules을 CommonJS 로 컴파일 하지 말기를 바랍니다.

## 이미지 크기 줄이기

[이미지는 페이지 크기의 절반 이상을 차지한다.](http://httparchive.org/interesting.php?a=All&l=Oct%2016%202017) 렌더링을 블락하는 자바스크립트 만큼은 치명적이지 않지만, 어쨌든 전체 네트워크 통신에서 많은 부분을 잡아먹는 것은 사실이다. `url-loader` `svg-url-loader` `image-webpack-loader` 등으로 최적화 할 필요가 있다.

`url-loader`는 작은 정적 파일을 앱에 삽입한다. 별도의 설정이 없을시 파일을 전달 받았다면, 해당 파일을 번들링하고 번들링된 주소를 리턴한다. 그러나 `limit`가 존재한다면, 해당 이미지를 더 작은 `Base64` 데이터로 인코딩하여 바꿔치기한다.

```javascript
// webpack.config.js
module.exports = {
  module: {
    rules: [
      {
        test: /\.(jpe?g|png|gif)$/,
        loader: 'url-loader',
        options: {
          // Inline files smaller than 10 kB (10240 bytes)
          limit: 10 * 1024,
        },
      },
    ],
  },
}
```

```javascript
// index.js
import imageUrl from './image.png'
// → If image.png is smaller than 10 kB, `imageUrl` will include
// the encoded image: 'data:image/png;base64,iVBORw0KGg…'
// → If image.png is larger than 10 kB, the loader will create a new file,
// and `imageUrl` will include its url: `/2fcd56a1920be.png`
```

`svg-url-loader`는 `url-loader`와 비슷하지만, 파일을 URL encoding한다는 점이 다르다. 이는 SVG파일에 유용한데, 그 이유는 SVG는 단순히 텍스트 이므로, 인코딩시 더 사이즈가 줄어들기 때문이다.

```javascript
// webpack.config.js
module.exports = {
  module: {
    rules: [
      {
        test: /\.svg$/,
        loader: 'svg-url-loader',
        options: {
          // Inline files smaller than 10 kB (10240 bytes)
          limit: 10 * 1024,
          // Remove the quotes from the url
          // (they’re unnecessary in most cases)
          noquotes: true,
        },
      },
    ],
  },
}
```

`image-webpack-loader`는 이미지 자체를 압축해준다. JPG, PNG, GIF, SVG를 지원한다. 이 옵션은 앞선 두 예시 처럼 따로 임베딩 해주지는 않는다. 함께 사용하기 위해서는, 아래처럼 하면 된다.

```javascript
// webpack.config.js
module.exports = {
  module: {
    rules: [
      {
        test: /\.(jpe?g|png|gif|svg)$/,
        loader: 'image-webpack-loader',
        // This will apply the loader before the other ones
        enforce: 'pre',
      },
    ],
  },
}
```

## 디펜던시 최적화하기

절반이상의 자바스크립트 번들 사이즈는 디펜던시에서 오며, 그 중 일부분은 불필요할 수 있다.

`Lodash`의 경우 번들 시에 72kb를 차지하지만, 몇가지 메소드를 사용하지 않는다면 크기를 줄일 수 있다. `Moment.js`는 무려 223KB를 차지하는데, 이는 평균 페이지당 자바스크립트 사이즈를 감안했을때 [452KB](http://httparchive.org/interesting.php?a=All&l=Oct%2016%202017) 엄청나게 큰 비중을 차지한다. 하지만 이중 170kb는 [Locale](https://github.com/moment/moment/tree/4caa268356434f3ae9b5041985d62a0e8c246c78/locale)관련 내용이다. 만약 Moment.js를 가지고 다양한 언어를 지원할 필요가 없다면, 이런 파일은 크기만 차지하게 된다.

이런 패키지들은 쉽게 최적화가 가능하다. [여기](https://github.com/GoogleChromeLabs/webpack-libs-optimizations)를 참고해보자.

## ES Module을 위한 module concatenation 켜두기 (aka 스코프 호이스팅)

웹팩에서 번들을 만들때, 각 모듈을 함수로 래핑한다.

```javascript
// index.js
import { render } from './comments.js'
render()

// comments.js
export function render(data, target) {
  console.log('Rendered!')
}
```

```javascript
// bundle.js (part  of)
/* 0 */
;(function (module, __webpack_exports__, __webpack_require__) {
  'use strict'
  Object.defineProperty(__webpack_exports__, '__esModule', { value: true })
  var __WEBPACK_IMPORTED_MODULE_0__comments_js__ = __webpack_require__(1)
  Object(__WEBPACK_IMPORTED_MODULE_0__comments_js__['a' /* render */])()
},
  /* 1 */
  function (module, __webpack_exports__, __webpack_require__) {
    'use strict'
    __webpack_exports__['a'] = render
    function render(data, target) {
      console.log('Rendered!')
    }
  })
```

과거 이러한 방식은 CommonJS나 AMD 모듈로 부터 분리시키기 위해 필요했다. 그러나 이러한 방식은 각 모듈의 사이즈를 키우고 퍼포먼스를 저하시킨다.

웹팩3 부터 [module concatenation](https://webpack.js.org/plugins/module-concatenation-plugin/)를 활용한 번들링이 가능해졌다. concatenation 모듈이 하는 것을 살펴보자.

```javascript
// index.js
import { render } from './comments.js'
render()

// comments.js
export function render(data, target) {
  console.log('Rendered!')
}
```

```javascript
// Unlike the previous snippet, this bundle has only one module
// which includes the code from both files

// bundle.js (part of; compiled with ModuleConcatenationPlugin)
/* 0 */
;(function (module, __webpack_exports__, __webpack_require__) {
  'use strict'
  Object.defineProperty(__webpack_exports__, '__esModule', { value: true })

  // CONCATENATED MODULE: ./comments.js
  function render(data, target) {
    console.log('Rendered!')
  }

  // CONCATENATED MODULE: ./index.js
  render()
})
```

차이가 보이는가? 플레인 번들에서는, 모듈 0이 모듈 1에 있는 `render`를 필요로 했다. module concatenation을 활용하면, `require` 대신 1번 모듈을 바로 호출하는 것을 볼 수 있다. 번들이 모듈의 수를 줄여주었고, 마찬가지로 오버헤드도 줄어들었다.

웹팩4

```javascript
// webpack.config.js (for webpack 4)
module.exports = {
  optimization: {
    concatenateModules: true,
  },
}
```

웹팩3

```javascript
// webpack.config.js (for webpack 3)
const webpack = require('webpack')

module.exports = {
  plugins: [new webpack.optimize.ModuleConcatenationPlugin()],
}
```

## 웹팩 코드와 웹팩으로 번들링 되지 않은 코드를 같이 슨다면 `externals`를 사용하라

만약 두개의 코드가 같은 디펜던시를 가지고 있다면, 이를 공유해서 여러번 같은 코드를 다운로드하는 것을 막을 수 있다.

### `window`에 있을 경우

```javascript
// webpack.config.js
module.exports = {
  externals: {
    react: 'React',
    'react-dom': 'ReactDOM',
  },
}
```

만약 이렇게 설정해둔다면, 웹팩은 `react`와 `react-dom`을 번들링하지 않는다. 대신 아래와 비슷한 일을 한다.

```javascript
// bundle.js (part of)
;(function (module, exports) {
  // A module that exports `window.React`. Without `externals`,
  // this module would include the whole React bundle
  module.exports = React
},
  function (module, exports) {
    // A module that exports `window.ReactDOM`. Without `externals`,
    // this module would include the whole ReactDOM bundle
    module.exports = ReactDOM
  })
```

### `AMD` 패키지의 경우

```javascript
// webpack.config.js
module.exports = {
  output: { libraryTarget: 'amd' },

  externals: {
    react: { amd: '/libraries/react.min.js' },
    'react-dom': { amd: '/libraries/react-dom.min.js' },
  },
}
```

웹팩은 위 라이브러리를 주소로 번들링 할 것이다.

```javascript
// bundle.js (beginning)
define(["/libraries/react.min.js", "/libraries/react-dom.min.js"], function () { … });
```

## 요약

- webpack4 이상의 버전에서는 `production`모드를 활성화 시켜라
- 번들 수준의 minifier와 loader 옵션을 활용하여 코드의 크기를 줄여라
- 개발 단계에서만 필요한 코드는 `NODE_ENV` `production`으로 관리하라
- 트리쉐이킹을 위해 ESModule을 사용하라
- 이미지를 압축하라
- 디펜던시 라이브러리를 최적화 하라
- module concatenation을 켜둬라
- 필요하다면 `externals`를 사용하라
