---
title: Webpack을 활용한 성능향상 - 캐싱 활용하기
tags:
  - javascript
  - browser
  - webpack
published: true
date: 2020-07-21 04:18:21
description: "[Make use of long-term
  caching](https://developers.google.com/web/fundamentals/performance/webpack/u\
  se-long-term-caching)을 번역한 글입니다. ```toc from-heading: 2 to-heading: 3 ```  앱
  로딩 속도를 향상시킬 수 있는 방법 중 ..."
category: javascript
slug: /2020/07/make-use-of-long-term-caching/
template: post
---

[Make use of long-term caching](https://developers.google.com/web/fundamentals/performance/webpack/use-long-term-caching)을 번역한 글입니다.

```toc
from-heading: 2
to-heading: 3
```

앱 로딩 속도를 향상시킬 수 있는 방법 중 하나는 캐싱을 활용하는 것이다. 캐싱을 활용하면, 클라이언트에서 매번 리소스를 다시 다운로드 하는 것을 방지해준다.

## 번들 버전과 캐시 해더 사용하기

캐싱을 하는 가장 일반적인 방법은 다음과 같다.

1. 브라우저에 해당 파일의 캐시 기간을 굉장히 길게 설정해 두는 것 (1년 쯤)

```
# Server header
Cache-Control: max-age=31536000
```

2. 파일의 이름을 바꿔서 강제로 다운로드 하게 하는 것

```html
<!-- Before the change -->
<script src="./index-v15.js"></script>

<!-- After the change -->
<script src="./index-v16.js"></script>
```

이러한 접근 법은 브라우저에 JS 파일을 다운로드 받게 하고, 이를 캐시하여 캐시된 복사본을 사용하게 한다. 브라우저는 파일명이 바뀌거나 1년이 지난 이후에야 새롭게 네트워크를 통해서 파일을 받을 것이다.

웹팩에서는 이와 동일한 작업을 할 수 있다. 버전명을 사요하는 대신, 파일 해시를 지정해서 사용할 수 있다. 파일명에 해시를 포함하기 위해서는 `[chuckhash]`를 사용하면 된다.

```javascript
// webpack.config.js
module.exports = {
  entry: './index.js',
  output: {
    filename: 'bundle.[chunkhash].js',
    // → bundle.8e0d62a03.js
  },
}
```

> 파일명만 바뀌거나, 번들링하는 OS의 버전이 다른 경우에도 다른 해시값이 나올 수도 있다. 이것은 웹팩의 버그로, 아직까지 [뚜렷한 해결책이 없는 듯 하다](https://github.com/webpack/webpack/issues/1479)

만약 클라이언트 사이드에 보낼 파일 명이 필요하다면 `HtmlWebpackPlugin` 이나 `WebpackManifestPlugin`을 사용하면 된다.

[HtmlWebpackPlugin](https://github.com/jantimon/html-webpack-plugin)은 사용법이 간단한 대신에 유연함이 떨어진다. 컴파일 하는 동안, 이 플러그인은 모든 리소스가 들어가 있는 HTML 파일을 만들어 낸다. 만약 서버의 로직이 복잡하지 않다면, 이정도로도 충분할 것이다.

```html
<!-- index.html -->
<!DOCTYPE html>
<!-- ... -->
<script src="bundle.8e0d62a03.js"></script>
```

[WebpackManifestPlugin](https://github.com/danethurber/webpack-manifest-plugin)은 서버사이드에서 복잡한 로직이 포함되어 있다면 사용하기에 좋다. 빌드 과정에서 JSON 파일을 만드는데, 이 파일에는 파일명과 해쉬되지 않는 값, 그리고 파일명과 해쉬된 값을 매핑해준다. 그리고 서버에서는 이 JSON을 활용해 어떤 파일을 사용해야하는지 찾는다.

```json
// manifest.json
{
  "bundle.js": "bundle.8e0d62a03.js"
}
```

## 디펜던시를 추출하여 런타임에서 별도로 실행하기

### 디펜던시

앱의 디펜던시 (의존성)은 실제 앱의 코드보다 변화가 덜 자주 일어난다. 만약 이것을 다른 파일로 분리한다면, 브라우저는 별도로 캐시하기가 한결 편해지고, 앱코드만 바뀐다고 하더라도 이들을 별도로 다운로드 받지 않을 것이다.

> 웹팩에서, 애플리케이션 코드를 각각 다른 파일로 나눈것을 `chunk`라고 부른다.

디펜던시를 별도의 chunk로 분리하기 위해서는, 아래 3가지 과정을 거치면 된다.

1. output 파일명을 `[name].[chunkname].js`로 바꾼다.
   ```javascript
   // webpack.config.js
   module.exports = {
     output: {
       // Before
       filename: 'bundle.[chunkhash].js',
       // After
       filename: '[name].[chunkhash].js',
     },
   }
   ```
2. `entry`를 object로 바꾼다.

```javascript
// webpack.config.js
module.exports = {
  // Before
  entry: './index.js',
  // After
  entry: {
    main: './index.js',
  },
}
```

위 코드에서, `main`은 chunk의 이름이다. 이 이름은 앞서 언급했던 `[name]`을 대체할 것이다. 그럼 지금부터, 앱을 빌드하게 되면 이 chunk는 모든 앱 코드에 포함되게 된다.

3. 웹팩4 부터는, `optimization.splitChunks.chunks.: 'all'`을 붙이면 된다.

```javascript
// webpack.config.js (for webpack 4)
module.exports = {
  optimization: {
    splitChunks: {
      chunks: 'all',
    },
  },
}
```

이 코드는 스마트 코드 스플리팅을 가능하게 해준다. 만약 벤더 코드가 30kb가 넘는다면 (최소화 및 gzip 이전에) 따로 추출해낸다. 그리고 이 단계에서 공통 코드도 추출하게 된다. 이는 빌드시에 여러개의 파일이 나올때 유용하다.

이렇게 바꾸고 나면, 매번 빌드시에 두개의 파일이 생성될 것이다. `main.[chunkhash].js` `vendor.[chunkhash].js` (웹팩 4의 경우 `vendors~main.[chunkhash].js`) 웹팩 4의 경우에는, 디펜던시가 그렇게 크지 않다면 벤더 번들을 만들어 내지 않는다.

```
$ webpack
Hash: ac01483e8fec1fa70676
Version: webpack 3.8.1
Time: 3816ms
                           Asset   Size  Chunks             Chunk Names
  ./main.00bab6fd3100008a42b0.js  82 kB       0  [emitted]  main
./vendor.d9e134771799ecdf9483.js  47 kB       1  [emitted]  vendor
```

이제 브라우저는 이 두 파일을 따로 캐싱할 것이며, 변화가 있는 파일만 별도로 다운로드 할 것이다.

### 웹팩 런타임 코드

애석하게도, 벤더 코드만 따로 추출하는 것으로는 부족하다. 만약 애플리케이션 코드에서 아래와 같이 변경이 있으면

```javascript
// index.js
…
…

// E.g. add this:
console.log('Wat');
```

그러면 `vendor`에도 변화가 발생했다는 것을 알 수 있다.

```
                           Asset   Size  Chunks             Chunk Names
./vendor.d9e134771799ecdf9483.js  47 kB       1  [emitted]  vendor
```

```
                            Asset   Size  Chunks             Chunk Names
./vendor.e6ea4504d61a1cc1c60b.js  47 kB       1  [emitted]  vendor
```

이는 모듈 코드와는 별개로, 웹팩 번들에 런타임(모듈 실행을 관리하는 코드 조각)이 포함되어 있기 때문이다. 코드를 여러 파일로 나누면, 이 파일들이 서로 chunk id로 각각 관련있는 파일들 끼리 연결되어 있기 때문이다.

```javascript
// vendor.e6ea4504d61a1cc1c60b.js
script.src =
  __webpack_require__.p +
  chunkId +
  '.' +
  {
    '0': '2f2269c7f0a55a5c1871',
  }[chunkId] +
  '.js'
```

웹팩은 런타임을 가작 마지막에 생성된 chunk에 넣는데, 우리의 경우에는 `vendor`가 그 파일이다. chunk가 각각 생길 때 마다, 코드 조각이 바뀌게되고, 이는 `vendor` 파일 전체의 변화를 초래한다.

이를 해결하기 위해서는, 런타임도 따로 분리해야 한다. webpack 4 버전에서는, `optimization.runtimeChunk`를 활성화 하여야 한다.

```javascript
// webpack.config.js (for webpack 4)
module.exports = {
  optimization: {
    runtimeChunk: true,
  },
}
```

이 작업까지 마치게 되면, 세 개의 파일이 생기게 된다.

```
$ webpack
Hash: ac01483e8fec1fa70676
Version: webpack 3.8.1
Time: 3816ms
                            Asset     Size  Chunks             Chunk Names
   ./main.00bab6fd3100008a42b0.js    82 kB       0  [emitted]  main
 ./vendor.26886caf15818fa82dfa.js    46 kB       1  [emitted]  vendor
./runtime.79f17c27b335abc7aaf4.js  1.45 kB       3  [emitted]  runtime
```

`index.html`은 위 순서의 반대로 생성되게 된다.

```html
<!-- index.html -->
<script src="./runtime.79f17c27b335abc7aaf4.js"></script>
<script src="./vendor.26886caf15818fa82dfa.js"></script>
<script src="./main.00bab6fd3100008a42b0.js"></script>
```

**더 알아보기**

- [웹팩의 장기간 캐싱](https://webpack.js.org/guides/caching/)
- [웹팩 런타임과 매니페스트에 관련된 문서](https://webpack.js.org/concepts/manifest/)
- [CommonsChunkPlugin을 최대한 활용하기](https://medium.com/webpack/webpack-bits-getting-the-most-out-of-the-commonschunkplugin-ab389e5f318)
- [`optimization.splitChunk`와 `optimization.runtimeChunk` 는 어떻게 작동하는가](https://gist.github.com/sokra/1522d586b8e5c0f5072d7565c2bee693)

## 웹팩 런타임을 인라인으로 처리해서 http request를 절약하기

webpack runtime을 인라인 코드로 넣는 것도 고려해볼만 하다.

```html
<!-- index.html -->
<script src="./runtime.79f17c27b335abc7aaf4.js"></script>
```

```html
<!-- index.html -->
<script>
  !function(e){function n(r){if(t[r])return t[r].exports;…}} ([]);
</script>
```

```html
<!-- index.html -->
<script>
  !function(e){function n(r){if(t[r])return t[r].exports;…}} ([]);
</script>
```

런타임 파일은 작기 때문에, 이를 인라인으로 처리하는 것이 http 요청을 줄이는데 도움을 준다.(http/1에서는 굉장히 중요하지만, HTTP/2에서는 그렇게 크지 않지만 - 아무튼 도움이 된다.)

[HtmlWebpackPlugin](https://github.com/jantimon/html-webpack-plugin)을 활용하여 html을 만든다면, [InlineSourcePlugin](https://github.com/DustinJackson/html-webpack-inline-source-plugin)을 활용하면 된다.

```javascript
// webpack.config.js
const HtmlWebpackPlugin = require('html-webpack-plugin')
const InlineSourcePlugin = require('html-webpack-inline-source-plugin')

module.exports = {
  plugins: [
    new HtmlWebpackPlugin({
      // Inline all files which names start with “runtime~” and end with “.js”.
      // That’s the default naming of runtime chunks
      inlineSource: 'runtime~.+\\.js',
    }),
    // This plugin enables the “inlineSource” option
    new InlineSourcePlugin(),
  ],
}
```

만약 커스텀 서버 로직을 사용하고 있다면,

1. [WebpackManifestPlugin](https://github.com/danethurber/webpack-manifest-plugin)을 추가하여 생성된 런타임 chunk의 이름을 알아낸다.

```javascript
// webpack.config.js (for webpack 4)
const ManifestPlugin = require('webpack-manifest-plugin')

module.exports = {
  plugins: [new ManifestPlugin()],
}
```

이 플러그인과 함께 빌드하면, 아래와 같은 파일이 만들어진다.

```json
// manifest.json
{
  "runtime~main.js": "runtime~main.8e0d62a03.js"
}
```

2. 런타임 chunk의 내용을 편한대로 인라인으로 적어둔다.

```javascript
// server.js
const fs = require('fs')
const manifest = require('./manifest.json')

const runtimeContent = fs.readFileSync(manifest['runtime~main.js'], 'utf-8')

app.get('/', (req, res) => {
  res.send(`
    …
    <script>${runtimeContent}</script>
    …
  `)
})
```

## 당장 필요하지 않은 코드는 레이지 로딩으로 처리하기

가끔은, 페이지를 중요한 부분과 덜 중요한 부분으로 나눌 수 있다.

- 만약 유튜브에서 영상을 로딩한다면, 댓글보다는 영상이 더 중요하다
- 만약 뉴스사이트에서 기사를 본다면, 기사가 광고보다는 더 중요하다

이러한 경우, 더 중요한 요소를 먼저 다운로드 하고, 덜 중요한 것은 나중에 다운로드 하여 페이지 성능 향상에 도움을 줄 수 있다. [import() 함수](https://webpack.js.org/api/module-methods/#import-)와 [code-splitting](https://webpack.js.org/guides/code-splitting/)을 아래와 같이 활용하자.

```javascript
// videoPlayer.js
export function renderVideoPlayer() { … }

// comments.js
export function renderComments() { … }

// index.js
import {renderVideoPlayer} from './videoPlayer';
renderVideoPlayer();

// …Custom event listener
onShowCommentsClick(() => {
  import('./comments').then((comments) => {
    comments.renderComments();
  });
});
```

`import()`를 활용하여 다이나믹 로딩을 할 모듈을 지정해둔다. 웹팩이 해당 코드를 만나게 되면, 이를 별도의 chunk로 분리하게 된다.

```
$ webpack
Hash: 39b2a53cb4e73f0dc5b2
Version: webpack 3.8.1
Time: 4273ms
                            Asset     Size  Chunks             Chunk Names
      ./0.8ecaf182f5c85b7a8199.js  22.5 kB       0  [emitted]
   ./main.f7e53d8e13e9a2745d6d.js    60 kB       1  [emitted]  main
 ./vendor.4f14b6326a80f4752a98.js    46 kB       2  [emitted]  vendor
./runtime.79f17c27b335abc7aaf4.js  1.45 kB       3  [emitted]  runtime
```

그리고 해당 코드를 `import()` 함수를 만날 때만 실행하게 된다.

이는 `main` 번들을 더 작게하여, 초기 로딩 타임을 줄여주는데 도움을 준다. 더 나아가 이는 캐싱에도 도움을 준다. main chunk의 코드에 변화가 있어도, `comments` chunk에는 변화가 생기지 않는다.

> 만약 바벨을 사용한다면, [syntax-dynamic-import](https://www.npmjs.com/package/babel-plugin-syntax-dynamic-import)를 사용해야 해당 코드를 사용할 수 있다.

**더 읽어보기**

- [import()관련 webpack 문서](https://webpack.js.org/api/module-methods/#import-)
- [import()](https://github.com/tc39/proposal-dynamic-import) 구현을 위한 자바스크립트 제안

## 코드를 라우팅과 페이지 단위로 나누기

애플리케이션에 다양한 페이지와 라우팅이 있는데, 만약 모든 자바스크립트 코드가 하나의 자바스크립트 파일 (`main`)에 의존하고 있다면, 각 요청마다 몇 바이트 씩 더 소비하고 있을 수 있다. 예를 들어, 사용자가 페이지에 방문했을 때

![](https://developers.google.com/web/fundamentals/performance/webpack/site-home-page.png)

다른 페이지에 있는 아티클과 관련된 코드를 미리 로딩할 필요가 없다. 만약 또한 사용자가 항상 특정 페이지에만 반복하고, 코드에 변화가 있을 경우에는 - 웹팩이 모든 번들의 무효화 시키므로 전체 앱을 다운로드 해야 하는 불편함이 존재한다.

만약 애플리케이션을 페이지 (SPA의 경우 라우팅) 단위로 나눈다면, 사용자는 해당 영역에 필요한 코드만 다운로드 할 수 있다. 나아가 브라우저는 캐시를 더욱 장녀스럽게 활용할 수 있다. 하나의 페이지에서만 코드가 변경 되었다면, 변경된 chunk만 무효화 할 것이다.

### 싱글페이지 애플리케이션의 경우

라우팅으로 관리하는 싱글 페이지 애플리케이션의 경우 `import()`를 활용하는 것이 좋다. 만약 프레임워크를 활용하고 있다면,

- 리액트 [react-router의 코드 스플리팅](https://reacttraining.com/react-router/web/guides/code-splitting)
- 뷰 [vue.js의 레이지 로딩 라우팅](https://router.vuejs.org/en/advanced/lazy-loading.html)

### 전통적인 멀티페이지 애플리케이션

webpack의 [entry points](https://webpack.js.org/concepts/entry-points/)를 활용한다. 만약 애플리케이션에 세개의 페이지가 있다면, 아래와 같은 방식으로 나누면 된다.

```javascript
// webpack.config.js
module.exports = {
  entry: {
    home: './src/Home/index.js',
    article: './src/Article/index.js',
    profile: './src/Profile/index.js',
  },
}
```

각 엔트리 파일별로, 웹팩은 각 엔트리에서 필요한 모듈을 별도의 의존성으로 나누어서 빌드 해준다.

```
$ webpack
Hash: 318d7b8490a7382bf23b
Version: webpack 3.8.1
Time: 4273ms
                            Asset     Size  Chunks             Chunk Names
      ./0.8ecaf182f5c85b7a8199.js  22.5 kB       0  [emitted]
   ./home.91b9ed27366fe7e33d6a.js    18 kB       1  [emitted]  home
./article.87a128755b16ac3294fd.js    32 kB       2  [emitted]  article
./profile.de945dc02685f6166781.js    24 kB       3  [emitted]  profile
 ./vendor.4f14b6326a80f4752a98.js    46 kB       4  [emitted]  vendor
./runtime.318d7b8490a7382bf23b.js  1.45 kB       5  [emitted]  runtime
```

예를 들어,`article` 페이지에 lodash가 들어 있다면, `home`과 `profile`에는 해당 라이브러리가 포함되지 않으므로, `home`만 방문하는 유저는 `lodash`를 받지 않게 된다.

그러나 이 방법도 단점이 존재한다. 만약 두개의 entry에서 lodash가 필요하고, 해당 의존성으로 `vendor`로 가져가지 않았다면 두개의 엔트리 포인트에서 모두 lodash를 가지게 된다. 이를 해결하기 위해서는 , wepback4의 `optimization.splitChunks.chunks: 'all'`를 웹팩 설정에 넣으면 된다.

```javascript
// webpack.config.js (for webpack 4)
module.exports = {
  optimization: {
    splitChunks: {
      chunks: 'all',
    },
  },
}
```

이 옵션은 스마트 코드 스플리팅을 활성화 시킨다. 이 옵션은 각 다른 파일에 있는 공통 코드를 자동으로 공통단위로 올려준다.

- [웹팩의 entry points](https://webpack.js.org/concepts/entry-points/)
- [웹팩의 CommonsChunkPlugin](https://webpack.js.org/plugins/commons-chunk-plugin/)

## 모듈 ID를 더욱 안정적으로 관리하기

코드를 빌드 할때, 웹팩은 각각의 모듈에 ID를 부여한다. 이 ID 는 번들 내의 `require()`로 사용 된다. 이러한 ID들은 모듈 경로 이전에 있는 빌드 결과물에서 볼 수 있다.

```
$ webpack
Hash: df3474e4f76528e3bbc9
Version: webpack 3.8.1
Time: 2150ms
                           Asset      Size  Chunks             Chunk Names
      ./0.8ecaf182f5c85b7a8199.js  22.5 kB       0  [emitted]
   ./main.4e50a16675574df6a9e9.js    60 kB       1  [emitted]  main
 ./vendor.26886caf15818fa82dfa.js    46 kB       2  [emitted]  vendor
./runtime.79f17c27b335abc7aaf4.js  1.45 kB       3  [emitted]  runtime
```

```
   [0] ./index.js 29 kB {1} [built]
   [2] (webpack)/buildin/global.js 488 bytes {2} [built]
   [3] (webpack)/buildin/module.js 495 bytes {2} [built]
   [4] ./comments.js 58 kB {0} [built]
   [5] ./ads.js 74 kB {1} [built]
    + 1 hidden module
```

기본값으로, ID는 카운터로 계산된다. (첫번째 모듈은 0, 두번째는 1...) 문제는 여기에서 모듈이 추가 된다면, 이 모듈이 모듈 리스트의 중간에 나타나서 모든 다음 모듈의 아이디를 바꿔 버린다는 것이다.

```
$ webpack
Hash: df3474e4f76528e3bbc9
Version: webpack 3.8.1
Time: 2150ms
                           Asset      Size  Chunks             Chunk Names
      ./0.5c82c0f337fcb22672b5.js    22 kB       0  [emitted]
   ./main.0c8b617dfc40c2827ae3.js    82 kB       1  [emitted]  main
 ./vendor.26886caf15818fa82dfa.js    46 kB       2  [emitted]  vendor
./runtime.79f17c27b335abc7aaf4.js  1.45 kB       3  [emitted]  runtime
   [0] ./index.js 29 kB {1} [built]
   [2] (webpack)/buildin/global.js 488 bytes {2} [built]
   [3] (webpack)/buildin/module.js 495 bytes {2} [built]
```

여기에 모듈을 추가했다고 하면

```
   [4] ./webPlayer.js 24 kB {1} [built]
```

`comments`는 아이디가 밀려서 5번으로 바뀌게 되었다.

```
   [5] ./comments.js 58 kB {0} [built]
```

그리고 `adsj.js`는 6번으로 밀린다.

```
   [6] ./ads.js 74 kB {1} [built]
       + 1 hidden module
```

이는 실제 코드가 바뀌지 않았음에도 불구하고 이후에 모든 모듈들을 무효화 시켜 버린다. 따라서 이를 해결하기 위해서는, 모듈 아이디를 계산하는 방법을 [HashedModuleIdsPlugin](https://webpack.js.org/plugins/hashed-module-ids-plugin/)으로 바꾸는 것이 있다. 이는 카운토를 기반으로 한 ID를 모듈 경로를 해쉬한 방식으로 고친다.

```
$ webpack
Hash: df3474e4f76528e3bbc9
Version: webpack 3.8.1
Time: 2150ms
                           Asset      Size  Chunks             Chunk Names
      ./0.6168aaac8461862eab7a.js  22.5 kB       0  [emitted]
   ./main.a2e49a279552980e3b91.js    60 kB       1  [emitted]  main
 ./vendor.ff9f7ea865884e6a84c8.js    46 kB       2  [emitted]  vendor
./runtime.25f5d0204e4f77fa57a1.js  1.45 kB       3  [emitted]  runtime
```

```
[3IRH] ./index.js 29 kB {1} [built]
[DuR2] (webpack)/buildin/global.js 488 bytes {2} [built]
[JkW7] (webpack)/buildin/module.js 495 bytes {2} [built]
[LbCc] ./webPlayer.js 24 kB {1} [built]
[lebJ] ./comments.js 58 kB {0} [built]
[02Tr] ./ads.js 74 kB {1} [built]
    + 1 hidden module
```

이 방법을 활용하면, 모듈의 ID는 모듈이 삭제되거나 이름이 변경될때만 바뀌게 된다. 새로운 모듈의 등장은 더이상 다른 모듈의 ID에 영향을 미치지 않는다.

```javascript
// webpack.config.js
module.exports = {
  plugins: [new webpack.HashedModuleIdsPlugin()],
}
```

## 요약

- 번들을 캐시하고, 번들명을 바꿔서 다른 버전을 관리하라
- 애플리케이션 코드를 app code, vender code, runtime으로 나누어라
- runtime코드는 인라인으로 관리해서 HTTP 요청을 줄여라
- 중요하지 않은 코드는 `import`로 레이지 로딩하라
- 불필요한 것의 로딩을 줄이기 위해 라우팅/페이지 단위로 코드를 나눠라.
