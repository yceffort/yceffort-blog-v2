---
title: Tensorflow.js - 02. 설치하기
date: 2019-12-20 10:49:01
published: true
tags:
  - machine-learning
  - ai
  - tensorflow
  - javascript
description: "[이전글 보기](/2019/12/20/tensorflowjs-01-get-started/) ## 설치  ### 브라우저
  설치  Tensorflow.js를 설치하는 방법은 두 가지가 있습니다.  - Script tag를 이용하는법 - npm을 이용해서 설치하고,
  Parcel, Webpack, Rollup 같은 빌드 툴을 사용  뉴비 웹 개발자거나, 위 에서..."
category: machine-learning
slug: /2019/12/20/tensorflowjs-02-setup/
template: post
---
[이전글 보기](/2019/12/20/tensorflowjs-01-get-started/)

## 설치

### 브라우저 설치

Tensorflow.js를 설치하는 방법은 두 가지가 있습니다.

- Script tag를 이용하는법
- npm을 이용해서 설치하고, Parcel, Webpack, Rollup 같은 빌드 툴을 사용

뉴비 웹 개발자거나, 위 에서 언급한 패키지 들을 전혀 모른다면 스크립트 태그를 활용하세요. 혹은 조금 경험 이 있거나, 큰 규모의 프로그램을 계획하고 있다면 빌드 툴 활용을 검토해보세요.

<!-- excerpt -->

#### 스크립트 태그 사용하기

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js"></script>
```

```javascript
// linear regression 모델
const model = tf.sequential()
model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

model.compile({ loss: "meanSquaredError", optimizer: "sgd" })

// 훈련을 위한 임의 데이터 생성
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1])
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1])

// 데이터 기반으로 훈련시키기
model.fit(xs, ys, { epochs: 10 }).then(() => {
  // 훈련한 모델을 기반으로 데이터 예측
  model.predict(tf.tensor2d([5], [1, 1])).print()
  // dev tool에 결과 값이 나온다.
})
```

### npm을 이용해서 설치하기

`npm cli`나 `yarn`둘다 활용할 수 있습니다.

```
yarn add @tensorflow/tfjs
```

```
npm install @tensorflow/tfjs
```

```javascript
import * as tf from "@tensorflow/tfjs"

// linear regression 모델
const model = tf.sequential()
model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

model.compile({ loss: "meanSquaredError", optimizer: "sgd" })

// 훈련을 위한 임의 데이터 생성
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1])
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1])

// 데이터 기반으로 훈련시키기
model.fit(xs, ys, { epochs: 10 }).then(() => {
  // 훈련한 모델을 기반으로 데이터 예측
  model.predict(tf.tensor2d([5], [1, 1])).print()
  // 결과 값이 나온다.
})
```

### Node.js 설치

`npm cli`나 `yarn`둘다 활용할 수 있습니다.

**Option1** native C++ 바인딩이 포함되어 있는 Tensorflow.js 설치

```
yarn add @tensorflow/tfjs-node
```

```
npm install @tensorflow/tfjs-node
```

**Option2** (리눅스 만 가능) 만약 시스템에서 CUDA NVIDIA GPU를 활용 가능하다면, 더 고성능 퍼포먼스를 위해 GPU 패키지를 사용할 수도 있습니다.

```
yarn add @tensorflow/tfjs-node-gpu
```

```
npm install @tensorflow/tfjs-node-gpu
```

**Option3** 순수 자바스크립트 버전 설치. 셋 중에 가장 느린 버전입니다.

```
yarn add @tensorflow/tfjs
```

```
npm install @tensorflow/tfjs
```

```javascript
const tf = require("@tensorflow/tfjs")

// 옵셔널
// '@tensorflow/tfjs-node-gpu' gpu와 사용하고 싶다면
require("@tensorflow/tfjs-node")

// Train a simple model:
const model = tf.sequential()
model.add(tf.layers.dense({ units: 100, activation: "relu", inputShape: [10] }))
model.add(tf.layers.dense({ units: 1, activation: "linear" }))
model.compile({ optimizer: "sgd", loss: "meanSquaredError" })

const xs = tf.randomNormal([100, 10])
const ys = tf.randomNormal([100, 1])

model.fit(xs, ys, {
  epochs: 100,
  callbacks: {
    onEpochEnd: (epoch, log) =>
      console.log(`Epoch ${epoch}: loss = ${log.loss}`),
  },
})
```

### TypeScript

타입스크립트 환경에서 사용한다면, 그리고 프로젝트에서 strict null 체킹을 한다면`skipLibCheck: true` 을 `tsconfig.json`에 포함시켜서 컴파일 도중에 에러가 나지 않도록 처리해야 합니다.
