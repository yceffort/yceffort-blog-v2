---

title: Tensorflow.js - 07. Build an image classifier
tags:
  - machine-learning
  - ai
  - tensorflow
  - javascript
published: true
date: 2020-02-04 05:41:09
description: "`toc tight: true, from-heading: 1 to-heading: 3 ` # Transfer
learning image classifier 본 튜토리얼에서는, 브라우저 환경에서 Tesnorflow.js를 활용하여 커스텀 이미지
분류기를 만드는 방법을 알아봅니다. 이번 장에서는, 최소한의 데이터를 활용하여 높은 정확도를 가진 모델..."
category: machine-learning
slug: /2020/02/tensorflowjs-07-build-an-image-classifier/
template: post
---

## Table of Contents

# Transfer learning image classifier

본 튜토리얼에서는, 브라우저 환경에서 Tesnorflow.js를 활용하여 커스텀 이미지 분류기를 만드는 방법을 알아봅니다.

이번 장에서는, 최소한의 데이터를 활용하여 높은 정확도를 가진 모델을 만들기 위하여 transfer learning(전이학습)을 활용해볼 것입니다. 우리는 이미 잘 학습되어 있는 MobileNet이라고 불리우는 이미지 분류기를 활용할 것입니다. 이 모델을 기반으로, 이미지 클래스를 사용자 정의 하여 학습해볼 것입니다.

## 1. Introduction

이번 튜토리얼에서는, 간단한 [teachable machine](https://teachablemachine.withgoogle.com/)을 만들어 볼 것입니다. teachable machine이란, 자바스크립트로 작성된 유연하고도 강력한 머신러닝 라이브러리인 tensorflow.js를 활용하여 브라우저에서 작동할 수 있는 커스텀 이미지 분류기 입니다. 먼저 MobileNet이라고 불리우는 모델을 브라우저 환경에서 불러오고 실행해볼 것입니다. 그 다음에는 전이학습을 활용하여 이미 학습된 MobileNet 모델을 커스터마이징하고 우리의 앱에서 실행할 수 있도록 할 것입니다.

이 튜토리얼에서는 teachable machine 애플리케이션을 만드는데 필요한 이론적 배경을 소개하지는 않습니다. 만약 궁금하다면, [이 튜토리얼](https://beta.observablehq.com/@nsthorat/how-to-build-a-teachable-machine-with-tensorflow-js)을 참고하시기 바랍니다.

### 배우게 될 것

- 이미 학습된 MobileNet 모델을 불러오고 새로운 데이터로 예측하는 방법
- 웹캠을 활용하여 예측하는 법
- MobileNet을 즉시 활성화하여, 웹캠에서 인식된 이미지를 분류하는 법

## 2. 요구 사항

1. 최신버전의 크롬 또는 모던 브라우저
2. 로컬 머신에서 실행할 수 있는 텍스트 에디터, 혹은 웹에서 이용할 수 있는 Codepen이나 Glitch
3. HTML, CSS, Javascript, Chrome Devtool에 대하한 기본적인 지식
4. 신경망에 대한 높은 수준의 이해. 만약 이에 관련된 지식이 필요하다면, [3blue1brown](https://www.youtube.com/watch?v=aircAruvnKk)이나 [video on Deep Learning in Javascript by Ashi Krishnan](https://www.youtube.com/watch?v=SV-cgdobtTA)를 보는 것을 추천합니다.

## 3. Tensorflow.js와 MobileNet 불러오기

index.html을 열고 아래 코드를 넣어주세요.

```html
<html>
  <head>
    <!-- Load the latest version of TensorFlow.js -->
    <script src="https://unpkg.com/@tensorflow/tfjs"></script>
    <script src="https://unpkg.com/@tensorflow-models/mobilenet"></script>
  </head>
  <body>
    <div id="console"></div>
    <!-- Add an image that we will use to test -->
    <img
      id="img"
      crossorigin
      src="https://i.imgur.com/JlUvsxa.jpg"
      width="227"
      height="227"
    />
    <!-- Load index.js after the content of the page -->
    <script src="index.js"></script>
  </body>
</html>
```

## 4. 브라우저에서 MobileNet을 로딩

`index.js`를만들고 연다음에, 아래 코드를 넣어주세요.

```javascript
let net

async function app() {
  console.log('Loading mobilenet..')

  // Load the model.
  net = await mobilenet.load()
  console.log('Successfully loaded model')

  // Make a prediction through the model on our image.
  const imgEl = document.getElementById('img')
  const result = await net.classify(imgEl)
  console.log(result)
}

app()
```

## 5. MobileNet을 테스트 하기

index.html을 웹브라우저에서 열어보세요.

자바스크립트 콘솔에, 사진의 강아지가 어떤 강아지인지 예측한 작업내역을 볼 수 있습니다. 이 작업은 모델을 다운로드 하는데 시간이 약간의 시간이 걸릴 수 있으므로 조금 기다리시기 바랍니다.

```json
[
  { "className": "kelpie", "probability": 0.5226836204528809 },
  {
    "className": "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier",
    "probability": 0.1948588341474533
  },
  { "className": "malinois", "probability": 0.11830379068851471 }
]
```

## 6. 브라우저에서 웹캠을 활용하여 MobileNet 모델을 실행하기

지금부터는 좀 더 실시간 데이터를 다뤄보도록 하겠습니다. 웹캠으로 부터 넘어온 이미지를 바탕으로 예측을 할 수 있도록 작업해보겠습니다.

먼저, 웹캠을 video element에 설정해보겠습니다. `index.html` 파일을 열어, `body` 에 아래 코드를 넣고, 강아지 이미지 로딩을 위해 사용했던 `img` 태그를 삭제하세요.

```html
<video autoplay playsinline muted id="webcam" width="224" height="224"></video>
```

`index.js`를 열고 파일의 최상단에 위 코드를 넣어주세요.

```javascript
const webcamElement = document.getElementById('webcam')
```

이제 `app()` function 내부에서, 이미지를 기반으로한 예측을 제거하고 대신 웹캠 엘리먼트를 통해 예측을 하는 무한루프 로직을 만들 수 있습니다.

```javascript
async function app() {
  console.log('Loading mobilenet..')

  // Load the model.
  net = await mobilenet.load()
  console.log('Successfully loaded model')

  // Create an object from Tensorflow.js data API which could capture image
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement)
  while (true) {
    const img = await webcam.capture()
    const result = await net.classify(img)

    document.getElementById('console').innerText = `
      prediction: ${result[0].className}\n
      probability: ${result[0].probability}
    `
    // Dispose the tensor to release the memory.
    img.dispose()

    // Give some breathing room by waiting for the next animation frame to
    // fire.
    await tf.nextFrame()
  }
}
```

웹페이지에서 콘솔창을 열어본다면, MobileNet에서 웹캠의 매 프레임을 바탕으로 예측한 결과를 볼수 있을 것입니다.

그러나 이는 ImageNet의 데이터 세트가 일반적으로 웹캠에 나타나는 이미지와 매우 많이 다르기 때문에 정확하지 않을 수 있습니다. 이것을 테스트 해보는 한가지 방법은, 노트북 카메라 (웹캠)에 강아지 사진이 있는 폰을 비추는 것입니다.

## 7. MobileNet에 Custom 분류기를 추가하기

이제, 조금 더 유용하게 만들어 봅시다. 우리는 웹캠을 활용하여 3개의 object로 분류하는 분류기를 만들어 볼 것입니다. 분류를 위해서 MobileNet을 활용할 것입니다만, 이번에는 특정 웹캠 이미지에 대한 모델의 내부활성화를 활용하여, (internal representation, activation) 분류에 사용해 볼 것입니다.

여기에서는 "K-Nearest Neighbors Classifier"라고 불리는 모듈을 사용할 것입니다. 이는 웹캠에서 입력되는 이미지를 서로다른 카테고리로 효과적으로 분리해주며, 사용자가 예측을 시도할 때 단순히 가장 확률이 높은 결과 하나만 리턴하게 할 것입니다.

KNN Classifier를 index.html의 head 태그 마지막에 import 하세요.

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/knn-classifier"></script>
```

index.html의 마지막에 세개의 버튼을 추가합니다. 각각의 버튼들은 이미지를 모델에서 훈련시키는데 사용됩니다.

```html
<button id="class-a">Add A</button>
<button id="class-b">Add B</button>
<button id="class-c">Add C</button>
```

`index.js`의 최상단에서는 classifier를 선언합니다.

```javascript
const classifier = knnClassifier.create()
```

app 함수를 업데이트 합니다.

```javascript
async function app() {
  console.log('Loading mobilenet..')

  // 모델을 로드한다.
  net = await mobilenet.load()
  console.log('Successfully loaded model')

  // 웹캠의 데이터를 tensor로 변환하는 tensorflwjs api를 만든다.
  const webcam = await tf.data.webcam(webcamElement)

  // 웹캠에서 이미지를 로딩하고, 특정 클래스와 연관짓는다.
  const addExample = async (classId) => {
    // Capture an image from the web camera.
    const img = await webcam.capture()

    // conv_preds라는 활성함수를 가져오고, KNN 분류기에 넘긴다.
    const activation = net.infer(img, 'conv_preds')

    // 중간 활성화 함수를 분류기에 넘긴다.
    classifier.addExample(activation, classId)

    // tensor를 메모리에 dispose 한다.
    img.dispose()
  }

  // 버튼을 클릭하면, 각 클래스별로 예제를 추가한다.
  document
    .getElementById('class-a')
    .addEventListener('click', () => addExample(0))
  document
    .getElementById('class-b')
    .addEventListener('click', () => addExample(1))
  document
    .getElementById('class-c')
    .addEventListener('click', () => addExample(2))

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture()

      // 웹켐의 mobilnet의 활성함수를 가져온다.
      const activation = net.infer(img, 'conv_preds')
      // 분류기에서 가장 근접한 결과를 가져온다.
      const result = await classifier.predictClass(activation)

      const classes = ['A', 'B', 'C']
      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `

      img.dispose()
    }

    await tf.nextFrame()
  }
}
```

이제 index.html 페이지를 로드 하면, 세 클래스에 각각에 대한 이미 캡처를 하기 위하여 공통 오브젝트를 사용하거나, 얼굴/바디 제스쳐를 활용할 수 있다. `add` 버튼을 클릭할 때마다, 하나의 이미지가 예제로서 해당 클래스에 추가됩니다. 이작업을 하는 동안, 모델은 웹캠 이미지에 대한 예측을 계속하고 결과를 실시간으로 보여줍니다.

## 8. Optional: 예제를 확장하기

이제 아무 동작도 나타내지 않는 클래스를 추가해 보세요.

## 9. 우리가 배운 것

이 예제에서, Tensorflow.js를 활용하여 간단한 머신러닝 학습 웹 애플리케이션을 구현했습니다. 웹캠에서 이미지를 분류하기 위해 미리 학습된 MobileNet 모델을 로드하고 사용했습니다. 그런 다음 모델을 커스텀하여 이미지를 세가지 사용자 정의 범주로 분류합니다.

[예제](https://codesandbox.io/s/tensorflowjs-07-build-an-image-classifier-28kc5)
