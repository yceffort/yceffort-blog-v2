---
title: Tensorflow.js - 03. Linear Regression
date: 2019-12-23 07:30:29
published: true
tags:
  - machine-learning
  - ai
  - tensorflow
  - javascript
description:
  '# Linear Regression 몇 번째 선형 회귀인지 알 수 없다.  ## 01. 2d data로
  예측해보기  이번 튜토리얼에서는 자동차 세트를 표현한 숫자 데이터로 예측하는 모델을 훈련시켜 봅니다.  이 연습에서는 다양한 종류의 모델을
  훈련하는 공통적인 단계를 보여주고, 이에 따라 작은 데이터 세트와 간단한 모델을 사용합니다. 1차적인 목표는 Te...'
category: machine-learning
slug: /2019/12/23/tensorflowjs-03-linear_regression/
template: post
---

# Linear Regression

몇 번째 선형 회귀인지 알 수 없다.

## 01. 2d data로 예측해보기

이번 튜토리얼에서는 자동차 세트를 표현한 숫자 데이터로 예측하는 모델을 훈련시켜 봅니다.

이 연습에서는 다양한 종류의 모델을 훈련하는 공통적인 단계를 보여주고, 이에 따라 작은 데이터 세트와 간단한 모델을 사용합니다. 1차적인 목표는 Tensorflow.js의 훈련 모델과 관련된 기본적인 용어, 개념 및 신택스를 숙지하고, 추가 학습을 위한 발판을 마련하는 것입니다.

## 02. Set up

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>TensorFlow.js Tutorial</title>
    <!-- Import TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js"></script>
    <!-- Import tfjs-vis -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@1.0.2/dist/tfjs-vis.umd.min.js"></script>
    <!-- Import the main script file -->
    <script src="script.js"></script>
  </head>
  <body>
    <h1>Tensorflow.js</h1>
  </body>
</html>
```

```javascript
console.log('Hello TensorFlow')
```

## 03. 데이터를 읽어오고, 포맷팅하고, 시각화 하기

가장 먼저, 데이터를 읽어오고 포맷팅하고, 시각화하여 우리가 훈련하기 좋은 모델 상태로 만들어 보자.

여기에서 `cars`데이터를 쓸 것이다. (https://storage.googleapis.com/tfjs-tutorials/carsData.json) 이 데이터에는 아주 다양한 자동차에 대한 기능들이 담겨 있다. 먼저 튜토리얼로, `Horsepower`와 `Miles per gallon` 데이터만 가져와 보고자 한다.

```javascript
/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const carsDataReq = await fetch(
    'https://storage.googleapis.com/tfjs-tutorials/carsData.json',
  )
  const carsData = await carsDataReq.json()
  const cleaned = carsData
    .map((car) => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    .filter((car) => car.mpg != null && car.horsepower != null)

  return cleaned
}
```

우리가 원하는 필드가 null 인 데이터들은 다 삭제했다. 이 데이터를 scatterplot에 표현해서 어떻게 나오는지 살펴보자.

```javascript
async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData()
  const values = data.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }))

  tfvis.render.scatterplot(
    { name: 'Horsepower v MPG' },
    { values },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300,
    },
  )

  // More code will be added below
}

document.addEventListener('DOMContentLoaded', run)
```

페이지를 새로고침하면, 오른쪽에서 아래와 같은 `scatterplot`이 나타날 것이다. 그 데이터는 아래와 같은 형태를 띄고 있을 것이다.

![](https://codelabs.developers.google.com/codelabs/tfjs-training-regression/img/6a7452e88f16d8e.png)

이 패널은 `visor`라고 불리우며, [tfjs-vis](https://github.com/tensorflow/tfjs-vis)에서 제공하는 것이다. 이 라이브러리는 데이터를 시각화하는데 도움을 준다.

일반적으로, 데이터를 다룰 때 데이터를 살펴보고 필요한 경우 정리하는 방법을 찾는 것이 좋다. 이번 데이터의 경우, 필요한 필드가 없는 데이터를 모두 제거해야 했다. 데이터를 시각화하면, 모델이 학습할 수 있는 데이터에 일정한 구조가 있는지 여부를 파악하는데 도움을 얻을 수 있다.

위 그래프에서, 마력과 MPG 사이에 부정적인 상관관계가 있음을 알 수 있었다. 즉 마력이 올라갈수록, 갤런당 마일 수가 줄어든다.

> 기억하자. 데이터에 일정한 구조가 없다면 (패턴이 없다면) 그 데이터로 부터 얻을 수 있는 것은 없다.

### 작업의 개념화

방금 분석한 데이터는 아래과 같이 생겼다.

```json
...
{
  "mpg":15,
  "horsepower":165,
},
{
  "mpg":18,
  "horsepower":150,
},
{
  "mpg":16,
  "horsepower":150,
},
...
```

여기서 목표는 하나의 숫자, 즉 마력을 가지고 갤런당 마일수를 예측하는 것을 학습하는 것이다. 다음 섹션에서는 1:1 매핑이 중요하므로, 꼭 기억해두록 하자.

마력과 MPG로 부터 학습하여 예측하는 이러한 예제들을 해결할 수 있는 신경망을 만들어 볼 것이다. 이번 예제 에서 처럼, 정답을 가지고 있는 예제에서 학습 하는 것을 [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning)이라고 한다.

## 4. 모델 아키텍쳐를 정의하기

이 섹션에서 우리는 모델 아키텍처를 설명하는 코드를 작성할 것이다. 모델 구조는 단지 "모델이 실행 중일 때 어떤 기능이 실행될 것인가" 또는 대안적으로 "모델이 답을 계산하기 위해 어떤 알고리즘을 사용할 것인가" 정도로 이해해 두면 된다.

ML 모델은 입력을 받고 결과를 만들어 내는 알고리즘이다. 신경망을 사용할 때, 알고리즘은 결과 값을 조절하는 '가중치' (숫자)를 가진 뉴런의 층이다. 훈련 과정은 그러한 무게에 대한 이상적인 '가중치'를 학습하게 된다.

```javascript
function createModel() {
  // sequential model 을 만든다.
  const model = tf.sequential()

  // 히든 레이어 하나를 추가한다.
  model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))

  // 아웃 풋 레이어 하나를 추가한다.
  model.add(tf.layers.dense({ units: 1, useBias: true }))

  return model
}
```

위 코드는 tensorflowjs에서 만들 수 있는 가장 단순한 예제다. 코드 하나씩 살펴보자.

### 모델 인스턴스화

```javascript
const model = tf.sequential()
```

이는 [tf.Model](https://js.tensorflow.org/api/latest/#class:Model)를 초기화 한다. 이 모델은 [sequential](https://js.tensorflow.org/api/latest/#sequential)인데, 그 이유는 입력값이 바로 출력값으로 이어지기 때문이다. 다른 종류의 모델의 경우에는 branch를 가질수 있으며, 혹은 여러개의 입력값과 출력값이 있을 수도 있다. 그러나 대부분의 경우에는 sequential일 가능성이 높다. Sequential Api는 사용하기도 더 쉽다.

### 레이어 추가

```javascript
model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }))
```

네트워크에 히든레이어를 추가한다. `dense` 레이어는 레이어의 일종으로, inputs에 matrix를 곱하고 (weight), 숫자를 더하는 (bias) 역할을 한다. 이 레이어가 네트워크에 첫번째에 위치하기 때문에, 우리의 입력값 `inputShape`를 정의할 필요가 있다. input으로 하나의 데이터가 들어가므로, `[1]`을 넣어둔다.

`units`은 weight matrix가 얼마나 클지 정하는 역할을 한다. 여기에서 1로 설정해 두어서, 우리는 데이터의 각 데이터의 input에 1의 weight가 있다고 전달할 수 있다.

> 알아두기: Dense Layer에서 useBias는 기본값으로 true이기 때문에 생략이 가능하다.

```javascript
model.add(tf.layers.dense({ units: 1 }))
```

위 코드는 아웃풋 레이어다. units을 1로 설정해서 한가지의 결과값만 나오게 한다.

> 알아두기: 위 예제에서, 히든레이어는 1개의 unit이 있다고 설정해 두었기 때문에 사실 위 아웃풋 레이어는 추가할 필요가 없다. 그러나 아웃풋 레이어를 따로 정의해 둠으로써, 입력과 출력의 일대일 매핑을 유지하면서 히든 레이어 계층의 units 수를 조절할 수 있다.

### 인스턴스 만들기

```javascript
// Create the model
const model = createModel()
tfvis.show.modelSummary({ name: 'Model Summary' }, model)
```

위 코드를 통해서 모델을 만들고, 각 레이어별 summary를 볼 수 있다.

## 5. 학습을 위해 데이터를 준비하기

Tensorflow.js 의 성능상으로 이점을 얻기 위해서는, 데이터를 [tensor](https://developers.google.com/machine-learning/glossary/#tensor)로 변환해야 한다. 또한 shuffling과 normalization를 활용하여 변환을 수행할 것이다.

```javascript
/**
 * 머신러닝을 위해 인풋값을 tensor로 변환한다.
 * 그리고 y축 데이터인 MPG 에 shuffling과 normalizing을 한다.
 */
function convertToTensor(data) {
  // tidy를 활용하면 중간에 만들어진 tensor들을 바로 해제할 수 있다.

  return tf.tidy(() => {
    // 1. 데이터를 섞는다.
    tf.util.shuffle(data)

    // 2. 데이터를 tensor로 변환한다.
    const inputs = data.map((d) => d.horsepower)
    const labels = data.map((d) => d.mpg)

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
    const labelTensor = tf.tensor2d(labels, [labels.length, 1])

    // 3. min-max scaling을 활용하여 데이터를 0-1사이로 만든다.
    const inputMax = inputTensor.max()
    const inputMin = inputTensor.min()
    const labelMax = labelTensor.max()
    const labelMin = labelTensor.min()

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin))
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin))

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // min-max 를 반환하여 나중에도 쓸 수 있게 한다.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  })
}
```

코드를 살펴보자

### 데이터 셔플

```javascript
// Step 1. Shuffle the data
tf.util.shuffle(data)
```

학습 알고리즘에 제공할 데이터를 무작위로 섞었다. Shuffling은 모델이 실제로 훈련될때 데이터 셋이 작은 단위인 batch로 쪼개지기 때문에 매우 중요한 단계다. Shuffling은 batch에 다양한 데이터가 섞여 들어갈 수 있도록 도움을 준다. 이 과정을 거침으로써

- 순서에 의존적이지 않는 데이터를 학습 시킬 수 있음
- subgroup에 민감하지 않는 데이터를 만들 수 있음 (훈련 초기에 마력이 높은 차량만 학습할 경우, 나머지 데이터 세트 훈련에 영향을 끼치지 않는 상관관계를 학습할 수 있음)

> Best Practice 1: Tensorflow.js에서 학습 알고리즘을 적용하기 전에 꼭 데이터를 셔플하도록 하자.

### Tensor로 변환

```javascript
// Step 2. Convert data to Tensor
const inputs = data.map((d) => d.horsepower)
const labels = data.map((d) => d.mpg)

const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
const labelTensor = tf.tensor2d(labels, [labels.length, 1])
```

여기서 두 개의 배열을 만들었는데, 하나는 input 이고 다른 하나는 ouput이다. 그리고 이를 각각 2d tensor로 변환하였다. 이 tensor는 각각 [num_examples, num_features_per_example]의 형태를 띌 것이다. 여기에서 inputs.length로 입력값의 개수를 넣을 수 있고, feature도 마력 하나 뿐이 므로, 1로 설정해둔다.

### 데이터 정규화

```javascript
const inputMax = inputTensor.max()
const inputMin = inputTensor.min()
const labelMax = labelTensor.max()
const labelMin = labelTensor.min()

const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))
```

다음으로 머신러닝의 또다른 관례중 하나인 정규화를 할 것이다. [min-max scaling](<https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)>)을 활용하여 데이터를 정규화하고, 0~1 사이에 위치하도록 한다. tensorflow.js는 너무 크지 않는 숫자로 작업하도록 되어 있기 때문에 정규화가 중요하다. 데이터를 표준화하여 0~1 -1~1 사이에 위치하게 하는 것이 보통이다. 어느정도 합리적인 수준까지 데이터를 정규화 하는 습관을 갖는다면, 더욱더 성공적으로 데이터를 학습 시킬 수 있다.

> Best Practice 2: 학습 전에 꼭 데이터 정규화를 염두해둬라.몇 몇 데이터셋은 정규화가 필요없을 수 있지만, 데이터를 정규화하면 효과적인 학습을 방해하는 클래스 문제를 제거할 수 있는 경우가 많다.
>
> 데이터를 텐서로 바꾸기 전에 정규화할 수 있다. 나중에 Tensorflow.js의 벡터화를 이용하여, 루프에 대한 명시적인 코드 없이 스케일링작업을 최소화 할 수 있다.

### 데이터와 정규화 범위 리턴

```javascript
return {
  inputs: normalizedInputs,
  labels: normalizedLabels,
  // Return the min/max bounds so we can use them later.
  inputMax,
  inputMin,
  labelMax,
  labelMin,
}
```

정규화 한 값, 정규화 후 값의 범위, 그리고 정규화 이전 값의 범위 모두를 리턴한다.

## 6. 모델 훈련

앞서 만든 모델과 tensor를 바탕으로 학습을 시켜보자.

```javascript
async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  })

  const batchSize = 32
  const epochs = 50

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] },
    ),
  })
}
```

### 학습 준비

```javascript
// Prepare the model for training.
model.compile({
  optimizer: tf.train.adam(),
  loss: tf.losses.meanSquaredError,
  metrics: ['mse'],
})
```

훈련 시키기에 앞서 모델을 컴파일 해야 한다. 이를 위해, 몇가지 중요한 사항을 짚고 넘어가야 한다.

- [optimizer](https://developers.google.com/machine-learning/glossary/#optimizer): 모델을 업데이트 할 때 이를 통제할 알고리즘 이다. Tensorflow.js에는 다양한 optimizer가 존재한다. 이 예제에서는 별도의 설정이 필요없고 빠르고 효과적인 adam optimizer를 사용한다.
- [loss](https://developers.google.com/machine-learning/glossary/#loss): 이 함수는 각 배치를 얼마나 잘 학습하고 있는지 알려주는 기능을 한다. 여기에서는 [meanSquaredError](https://developers.google.com/machine-learning/glossary/#MSE)를 사용하여 예측과 실제가 참인지 비교한다.

```javascript
const batchSize = 32
const epochs = 50
```

batchSize와 epcoch을 설정한다.

- [batchSize](https://developers.google.com/machine-learning/glossary/#batch_size) 란 매 훈련시에 사용할 subset 데이터 사이즈를 의미한다. 일반적으로 32~512 정도의 사이즈를 둔다. 여기에 이상적인 크기란 따로 없으며, 다양한 배치 크기에 대한 수학적 함의는 본 튜토리얼의 범위를 벗어나는 주제다.
- [epoch](https://developers.google.com/machine-learning/glossary/#epoch) 는 모델이 전체 데이터넷을 얼마나 살펴볼 것인지 횟수를 의미한다. 여기에서는 50으로 설정하여 50회 훈련을 하도록 한다.

### 훈련 loop 시작

```javascript
return await model.fit(inputs, labels, {
  batchSize,
  epochs,
  callbacks: tfvis.show.fitCallbacks(
    { name: 'Training Performance' },
    ['loss', 'mse'],
    { height: 200, callbacks: ['onEpochEnd'] },
  ),
})
```

`model.fit`은 훈련루프를 호출하는 함수다. 이는 비동기 함수이므로, promise가 리턴되며 호출하는 측에서는 언제 학습이 끝나는지 알 수 있다.

훈련과정을 모니터링 하기 위해, `model.fit`에 콜백함수를 넘길 수 있게 해준다. 여기에서는 [tfvis.show.fitCallbacks](https://js.tensorflow.org/api_vis/latest/#show.fitCallbacks) 를 활용하여 `loss`와 `mse`를 plot 차트로 그려본다.

### Put it all together

위에서 만든 함수들을 `run`함수에서 호출 하도록 해보자.

```javascript
// Convert the data to a form we can use for training.
const tensorData = convertToTensor(data)
const { inputs, labels } = tensorData

// Train the model
await trainModel(model, inputs, labels)
console.log('Done Training')
```

새로고침하면, 아래와 같이 뜰 것이다.

![tfjs-training-performance](../images/tfjs-training-preformance.gif)

이는 앞서서 선언한 콜백의 작품이다. 매 epoch마다 전체 데이터의 loss와 mse의 평균을 보여주고 있다. 모델을 훈련시킬 때 마다, 점차 내려가고 있는 것을 알 수 있다. 이 경우 우리의 측정 지표는 error 이므로(mse) 점차 내려가는 것을 보아야 한다.

> 경사하강에 대해 알고 싶으면, [이 비디오](https://www.youtube.com/watch?v=IHZwWFHWa-w)를 참고하라.

## 7. 예측 모델 만들기

모델이 훈련되었으니, 이제 예측을 한번 해볼 차례다. 저력? 에서 고력? 까지의 균일한 범위의 마력을 예측하는 것을 보고 모델이 어떤지 한번 평가해보자.

```javascript
function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData

  // 0과 1사이에서 균일한 숫자를 생성하여 예측
  // min-max 스케일링을 거꾸로 다시 적용하여 데이터를 비정규화 (원래 보던 데이터) 한다.
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100)
    const preds = model.predict(xs.reshape([100, 1]))

    const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin)

    const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin)

    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()]
  })

  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] }
  })

  const originalPoints = inputData.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }))

  tfvis.render.scatterplot(
    { name: 'Model Predictions vs Original Data' },
    {
      values: [originalPoints, predictedPoints],
      series: ['original', 'predicted'],
    },
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300,
    },
  )
}
```

위 코드에서 주의해야 할 몇가지가 있다.

```javascript
const xs = tf.linspace(0, 1, 100)
const preds = model.predict(xs.reshape([100, 1]))
```

이 코드에서는 새로운 100개의 예제를 만들어 모델에 제공했다. `Model.predict`는 이 예제들을 어떻게 모델에 적용하는지를 보여준다. 명심해야 할 것은, 학습시킬 때와 마찬가지의 데이터 형태 `[num_examples, num_features_per_example]`를 띄어야 한다는 것이다.

```javascript
// Un-normalize the data
const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin)

const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin)
```

0~1 형태가 아닌 원래 데이터 형태로 돌아오기 위해, 정규화 하는 과정을 거꾸로 다시 거쳤다.

```javascript
return [unNormXs.dataSync(), unNormPreds.dataSync()]
```

[.dataSync](https://js.tensorflow.org/api/latest/#tf.Tensor.dataSync)는 tensor 내에 저장되어 있는 `typedarray`를 가져올 때 쓰는 메소드다. 이 작업을 통해 텐서 값들을 자바스크립트가 이해할 수 있는 값으로 변환할 수 있다. 이 함수는 보통 더 자주 쓰이는 [.data](https://js.tensorflow.org/api/latest/#tf.Tensor.data)의 동기 버전이라고 보면 된다.

마지막으로 `tfjs-vis`를 통해 원래 데이터와 모델이 예측한 값을 시각화 해서 볼 수 있다.

```javascript
// 예측 값을 만들어서 원래 데이터와 비교
testModel(model, data, tensorData)
```

페이지를 새로고침하면, 이제 아래와 같이 모델이 훈련해서 예측한 내용을 볼 수 있다.

![](https://codelabs.developers.google.com/codelabs/tfjs-training-regression/img/210afe5891514fb2.png)

축하합니다. 방금 우리는 간단한 머신러닝 모델을 훈련해보았습니다. 이는 선형회귀라고 하고 알려진 모델로, 주어진 데이터를 바탕으로 선형 예측 모델을 만들어보는 예제 입니다.

## 8. 주요 시사점

이번 머신러닝 모델 학습 모델에서는 아래와 같은 것을 배웠습니다.

작업의 공식화:

- regression / classification 문제인가?
- supervised / unsupervised learning인가?
- 입력 데이터의 형태는 어떤가? 출력 데이터는 어떤 형태를 가져야 하는가?

데이터 준비하기:

- 데이터를 클렌징하고, 데이터에서 패턴이 보일 수 있는지 조사하여라
- 학습 전에 데이터를 무작위로 섞어라
- 신경망에 학습시키기 용이 하도록 데이터를 정규화 하기. 보통 0~1 또는 -1~1 정도로 한다.
- 데이터를 텐서로 변환하여라

모델을 만들고 실행시키기:

- `tf.sequential`과 `tf.model`을 사용하여 모델을 정의하고, `tf.layers.*`로 레이어를 추가해라
- optimizer(보통 adam을 많이 쓴다), 배치크기, epoch횟수와 같은 파라미터를 정하라
- 문제해결에 적합한 [loss function](https://developers.google.com/machine-learning/glossary/#loss)를 선택하고, 진행률을 예측하는데 도움이 되는 accuracy metric을 선택하기. [meanSquaredError](https://developers.google.com/machine-learning/glossary/#MSE)가 보통 회귀 문제에서 가장 많이 이용되는 손실함수다.
- 학습 과정에서 손실이 감소하는지 지켜보기

모델 평가하기:

- 학습과정에서 모니터링할 수 있도록 모델에 적합한 evaluation metric 을 선택해라. 한번 학습된 뒤에는, 예측 정확도가 맞는지 확인하기 위해 테스트 예측을 해보아라.

[코드보기](https://codesandbox.io/embed/tensorflowjs-03-linear-regression-65lku?fontsize=14&hidenavigation=1&theme=dark)

## 9. 추가로 해볼만한 것들

- epochs 횟수를 변경해 보아서 실험해보자. 그래프가 평평해지기 위해서는 epochs이 몇번이 필요할까?
- 히든레이어의 unit수를 늘려보자.
- 입출력 레이어 사이에 히든레이어를 몇개 더 추가해보자. 추가될 레이어는 예시로 아래와 같은 형태가 될 수도 있다.

```javascript
model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }))
```

여기에서 중요한 것은, 히든레이어로 비선형 활성화 함수인 [sigmoid](https://developers.google.com/machine-learning/glossary/#sigmoid_function)를 활용했다는 사실이다. 활성화 함수에 더 알아보고 싶다면, [여기](https://developers.google.com/machine-learning/crash-course/introduction-to-neural-networks/anatomy)를 참조하자.

위 실험을 거친다면, 아래와 같은 모습이 나타날 것이다.

![](https://codelabs.developers.google.com/codelabs/tfjs-training-regression/img/fe7afd4c351901f6.png)

[코드보기](https://codesandbox.io/embed/tensorflowjs-03-linear-regressio-extra-credit-wwg16?fontsize=14&hidenavigation=1&theme=dark)

[출처](https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html)
