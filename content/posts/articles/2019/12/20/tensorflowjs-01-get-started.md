---
title: Tensorflow.js - 01. 시작하기
date: 2019-12-20 10:49:06
published: true
tags:
  - machine-learning
  - ai
  - tensorflow
  - javascript
description: "[https://www.tensorflow.org/js/tutorials](https://www.tensorflow.\
  org/js/tutorials)을 개인적인 학습을 위해 번역한 글입니다. 정확한 번역을 위해서가 아니라, 개인적인 공부를 위해서 하는
  것입니다. 오해 ㄴㄴ ## 시작하기  Tensorflow.js는 브라우저와 Node.js에서 머신러닝 모델..."
category: machine-learning
slug: /2019/12/20/tensorflowjs-01-get-started/
template: post
---
[https://www.tensorflow.org/js/tutorials](https://www.tensorflow.org/js/tutorials)을 개인적인 학습을 위해 번역한 글입니다. 정확한 번역을 위해서가 아니라, 개인적인 공부를 위해서 하는 것입니다. 오해 ㄴㄴ

## 시작하기

Tensorflow.js는 브라우저와 Node.js에서 머신러닝 모델을 사용하고 훈련시킬 수 있는 자바스크립트 라이브러리입니다.

시작하는 다양한 방법은 아래 섹션들을 참고하세요.

<!-- excerpt -->

### 직접 Tensors를 사용하지 않고 머신러닝 프로그램을 작성하기

저레벨의 Tensors나 Optimizers 등을 고려하지 않고 머신러닝을 시작하고 싶나요? Tensorflow.js를 기반으로 구축한 ml5.js 라이브러리는 간결하고 접근 가능한 API 를 통해 브라우저 환경에서 머신러닝 알고리즘 및 모델에 접근할 수 있게 해줍니다.

[ml5.js](https://ml5js.org/)

### Tensorflow.js 설치하기

Tensors, layers, optimizers, loss functions 등의 개념에 익숙하신가요? Tensorflow.js는 자바스크립트 환경에서 신경망 네트워크 구축을 위한 유연함을 제공합니다.

Tensorflow.js를 브라우저와 node.js환경에서 어떻게 실행할 수 있는지 아래에서 확인해보세요.

[설치하기](/2019/12/20/tensorflowjs-02-setup/)

### 이미 훈련된 모델을 Tensorflow.js로 변환하기

파이썬으로 사전에 훈련된 모델을 어떻게 Tensorflow.js로 변환할 수 있는지 알아보세요.

[Keras Model (번역예정)](https://www.tensorflow.org/js/tutorials/conversion/import_keras)
[GraphDef Model (번역예정)](https://www.tensorflow.org/js/tutorials/conversion/import_saved_model)

### 이미 존재하는 Tensorflow.js 코드에서 배우기

`tfjs-examples`는 다양한 머신러닝 과제들을 Tensorflow.js로 구현한 작은 코드 예제들을 보여줍니다.

[Github에서 보기](https://github.com/tensorflow/tfjs-examples)

### Tensorflow.js 모델을 시각화 하기

`tfjs-vis` 는 브라우저에서 Tensorflow.js를 시각화 할 수 있는 작은 라이브러리 입니다.

### 데이터를 Tensorflow.js에서 처리하기

`TensorFlow.js`는 머신러닝을 활용해 데이터를 처리할 수 있도록 도와줍니다.
