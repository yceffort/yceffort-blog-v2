---
title: Deep Learning 01) - Neuron
date: 2019-03-06 03:08:28
published: true
tags:
  - pytorch
mathjax: true
description:
  'Deep Learning 01) Neuron ## Neron  뉴런은 딥러닝의 핵심이 되는 개념이다. 뉴론 하나
  자체로는 별다른 기능을 할 수 없지만, 여러개의 뉴런이 모여야 진정하게 기능을 할 수 있다.   일단 뉴런의 구조를
  살펴보자.  ![neuron](https://cdn-images-1.medium.com/max/1600/1*_Zy1C83cnmY...'
category: pytorch
slug: /2019/03/05/deep-learning-1-neuron/
template: post
---

Deep Learning 01) Neuron

## Neron

뉴런은 딥러닝의 핵심이 되는 개념이다. 뉴론 하나 자체로는 별다른 기능을 할 수 없지만, 여러개의 뉴런이 모여야 진정하게 기능을 할 수 있다.

일단 뉴런의 구조를 살펴보자.

![neuron](https://cdn-images-1.medium.com/max/1600/1*_Zy1C83cnmYUdETCeQrOgA.png)

뉴런은 보이는 것처럼 n개의 input signal을 받아서 output signal을 내보내는 구조를 가지고 있다. input value는 독립변수이며, 이를 받아서 어떻게 output을 내보낼 것인가가 핵심이다. 일단 이러한 input value들의 값은 특성에 따라 제각각 일 것이므로, 일반적으로 표준화하여 0과 1사이에서 존재하도록 처리한다. 이렇게 처리된 input value 에는 가중치 (weight)를 각각 적용한다. 이는 이 input value가 결과에 얼마나 영향을 미치는지 결정하게 되는데, 이 weight값은 값을 예측하는 과정에서 점차 보정이 이루어진다.

output value는 연속적인 값이 될수도, binary한 값이 될수도, categorical한 값이 될 수 있다. 이는 우리가 무엇을 예측하려고 하느냐에 따라 다르다.

### Weight

weight은 neural network에서 굉장히 중요한 부분이다. neural network에서 weight을 조정하면서 뉴런은 이 값들을 어떻게 가중치를 가해서 결과 값을 냘지 결정한다. 그리고 이 weight 을 조정하는 과정을 학습하는 과정이라고 볼 수 있다. wiehgt을 조정하면서, 어떤 input이 중요한지, 덜 중요한지를 판단하게 된다.

![01](../images/01.png)

이 weight을 조정하는 과정이 전체 neural network에서 이뤄지게 되며, 이를 위해서 나중에 설명할 gradient descent나 backpropagation이 이뤄지게 된다.

### 뉴런안에서는 무슨일이 일어날까?

1. 모든 input value에 각각의 가중치를 곱한 값을 합한다.

$$ \sum\_{i=1}^{m} WiXi $$

2. activation function을 적용한다.

$$ \text{Activation Function}(\sum\_{i=1}^{m} WiXi) $$

activation function, 활성함수는 1번에서 처리된 값을 어떻게 내보낼지 결정하는 함수다. 이 함수들은 주로 비선형 함수이다. (선형함수는 아무리 여러 계층을 쌓아봐야 결국 선형함수일 뿐이므로, 즉 층을 깊게하는 의미가 없어지므로) 이 활성함수의 종류에 대해서는 나중에 다루도록 한다.

3. 다음 neuron으로 값을 전달한다.
