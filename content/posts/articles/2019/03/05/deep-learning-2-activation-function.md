---
title: Deep Learning 02) - Activation Function
date: 2019-03-06 03:41:30
published: true
tags:
  - pytorch
mathjax: true
description: "Deep Learning 02) Activation Function ## Activation Function  활성화
  함수  ![02](../images/02.png)  활성화 함수는 가중치가 더해진 input
  value를 어떻게 처리할 것인지 결정하는 함수다. 이러한 활성화 함수에는 몇가지 종류가 있다.  ### 1..."
category: pytorch
slug: /2019/03/05/deep-learning-2-activation-function/
template: post
---
Deep Learning 02) Activation Function

## Activation Function

활성화 함수

![02](../images/02.png)

활성화 함수는 가중치가 더해진 input value를 어떻게 처리할 것인지 결정하는 함수다. 이러한 활성화 함수에는 몇가지 종류가 있다.

### 1. Threshold Function

![threshold-function](https://www.saedsayad.com../../../images/ANN_Unit_step.png)

$$ 1 f x \geq 0 $$

$$ 0 f x < 0 $$

매우매우 간단한 형태의 함수다. 

### 2. Sigmoid Function

![sigmoid](https://t1.daumcdn.net/cfile/tistory/275BAD4F577B669920)

$$ \frac{1}{1 + e^{-x}}$$

위 함수와 다르게 부드러운 형태를 띄고 있다. 0과 1을 구분하는 곳은 경사가 급하고, 나머지 부분에서는 경사가 매우 완만하다. 이 함수는 결과값으로 확률을 구해야할 때 굉장히 유용하게 사용되고 있다.

### 3. ReLU

![relu](https://cdn-images-1.medium.com/max/937/1*oePAhrm74RNnNEolprmTaQ.png)

$$ \text{max}(0, x) $$

딥러닝에서 가장 유명한 활성화 함수중 하나다. 0이하의 값에서는 0을, 0이상에서는 x의 값을 그대로 가져간다. 다른함수에 비해 속도가 빠르고, 구현도 쉽고 따라서 연산비용도 저렴해서 많이 애용하는 함수다. 그러난 x가 0보다 작아지면 그 값을 모두 무시해버리기 때문에 (0으로 처리하기 때문에) 뉴런이 죽어버릴수도 있다는 단점이 존재한다.

### 4. tanh

![tanh](https://www.medcalc.org/manual/_help/functions/tanh.png)

$$ \frac{1 - e^{-2x}}{1+e^{-2x}} $$

시그모이드와 비슷하게 생긴 쌍곡선 함수다. sigmoid가 중간값이 $$\frac{1}{2}$$인데 반에 이 함수는 중간값을 0으로 옮겨왔다. 따라서 값이 다른 함수와는 다르게 -1에서 1사이에서 형성된다.

- 참고논문: [Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)