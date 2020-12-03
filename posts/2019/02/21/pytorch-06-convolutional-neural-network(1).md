---
title: Pytorch 06) - Convolutional Neural Network (1)
date: 2019-02-21 09:05:35
published: true
tags:
  - pytorch
  - python
mathjax: true
description: 'Pytorch - 06) Convolutional Neural Network (1) ## Convolutional
  Neural Network (CNN)  ### Fully Connected Layer의 문제점  Convolutional Neural
  Network (이하 CNN)은 이미지, 비디오, 텍스트, 사운드를 분류하는 딥러닝에서 가장 많이 사용되는 ...'
category: pytorch
slug: /2019/02/21/pytorch-06-convolutional-neural-network(1)/
template: post
---

Pytorch - 06) Convolutional Neural Network (1)

## Convolutional Neural Network (CNN)

### Fully Connected Layer의 문제점

Convolutional Neural Network (이하 CNN)은 이미지, 비디오, 텍스트, 사운드를 분류하는 딥러닝에서 가장 많이 사용되는 알고리즘 중 하나다. CNN은 패턴을 찾는데 특히 유용하며, 데이터에서 직접 학습하며, 패턴을 사용하여 이미지를 분류하여 특징을 수동으로 추출할 필요가 없다.

기존 신경망은 인접하는 계층의 모든 뉴런이 연결되어 있는 fully-connected 구조다. 이전에 28*28*1 짜리 이미지를 학습했을때, 3차원적인 정보는 모두 무시하고 1차원 배열로 변환하여 학습하였다. 이는 공간적인 정보를 모두 잃게 된다.

![cnn1](../images/cnn1.png)

예를 들어 가까운 픽셀은 값이 비슷하거나, rgb값이 비슷하거나 하는 ㄹ등의 정보가 있을 수 있지만, 1차원으로 쭉 늘어트리게 되면 이러한 정보 (패턴)가 모두 무시되어 버린다.

또한, 큰이미지가 인풋으로 들어오게되면 weight로 설정해야 하는 값이 기하급수적으로 많아지게 되며, 이는 성능저하를 불러온다.

### Convolution

합성곱연산은, 특징정 크기 (width, height)를 갖는 필터(filter, kernel)를 일정간격(stride)로 이동해가며 입력데이터에 적용하는 것을 의미한다.

![cnn](https://cdn-images-1.medium.com/max/1600/1*7S266Kq-UCExS25iX_I_AQ.png)

![filter-stride](http://deeplearning.stanford.edu/wiki../../../images/6/6c/Convolution_schematic.gif)

원본이미지에서 필터를 곱하는 것을 합성곱(convolution)이라고 한다. 이렇게 주어진 필터를 이미지 전체에 stride에 따라서 적용하면서, 이미지의 특징을 찾아내는 것이다.

이렇게 하게 되면, 해당 필터는 필터가 갖고 있는 특징이 데이터에 있는지 없는지를 검출 해 줄 수 있다.

![filter](https://adeshpande3.github.io/assets/Filter.png)

왼쪽이 곡선의 특징을 찾는 필터라면, 오른쪽과 같은 모양을 필터가 찾아내는 것이라고 보면 된다.

![filter1](https://adeshpande3.github.io/assets/OriginalAndFilter.png)

![filter2](https://adeshpande3.github.io/assets/FirstPixelMulitiplication.png)

이런 식으로, 쥐의 엉덩이에 있는 곡선의 특징을 이미지에서 찾아내게 되는 것이다. Linear에 비유하자면, weight가 filter가 되는 것이다. 마찬가지로 bias (편향) 도 포함될 것이다.

### padding

cnn을 하기전에, 입력데이터 주변을 특정값으로 채우는 단계로, 이는 주로 입력 데이터와 출력데이터의 크기를 맞추기 위해서 쓴다.

![padding](https://adeshpande3.github.io/assets/Pad.png)

32x32x3의 입력값에서 5x5x3 필터를 적용 시키게 되면, feature map의 크기는 28x28x3이 된다. 이렇게 사이즈가 안 맞는것을 막기 위해서 Padding을 쓴다. 입력데이터 주위에 2의 두께로 0을 둘러 쌓아주면, 36x36x3이 되고, 5x5x3 필터를 적용하더라도 결과값은 32x32x3 로 유지된다.

이러한 패딩은 특징이 유실되는 것을 방지해주고, 또한 과적합도 방지하는 효과가 있다. 주로 0을 적용하는 zero-padding을 쓴다.

### stride

필터를 적용하는 위치의 간격을 의미한다. stride가 커지면, 당연히 간격이 넓어지므로 출력크기가 작아진다.

![stride](http://deeplearning.stanford.edu/wiki../../../images/6/6c/Convolution_schematic.gif)

위의 이미지에서는 한칸씩 (1의 크기로) stride를 적용하고 있다.

### 출력 크기 계산

- 입력크기: H, W
- 필터크기: FH, FW
- 출력크기: OH, OW
- 패딩: P
- 스트라이드 S

$$ OH = \frac{H + 2P - FH}{S} + 1 $$

$$ OW = \frac{W + 2P - FW}{S} + 1 $$

두 결과는 모두 정수로 나누어 떨어지는 값이어야 한다.

### 3차원 데이터 (RGB 이미지)

3차원 이미지 데이터에 대해서는, 필터도 이미지와 마찬가지로 같은 수의 채널을 갖고 있어야 한다.

### Pooling

이렇게 input에서 filter 를 적용하여 특징을 추출했다면, 이를 어떻게 판단할지가 중요 하다. 이렇게 추출된 map을 인위적으로 줄이는 작업을 pooling이라고 한다.

![pooling](https://upload.wikimedia.org/wikipedia/commons/e/e9/Max_pooling.png)

pooling 하는 방법은 크게 max와 average가 있는데, 위 샘플 이미지에서는 최대값을 가져오는 MaxPooling 을 수행했다. 이는 큰값이 다른 주변 값(특징)을 대표한다는 개념을 적용시키게 된다. 이는 과적합을 방지하고, 컴퓨팅 리소스를 줄이는 효과를 가져온다.

## Convolutaional Layer

![cnn-sample](https://image.slidesharecdn.com/deeplearning-161020090534/95/deep-learning-stm-6-19-638.jpg?cb=1476964837)

자동차 인식을 위해 CNN을 적용한 모습이다. Conv-Relu-Conv-Relu-Pool-Conv-Relu-Conv-Relu-Pool-Conv-Relu-Conv-Relu-Pool를 적용해서 인식을 해낸 모습이다.

정리하자면, 이미지에서 특징을 추출해내는 작업을 몇번 거치고, 그뒤에 이를 fully connect 하여 perceptron을 바탕으로 최종적으로 예측을 하는 것이라고 할 수 있다.

### ReLU를 쓰는이유

- 대부분의 실제 예제는 linear한 상황이 별로 없다.
- ReLU는 negative값을 모두 무시하기 때문에, 조금 더 낫다.
- tanh와 sigmoid는 vanishing gradient 문제를 갖고 있다.
  - Deep Neural Netwrok는 Backpropagation (역전파)를 하기 위해서 gradient descent (경사하강법)을 적용하여 parameter를 조정한다.
  - 그러나 이 gradient 값이 매우 작아지게 되면, 이것을 효율적으로 개선해나가는 것이 쉽지 않아지고, 느려진다.
  - sigmoid는 중간 값에서는 경사가 있지만, 값이 작거나 커질 수록 극도로 경사가 작은 값을 갖게 된다. (0과 가까워 진다.)
  - 따라서 아무리 계산을 누적해도 굉장히 작은 값이 나오게 된다.
- Relu는 input이 음수면 그냥 0을 리턴, 양수면 같은 값을 리턴한다.
- 따라서 값이 극도로 작아지거나 커진다고 해서 gredient가 사라지지 않는다.

![cnn2](../images/cnn2.png)

MNIST 손글씨 데이터셋에 CNN을 적용한다면 위의 이미지와 같은 \모습이 될것이다.
