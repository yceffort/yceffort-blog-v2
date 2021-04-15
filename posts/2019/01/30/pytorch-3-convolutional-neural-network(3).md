---
title: 'Pytorch (3-2) - CNN: Convolutional Neural Network '
date: 2019-01-30 06:17:08
published: true
tags:
  - pytorch
mathjax: true
description:
  '앞서 CNN네트워크를 진행하면서 모르는 부분이 좀 있어서, 이론적인 측면을 좀더 강조해서 글을 써보려고 한다. ##
  구조  일단 중간층이 Convolutional Layer, Pooling Layer, Fully-connted Layer로 구성되어 있다.
  그리고 마지막엔 Dropouut Layer를 넣는 경우도 있고, Softmax 함수로 마무리 한다. ...'
category: pytorch
slug: /2019/01/30/pytorch-3-convolutional-neural-network(3)/
template: post
---

앞서 CNN네트워크를 진행하면서 모르는 부분이 좀 있어서, 이론적인 측면을 좀더 강조해서 글을 써보려고 한다.

## 구조

일단 중간층이 Convolutional Layer, Pooling Layer, Fully-connted Layer로 구성되어 있다. 그리고 마지막엔 Dropouut Layer를 넣는 경우도 있고, Softmax 함수로 마무리 한다.

![structure](https://navoshta.com../../../images/posts/traffic-signs-classification/traffic-signs-architecture.png)

### Convolutional Layer

이른바 합성곱층이다. 입력데이터에 필터를 적용해서 특징값을 추출하는 레이어다. 필터를 적용해서 특징값을 추출하고, 이 필터의 값을 비선형 값으로 바꾸어주는 Activation 함수 (일반적으로 ReLU)로 이루어져 있다.

![convolutional-layer](https://t1.daumcdn.net/cfile/tistory/23561441583ED6AB29)

### Filter

필터는 찾으려는 특징이 입력 데이터에 존재하는지 여부를 검출해주는 함수다.

![filter](https://cdn-images-1.medium.com/max/1600/1*7S266Kq-UCExS25iX_I_AQ.png)

먼저 왼쪽 과 같은 그림 이 있다고 가정하자. 그리고 해당 특징을 찾으려는 필터를 곱하면 (Convolutaionl) 일정한 결과 값이 나오게 될 것이다. 찾으려는 특징이 존재한다면 큰값이 나오고, 찾으려는 특징이 존재하지 않으면 0에 가까운 값이 나올 것이다. (Activation Layer로 처리하겠지만)

물론 적용하는 필터는 단순히 한개가 아니다. 여러 다양한 필터를 조합한다면, 원본 입력 데이터가 어떤 형태의 특징을 가지고 있는지를 판단할 수 있다.

### Stride

![image-filter](http://deeplearning.stanford.edu/wiki../../../images/6/6c/Convolution_schematic.gif)

필터를 적용하는 간격의 값을 Stride라고 한다. 그리고 필터를 적용해서 얻어낸 결과를 Feature Map이라고 한다.

### Padding

위 움짤에서 보이는 것처럼, 필터를 적용하고 나면 그 크기가 필터 적용 이전보다 작아지게 된다. 5X5이미지는 3X3 필터의 1 stride를 적용하고 났더니, 결과 크기는 3X3으로 쪼그라 들었다. 그러나 문제는 단 한개의 레이어가 아니고, 여러개의 필터를 적용해서 특징을 추출해 나간다는 것이다. 이 과정이 반복되서 결과 크기가 줄어들게 되면, 처음에 비해서 그 특징을 많이 잃어버릴 수 가 있다. 이를 방지 하기 위한 기법으로 Padding을 쓴다. Padding은 입력값 주위로 특정 값을 넣어서 크기가 줄어드는 것을 인위적으로 방지한다. Padding에는 주로 0을 쓰는 Zero Padding을 많이 쓰게 된다.

![padding](https://adeshpande3.github.io/assets/Pad.png)

32x32x3의 입력값에서 5x5x3 필터를 적용 시키게 되면, feature map의 크기는 28x28x3이 된다. 이렇게 사이즈가 작아지는 걸 막기 위해서 Padding을 쓴다. 입력데이터 주위에 2의 두께로 0을 둘러 쌓아주면, 36x36x3이 되고, 5x5x3 필터를 적용하더라도 결과값은 32x32x3 로 유지된다.

이러한 패딩은 특징이 유실되는 것을 방지해주고, 또한 과적합도 방지하는 효과가 있다.

### Activation Function

이렇게 해서 나온 Feature Map에 Activation Function을 적용한다. 신경망에는 보통 sigmoid 함수보다는 ReLU를 많이 쓴다. 신경망이 깊어질 수록 학습이 어려워 지기 때문에 역전파를 통해 오차를 다시 계산한다. 만약 sigmoid를 활성함수로 사용시, 레이어가 깊어지게 되면 역전파가 제대로 되지 않기 때문에 (Gradient Vanishing) ReLU를 보통 많이 쓴다.

![relu](https://cdn-images-1.medium.com/max/1600/1*DfMRHwxY1gyyDmrIAd-gjQ.png)

### Pooling Layer

이전에 Covolutional Layer에서 특징을 추출했다면, 그 특징을 어떻게 판단할지가 중요하다. 이렇게 인위적으로 추출된 Actiavtion Map을 인위적으로 줄이는 작업을 Pooling 이라고 한다. 이러한 Pooling에는 Max Pooling, Average Pooling등 다양한 것이 있는데, 보통 Max를 많이 사용한다.

![max-pooling](https://upload.wikimedia.org/wikipedia/commons/e/e9/Max_pooling.png)

Max Pooling 의 예다. 각 섹션 별로 큰 값만 추출해 낸 모습이다. Average Pooling 이라면 평균값을 추출 할 것이다. 이렇게 함으로써, 큰 값이 다른 주변의 값을 (특징을) 대표한다는 개념을 적용시킬 수 있다. 이렇게 Pooling 을 적용하여 과적합을 방지하고, 리소스를 어느정도 줄일 수 있다.

## Convolutaionl Layer

![example](https://image.slidesharecdn.com/deeplearning-161020090534/95/deep-learning-stm-6-19-638.jpg?cb=1476964837)

자동차 인식을 위해 CNN을 적용한 모습이다. Conv-Relu-Conv-Relu-Pool-Conv-Relu-Conv-Relu-Pool-Conv-Relu-Conv-Relu-Pool를 적용해서 인식을 해낸 모습이다.

### Softmax Function

마지막에는 최종적으로 Softmax함수를 적용한다. Softmax도 일종의 Activation 함수인데, Sigmoid와 ReLU가 True, False (0, 1)만 표현한다면, Softmax Function은 여러개의 분류를 가질 수 있는 함수다.

### Dropout

드롭아웃은 Overfitting을 방지하기 위한 방법이다.

![drop-out](https://cdn-images-1.medium.com/max/1200/1*iWQzxhVlvadk6VAJjsgXgg.png)

일부 노드들을 훈련에 참여시키지 않고 몇개의 노드를 끊어서, 남은 노드들을 통해서만 훈련시키는 방식이다. 이 때 끊어버리는 노드는 랜덤으로 선택한다. pytorch에서는 기본값이 0.5 다. 즉 절반의 노드를 dropout하고 계산한다. 이렇게 함으로써, training하는 과정에서 Overfitting이 발생하지 않게 할 수 있다.

<!-- ## 이미지 구별하기 예제

옷 이미지를 구별하는 예제에 CNN을 적용해 보려고 한다. 데이터 출처는 [여기](https://github.com/zalandoresearch/fashion-mnist)다.

```python
with gzip.open(path + image_file_name, 'rb') as f:
  mnist_data = np.frombuffer(f.read(), np.uint8, offset=16)
  mnist_data = mnist_data.reshape(-1, 28*28)

# 데이터 0, 1로 정규화
mnist_data = mnist_data / 255


with gzip.open(path + label_file_name, 'rb') as f:
  mnist_label = np.frombuffer(f.read(), np.uint8, offset=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")3
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(mnist_data, mnist_label, test_size=0.2)

train_X = train_X.reshape((len(train_X), 1, 28, 28))
test_X = test_X.reshape((len(test_X), 1, 28, 28))

train_X = torch.from_numpy(train_X).float().to(device)
train_Y = torch.from_numpy(train_Y).long().to(device)

test_X = torch.from_numpy(test_X).float().to(device)
test_Y = torch.from_numpy(test_Y).long().to(device)

train = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train, batch_size=100, shuffle=True)
```

### 신경망

- 입력 데이터 높이: H = 28
- 입력 데이터 폭: W = 28
- 필터 높이: FH = 5
- 필터 폭: FW = 5
- stride 크기: S = 1

padding 크기는 출력결과가 원본과 같은 크기를 만들 수 있다면 얼마든지 가능하다.

$$ \text{Padding} = \frac{\text{FilterSize}-1}{2}  = 2 $$

- 패딩사이즈: P = 2

이를 기반으로 출력 크기를 계산해보자.

$$ \text{OutputHeight} =  \frac{(H + 2P - FH)}{S} + 1  $$

$$ \text{OutputHeight} = \frac{28 + 2 * 2 - 5 }{1}  + 1 = 28  $$

두번째 출력은 필터 사이즈를 조금더 줄이고 (2), stride를 (2)로 설정한다.

$$ \text{OutputHeight} = \frac{28 + 2 * 2 - 2 }{2}  + 1 = 16 $$

도 같을 것이다.

`Conv-Relu-Pool-Conv-Relu-Pool` 이정도 느낌으로 만들어보자. -->
