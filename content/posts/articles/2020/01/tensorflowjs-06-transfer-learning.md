---
title: Tensorflow.js - 06. What is Transfer Learning
tags:
  - machine-learning
  - ai
  - tensorflow
  - javascript
published: true
date: 2020-01-23 07:06:36
description: "```toc tight: true, from-heading: 1 to-heading: 3 ``` # 전이 학습이란
  무엇인가?  정교한 딥러닝 모델은 수백만개의 파라미터 (가중치)를 가지고 있으며, 이를 처음 부터 훈련하려면 엄청난 양의 컴퓨팅 자원이
  필요한 경우가 많습니다. 전이학습은, 이미 관련된 작업에 대해 훈련받은 모델의 한 부분을 취해서, 새로운 ..."
category: machine-learning
slug: /2020/01/tensorflowjs-06-transfer-learning/
template: post
---
```toc
tight: true,
from-heading: 1
to-heading: 3
```

# 전이 학습이란 무엇인가?

정교한 딥러닝 모델은 수백만개의 파라미터 (가중치)를 가지고 있으며, 이를 처음 부터 훈련하려면 엄청난 양의 컴퓨팅 자원이 필요한 경우가 많습니다. 전이학습은, 이미 관련된 작업에 대해 훈련받은 모델의 한 부분을 취해서, 새로운 모델에서 이를 재사용함으로써 이러한 컴퓨팅 자원의 소모를 단축시키는 기술입니다.

예를 들어, 다음 튜토리얼은 이미 이미지내에서 1000 여가지의 다른 종류의 이미지를 인식하도록 훈련된 모델을 활용해, 자신만의 인식기를 만드는 방법을 보여줍니다. 사전에 학습된 모델의 기존의 지식을 활용하여, 원래 그 모델이 썼던것 보다 훨씬 더 적은 학습용 데이터를 활용하여, 자신만의 이미지를 인식하도록 할 수 있습니다.

이는 브라우저나 모바일과 같은 리소스가 제약되어 있는 환경에서 모델을 커스터마이징 하는 것 뿐만 아니라, 새로운 모델을 빠르게 개발하는데에도 굉장히 유용합니다.

전이학습을 할 때, 보통 원래 모델의 가중치를 조정하지 않습니다. 대신, 우리는 최종 레이어를 제거하여, 이렇게 잘린 모델의 출력위에 새로운 (얇은) 모델을 학습시킵니다. 그리고 이 기법은 다음 두 튜토리얼에서 진행할 것입니다.

- [Build a transfer-learning based image classifier](/2020/01/tensorflowjs-07-build-an-image-classifier/) [원문](https://www.tensorflow.org/js/tutorials/transfer/image_classification)
- [Build a transfer-learning based audio recognizer](/2020/01/tensorflowjs-08-build-an-audio-recognizer/) [원문](https://www.tensorflow.org/js/tutorials/transfer/audio_recognizer)
