---
title: Pytorch 05) - Image Recognition
date: 2019-02-21 06:58:00
published: true
tags:
  - pytorch
description:
  'Pytorch - 05) Image Recognition 마지막으루다가 손글씨 분류해보는 실습을 해보겠습니다.  ###
  Dataset  ```python import torch import matplotlib.pyplot as plt import numpy
  as np  from torch import nn from torch.nn import functi...'
category: pytorch
slug: /2019/02/21/pytorch-05-image-recognition/
template: post
---

Pytorch - 05) Image Recognition

마지막으루다가 손글씨 분류해보는 실습을 해보겠습니다.

### Dataset

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms


# 리사이즈
# 텐서화
# channel, height, width를 정규화 한거임
# 처음 세개는 mean
# 그 다음 세개는 sd
transformer = transforms.Compose([transforms.Resize((28, 28)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])


training_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transformer)
validation_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transformer)

# 6000개를 100개 배치로 나눠서 돌림
training_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=100, shuffle=False)
```

텐서를 받아서 image로 보여주는 메소드를 만들자.

```python
def im_convert(tensor):
  # 복제하고, 자동미분 끄고, numpy로
  image = tensor.clone().detach().numpy()
  # 데이터 형태는 color channel 1 28 px 28 px , 즉 1, 28, 28로 되어있음
  # 이거를 28, 28, 1 로 변경
  image = image.transpose(1, 2, 0)
  # denormalize
  image = image * np.array([0.5, 0.5, 0.5] + np.array([0.5, 0.5, 0.5]))
  # 데이터를 0과 1사이로만 있도록 보정
  image = image.clip(0, 1)
  return image
```

맛보기로 이미지를 한번 보자.

```python
dataiter = iter(training_loader)
images, labels = dataiter.next()

fig = plt.figure(figsize=(25, 4))

for i in np.arange(20):
  # row 2 column 10
  ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[i]))
  ax.set_title(labels[i].item())
```

![image-recognition1](../images/image-recognition1.png)

### Model

```python
class Classifier(nn.Module):

  def __init__(self, n_input, H1, H2, n_output):
    super().__init__()
    self.linear1 = nn.Linear(n_input, H1)
    self.linear2 = nn.Linear(H1, H2)
    self.linear3 = nn.Linear(H2, n_output)

  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)
    return x
```

hidden layer2개에, linear와 활성화 함수로 relu를 사용했다. sigmoid와 다르게 relu는 값 그자체가 확률을 나타내지 않으므로, 마지막에는 ReLU를 쓰지 않았다.

```python
model = Classifier(784, 125, 65, 10)
model
```

```bash
Classifier(
  (linear1): Linear(in_features=784, out_features=125, bias=True)
  (linear2): Linear(in_features=125, out_features=65, bias=True)
  (linear3): Linear(in_features=65, out_features=10, bias=True)
)
```

이미지가 28\*28이므로 784개의 input, 총 10개 (0~9)로 구별해야 하므로 output은 10개로 나타넀다. 가운데 125와 65는 그냥 임의로 넣어봤다. 해보면서 조절해보면 된다.

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
```

classification 문제라서 CrossEntroypyLoss를, 그리고 optimizer로는 Adam을 사용하였다.

### training

```python
epochs = 12
running_loss_history = []
running_correct_history = []
validation_running_loss_history = []
validation_running_correct_history = []

for e in range(epochs):

  running_loss = 0.0
  running_correct = 0.0
  validation_running_loss = 0.0
  validation_running_correct = 0.0

  for inputs, labels in training_loader:

    inputs = inputs.view(inputs.shape[0], -1)
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    _, preds = torch.max(outputs, 1)

    running_correct += torch.sum(preds == labels.data)
    running_loss += loss.item()



  else:
    # 훈련팔 필요가 없으므로 메모리 절약
    with torch.no_grad():

      for val_input, val_label in validation_loader:
        val_input = val_input.view(val_input.shape[0], -1)
        val_outputs = model(val_input)
        val_loss = criterion(val_outputs, val_label)

        _, val_preds = torch.max(val_outputs, 1)
        validation_running_loss += val_loss.item()
        validation_running_correct += torch.sum(val_preds == val_label.data)


    epoch_loss = running_loss / len(training_loader)
    epoch_acc = running_correct.float() / len(training_loader)
    running_loss_history.append(epoch_loss)
    running_correct_history.append(epoch_acc)

    val_epoch_loss = validation_running_loss / len(validation_loader)
    val_epoch_acc = validation_running_correct.float() / len(validation_loader)
    validation_running_loss_history.append(val_epoch_loss)
    validation_running_correct_history.append(val_epoch_acc)

    print("===================================================")
    print("epoch: ", e + 1)
    print("training loss: {:.5f}, {:5f}".format(epoch_loss, epoch_acc))
    print("validation loss: {:.5f}, {:5f}".format(val_epoch_loss, val_epoch_acc))
```

```bash
===================================================
epoch:  1
training loss: 0.34698, 89.241669
validation loss: 0.18939, 94.029999
===================================================
epoch:  2
training loss: 0.17725, 94.583336
validation loss: 0.16674, 94.790001
===================================================
epoch:  3
training loss: 0.15004, 95.341667
validation loss: 0.16987, 94.989998
===================================================
epoch:  4
training loss: 0.12949, 95.991669
validation loss: 0.13101, 96.040001
===================================================
epoch:  5
training loss: 0.11537, 96.478333
validation loss: 0.14699, 95.970001
===================================================
epoch:  6
training loss: 0.10967, 96.628334
validation loss: 0.11986, 96.410004
===================================================
epoch:  7
training loss: 0.10788, 96.681664
validation loss: 0.16216, 95.400002
===================================================
epoch:  8
training loss: 0.10198, 96.928337
validation loss: 0.13875, 96.029999
===================================================
epoch:  9
training loss: 0.09768, 97.044998
validation loss: 0.13843, 96.129997
===================================================
epoch:  10
training loss: 0.09128, 97.230003
validation loss: 0.14281, 96.349998
===================================================
epoch:  11
training loss: 0.09073, 97.208336
validation loss: 0.15617, 95.629997
===================================================
epoch:  12
training loss: 0.09084, 97.209999
validation loss: 0.13772, 96.470001
```

### Loss

```python
plt.plot(running_loss_history, label='training loss')
plt.plot(validation_running_loss_history, label='validation loss')
plt.legend()
```

![image-recognition2](../images/image-recognition2.png)

### Accuracy

```python
plt.plot(running_correct_history, label='training accuracy')
plt.plot(validation_running_correct_history, label='validation accuracy')
```

![image-recognition3](../images/image-recognition3.png)

정확도는 95% 안팎으로 나왔다. 그러나 validation loss가 training loss보다 커지는 것을 보아, 훈련하는 과정에서 overfitting이 일어났다고 봐야할 것이다. hyperparameter를 조금씩 조절할 필요가 있어 보인다.

### Test

```python
import requests
from PIL import Image
import PIL

url = 'https://images.homedepot-static.com/productImages/007164ea-d47e-4f66-8d8c-fd9f621984a2/svn/architectural-mailboxes-house-letters-numbers-3585b-5-64_1000.jpg'
response = requests.get(url, stream=True)
img = Image.open(response.raw)
plt.imshow(img)
```

![image-recognition4](../images/image-recognition4.png)

```python
# 반전
img = PIL.ImageOps.invert(img)
# 흑백으로 좀더 또렷하게
img = img.convert('1')
img = transformer(img)

plt.imshow(im_convert(img))
```

![image-recognition5](../images/image-recognition5.png)

```python
img = img.view(img.shape[0], -1)
outputs = model(img)
_, pred = torch.max(outputs, 1)
print(pred.item())
```

```bash
5
```

맞췄다. 한번 validation dataset에 다시 해보자.

```python
dataiter = iter(validation_loader)
images, labels = dataiter.next()
images_ = images.view(images.shape[0], -1)
output = model(images_)
_, preds = torch.max(output, 1)

fig = plt.figure(figsize=(25, 4))

for i in np.arange(20):
  # row 2 column 10
  ax = fig.add_subplot(2, 10, i+1, xticks=[], yticks=[])
  plt.imshow(im_convert(images[i]))
  ax.set_title("{} ({})".format(str(preds[i].item()), str(labels[i].item())), color=('green' if preds[i] == labels[i] else 'red'))
```

![image-recognition5](../images/image-recognition6.png)

하나 빼고 다 맞췄는데, 저건 내가봐도 6으로 볼 수도 있을 것 같다. 착한 분류기 ㅇㅈ 합니다.
