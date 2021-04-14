---
title: Computer Vision 01) - Image Representation
date: 2019-04-01 06:35:16
published: true
tags:
  - python
description: "## Image Representation & Classification ### Images as Grids of
  Pixels  ```python import numpy as np from skimage import io import
  matplotlib.image as mpimg  import matplotlib.pyplot as plt import cv..."
category: python
slug: /2019/04/01/computer-vision-1-image-representation/
template: post
---
## Image Representation & Classification

### Images as Grids of Pixels

```python
import numpy as np
from skimage import io
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import cv2

import urllib

%matplotlib inline
```

먼저 이미지를 불러온다.

```python
waymo_car_url = 'https://zdnet2.cbsistatic.com/hub/i/r/2018/01/22/e270d68c-c028-421a-bc5b-5d2a9a9458d1/resize/770xauto/50e9d2f0fc86841ba455489d50651291/google-waymo-self-driving-atlanta.png'
f = urllib.request.urlopen(waymo_car_url)
image = mpimg.imread(f)

print('Image dimensions:', image.shape)
```

```
Image dimensions: (410, 770, 4)
```

```python
# 이미지를 회색으로 바꾼다.
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_image, cmap='gray')
```

![waymo-gray](./images/01.png)

```python
# 특정 좌표의 grayscale
x = 400
y = 300

print(gray_image[y,x])
```

```
0.543404
```

```python
# 맥시멈 / 미니멈 grayscale

max_val = np.amax(gray_image)
min_val = np.amin(gray_image)

print('Max: ', max_val)
print('Min: ', min_val)
```

```
Max:  0.9646824
Min:  0.040556863
```

### RGB colorspace

```python
plt.imshow(image)
```

![waymo](../images/02.png)

```python
## 각각의 RGB영역 확보
r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]
```

```python
fx, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
ax1.set_title('R channel')
ax1.imshow(r, cmap='gray')
ax2.set_title('G channel')
ax2.imshow(g, cmap='gray')
ax3.set_title('B channel')
ax3.imshow(b, cmap='gray')
```

![waymo-rgb](../images/03.png)

### Color Threshold, Bluescreen

파란색 크로마키에 있는 사진을 합성하는 과정이다.

![image](https://ak5.picdn.net/shutterstock/videos/548875/thumb/1.jpg)

```python
cromakey_blue = 'https://ak5.picdn.net/shutterstock/videos/548875/thumb/1.jpg'
f = urllib.request.urlopen(cromakey_blue)
image = mpimg.imread(f, 0)
print('This image is:', type(image), 'with dimensions:', image.shape)
```

```
This image is: <class 'numpy.ndarray'> with dimensions: (480, 852, 3)
```

```python
# 이미지 복사
image_copy = np.copy(image)

# BGR이미지를 RGB로 변환
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

# 복사된 이미지 보여주기
plt.imshow(image_copy)
```

![red](../images/04.png)

```python
lower_blue = np.array([0,0,200])
upper_blue = np.array([250,250,255])

# 마스킹될 영역을 선택한다.
mask = cv2.inRange(image, lower_blue, upper_blue)

# 마스킹 이미지 표시
plt.imshow(mask, cmap='gray')
```

![mask](../images/05.png)

```python
# 이미지를 카피한다음
masked_image = np.copy(image_copy)
# 마스킹 영역을 제외하고 모두 검정색으로 바꾼다면
masked_image[mask != 0] = [0, 0, 0]
plt.imshow(masked_image)
```

![mask](../images/06.png)

효과적으로 파란색 영역을 제거한 것을 알 수 있다. 여기에 배경을 합성해보자.

```python
space_url = 'https://images.unsplash.com/photo-1496715976403-7e36dc43f17b?ixlib=rb-1.2.1&w=1000&q=80'
sf = urllib.request.urlopen(space_url)
image = np.asarray(bytearray(sf.read()), dtype="uint8")
background_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

crop_background = background_image[0:480, 0:852]
# 이번엔 반대로 마스킹 해야할 영역을 검정색으로 처리한다.
crop_background[mask == 0] = [0, 0, 0]
plt.imshow(crop_background)
```

![background](../images/07.png)

```python
# 두 이미지를 합치면
complete_image = masked_image + crop_background
plt.imshow(complete_image)
```

![complete](../images/08.png)
