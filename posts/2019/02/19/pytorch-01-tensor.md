---
title: Pytorch 01) - Tensor
date: 2019-02-19 06:16:25
published: true
tags:
  - pytorch
description:
  'Pytorch - 01) Tensor ## Tensor  이제는 더 이상 기초를 다루지 않겠다. (마지막 기초
  공부)  ```python # 기초적인 배열 선언 v = torch.tensor([1, 2, 3]) v ```  ``` tensor([1,
  2, 3]) ```  ```python # 타입 확인 v.dtype ```  ``` torch.int64 ...'
category: pytorch
slug: /2019/02/19/pytorch-01-tensor/
template: post
---

Pytorch - 01) Tensor

## Tensor

이제는 더 이상 기초를 다루지 않겠다. (마지막 기초 공부)

```python
# 기초적인 배열 선언
v = torch.tensor([1, 2, 3])
v
```

```bash
tensor([1, 2, 3])
```

```python
# 타입 확인
v.dtype
```

```bash
torch.int64
```

```python
# pytohn 배열과 마찬가지로 슬라이싱이 가능하다
v = torch.tensor([1, 2, 3, 4, 5, 6])
v[1:]
v[1:4]
```

```python
# Float 형태로 선언
f = torch.FloatTensor([1, 2, 3, 4, 5, 6])
f
```

```bash
tensor([1., 2., 3., 4., 5., 6.])
```

```python
# 크기 확인
f.size()
```

```bash
torch.Size([6])
```

```python
# view를 써서 배열 형태를 조작할 수도 있다.
v.view(6, 1)
```

```bash
tensor([[1],
        [2],
        [3],
        [4],
        [5],
        [6]])
```

```python
# 3만 줄테니 알아서 사이즈를 조절해라
v.view(3, -1)
```

```bash
tensor([[1, 2],
        [3, 4],
        [5, 6]])
```

```python
# numpy array를 tensor array를  상호간에 변환하는 것이 가능하다.
a = np.array([1, 2, 3, 4, 5, ])
tensor_cnv = torch.from_numpy(a)
print(tensor_cnv, tensor_cnv.type())
```

```bash
tensor([1, 2, 3, 4, 5]) torch.LongTensor
```

```python
numpy_cnv = tensor_cnv.numpy()
numpy_cnv
```

```bash
array([1, 2, 3, 4, 5])
```

```python
# 0부터 10을 100개로 쪼갠다.
torch.linspace(0, 10)
# 0부터 10을 5개로 쪼갠다.
torch.linspace(0, 10, 5)
```

```python
# 이런 것도 가능하다.
x = torch.linspace(0, 10, 100)
y = torch.exp(x)
y
```

```bash
tensor([1.0000e+00, 1.1063e+00, 1.2239e+00, 1.3540e+00, 1.4979e+00, 1.6571e+00,
        1.8332e+00, 2.0280e+00, 2.2436e+00, 2.4821e+00, 2.7459e+00, 3.0377e+00,
        3.3606e+00, 3.7178e+00, 4.1130e+00, 4.5501e+00, 5.0337e+00, 5.5688e+00,
        6.1606e+00, 6.8154e+00, 7.5398e+00, 8.3412e+00, 9.2278e+00, 1.0209e+01,
        1.1294e+01, 1.2494e+01, 1.3822e+01, 1.5291e+01, 1.6916e+01, 1.8714e+01,
        2.0704e+01, 2.2904e+01, 2.5338e+01, 2.8032e+01, 3.1011e+01, 3.4307e+01,
        3.7954e+01, 4.1988e+01, 4.6450e+01, 5.1387e+01, 5.6849e+01, 6.2892e+01,
        6.9576e+01, 7.6971e+01, 8.5153e+01, 9.4203e+01, 1.0422e+02, 1.1529e+02,
        1.2755e+02, 1.4110e+02, 1.5610e+02, 1.7269e+02, 1.9105e+02, 2.1135e+02,
        2.3382e+02, 2.5867e+02, 2.8616e+02, 3.1658e+02, 3.5023e+02, 3.8745e+02,
        4.2864e+02, 4.7419e+02, 5.2459e+02, 5.8035e+02, 6.4204e+02, 7.1028e+02,
        7.8577e+02, 8.6929e+02, 9.6168e+02, 1.0639e+03, 1.1770e+03, 1.3021e+03,
        1.4405e+03, 1.5936e+03, 1.7630e+03, 1.9503e+03, 2.1576e+03, 2.3870e+03,
        2.6407e+03, 2.9213e+03, 3.2318e+03, 3.5753e+03, 3.9554e+03, 4.3758e+03,
        4.8409e+03, 5.3554e+03, 5.9246e+03, 6.5543e+03, 7.2510e+03, 8.0216e+03,
        8.8742e+03, 9.8175e+03, 1.0861e+04, 1.2015e+04, 1.3292e+04, 1.4705e+04,
        1.6268e+04, 1.7997e+04, 1.9910e+04, 2.2026e+04])
```

```python
plt.plot(x.numpy(), y.numpy())
```

![exp](../images/tensor1.png)

```python
x = torch.linspace(0, 10, 100)
y = torch.sin(x)
plt.plot(x.numpy(), y.numpy())
```

![sin](../images/tensor2.png)

## n차원 배열

```python
one_d = torch.arange(0, 9)
two_d = one_d.view(3, 3)
two_d
```

```bash
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8])
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])
```

```python
# dim으로 차원을 확인할 수 있다.
two_d.dim()
```

```bash
2
```

```python
# 2개의 블록, 3로우, 3컬럼의 형태로 만들어진다.
x = torch.arange(18).view(2, 3, 3)
x
```

```bash
tensor([[[ 0,  1],
         [ 2,  3],
         [ 4,  5]],

        [[ 6,  7],
         [ 8,  9],
         [10, 11]],

        [[12, 13],
         [14, 15],
         [16, 17]]])
```

```python
mat_a = torch.tensor([0, 3, 5, 5, 5, 2]).view(2, 3)
mat_b = torch.tensor([3, 4, 3, -2, 4, -2]).view(3, 2)
torch.matmul(mat_a, mat_b)
mat_a @ mat_b
```

matmul 과 @ 은 서로 곱할 수 있는 크기의 매트릭스를 곱하는 식이다. 고등학교때 행렬 열심히 하기를 잘했다.

## autograd

pytorch의 장점은 자동미분(autograd)을 지원한다는 점이다.

$$x=2$$

$$y = 9x^4 + 2x^3 + 3x^2 + 6x+1$$

이식을 미분해보자.

```python
x = torch.tensor(2.0, requires_grad=True)
y = 9*x**4 + 2*x**3 + 3*x**2 + 6*x + 1
```

```bash
tensor(2., requires_grad=True)
tensor(185., grad_fn=<AddBackward0>)
```

자동미분을 True로 세팅하여, 모든 연산에 대해 추적을 할 수 있게 해둔 것이다. 계산작업이 모두 수행 되었으므로 (y) .backward를 수행하여, 모든 그라디어트를 자동으로 계산하게 할 수 있다. 그리고 그 그라디언트는 .grad에 누적되어 저장된다.

```python
y.backward()
x.grad
```

$$ 4 \times 9x^3 + 3 \times 2x^2 + 2 \times 3 x + 6$$

여기에 2를 대입하면

```bash
tensor(330.)
```

```python
x = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(2.0, requires_grad=True)
y = x**2 + z**3
y.backward()
```

```python
x.grad
z.grad
```

각각 2와 12가 나올 것이다.
