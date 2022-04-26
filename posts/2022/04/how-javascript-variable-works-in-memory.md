---
title: 'V8에서 관리되는 자바스크립트 변수'
tags:
  - javascript
  - nodejs
  - V8
published: true
date: 2022-04-21 22:10:10
description: 'V8 내부 코드를 자유롭게 읽을 수 있는 그날까지'
---

## Table of Contents

## 거의 대부분의 변수는 힙에 존재한다.

**일반적으로 원시값은 스택에, 객체는 힙에 할당된다고 알려져있지만, 이와 반대로 자바스크립트의 모든 원시값도 힙에 할당되어 있다.** 이는 굳이 V8 소스코드를 까지 않고도 알 수 있는 방법이 존재한다.

1. 먼저 자신의 로컬 머신에 `node --v8-options` 명령어를 날려보자. 여기에는 다양한 노드 v8과 관련된 옵션값이 존재한다. 그 중에 `stack-size`라는 옵션을 활용하면, 로컬머신의 V8 스택 사이즈를 확인할 수 있다. 나는 최신버전의 (20220422 기준) 노드 v16.6.2를 사용하고 있는데, 864kb가 나왔다.
2. 자바스크립트 파일을 생성한다음, 엄청나게 큰 string을 만들고, `process.memoryUsage().heapUsed`를 활용해서 힙을 얼마나 차지하고 있는지 확인해보자.

```javascript
function memoryUsed() {
  const mbUsed = process.memoryUsage().heapUsed / 1024 / 1024
  console.log(`Memory used: ${mbUsed} MB`)
}

function memoryUsed() {
  const mbUsed = process.memoryUsage().heapUsed / 1024 / 1024
  console.log(`Memory used: ${mbUsed} MB`)
}

console.log('before')
memoryUsed() // Memory used: 4.7296905517578125 MB

const bigString = 'x'.repeat(10 * 1024 * 1024)
console.log(bigString) // 컴파일러가 bitString을 최적화하여 날려먹지 않게 필요하다.

console.log('after')
memoryUsed() // Memory used: 14.417839050292969 MB
```

엄청나게 큰 10mb짜리 string을 선언한 뒤 확인해보니, 해당 문자열 만큼의 약10mb의 차이가 있는 것을 확인할 수 있었다. 앞서 언급했던 스택 사이즈의 경우, 오직 864kb 밖에 없었다. 그렇다. 스택에는 이렇게 큰 문자열을 저장할 공간이 없다.

## 자바스크립트의 원시 값은 대부분 재활용 된다.

만약 아까 만들었던 `'x'.repeat(10 * 1024 * 1024)`를 또다른 변수에 할당한다면, 메모리에 그것을 그대로 복사해서 힙에 총 20mb 를 차지하게 될까?

정답은 그렇지 않다. 중복된 문자열은 별도로 할당되지 않는다. 우리가 일반적으로 알고 있는 것 처럼, 자바스크립트에서 변수를 할당 하는 동작은 실제 값의 크기에 비례하는 비용이 드는 것은 아니다. 자바스크립트 변수의 대부분은 포인터로 이루어져 있다.

이러한 사실을 Chrome DevTools를 활용한 메모리 프로파일링을 통해서 확인할 수 있다.

```html
<html>
  <body>
    <button id="button">button</button>
    <script>
      const button = document.querySelector('#button')

      button.addEventListener('click', function () {
        const string1 = 'hello'
        const string2 = 'hello'
      })
    </script>
  </body>
</html>
```

다음과 같은 html 문서를 만들어 저장하고, Chrome Devtool의 Memory 탭에서 확인해보자.

![chrome-devtool1](./images/chrome-devtool1.png)

![chrome-devtool2](./images/chrome-devtool2.png)

클릭을 여러번해도, string "hello"`는 힙에 단 하나만 존재하는 것을 알 수 있다.

이러한 것을 [String interning](https://en.wikipedia.org/wiki/String_interning)이라고 한다. 각 문자열의 값을 복사본 하나만 저장하는 방법으로, 불변해서 관리하는 것을 의미한다. 이러한 방법을 활용하면, 문자열이 생성되거나 인터닝 될때, 이로인한 시간 소요나 공간을 효율적으로 관리할 수 있게 된다.

v8 내부에서는, 이를 [string-table](https://chromium.googlesource.com/v8/v8/+/fc0cbc144530662db5ef27406e1c7302760e8461/src/objects/string-table.h) 이라는 코드의 형태로 관리한다.

여기에 추가로, V8에는 [oddball](https://chromium.googlesource.com/v8/v8/+/master/src/builtins/base.tq#506) 이라고 불리는 것이 존재한다.

```
type TheHole extends Oddball;
type Null extends Oddball;
type Undefined extends Oddball;
type True extends Oddball;
type False extends Oddball;
type Exception extends Oddball;
type EmptyString extends String;
type Boolean = True|False;
```

이들은 스크립트의 첫번째 라인이 실행되기전에 V8에 의해 힙에 미리 할당되는 값이다. 즉, 자바스크립트 프로그램에서 이러한 값이 사용되던 말건 상관없이 미리 할당해두는 값이라고 보면 된다

그리고 이 값들은 항상 재사용된다. 즉,각 `oddball` 별로 하나의 값만 가지고 있다.

```javascript
function Oddballs() {
  this.undefined = undefined
  this.true = true
  this.false = false
  this.null = null
  this.emptyString = ''
}
const obj1 = new Oddballs()
const obj2 = new Oddballs()
```

앞서 실행해보았던 코드에 위 코드를 추가하고, 다시한번 스냅샷을 찍어보자.

![chrome-devtool3](./images/chrome-devtool3.png)

클릭을 두번해서 별개의 객체가 생성되었음에도, 각각의 값들은 같은 주소를 가리키고 있는 것을 볼 수 있다.

자바스크립트가 `oddball`을 가지고 있는 변수를 만들경우, 이들은 값을 생성하거나 파괴하는 동작을 거치는게 아닌 미리 만들어둔 값을 부르는 방식으로 관리하는 것을 볼 수 있다.

## 자바스크립트 변수들은 대부분 포인터다.

V8 소스코드를 더 깊게 파고들어가 보면, 자바스크립트 프로그램에서 생성한 변수가, 힙에 위치한 C++ 객체를 가리키는 메모리 주소라는 사실을 알 수 있다.

예를 들어, `undefined`는 V8에서 [다음](https://chromium.googlesource.com/v8/v8/+/a684fc4c927940a073e3859cbf91c301550f4318/include/v8-primitive.h#830)과 같이 구현되어 있다.

```
V8_INLINE Local<Primitive> Undefined(Isolate* isolate) {
  using S = internal::Address;
  using I = internal::Internals;
  I::CheckInitialized(isolate);
  S* slot = I::GetRoot(isolate, I::kUndefinedValueRootIndex);
  return Local<Primitive>(reinterpret_cast<Primitive*>(slot));
}
```

우리가 주목해야할 것은, `GetRoot`다. `GeetRoot`는 [다음](https://chromium.googlesource.com/v8/v8/+/a684fc4c927940a073e3859cbf91c301550f4318/include/v8-internal.h#388)과 같이 구현되어 있다.

```
V8_INLINE static internal::Address* GetRoot(v8::Isolate* isolate, int index) {
    internal::Address addr = reinterpret_cast<internal::Address>(isolate) +
                             kIsolateRootsOffset +
                             index * kApiSystemPointerSize;
    return reinterpret_cast<internal::Address*>(addr);
  }
```

## 숫자는 조금 복잡

숫자의 경우에는 다른 객체와는 조금 다르다. 숫자는 사용 빈도가 매우 잦기 때문에, 앞선 방식으로 관리하게 되면 새로운 개체를 할당해야 하는데 이는 V8입장에서 너무 큰 부담이된다. 여기서 사용하는 방법이 `포인터 태깅`이고, 64비트 아키텍쳐를 기반으로 $$-2^{31}$$ 에서 $$2^{31}-1$$ 까지 구성되어 있는 이 숫자를 V8에서는 이를 `smi` 라고 부른다.

이들은 내부적으로 최적화가 되어 있어 포인터에서 추가적인 스토리지를 할당할 필요 없이 포인터 내부에서 인코딩할 수 있다. 이는 V8만의 특징은 아니고, 다른 언어에서도 발견할 수 있다.

이 방법이 조금 복잡한데, 간단하게 요약하자면 SMI에는 포인터가 아닌 부호있는 정수값을 직접 저장한다. SMI는 힙메모리에 할당하지 않는 즉치 값(Immediate values)라고 부른다.

그러나 모든 숫자가 smi인 것은 아니다. 범위를 넘어서는 정수값, double 형식, 값을 박싱해야 하는 경우에는 여전히 힙에 개체로 저장되며 이는 힙숫자라고 부른다.

숫자가 가지고 있는 또다른 복잡한 특징은 앞서 언급했던 문자열과는 반대로, 재사용되지 않을 수도 있다는 사실이다.

이러한 사실들을 실제로 메모리 스냅샷을 통해 알아보자.

```javascript
function MyNumbers() {
  ;(this._integer = 1), (this._double = 1.1)
}
// 전역변수
var _global_integer = 1
var _global_double = 1.1
const num1 = new MyNumbers()
const num2 = new MyNumbers()
```

이제 이 코드를 추가해서 또 확인해보자.

![chrome-devtool4](./images/chrome-devtool4.png)

immediate value인 1은 `smi`가 되어서 관리되고 있어 속성 값에서 나타나지 않았다. 앞서 말한 것처럼, `smi`값 1은 힙메모리에 할당되지 않기 때문이다.

![chrome-devtool5](./images/chrome-devtool4.png)

위 스크린샷은 앞서 선언했던 전역변수 두개를 살펴본 것이다. `smi`인 값은 역시 나타나지 않았고, `_global_double`의 값만 메모리 힙 스냅샷에 나타난다.

이렇게 힙에서 관리하고 있는 숫자를 `HeapNumber` 라고 하는데, 이 숫자의 특징은 재사용되지 않는 다는 것이다.

![chrome-devtool6](./images/chrome-devtool6.png)

> heap number가 각각 다른 포인트를 가리키고 있는 모습

예를 들어, $$3 + 0.14$$ 나 $$\frac{314}{100}$$ 과 같은 연산은 3.14라는 `HeapNumber`가 이미 존재하는지 확인할 필요가 없기 때문에 (이미 연산했으므로) 각각의 다른 `HeapNumber`가 할당된다.

## 정리

```javascript
const a = 'foo'
const b = 123
const c = false
const d = { name: 'foo', number: 123 }
```

컴파일러를 거치면, 이러한 변수는 메모리에 위치하게 된다.

```
a: 0x000
b: 0x010
c: 0x020
d: 0x030
```

자바스크립트 변수는 스택/힙/레지스터 등에 위치하거나 이미 알려져있는 힙 메모리 위치에 존재할 수 있다.

```
0x000: 0x100
0x010: 123
0x020: 0x200
0x030: 0x300
```

실제 자바스크립트의 값은 힙에 위치한다.

```
0x100: 'foo'
0x200: false
0x300: {name: 0x100, number: 123 }
```

컴퓨터 메모리는 엄청나게 복잡한 주제다. 그리고 이러한 주제를 다루기에 아직 부족한 면도 있고, 또 메모리와 관련된 질문에 대한 대부분의 답은 컴파일러와 프로세서 아키텍처 마다 다르다. 예를 들어, 변수는 항상 메모리(RAM)에 있는 것은 아니다. 즉, 대상 레지스터에 직접 로드될 수도 있고, 즉각적인 값으로 명령의 일부가 될 수도, 심지어 완전히 무의 상태로 최적화 될 수 있다. V8과 같은 자바스크립트 엔진은 너무 복잡하고, 제공하는 기능이 강력하므로, 만약 메모리 레이아웃과 같은 저수준의 세부 정보를 공부하기 위해서는 C, C++로 시작하여 소스코드가 기계코드가 되는 방법을 이해하는 것이 좋지 않을까?

## 더 읽어보기

- https://www.zhenghao.io/posts/javascript-variables
- http://www.egocube.pe.kr/lecture/content/html-javascript/202003240001
