---
title: '크롬 메모리 프로파일러 사용하는 방법'
tags:
  - javascript
  - chrome
published: true
date: 2022-04-28 20:18:04
description: '스냅샷 해석과 디버깅의 책임은 본인에게 있습니다'
---

## Introduction

웹 애플리케이션 성능 최적화 내지는 메모리 이슈를 해결하기 위해서 이것저것 뒤지다보면, 결국 최종적으로 확인해봐야 할 것은 바로 이 메모리 프로파일링 탭이다. 이 탭을 통해 웹 애플리케이션에서 메모리 누수가 일어나고 있는지, 또 메모리를 최대한 효율적으로 사용하고 있는지를 확인하기 위해서는, 이 메모리 프로파일링 탭을 읽을 수 있어야 한다. 마침 크롬 버전이 업데이트 되면서 (꽤 되긴했지만) 친절하게 크롬의 디버그 도구가 한글로 번역까지 되어 있다.

본격적으로 탭을 살펴보기에 앞서, 크롬 메모리 프로파일러를 처음 보면 매우 혼란스럽다. 친절한 자바스크립트 코드를 보다가 포인터와 각종 희한한 정보들을 살펴보다면 매우 혼란스러울 것이다. 그래서 본격적으로 우리가 (혹은 내가) 만든 페이지를 디버깅 하기에 앞서, 빈 html 태그만 있는 페이지를 기준으로 살펴보려고 한다.

HTML의 역사는 매우매우 오래되었고 또 갖가지 문법들이 유서깊게 짬뽕되어 있기 때문에, 우리가 빈 `<html/>` 문서만 만들어도 크롬은 알아서 `head`와 `body` 삽입해서 기본적인 HTML 트리를 만들어 준다. 아무튼, 이 빈 `<html/>`을 시작으로 크롬 메모리 탭에서 무슨 일이 일어나고 있는지 살펴보자.

```html
<html />
```

위 파일을 별도 html로 저장한 다음에 시크릿탭으로 페이지를 한번 열어보자. 꼭 시크릿탭으로 여는 것이 좋다. 그렇지 않으면 안그래도 정신사나운 정보들에 추가로 익스텐션 정보들까지 달라붙어 매우 읽기 어려워진다.

![chrome-memory-profiler1](./images/chrome-memory-profiler1.png)

힙 스냅샷 하단에 있는 숫자값을 포함할지 여부를 꼭 선택하자. [이전 포스트](/2022/04/how-javascript-variable-works-in-memory#숫자는-조금-복잡)에서 살펴보았던 것처럼, `smi` 숫자는 관리 방식이 달라 이것을 체크하지 않으면 숫자값을 볼 수 없다.

![chrome-memory-profiler2](./images/chrome-memory-profiler2.png)

놀랍게도, 아무것도 없는 말그대로 빈페이지 주제에 단순히 빈 페이지를 렌더링하는데에도 많은 오브젝트가 관여되어 있는 것을 볼 수 있다. 이 페이지가 로드된 이후, 인스턴스화된 각 자바스크립트 객체는 해당 생성자 클래스 아래에 그룹화되어 있는 것을 볼 수 있다. 괄호로 쳐져있는 그룹 `()`은 직접 호출할 수 없는 네이티브 생성자를 나타낸다. 위 그림에서 보면 많은 `(compiled code)` `(system)` 등도 볼 수 있고, 그리고 `Date` `String` `RangeError` 과 같은 전통적인 자바스크립트 객체도 볼 수 있다.

이 모든 것을 이해하기 위해서, 유저가 간단하게 버튼을 눌러서 동작하는 작업을 추가해보자.

```html
<html>
  <head>
    <script>
      var counter = 0
      var instances = []

      function X() {
        this.i = counter++
      }

      function allocate() {
        instances.push(new X())
      }
    </script>
  </head>
  <body>
    <button onclick="allocate()">Allocate</button>
  </body>
</html>
```

위 코드에서 버튼을 클릭하고 메모리 프로파일러를 열어보자.

![chrome-memory-profiler3](./images/chrome-memory-profiler3.png)

`X`라는 객체가 할당되어 있는 것을 볼 수 있다.

이를 조금 더 찾기 쉽게 하는 방법은, 먼저 첫번째 스냅샷을 찍은 뒤에, 다시 동그라미 버튼을 눌러 두번쨰 스냅샷을 찍는 것이다. 그리고 드롭다운에서, 이 스냅샷 사이에 생성된 생성자만 보는 방법이 있다.

![chrome-memory-profiler5](./images/chrome-memory-profiler5.png)

버튼을 클릭한 이벤트만 했을 뿐이라, `X`만 보였을 것이라 예상하였지만, 몇가지 추가적인 작업이 발생했음을 알 수 있다. 크롬의 경우 레이지 로딩 객체에 대해 최적화를 하는 작업이 있다. 이 경우에는 HTML 버튼 엘리먼트을 클릭하기 전까지 메모리가 주어지지 않았음을 알 수 있다. 즉 클릭이 실제로 일어났을 때 그때서야 비로소 메모리를 할당해서 작업을 한 것이다.

이를 확인해보는 방법은 버튼을 여러번 클릭해보는 것이다. 여러번 클릭한후 스냅샷을 찍어두면, 아까와 다르게 딱 필요한 `X`만 할당해서 작업이 이뤄지고 있음을 알 수 있다.

![chrome-memory-profiler6](./images/chrome-memory-profiler6.png)

> 여기서 보여주는 아이디는 객체 인스턴스를 구별하기 쉽게 도와주는 아이디 값으로, 실제로 메모리 주소를 가리키는 것은 아니다.

각 인스턴스는, 클래스 이름이 아래에 내열되고, 실제로 그 객체를 클릭해보면 객체에 대한 상세한 정보가 나와있는 것을 알 수 있다.

![chrome-memory-profiler7](./images/chrome-memory-profiler7.png)

여기서 주목해야할 것은 `얕은 크기`라고 작성되어 있는 열이다. 이 `얕은 크기`라는 것은 객체가 유지하고 있는 바이트의 크기를 나타낸다. 자바스크립트는, 이 자바스크립트를 만든 C언어와는 다르게 객체 하나를 지하는데 더 많은 크기를 차지한다.

`유지된 크기`는 객체가 참조를 보유하고 있는 객체 외에 객체 자체의 내부 메모리 때문에 이 객체가 보유하고 있는 바이트 수를 의미한다. 그리고 이 메모리는 가비지 콜렉팅 되지 않는다. 무슨 말인지 이해하기 위해 다음 예제를 실행해보자.

```html
<html>
  <head>
    <script>
      var counter = 0
      var instances = []

      function Y() {
        this.j = 5
      }

      function X() {
        this.i = counter++
        this.y = new Y()
      }

      function allocate() {
        instances.push(new X())
      }
    </script>
  </head>
  <body>
    <button onclick="allocate()">Allocate</button>
  </body>
</html>
```

![chrome-memory-profiler8](./images/chrome-memory-profiler8.png)

위 예제는, X 인스턴스가 생성될 때 마다 `y`에 `Y` 인스턴스를 초기화 하고 할당한다. `X`자체는 52 바이트 밖에 없지만, `X`내부가 순수 이 객체외에 참조를 유지하기 위해 이용하는 메모리, 즉 `Y`의 사이즈인 48 바이트를 추가로 사용하므로 `유지된 크기`는 100 바이트임을 알 수 있다.

```html
<html>
  <head>
    <script>
      var counter = 0
      var instances = []

      function X() {
        this.i = counter++
        if (instances.length) {
          this.ref = instances[instances.length - 1]
        }
      }

      function allocate() {
        instances.push(new X())
      }
    </script>
  </head>
  <body>
    <button onclick="allocate()">Allocate</button>
  </body>
</html>
```

크롬의 메모리 프로파일러는 `유지된 크기`를 계산하는데 있어 매우 영리한 모습을 보여준다.

![chrome-memory-profiler9](./images/chrome-memory-profiler9.png)

예제를 스냅샷 찍은 모습에서 알 수 있는 것처럼, X에 새로운 Y를 할당하는 대신에 인스턴스 베열에서 이전에 만들었던 인스턴스에 대한 참조를 유지하여 메모리를 효율적으로 관리하는 것을 볼 수 있다.

여기서 언급하지 않은 열 하나가 바로 `거리`다. 자바스크립트의 메모리 누수는 다른 객체가 참조를 보유하고 있어서 가비지 콜렉터가 객체 인스턴스를 수집하지 못할 때 발생한다.다른 객체가 해당 객체에 대한 참조를 보유하고 있기 때문에 발생한다.

![chrome-memory-profiler10](./images/chrome-memory-profiler10.png)

지금 위 그림에서 보여준 객체는 가비지 콜렉팅에 적합하지 않아서 힙에서 계속 머물러 있는 것을 볼 수 있다. 여기에서 객체 항목을 보면 이 인스턴스가 메모리를 계속해서 차지하고 있는 이유를 알 수 있다. 이 객체는 글로벌, 즉 `window` 객체가 소유하는 인스턴스 배열이 유지되어야 하기 때문에 계쏙해서 존재하는 것을 볼 우 있다. 웹 애플리케이션에서 메모리 누수가 발생하는 주요 원인은 잘못된 변수 선언에 있으며, 이러한 잘못 선언된 변수가 글로벌에 계속 머물러 있기 때문이다. 이 경우에는 이 객체 뷰를 통하여 신속하게 왜 유지되고 있는지를 확인할 수 있다.

다음으로 살펴볼 메뉴는 타임라인 할당 계측 이다. 이 메뉴는 앞선 힙 스냅샷과 유사하다. 한가지 차이점이라면 지속적으로 실행되어 멈추기전까지 이 메모리 스냅샷에서 일어나는 변화를 알 수 있다는 것이다. 사용자의 인터랙션에 따른 메모리의 상황을 점검하기 위해 유용하다.

![chrome-memory-profiler11](./images/chrome-memory-profiler11.png)

![chrome-memory-profiler12](./images/chrome-memory-profiler12.png)
