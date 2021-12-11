---
title: ''
tags:
  - javascript
  - web
  - browser
published: false
date: 2021-12-08 19:45:17
description: ''
---

## Table of Contents

## 바이트코드 번들

요즘 모던 웹 애플리케이션의 경우, 브라우저가 최초로 보게되는 자바스크립트는 사실 우리가 알고 있는 개발자가 작성한 자바스크립트가 아니다. 일반적으로 브라우저가 처음 마주하는 자바스크립트는 webpack과 같은 도구에 의해 생산된 번들 파일일 것이다. 그리고 이 번들에는 리액트와 같은 ui 프레임워크, 그리고 다양한 polyfill, 그리고 npm을 통해 제공되는 다양한 패키지들이 포함된 다소 큰 번들일 것이다. 브라우저에 존재하는 자바스크립트 엔진의 첫번째 임무는 이렇게 받은 큰 자바스크립트 텍스트 묶음을 가상머신에서 실행할 수 있는 명령어로 변환하는 것이다. 자바스크립트 엔진은 코드를 파싱해야 하는데, 사용자가 자바스크립트에서 인터랙션을 기다리고 있기 때문에 이 작업을 빠르게 할 필요가 있다.

높은 수준에서, 자바스크립트엔진은 다른 프로그래밍 언어 컴파일러처럼 코드를 파싱한다. 먼저 자바스크립트 코드 텍스르트를 토큰이라고 불리는 청크로 나눈다. 각 토큰은 일반적인 우리 언어와 유사하게 구문 구조내에서 의미있는 단위로 쪼개진다. 그런 다음 이 토큰들은, 프로그램을 나타내는 트리구조를 생성하는 탑다운 파서로 공급된다. 자바스크립트 언어 디자이너와 컴파일러 엔지니어들은 이 트리 구조를 [AST (abstract syntax tree)](https://gyujincho.github.io/2018-06-19/AST-for-JS-devlopers) 라고 부른다. 그 결과 AST를 분석하여 바이트 코드라고 불리는 가상 머신 명령어 목록을 생성할 수 있다.

![js-ast-bytecode](https://i0.wp.com/alistapart.com/wp-content/uploads/2018/11/fig1.png?w=960&ssl=1)

AST를 생성하는 과정은 자바스크립트에서 비교적 간단한 임무 중 하나다. 그러나 불행하게도, 이 작업은 느릴 수 있다. 자바스크립트 엔진은 사용자가 인터랙션을 시작하기전에 전체 번들에 대한 구문 트리를 분석하고 빌드해야 한다. 이러한 코드의 대부분은 초기 페이지 로딩에 불필요할 수 있으며, 일부는 아예 실행되지도 않을 수도 있다.

다행히도(?) 컴파일러 엔지니어들은 이러한 느린 속도를 높이기 위해 다양한 트릭을 만들었다. 먼저, 일부 엔진들은 백구라운드 스레드에서 코드를 파싱하여 다른 작업을 위한 메인 UI 스레드를 확보한다. 그리고 요즘 모던 엔지는 지연 구문 분석 또는 지연 컴파일이라는 기술을 사용하여 가능한 장시간 메모리 내 구문 트리 생성을 지연한다. 이 동작 방법을 간단히 살펴보자. 엔진이 한동한 실행 되지 않게 될 함수를 보게 되면, 함수 몸체를 가능한 빠르게 'throwaway' 한다. 나중에, 이 함수를 처음으로 호출하게 되면 코드가 다시 파싱된다. 이 때 엔진은 실행에 필요한 전체 AST 및 바이트를 생성한다. 자바스크립트의 세계에서는, 어떤 것을 두번하는 것이 한번하는 것보다 빠를 때도 종종있다.

여기서 가장 좋은 최적화는, 우리가 어떤 작업이든 바이패스 할 수 있게 해주는 것이다. 자바스크립트 컴파일에서는 파싱 단계를 완전히 건너뛴다. 일부 자바스크립트 엔진은 사용자가 사이트를 다시 방문할 경우 나중에 재사용할 수 있도록 생성된 바이트 코드를 캐시하려고 시도한다. 물론 이는 말처럼 간단하지 않다. 자바스크립트 번들은 웹사이트가 업데이트 됨에 따라서 자주 변경될 수 있으며, 브라우저는 캐싱에 오는 성능 향상과 비교해서 바이트코드를 직렬화하는 비용을 신중히 따져 봐야 한다.

## 바이트코드 런타임

이제 바이트 코드를 획득했으니 실행할 준비가 되었다. 오늘날 자바스크립트 엔진은 파싱 중에 생성한 코드를 가장 먼저 인터프리터라는 가상 머신에 제공한다. 인터프리터는 소프트웨어에 구현된 CPU와 약간 비슷하다. 각 바이트 코드 명령어를 한번에 하나씩 살펴보고, 실제 기계 명령어를 실행한다음, 다음에 수행해야 하는 작업을 결정한다.

자바스크립트라 불리는 이 프로그래밍 언어의 구조와 동작은 [ECMA-262](https://tc39.github.io/ecma262) 문서에 공식적으로 정의 되어 있다. 프로그래밍 언어 디자이너 들은 구조 부분을 "syntax"라고 부르고, 행동/작업 (behavior)을 "semantics"이라고 한다. 이 "semantics" 의 거의 대부분은 의사 코드를 사용하여 작성된 알고리즘에 의해 정의 된다. 예를 들어, `>>`, [signed right shift operator](<https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Bitwise_Operators#%3E%3E_(Sign-propagating_right_shift)>)를 구현하는 컴파일러 엔지니어라고 가정해보자. 여기에서 정의된 [스펙](https://tc39.github.io/ecma262/#sec-signed-right-shift-operator-runtime-semantics-evaluation)을 따르면 아래와 같이 구현하게 된다.

ShiftExpression : ShiftExpression >> AdditiveExpression

1. Let _lref_ be the result of evaluating ShiftExpression.
2. Let _lval_ be ? [GetValue](https://tc39.github.io/ecma262/#sec-getvalue)(_lref_).
3. Let _rref_ be the result of evaluating AdditiveExpression.
4. Let _rval_ be ? [GetValue](https://tc39.github.io/ecma262/#sec-getvalue)(_rref_).
5. Let _lnum_ be ? [ToInt32](https://tc39.github.io/ecma262/#sec-toint32)(_lval_).
6. Let _rnum_ be ? [ToUint32](https://tc39.github.io/ecma262/#sec-touint32)(_rval_).
7. Let shiftCount be the result of masking out all but the least significant 5 bits of _rnum_, that is, compute _rnum_ & 0x1F.
8. Return the result of performing a sign-extending right shift of _lnum_ by shiftCount bits. The most significant bit is propagated. The result is a signed 32-bit integer.

처음 여섯 단계를 먼저 보면, 피연산자 (`>>`의 양쪽 값)을 32비트 정소로 변환한 다음, 실제 시프트 연산을 수행했다.

그런데, 우리가 이 스펙 곧이 곧대로 알고리즘을 구현한다면, 결과적으로 매우 느린 인터프리터를 만들게 될 것이다. 자바스크립트 객체에서 속성 값을 가져오는 가주 간단한 작업을 상상해보자.

자바스크립트의 객체는 개념적으로 일종의 `dict`(사전)과 같다. 객체는 또한 프로토타입 객체도 가질 수 있다.

![js-object](https://i0.wp.com/alistapart.com/wp-content/uploads/2018/11/fig2.png?w=960&ssl=1)

그래서, 만약 객체에 주어진 문자열 키에 대한 엔트리가 없다면 우리는 프로토타입에서 그 키를 찾아야 한다. 이 작업은 원하는 키를 찾거나, 프로토타입의 체인 끝까지 갈 때까지 반복된다.

따라서 객체에서 특정 값을 찾으려고 할 때마다 수행해야 하는 작업이 많아질 수도 있다.

이러한 동적 속성 조회 속도를 높이기위해, 자바스크립트 엔진에 사용되는 전략을 인라인 캐싱이라고 한다. 인라인 캐싱은 1980년도에 스몰토크 언어용으로 처음개발되었다. 기본적인 아이디어는, 이전 키 조회 작업 결과를 바이트코드 명령어에 직접 저장하는 것이다.
