---
title: 'HTML은 프로그래밍 언어인가? 라는 논쟁보다 중요한 것'
tags:
  - html
  - programming
published: true
date: 2021-10-03 12:57:23
description: '더이상 HTML 논란은,, 네이버,,,'
---

## Table of Contents

## meme?

![meme1](https://i.redd.it/m41loixjno811.jpg)

![meme2](https://i.kym-cdn.com/photos/images/original/001/382/372/e8b.jpg)

![meme3](https://pbs.twimg.com/media/EPgQXItUcAAQE9V.jpg)

## Introduction

> HTML은 프로그래밍 언어가 아닙니다.

라는 말은 프론트엔드 개발을 시작하면서 지겹게도 들어보았다. 정말로 프로그래밍 언어가 아니라는 것을 강조하고 싶었던 것인지, 자바스크립트 개발자로서 HTML은 매우 쉬운 언어이기 때문에 차별성을 두고 싶었던건지, 어쨌는 건지 모르겠지만 흔히들 하는 말은 '로직이 없다' 라든가 [튜링 완전(turing completeness)](https://ko.wikipedia.org/wiki/%ED%8A%9C%EB%A7%81_%EC%99%84%EC%A0%84) 이 없다 라든가 등의 논리를 들며 HTML은 프로그래밍 언어가 아니고 마크업 언어라고 주장한다.

> 튜링 완전(turing completeness)이란 어떤 프로그래밍 언어나 추상 기계가 튜링 기계와 동일한 계산 능력을 가진다는 의미이다. 이것은 튜링 기계로 풀 수 있는 문제, 즉 계산적인 문제를 그 프로그래밍 언어나 추상 기계로 풀 수 있다는 의미이다.

그렇다면 정말 프로그래밍 언어로서 HTML은 부족한 점이 있는 것일까? HTML은 정말 프로그래밍 언어가 아닌 것일까? 흔히들 HTML이 프로그래밍 언어로서 무언가 결함이 있거나 부정확하다고 주장할 때 언급되는 논리에 대해서 하나씩 살펴보자.

## HTML이 프로그래밍 언어가 아니라고 하는 주장들

대표적인 주장들을 살펴보자.

### HTML은 마크업 언어지, 프로그래밍 언어가 아니예요.

이 문장 자체는 맞는 것 같아 보이지만 틀렸다고 생각한다. 마크업 언어도 프로그래밍 언어가 될 수 있다. 모든 마크업 언어가 프로그래밍 언어인것은 아니지만, 그럴 수도 있다. 두 개의 벤다이어그램을 그린다면, 약간은 교차하는 모습이 나올 것이다.

변수, 제어, 루프 등을 가지고 있는 마크업 언어 또한 프로그래밍 언어가 될 수 있다. 이는 상호 배타적인 개념이 아니다. 마찬가지로 수학 수식을 그릴 때 사용하는 `Tex` 와 `LaTex` 도 프로그래밍 언어로 볼 수 있는 마크업 언어다. 이 두언어를 가지고 개발을 하는 것은 일반적인 상황은 아니지만, 온라인에서 이러한 예시를 찾아볼 수는 있다.

- https://www.ctan.org/tex-archive/macros/generic/basix
- https://sdh33b.blogspot.com/2008/07/icfp-contest-2008.html

따라서 마크업 언어라고 해서 프로그래밍 언어가 아니라고 하는 것은 잘못되었다. 요점은 두개의 개념이 별개가 아니라는 것이다. 마크업 언어는 프로그래밍 언어일 수 있다. 따라서 HTML이 마크업 언어이기 때문에 프로그래밍 언어가 아니라고 말하는 것은 애초에 전재 조건이 잘못되었다고 볼 수 있다.

### HTML은 로직을 가질 수 없어서 프로그래밍 언어가 아니예요.

로직이란 무엇일까? 튜링 완전과 마찬가지로, 이 주장을 다루기 전에 로직이란 무엇인지 확실히 정의내릴 필요가 있다.

보통 프로그래밍 언어에서의 로직은, 변수나 조건, 루프 등이 있어 사용할 수 있는 것을 의미한다. HTML은 그런데 이러한 것을 사용할 수 없기 때문에 프로그래밍 언어가 아니라고 생각할 수 있다. 하지만 HTML 속성 중에는 변수가 있으며, 변수와 함께 사용할 수 있는 제어구조가 있다. 내부 논리를 제어하는데 자바스크립트나 CSS 가 필요없는 HTML element 가 있다. 여기서 말하는 것은 과거 표준이었던 `<link>` `<noscript>`가 아니다. 사용자의 입력에 따라서 element의 현재 상태와 변수에 값에 따라 조건부 작업을 수행하는 element를 말하는 것이다.

아래 예시를 살펴보자.

```html
<dialog id="dialog1" open>
  <h2>This is an HTML <code>&lt;dialog&gt;</code></h2>
  <p>Dialog</p>
  <p>CSS나 자바스크립트 없이도 닫을 수 있습니다?!</p>
  <form method="dialog" action="#dialog1">
    <button>닫기</button>
  </form>
</dialog>

<details>
  <summary>화살표를 눌러보세요</summary>
  <div>
    <p>띠요옹</p>
    <p>CSS나 자바스크립트 없이도 여닫기를 할 수 있다?!</p>
  </div>
</details>
```

https://6710t.csb.app/

- https://developer.mozilla.org/ko/docs/Web/HTML/Element/dialog
- https://developer.mozilla.org/ko/docs/Web/HTML/Element/summary

따라서 HTML에는 로직이 없어서 프로그래밍 언어가 아니라고 말하는 것도 오해의 소지가 있다. HTML이 사용자의 입력 (클릭)에 따른 결정을 내릴 수 있다. 물론 HTML이 로직을 가지고는 있지만, 데이터를 조작하도록 설계된 다른 언어의 로직과는 본질적으로 다르긴하다. 어쨌거나, HTML이 프로그래밍 언어가 아니라고 말하기 위해서는 더 논리적인 주장이 필요하다.

### 튜링 완전함이 없어서 프로그래밍 언어가 아닙니다.

튜링 완전성이 무엇인지에 대한 논의는 여기저기서 많이 다뤄지고 있기 때문에 굳이 언급하지는 않겠다.

> In the simplest terms, for a language or machine to be Turing complete, it means that it is capable of doing what a Turing machine could do: perform any calculation, a.k.a. universal computation. After all, programming was invented to do math although we do a lot more with it now, of course!

> 언어나 머신이 튜링 완전성을 가지기 위해서는, 튜링 기계가 할 수 있는 것을 똑같이 할 수 있어야 한다. 모든 연산, 즉 보편적인 연산을 수행할 수 있어야 한다. 프로그래밍은 비록 우리가 수학으로 할 수 있는 것 보다 더 많은 것을 하고 있지만, 본질적으로는 수학을 하기 위해서 발명되었다.

> https://notlaura.com/is-css-turing-complete/

대부분의 현대 프로그래밍 언어들이 튜링 완전하기 때문에 사람들은 이를 프로그래밍 언어의 정의로 사용한다. 하지만 사실 튜링 완전성의 정의는 그것이 아니다. 앞서 위키피디아에서 본것 처럼, 튜링 기계를 시뮬레이션 할 수 있는지 여부를 나타내는 것이다. 튜링 완전성은 프로그래밍 언어를 분류하는데 사용할 수 있지만, 튜링 완전성을 가지고 프로그래밍 언어의 정의를 내리는데 사용하기엔 조금 무리가 있다. 예를 들어 마인크래프트나 매직더 게더링도 튜링 완전성을 가지고 있지만, 이를 프로그래밍 언어라고 생각하는 사람은 없다.

튜링 완전성은 아주 과거 일부 사람들이 컴파일 언어와 인터프리터 언어의 차이를 '좋은 언어'의 차이로 나눴던 것과 마찬가지 방식으로 사용되고 있는 것 같다. 백엔드 개발자가, 프론트엔드 개발자를 '가짜 프로그래밍을 하는 사람'으로 평가 절하했던 것과 마찬가지로.

프로그래밍의 정의는 시간에 따라 달라진다. 천공 카드에 어셈블리 코딩을 입력하는 것이 진짜 프로그래밍이라고 주장하는 사람들도 있을 것이다. 보편적이거나, 모세의 그것 처럼 돌로 쓰여진 것은 아무것도 없다.

튜링 완전성은 공정한 기준이라고 볼수도 있지만, 편향되고 주관적인 기준이라고 보는 것이 맞다. 튜링 완전 기계를 생성할 수 있는 언어는 프로그래밍 언어로 판단되는 반면, [유한 상태 기계](https://ko.wikipedia.org/wiki/%EC%9C%A0%ED%95%9C_%EC%83%81%ED%83%9C_%EA%B8%B0%EA%B3%84)를 생성하는 언어는 그렇지 않는 것으로 보는 이유는 무엇인가? 이는 완전히 주관적이라고 생각한다.

마찬가지로, HTML은 튜링 완전하지 않다라고 하는 사람들도 정작 튜링 완전성이 무엇인지 모르거나 이해하지 못하는 것 또한 사실이다. 튜링 완전성은 품질 보증서가 아니다. 프로그래밍 언어를 정의하는 것이 아니라 분류하는 방법이다. 프로그래밍언어는 컴파일/인터프리터일수도, 명령형/선언형 일수도, 절차적이거나/객체지향적 일수도있다. 그리고 마찬가지로 튜링완전할 수도 그렇지 않을 수도 있다.

## 그래서 HTML이 프로그래밍 언어라구요? 아뇨, 더 중요한 것은..

그래서, 프로그래밍 언어가 아니라고 하는 주장을 반박하면 HTML이 프로그래밍 언어가 되는건가? 그런 건 아니다. HTML 표준이 상상 이상으로 발전하거나, 프로그래밍 언어의 정의가 바뀌지 않는 이상 이 논쟁은 아마두 계속 될 것이다.

그러나 중요한 것은 개발자로서 이러한 질문을 받아드릴 경우 심각한 논쟁을 야기하는 것이 아니라, 이러한 논쟁으로부터 개발 생태계를 분리시키는 것이다. 이러한 논쟁은 의도를 숨긴채 쓸데없는 논란을 불러 오기 때문에 매우 경계해야 한다.

> They say you’re not a real programming language like the others, that you’re just markup, and technically speaking, I suppose that’s right. Technically speaking, JavaScript and PHP are scripting languages. I remember when it wasn’t cool to know JavaScript, when it wasn’t a “real” language too. Sometimes, I feel like these distinctions are meaningless, like we built a vocabulary to hold you (and by extension, ourselves as developers) back. You, as a markup language, have your own unique value and strengths.

> 사람들은 여러분이 다른 언어처럼 진짜 프로그래밍 언어라고 말합니다. 이는 단지 마크업이고, 엄밀히 말하면 저는 이것이 옳다고 생각합니다. 더 기술적으로 이야기 하자면, 자바스크립트와 PHP는 스크립트 언어 입니다. 자바스크립트를 아는 것이 자랑이 아니었을 때, 자바스크립트가 진짜 언어 취급을 받지 않았을 때를 기억합니다. 때때로 이런 구별과 논의가 무의미하다고 생각합니다. 마치 이러한 논쟁이 마크업 개발자를 자바스크립트 개발자 뒤로 붙들기 위한 어휘 처럼 느껴집니다. 마크업 언어로서, 여러분 (마크업 개발자 분들)들은 자신만의 고유한 가치와 장점을 가지고 있습니다.

https://css-tricks.com/a-love-letter-to-html-css/

## 결론

HTML 은 프로그래밍 언어인가 아닌가는 별로 중요한 논의도 아니고, 누군가 결정을 내려줘야할 문제도 아니다. 과거 프론트엔드 개발자들이 백엔드 개발자들에게, 나아가 개발자 커뮤니티에서 알게 모르게 무시 당했던 것과 비슷한 느낌이다. HTML과 프로그래밍 언어 이 논쟁은 의미도 없고 아무런 가치도 가지고 있지 않다. 개발을 하는데 개발자가 컴퓨터 공학과 인지, 동양인지 전혀 상관없는 것 처럼 개발자는 개발자 일 뿐이고, HTML은 HTML 일 뿐이다. 웹 생태계, 나아가 현재의 소프트웨어 생태계를 보았을 때 HTML이 차지하는 비중이 결코 작지 않다. HTML은 다른 것들과 마찬가지로 방대한 문서와 광범위한 문법을 가지고 있고, 간단하게 배울 수 있는 것도 아니며, 그 복잡성으로 인해 숙달하는데 있어 몇년이 걸린다. 프로그래밍 언어든 아니든 중요한 것은 HTML이 있다는 것, 그리고 좋은 품질의 웹 애플리케이션을 만들기 위해 반드시 사용해야 한다는 것, 그 것 뿐이다.

## 살펴보기

- https://notlaura.com/is-css-turing-complete/
- https://css-tricks.com/a-love-letter-to-html-css/
- https://css-tricks.com/html-is-not-a-programming-language/
