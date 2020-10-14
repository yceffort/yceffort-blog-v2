---
title: 프론트엔드 개발자가 알아야 하는 Angular와 React의 Change Detection
tags:
  - javascript
  - react
  - angular
  - web
published: true
date: 2020-07-04 07:06:10
description: "[What every front-end developer should know about change detection
  in Angular and
  React](https://indepth.dev/what-every-front-end-developer-should-know-about-c\
  hange-detection-in-angular-and-react/)를 번..."
category: javascript
slug: /2020/07/change-detection-in-angular-react/
template: post
---

[What every front-end developer should know about change detection in Angular and React](https://indepth.dev/what-every-front-end-developer-should-know-about-change-detection-in-angular-and-react/)를 번역/요약한 것입니다.

## Table of Contents

요즘 거의 대부분의 웹 애플리케이션에서 Change Detection을 찾을 수 있다. 이는 인기있는 웹 프레임워크의 필수적인 부분이다. Data Grids나 stateful jquery 플러그인 등도 충분히 발전된 Change Detection를 가지고 있다. 그리고 아마도 대부분의 애플리케이션 코드 베이스에는 Change Detection가 존재할 가능성이 크다.

소프트웨어 설계에 관심있는 사람은 이 메커니즘을 잘 이해 해야 한다. DOM 업데이트와 같이 눈에 띄는 부분을 담당하기 때문에, Change Detection이 아키텍쳐의 가장 중요한 부분이라고 생각한다. 응용프로그램의 성능에 영향을 미치는 영역이기도 하다.

전반적인 Change Detection을 알아보는 것을 시작으로, 메커니즘을 구현해볼 것이다. 그리고 이것이 리액트와 앵귤러에서 어떻게 구현되는지 심층적으로 살펴볼 것이다.

## Change Detection 은 무엇인가?

> Change Detection은 애플리케이션 상태(state) 변경을 추적하고 이러한 업데이트된 상태를 화면에 렌더링 하도록 설계한 메커니즘이다. 이는 사용자 인터페이스가 항상 내부 상태와 동기화 되도록 한다.

이 정의에서 알 수 있듯, Change Detection이 `변경 추적` 과 `렌더링` 이라는 중요한 두부분을 가지고 있다는 것을 알 수 있다.

먼저 렌더링을 알아보자. 어떤 애플리케이션이든, 렌더링 프로세스는 프로그램 내부 상태를 파악하고, 화면에서 이를 볼 수 있도록 한다. 웹 개발에서 객체나 배열 같은 데이터 구조를 가지고 있고, 이를 이미지, 버튼, 등 기타 시각적인요소의 형태로 해당 데이터의 DOM을 표현하게 된다. 렌더링 로직구현이 사소한 것은 아니지만 - 꽤 간단한 형태를 취한다.

시간이 지남에 따라 변하는 데이터를 표시하기 시작하면 상황은 훨씬 더 정교함을 요구하게 된다. 오늘날의 웹 애플리케이션은 상호작용한다. 즉 애플리케이션의 상태는 사용자의 상호작용의 결과로 언제든지 변경 될 수 있다. 또는 서버에서 데이터를 가져와서 변경될 수도 있다.

상태 변화는, 이것을 감지해야 하며 이러한 변화를 반영해야 한다.

![](https://admin.indepth.dev/content/images/2019/11/image-10.png)

예제를 살펴보자.

## Rating Widget

별점 위젯을 만든다고 가정해보자. 대략 아래와 같은 모습을 취할 것이다.

![rating-widget](https://admin.indepth.dev/content/images/2020/01/1_4OWcFci2w7bTNpEDaZHoEQ.gif)

별점을 추적하기 위해, 현재의 값을 어딘가에 저장해두어야 한다. 아래와 같은 private `_rating` 프로퍼티로 현재 상태를 정의해 두자.

```javascript
export class RatingsComponent {
  constructor() {
    this._rating = 1
  }
}
```

위젯의 상태가 변화하게되면, 이를 화면에 반영해야 된다.

```html
<ul class="ratings">
  <li class="star solid"></li>
  <li class="star solid"></li>
  <li class="star solid"></li>
  <li class="star outline"></li>
  <li class="star outline"></li>
</ul>
```

### 초기화

먼저, 구현하는데 필요한 돔 노드를 만들어야 한다.

```javascript
export class RatingsComponent {
  // ...
  init(container) {
    this.list = document.createElement('ul')
    this.list.classList.add('ratings')
    this.list.addEventListener('click', (event) => {
      this.rating = event.target.dataset.value
    })

    this.elements = [1, 2, 3, 4, 5].map((value) => {
      const li = document.createElement('li')
      li.classList.add('star', 'outline')
      li.dataset.value = value
      this.list.appendChild(li)
      return li
    })

    container.appendChild(this.list)
  }
}
```

### Change Detection

별점에 변화가 있을 때 마다 이를 알아채야 한다. Change Detection의 기본적인 구현에서는, 자바스크립트에서 제공하는 [setter](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Functions/set)를 활용할 것이다. 따라서 별점을 위한 setter를 정의하고, 그 값이 변경될 때 마다 업데이트를 트리거 한다. DOM 업데이트는 목록 항목의 클래스를 변경하면서 수행한다.

```javascript
export class RatingsComponent {
  // ...
  set rating(v) {
    this._rating = v

    // triggers DOM update
    this.updateRatings()
  }

  get rating() {
    return this._rating
  }

  updateRatings() {
    this.elements.forEach((element, index) => {
      element.classList.toggle('solid', this.rating > index)
      element.classList.toggle('outline', this.rating <= index)
    })
  }
}
```

예제는 [여기](https://stackblitz.com/edit/js-aufmae)에서 찾아볼 수 다.

이렇게 매운 간단한 위젯을 구현하기 위해 작성해야 하는 코드의 양을 보자. 일부 시각적 요소를 표시하거나, 숨길 수 있는 목록, 조건부 로직 등 훨씬더 정교한 기능을 상상한다면, 코드의 양과 복잡성은 계속해서 증가할 것이다. 이상적인 상황이라면, 일반적인 개발에서 우리는 애플리케이션의 논리에 초점을 맞춰야 한다. 다른 누군가가 상태를 추적하고, 화면을 업데이트 하는 것을 맡기를 원한다. 그리고 이것이 프레임워크가 필요한 지점이다.

## 프레임워크

프레임워크에서 애플리케이션 내부 상태와 사용자 인터페이스 사이의 동기화를 관리한다. 이들은 우리의 부담을 줄여줬을 뿐 만 아니라, 상태 주적과 DOM 업데이트를 매우 효율적으로 처리한다.

다음은 리액트와 앵귤러에서 동일한 위젯을 구현하는 방법이다. UI와 관련한 사용자 관점에서, 템플릿은 컴포넌트 구성요소에서 매우 중요하다. 이러한 프레임워크에서 비슷한 방식으로 처리한 것은 매우 흥미롭다.

### Angular

```html
<ul class="rating" (click)="handleClick($event)">
  <li [className]="'star ' + (rating > 0 ? 'solid' : 'outline')"></li>
  <li [className]="'star ' + (rating > 1 ? 'solid' : 'outline')"></li>
  <li [className]="'star ' + (rating > 2 ? 'solid' : 'outline')"></li>
  <li [className]="'star ' + (rating > 3 ? 'solid' : 'outline')"></li>
  <li [className]="'star ' + (rating > 4 ? 'solid' : 'outline')"></li>
</ul>
```

### React

```html
<ul className="rating" onClick={handleClick}>
    <li className={'star ' + (rating > 0 ? 'solid' : 'outline')}></li>
    <li className={'star ' + (rating > 1 ? 'solid' : 'outline')}></li>
    <li className={'star ' + (rating > 2 ? 'solid' : 'outline')}></li>
    <li className={'star ' + (rating > 3 ? 'solid' : 'outline')}></li>
    <li className={'star ' + (rating > 4 ? 'solid' : 'outline')}></li>
</ul>
```

구문은 약간 다르다. DOM의 속성에 값을 사용한다는 아이디어는 동일하다. 위 템플릿에서, DOM 속성인 `className`이 컴포넌트의 속성값인 rating에 의존하고 있다고 볼 수 있다. 따라서 rating이 변할때 마다, 해당 expression등은 다시 계산된다. 만약 변화가 감지되면, className 속성 값이 바뀌는 것이다.

> click 이벤트 리스너는 앵귤러와 리액트에서 change detection의 일부가 아니다. 이들은 chagne detection을 트리거 할 지언정, 이러한 과정에서 포함되어 있지는 않다.

## Change Detection구현

DOM 요소의 속성에 값을 준다는 것은 기본적으로 두 프레임워크 모두 동일하지만, 기본적인 메커니즘에서 차이가 있다.

### Angular

컴파일러가 템플릿을 분석하면, DOM 요소와 관련된 property를 식별한다. 여기서 연결된 구성마다, 컴파일러는 일종의 명령의 형태로 바인딩을 만든다. 바인딩은 앵귤러에서 변화를 감지하는 핵심요소다. 컴포넌트의 속성과 DOM 요소 속성 사이에 연관관계를 정의한다.

이렇게 바인딩이 만들어지면, 앵귤러는 더이상 템플릿과 함께 동작하지 않는다. 변경 감지 메커니즘은 바인딩을 처리하는 명령을 실행한다. 이러한 작업은 속성이 있는 표현식의 값이 변경되었는지 확인하고, 필요한 경우 DOM 업데이트를 수행한다.

본 예제에서는, `rating` 속성이 템플릿의 `className`에 바인딩된다.

`[className]="'star ' + ((ctx.rating > 0) ? 'solid' : 'outline')"`

템플릿의 이부분에 대해 컴파일러는 바인딩을 설정하고, 더티 체크를 수행하며 필요한 경우 DOM을 업데이트 한다.

```javascript
if (initialization) {
    elementStart(0, 'ul');
        ...
        elementStart(1, 'li', ...);

        // sets up the binding to the className property
        elementStyling();
        elementEnd();
        ...
    elementEnd();
}

if (changeDetection) {

    // checks if the value of the expression has changed
    // if so, marks the binding as dirty and update the value
    elementStylingMap(1, ('star ' + ((ctx.rating > 0) ? 'solid' : 'outline')));
    elementStylingApply(1);
    ...
}
```

> 위 코드는 `Ivy`라고 알려진 새로운 컴파일러의 결과물이다. 이전 버전의 앵귤러는 바인딩과 더티체크에 대해 같은 아이디어를 사용하긴 했지만, 구현이 약간 다르다.

예를 들어 앵귤러가 `className`을 위한 바인딩을 만들었고, 현재 그 값은 대충 이럴 것이다.

`{ dirty: false, value: 'outline' }`

별점이 변화하게 되면, 앵귤러는 변화감지를 실행하게 될 것이다. 먼저 계싼된 값의 결과를 받아서 바인딩에 의해 가지고 있는 이전 값과 비교한다. 여기에서 `dirty check`라는 말이 유래되었다. 값이 변경되었다면, 현재 값을 업데이트 하고 이 바인딩을 dirty로 표시한다.

`{ dirty: true, value: 'solid' }`

그 다음엔 바인딩 된 값이 dirty인지 확인된다음에, 만약 dirty라면 (true라면) 새로운 값으로 DOM을 업데이트 한다. 우리의 예제에서는 `className` 프로퍼티가 업데이트 될 것이다.

더티체크를 수행하고, DOM의 관련된 부분을 업데이트 하는 바인딩을 처리하는 것이 앵귤러의 Change Detection의 핵심 작업이다.

### React

앞서 얘기했던 것 처럼, 리액트는 전혀 다른 접근 법을 사용한다. 리액트는 바인딩을 사용하지 않는다. 리액트에서 가장 중요한 변경 감지 매커니짐은 가상 DOM 비교다.

모든 리액트의 컴포넌트들은 JSX 템플릿을 반환하는 렌더링을 구현한다.

```javascript
export class RatingComponent extends ReactComponent {
    ...
    render() {
        return (
            <ul className="rating" onClick={handleClick}>
                <li className={'star ' + (rating > 0 ? 'solid' : 'outline')}></li>
                ...
            </ul>
        )
    }
}
```

리액트에서, 템플릿은 `React.createElement` 함수를 호출해서 컴파일 된다.

```javascript
const el = React.createElement;

export class RatingComponent extends ReactComponent {
    ...
    render() {
        return el('ul', { className: 'ratings', onclick: handleClick}, [
                 el('li', { className: 'star ' + (rating > 0 ? 'solid' : 'outline') }),
                    ...
        ]);
    }
}
```

`React.createElement` 함수가 호출될 때마다, 가상 DOM 노드라고 하는 데이터 구조를 생성하게 된다. 전혀 새로울 것이 없는, HTML 요소를 표현하는 일반적인 자바스크립트 오브젝트다. 이게 여러번 호출되면, 가상 돔 트리를 만들게 된다. 결과적으로, render 메소드는 가상 돔트리를 만들어 낸다.

```javascript
export class RatingComponent extends ReactComponent {
   ...
   render() {
       return {
           tagName: 'UL',
           properties: {className: 'ratings'},
           children: [
               {tagName: 'LI', properties: {className: 'outline'}},
               ...
           ]
       }
   }
}
```

render 함수가 호출될 때마다, 컴포넌트에 있는 프로퍼티를 보게 된다. 가상 돔 노드 속성은 이러한 계산된 값을 포함하게 된다. 우리 예제에서 별점이 0이라고 가정해보자. 이는 아래와 같은 표현식을 갖게 된다.

`{ className: rating > 0 ? 'solid' : 'outline' }`

![](https://admin.indepth.dev/content/images/2019/11/image-12.png)

이 값은 가상 돔의 `className`에 사용되는 값이다. 이 가상 돔 트리에 기반하여, 리액트는 class 값을 포함하는 리스트 아이템을 생성하게 된다.

만약 값이 0에서 1로 바뀌었다면

`{ className: rating > 0 ? 'solid' : 'outline' }`

이 값은 이제 solid가 될 것이다. 리액트가 변화감지를 수행하면, render 함수를 호출하여 새로운 버전의 가상 돔 트리를 만들어 낸다. `className` 의 속성은 이제 `solid`값으로 변경되었다. **각 change detection이 호출될 때 마다 render function이 호출된다는 것은 굉장히 중요한 사실이다.** 이 말은, 함수가 호출될때마다, 완전히 다른 가상 돔트리를 리턴한다는 것이다.

![](https://admin.indepth.dev/content/images/2019/11/image-13.png)

이렇게 만들어진 두 가상 DOM에서 비교 알고리즘을 실행하여, 두 가상 DOM 사이의 변경사항 집합을 얻는다. 우리의 경우에는 `className`의 차이 일 것이다. 차이점이 발견되면, 알고리즘은 해당 DOM 노드를 수정하는 패치를 생성한다. 이경우 패치는 `className`속성을 새 가상 돔에서 solid라는 값으로 변경할 것이다. 그리고 업데이트 된 버전에서 가상 돔은 다음 변경 감지 주기 동안 비교 대상으로 사용될 것이다.

컴포넌트에서 새로운 가상 DOM 트리를 가져와 이전 버전의 트리와 비교하고, DOM의 관련된 부분을 업데이트 하기 위한 패치를 생성하고 업데이트를 수행하는 것이 리액트 Change Detection의 핵심 요소다.

## 언제 Change Detection이 실행되는가?

Change Detection에 대한 이해를 하기 위해서는, React의 렌더 함수 또는 Angular의 측정이 언제 호출되는지 알아야 한다.

생각해보면, 변화를 감지하는 방법엔 두가지가 있다. 먼저 프레임워크에 변화가 있거나 혹은 변화가 있을 수 있는 것들을 알려서, Change Detection을 실행해야 한다는 것을 알리는 것이다.

### React

리액트에서는 change detection을 수동으로 해야 한다. 그것은 바로 `setState`다.

```javascript
export class RatingComponent extends React.Component {
    ...
    handleClick(event) {
        this.setState({rating: Number(event.target.dataset.value)})
    };
}
```

리액트에서는 이를 자동으로 하는 방법이 없다. 모든 변경감지 사이클은 `setState`로 부터 시작된다.

### Angular

앵귤러에서는 두가지 옵션이 다 있다. changeDetector 서비스를 활용해서 수동으로 트리거할 수도 있다.

```javascript
class RatingWidget {
  constructor(changeDetector) {
    this.cd = changeDetector
  }

  handleClick(event) {
    this.rating = Number(event.target.dataset.value)
    this.cd.detectChanges()
  }
}
```

그러나, 프레임워크에서 자동으로 Chnage Detection을 하게 할 수 있다. 여기에서는, 단순히 property를 업데이트 해야 한다.

```javascript
class RatingWidget {
  handleClick(event) {
    this.rating = Number(event.target.dataset.value)
  }
}
```

하지만 앵귤러에서는 어떻게 change detection을 실행해야 한다는 것을 알까?

앵귤러가 제공하는 메커니즘을 활용하여, 템플릿의 UI 이벤트에 바인딩 하기 때문에 모든 UI 이벤트 리스너에 대해 알수 있다.이 이벤트 리스너를 가로챈다는 것은, 애플리케이션 코드 실행이 끝난후 변경 탐지 실행을 스케줄링할 수 있다는 것을 의미한다. 이것은 기발한 아이디어지만, 이 메커니즘으로 모든 비동기 이벤트를 가로챌수는 없다.

`setTimout`이나 `XHR` 과 같은 이벤트에 앵귤러 매커니즘을 바인딩 할 수 없으므로, Change Detection이 자동으로 이루어질 수 없다. 이러한 문제를 해결하기 위해 zone.js라는 라이브러리를 사용한다. 브라우저의 모든 비동기 이벤트를 패치한다음, 특정 이벤트가 발생할때 앵귤러에 알릴 수 있다. UI 이벤트와 마찬가지로, 앵귤러는 애플리케이션 의 실행이 완료될 때 까지 기다렸다가 자동으로 변경을 탐지할 수 있다.
