---
title: 'HTML과 CSS를 활용해서 콘텐츠를 숨기는 10가지 방법'
tags:
  - html
  - css
  - browser
published: true
date: 2021-03-05 21:14:45
description: '원래 알던건 두 세가지밖에 안됨'
---

## 요약

|                              	| Visible   	| Accessible 	|
|------------------------------	|-----------	|------------	|
| `.sr-only`                   	| X         	| O          	|
| `aria-hidden="true"`         	| O         	| X          	|
| `hidden=""`                  	| X         	| X          	|
| `display: none`              	| X         	| X          	|
| `visibility: hidden`         	| X (Space) 	| X          	|
| `opacity: 0`                 	| X (Space) 	| Depends?   	|
| `clip-path: circle(0)`       	| X (Space) 	| Depends?   	|
| `transform: scale(0)`        	| X (Space) 	| O          	|
| `width: 0`+`height: 0`       	| X         	| X          	|
| `content-visibility: hidden` 	| X         	| X          	|

## `.sr-only`

`.sr-only`는 페이지에서는 가리지만, 스크린 리더에서는 접근 가능하도록 하는 일종의 CSS 선언이다. [bootstrap에 이와 관련된 코드가 있다](https://getbootstrap.com/docs/4.0/utilities/screenreaders/)

```css
.sr-only {
  border: 0 !important;
  clip: rect(1px, 1px, 1px, 1px) !important;
  -webkit-clip-path: inset(50%) !important;
  clip-path: inset(50%) !important;
  height: 1px !important;
  overflow: hidden !important;
  padding: 0 !important;
  position: absolute !important;
  width: 1px !important;
  white-space: nowrap !important;
}
```

이 방법은 텍스트를 마스킹하는데만 사용해야 한다. 다시 말해, 숨겨진 요소안에 focus 가능한 엘리먼트가 있어서는 안된다. 그렇지 않으면, 보이지 않는 엘리먼트가 스크롤이 되는 등의 귀찮은 일이 있을 수 있다.

## `aria-hidden` 속성

[aria-hidden](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Techniques/Using_the_aria-hidden_attribute)이 `true`로 설정되면, accessibility tree에서 해당 콘텐츠를 가리지만, 여전히 시각적으로는 볼수 있다. `aria-hidden="true"` 엘리먼트에 기본 스타일을 적용하는 브라우저는 없다.

주의 할 점은 , `aria-hidden="true"`가 focusable한 요소가 되어서는 안된다는 것이다. 스크린리더에는 보이지 않지만, focus가 되기 때문이다.

```html
<!-- 안됨 -->
<button aria-hidden="true">press me</button>
```

## `display: none`과 `hidden`속성

이 두가지 방법은 모두 렌더링트리와 접근성 트리에서 사라지게 한다. `hidden`은 별도의 CSS가 없이도 HTML을 통해서 완전히 마스킹할 수 있어서 편리한 점이 있다.

## `visibility :hidden` 

이 css 속성은 레이아웃에 영향을 미치지 않고 엘리먼트를 감춘다. 사라진 공간은 빈 공간으로 남으며, 리플로우가 일어나지 않는다. 접근성 관점에서 보았을 때는 위의 `display:none`과 동일하다.

## `opacity:0`, `clip-path: circle(0)`

이 두 속성은 엘리먼트의 요소를 안보이게 하지만, `visibility: hidden`과 마찬가지로 빈공간이 남아 있게 된다. 이 콘텐츠에 접근할 수 있는지 여부는 접근성 기술에 따라 다르다. 따라서 일관되게 숨시려면 이것을 사용하지 않는 것이 좋다.

## `transform: scale(0)`

시각적으로 엘리먼트를 감추지만, 위 두속성과 마찬가지로 빈공간이 남는다. 그러나 스크린 리더에서는 이 요소에 접근할 수가 있다.

## `width: 0`, `height: 0`

특정 요소의 너비와 높이를 0으로 설정하여 숨기는 방식으로, 스크린리더 또한 이 엘리먼트가 접근 가능하지 않은 것으로 간주하고 건너뛴다. 그러나 이 기술은 일종의 낚시(?ㅋㅋ)같은 수상한 기술이고, SEO 차원에서도 좋지 못하다.

## `content-visibility: hidden`

[이 속성](https://developer.mozilla.org/en-US/docs/Web/CSS/content-visibility)은 크롬브라우저에서 특정 요소의 렌더링을 뷰포트내에 있기전까지는 가리는 방법으로 도입되었다. 이 속성은 `display: none`과 마찬가지로 접근성 트리에서 사라진다. 접근성차원에서 봤을 때는 그다지 좋은 기술은 아니므로 쓰지 않는 것이 좋다. 

- https://web.dev/content-visibility/
- https://wit.nts-corp.com/2020/09/11/6223

## 요약

일반적으로 말하자면, 시각적인 내용과 접근성으로 노출되는 내용간에 너무 많은 불일치가 존재해서는 안된다. 둘 모두에게 잘 동기화된 내용을 보여주어야 한다.

- 만약 화면과 접근성 모두에서 가리고 싶다면, `display:none` `hidden`을 사용하는 것이 좋다. (위젯을 토글하거나, 다이얼러그를 닫는 등)
- 접근성에서는 감추지만 시각적으로 보이고 싶을 경우에는, `aria-hidden="true"`를 사용하자. (아이콘과 같은 경우)
- 화면에서는 가리지만 접근성에서는 보이고 싶을 경우, `.sr-only`를 쓰자. (링크나 아이콘 버튼과 같이 접근성요소로 정보를 제공하고 싶은 경우)