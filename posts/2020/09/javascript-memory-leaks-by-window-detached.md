---
title: detached window로 인한 자바스크립트 메모리 누수
tags:
  - javascript
published: true
date: 2020-09-29 23:43:18
description: '면접에서 들었던 거지같은 질문에 대한 해답'
category: javascript
template: post
---

## Table of Contents

## 자바스크립트 메모리 누수란 무엇인가

일반적으로 메모리 누수라함은 애플리케이션을 실행하는 과정에서 발생하는 의도치 않게 메모리 사용량이 증가 하는 것을 의미한다. 자바스크립트에서는 일반적으로 더 이상 필요하지 않은 객체를 다른 객체나 함수에서 참조하고 있을 때 발생한다. 이러한 참조는 더 이상 필요없는 객체가 가비지 콜렉터에 의해 회수되는 것을 저지한다.

> 자바스크립트 메모리 누수에 대해서 예전에 한번 글을 올린 적이 있습니다용 https://yceffort.kr/2020/07/memory-leaks-in-javascript/

가비지 컬렉터의 역할은 애플리케이션에서 더 이상 사용하지 않거나 사용될 수 없는 객체를 알아내고 처리하는 것이다. 이는 객체가 자기 자신을 참조하거나, 순환참조로 서로를 참조할 때에도 정상적으로 작동한다. 애플리케이션에서 객체에 접근할 수 있는 참조가 더 이상 존재 하지 않는다면, 이는 가비지 컬렉팅 될 것이다.

```javascript
let A = {}
console.log(A) // 지역변수를 참조

let B = { A } // B.A로 A를 참조

A = null // 참조를 해제함.

console.log(B.A) // A는 여전히 B에서 참조되고 있음

B.A = null // B에서 A참조를 제거

// A를 참조하는 것이 더 이상 없다. 따라서 메모리에서 해제 된다.
```

애플리케이션이 DOM 요소나 팝업 창과 같이 자체 라이프 사이클이 있는 객체를 참조하게 되면 굉장히 까다로운 메모리 누수가 발생할 수 있다. 이러한 유형의 객체들은 애플리케이션이 알지 못하는 사이에 사용되고 있을 수 있다. 즉, 애플리케이션 코드는 가비지를 수집할 수 있는 객체에 대한 유일한 참조를 가질 수 있다. (유일한 참조를 가지고 있어서 가비지 컬렉팅을 방해할 수 있다.)

## 분리된 윈도우 (detached window)란 무엇인가?

아래 예제는, 슬라이드 쇼 뷰어 애플리케이션에는 발표자 노트 팝업을 열고 닫기 위한 버튼이 포함되어 있다. 만약 사용자가 `표시`를 누른후 `숨기기`를 누르지 않고 팝업창을 직접 닫았다고 가정해보자. `notesWindow` 변수는 여전히 팝업이 닫혀서 더 이상 사용할 수 없음애도 불구하고 팝업에 대한 접근 가능한 참조를 가지고 있을 것이다.

```html
<button id="show">Show Notes</button>
<button id="hide">Hide Notes</button>
<script type="module">
  let notesWindow
  document.getElementById('show').onclick = () => {
    notesWindow = window.open('/presenter-notes.html')
  }
  document.getElementById('hide').onclick = () => {
    if (notesWindow) notesWindow.close()
  }
</script>
```

이러한 예제를 분리된 윈도우 (detached window)라고 할 수 있다. 팝업 윈도우는 닫혔지만, 코드 상에서는 참조를 가지고 있어 브라우저가 해당 변수에 대한 메모리를 회수하고 있지 못하는 모습이다.

`window.open()`을 이용해 브라우저나 탭이 열렸을 경우, 열린 윈도우나 탭에 대한 [Window](https://developer.mozilla.org/en-US/docs/Web/API/Window) 객체를 리턴하게 된다. 이 윈도우가 닫혀버리더라도, `window.open`을 통해 리턴된 `Window`객체는 여전히 참조할 수 있는 정보를 가지고 있다. 이것이 분리된 윈도우 문제의 한 종류다. 자바스크립트 객체는 닫혀버린 `window` 객체에 대해 접근할수 있기 때문에, 이는 메모리에서 회수되지 않는다. 만약 이러한 팝업에 많은 양의 자바스크립트 코드나 iframe이 존재한다면, 해당 메모리는 해당 window 속성에 대한 자바스크립트 참조가 남아 있지 않을 때까지 회수할 수 없는 상태로 남아 있게 된다.

```javascript
// 창을 열고
let win = window.open(
  '/heavy.html',
  '',
  'left=200,top=200,width=200,height=200',
)
// 무언가 무거운 내용을 넣는다.
win.document.body.innerHTML = 'Heavy HTML'
// 창을 닫아버리지만
win.closed
// 여전히 팝업에 대한 정보가 담겨져 있다.
win.document.body.innerHTML
```

https://storage.googleapis.com/web-dev-assets/detached-window-memory-leaks/example-detached-window.webm

이러한 문제는 `<iframe>`에서도 동일하게 나타난다. `iframe`은 기본적으로 document 안에 window를 가지고 있는 것처럼 동작하고 있으며, `contentWindow`속성은 `Window`객체에 대한 접근을 가능하게 한다. 그리고 이 속성은, `window.open()`과 같이 동작한다. 마찬가지로 자바스크립트 코드는 iframe 의 `contentWindow`나 `contentDocument`에 대한 참조를 가질 수 있으므로, 주소나 DOM에 의해 iframe이 사라지더라도, 여전히 참조가 남아 있게되므로 메모리를 회수할 수 없게 된다.

https://storage.googleapis.com/web-dev-assets/detached-window-memory-leaks/example-detached-iframe.webm

자바스크립트에서 window 또는 iframe에 대한 참조가 유지되는 경우, 이 window나 iframe이 새로운 URL로 이동하도라도, 해당 document는 메모리에 저장된다. 이는 특히 자바스크립트가 document를 메모리에 보관하는 마지막 참조가 되는 시기를 모른다면, 해당 참조를 보관하는 자바스크립트가 window/iframe 이 새로운 URL로 이동한 것을 감지하지 못할 때 문제가 될 수 있다.

## 어떻게 분리된 윈도우가 메모리 누수를 유발하는가

기본 페이지와 동일한 도메인에서 window 및 iframe으로 작업을 할 때, document의 경계를 넘어서 이벤트 리스너를 추가하거나 속성을 엑세스 하는 것이 일반적이다. 예를들어, 앞서서 예시로 들었던 프레젠테이션 뷰어를 살펴보자. 뷰어에서 스피커 노트를 표시할 수 있는 두번째 창을 열고, 스피커느 다음 슬라이드로 이동하기 위한 신호로 클릭 이벤트를 받는다고 가정해보자. 사용자가 이 노트 창을 닫아도, 원래 상위 창에서 실행되는 자바스크립트는, 여전히 스피커 노트 문서 전체에 대한 엑세스 권한을 갖게 된다.

```html
<button id="notes">Show Presenter Notes</button>
<script type="module">
  let notesWindow
  function showNotes() {
    notesWindow = window.open('/presenter-notes.html')
    notesWindow.document.addEventListener('click', nextSlide)
  }
  document.getElementById('notes').onclick = showNotes

  let slide = 1
  function nextSlide() {
    slide += 1
    notesWindow.document.title = `Slide  ${slide}`
  }
  document.body.onclick = nextSlide
</script>
```

`showNotes()` 함수로 생성된 브라우저의 윈도우를 닫았다고 가정해보자. `window` 가 닫혔는지를 판단하는 이벤트 리스너는 존재하지 않으므로, 코드에 해당 참조를 제거해야하는지 여부를 알려줄 수 있는 방법이 없다. 따라서 `nextSlide()`함수는 여전히 메인 페이지의 클릭핸들러로써 존재하게 되며, `nextSlide`가 가지고 있는 `notesWindow`는 메모리 회수 대상에서 제외되어 버린다.

https://storage.googleapis.com/web-dev-assets/detached-window-memory-leaks/animation.webm

이 외에도 분리된 윈도우 문제로 인하여 메모리 회수를 막는 다양한 케이스가 존재할 수 있다.

- 이벤트 핸들러는 iframe이 의도된 URL로 미처 이동하기 전에, iframe에 등록 할 수 있으므로, 다른 참조가 정리된 후에도 document 및 iframe에 대하여 우발적인 참조가 지속되는 경우
- 많은 양의 메모리를 차지하는 iframe 또는 window는 새로운 URL로 이동한 후에도 오래동안 우연히 메모리에 저장되어 있을 수 있다. 이는 종종 리스너 제거를 허용하기 위해 문서에 대한 참조를 유지하는 부모 페이지에 의해 발생된다.
- 자바스크립트 객체를 다른 창 또는 iframe에 넘길 때, 객체 프로토타입체인에는 window을 포함하여 생성한 환경에 대한 참조가 포함된다. 이는 window 객체 에 대한 참조를 피하는 것 만큼, 다른 window에서 객체에 대한 참조를 피하는 것이 중요하다는 것을 의미한다.

`index.html`

```html
<script>
  let currentFiles
  function load(files) {
    currentFiles = files
  }
  window.open('upload.html')
</script>
```

`upload.html`

```html
<input type="file" id="file" />
<script>
  file.onchange = () => {
    parent.load(file.files)
  }
</script>
```

## 분리된 윈도우로 인한 메모리 누수를 감지하는 법

메모리 누수를 찾는 과정은 어렵다. 메모리가 누수되는 과정을 재현하는 것은 어렵고, 특히 많은 document 와 window가 얽혀있으면 더욱 어렵다. 더 골때리는 것은, 잠재적인 메모리 누수를 유발하는 참조를 조사하다가 또 다른 메모리 누수를 유발하는 객체를 만드는 것이다.

메모리 문제를 디버깅하기 위한 최적의 방법은 [heap sanpshot을 찍는 것이다](https://developers.google.com/web/tools/chrome-devtools/memory-problems#discover_detached_dom_tree_memory_leaks_with_heap_snapshots). 이는 현재 애플리케이션에서 사용되는 메모리, 생성되었지만 아직 수집되지 않은 모든 객체에 대한 시점별 뷰를 제공한다. heap snapshot에는 객체의 크기, 변수 목록, 객채를 참조하는 클로져등 객체에 대한 유용한 정보가 포함되어 있다.

![heap snap shot](https://webdev.imgix.net/detached-window-memory-leaks/heap-snapshot.png)

heap snap shot 녹화를 하기 위해서는, 크롬 개발자도구에서 메모리 탭을 누르고, 가능한 프로파일링 타입들 중에서 heap snapshot을 클릭한다. 녹화가 끝나면, 요약에서 현재 메모리에 있는 객체들이 그룹핑 되어 보일 것이다.

https://storage.googleapis.com/web-dev-assets/detached-window-memory-leaks/take-heap-snapshot.webm

힙 덤프를 분석하는 것은 굉장히 어려운 작업이기에, 디버깅을 하기 위한 적절한 정보를 찾는 것이 꽤 어려운 작업이 될 수 있다. 이를 위해 [Heap Cleaner](https://github.com/ykahlon/heap-cleaner)를 설치해서 개발하는 것이 도움이 될 수 있다. 이를 사용하면 그래프에서 다른 불필요한 정보가 제거되므로 보다 쾌적하게 추적할 수 있다.

### 코드로 메모리 계산하기

힙 스냅샷은 고수준의 세부정보를 제공하며, 누출 발생 지점을 파악하는데 탁월하다. 그러나 힙 스냅샷을 만들어 보는 것은 수동적인 절차를 거쳐야 한다. 이를 확인할 수 있는 다른 방법이 [performance.memory API](https://developer.mozilla.org/en-US/docs/Web/API/Performance/memory)를 이용하는 것이다.

![performance memory api](https://webdev.imgix.net/detached-window-memory-leaks/performance-memory.png)

이 `performance.memory` API는 오직 자바스크립트 힙사이즈의 정보만 제공하므로, 팝업의 document나 리소스에 대한 메모리는 포함되어 있지않다. 따라서 정확한 그림을 보기 위해서는 [performance.measureMemory API](https://web.dev/monitor-total-page-memory-usage/)를 봐야 한다.

## 분리된 윈도우의 메모리 누수 해결

가장 일반적인 두가 지 경우

- 상위 문서가 닫힌 팝업 또는 제거된 iframe 에 대한 참조를 가지고 있을때
- window나 iframe의 예상치 못한 네비게이션으로 인해 이벤트 핸들러가 등록되지 않는 경우

에 대해 알아보도록 하자.

### 예제1) 팝업 닫기

```html
<button id="open">Open Popup</button>
<button id="close">Close Popup</button>
<script>
  let popup
  open.onclick = () => {
    popup = window.open('/login.html')
  }
  close.onclick = () => {
    popup.close()
  }
</script>
```

코드를 얼핏 보면, 일반적인 함정을 피하는 것처럼 보인다. 팝업에 대한 참조는 유지 않고, 팝업 창에 이벤트 핸들러를 등록하고 있지 않다. 그러나 팝업 열기 버튼을 클릭하면 팝업 변수가 열어 버린 창을 참조하며, 해당 변수는 팝업 닫기 버튼 클릭 핸들러 내에서 액세스가 가능하다. `popup`이 재 할당 되거나, 클릭 핸들러가 제거 되지 않는 한, 해당 핸들러에 들어가 있는 참조는 메모리 수집이 되지 않는 다는 것을 의미한다.

#### 해결책: 참조 해제하기

가장 간단한 해결책은, 닫히는 순간 `popup` 변수를 재할당하여 해제하는 것이다.

```javascript
let popup
open.onclick = () => {
  popup = window.open('/login.html')
}
close.onclick = () => {
  popup.close()
  popup = null
}
```

당장, 이는 도움이 되는 것 처럼 보이지만, 만약 사용자가 닫기 버튼 대신 윈도우에 있는 X버튼을 눌러 닫아버리면 어떻게 되는가? 열어놓은 창에서 다른 웹사이트를 탐색한다면? 정리하자면, 닫기 버튼 이외에 다른 행동을 사용가자 취할 경우 여전히 메모리 누수 가능성이 존재한다.

#### 해결책: 닫히는지 확인하고 해제하기

많은 상황에서 창문을 열거나 프레임을 만드는 일을 담당하는 자바스크립트는 그들의 라이프사이클에 대한 독점적인 통제권을 가지고 있지 않다. 사용자가 팝업을 닫거나, 새로운 문서로 이동하면 이전에 창이나 프레임에 의해 포함된 문서가 분리될 수 있다. 두 경우 모두 브라우저는 `pageHide` 이벤트를 실행하여 문서가 언로드되고 있음을 알린다.

> 주의:`pageHide` 대신 [unload](https://developers.google.com/web/updates/2018/07/page-lifecycle-api#the-unload-event)도 비슷한 일을 할 것 같이 생겼지만, 이는 레거시 API 이므로 사용하면 안된다.

`pageHide` 이벤트는 윈도우가 닫히거나, 현재 document 에서 다른 페이지로 넘어가는 이벤트를 감지할 수 있다. 그러나 한가지 주의 할 것이 있다. 새로 생성된 모든 window와 iframe은 빈문서를 포함하고 있으며, URL이 제공되는 경우 비동기형태로 이동한다. 따라서 대상 문서가 로드되기 직전이나, window나 프레임을 작성한 직후에 `pageHide`이벤트가 실행된다. 대상 document가 언로드 될 때 참조 정리 코드가 실행되어야 하기 때문에, 우리는 이 첫 `pageHide` 이벤트를 무시하는 코드를 넣어야 한다. 이를 위해 여러가지 트릭들이 존재하지만, 가장 간단한 방법은 바로 이것이다.

```javascript
let popup
open.onclick = () => {
  popup = window.open('/login.html')

  // listen for the popup being closed/exited:
  popup.addEventListener('pagehide', () => {
    // ignore initial event fired on "about:blank":
    if (!popup.location.host) return

    // remove our reference to the popup window:
    popup = null
  })
}
```

여기서 한가지 또 기억해야 할 것은, 이러한 방식은 window나 frame이 동일한 origin에서 실행되어야 한다는 것이다. 만약 다른 origin에서 열릴 경우, `location.host`와 `pageHide` 이벤트는 보안상의 이슈로 인해 실행되지 않는다. 이를 위해 `window.closed`나 `frame.isConnected`속성을 모니터링 해야 한다. 각각의 속성은 창이 닫히거나 iframe이 제거되는 경우 모두를 확인할 수 있다.

```javascript
let popup = window.open('https://example.com')
let timer = setInterval(() => {
  if (popup.closed) {
    popup = null
    clearInterval(timer)
  }
}, 1000)
```

#### 해결책: WeakRef 사용하기

> [WeakRef](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/WeakRef)는 자바스크립트의 새로운 기능으로, 아직 지원한느 브라우저가 많지 않다. (크롬, 파이어폭스) 이 방법은 문제를 해결하는 방법이라기 보다, 문제를 디버깅 하는 방법에 더 가깝다.

자바스크립트에서 참조 객체를 가비지 콜렉팅 가능하게 하기 위한 방법으로 `WeakRef`를 새롭게 지원하고 있다. `WeakRef`는 객체를 직접적으로 참조하는 것이 아니라, `.deref()`라고 불리우는, 가비지 수집되지 않는 한 객체에 대한 참조를 반환하는 새로운 방법을 제공한다. `WeakRef`를 사용하면 창이나 문서의 현재 값에 액세스 하면서도 동시에 가비지 콜렉팅이 가능하다. `pageHide`, `window.closed`와 같은 속성에 대응하여 수동으로 해제해야 하는 창에 대한 참조를 유지하는 대신, 필요에 따라 창에 대한 액세스를 얻는다. 창이 닫히면 메모리 수집이 가능해져 `.deref()`메소드가 `undefined`를 리턴한다.

```html
<button id="open">Open Popup</button>
<button id="close">Close Popup</button>
<script>
  let popup
  open.onclick = () => {
    popup = new WeakRef(window.open('/login.html'))
  }
  close.onclick = () => {
    const win = popup.deref()
    if (win) win.close()
  }
</script>
```

한가지 흥미로운 것은, 일반적으로 창이 닫히거나 `iframe`이 제거 된 후 짧은 시간동안 참조를 사용할 수 있다는 것이다. 이는 `WeakRef`가 관련된 객체를 가비지 콜렉팅하기 전까지 값을 계속 반환하기 때문인데, 이는 자바스크립트에서 메모리 수집이 유휴 시간 동안 비동기적으로 일어나기 때문이다. 크롬 개발자 도구에서 메모리 탭을 확인하면, 실제로 가비지 콜렉팅이 트리거 되고 약하게 참조된 window가 폐기되는 것을 볼 수 있다. 또한 `deref()`가 `undefined`를 반환하거나, [FinalizationRegistry API](https://v8.dev/features/weak-references#:~:text=FinalizationRegistry)를 활용하여 `WeakRef`를 통해 참조된 객체가 자바스크립트에서 삭제되엇는지 확인할 수 있다.

```javascript
let popup = new WeakRef(window.open('/login.html'))

// Polling deref():
let timer = setInterval(() => {
  if (popup.deref() === undefined) {
    console.log('popup was garbage-collected')
    clearInterval(timer)
  }
}, 20)

// FinalizationRegistry API:
let finalizers = new FinalizationRegistry(() => {
  console.log('popup was garbage-collected')
})
finalizers.register(popup.deref())
```

#### 해결책: postMessage로 통신하기

위에서의 해결책 보다 더 근본적인 방식에 대해 고민해볼 필요가 있다. 바로 두 페이지 간의 통신이다.

window와 문서 사이에서 구질구질하게 참조를 방지하는 것보다 좋은 해결책은 바로 문서간 통신을 [postMessage()](https://developer.mozilla.org/en-US/docs/Web/API/Window/postMessage)로 제한하는 것이다.

```javascript
let updateNotes
function showNotes() {
  // popup에 대한 참조를 클로져로 제한하여 밖에서 참조되는 것을 막는다.
  let win = window.open('/presenter-view.html')
  win.addEventListener('pagehide', () => {
    if (!win || !win.location.host) return // ignore initial "about:blank"
    win = null
  })
  // 다른 함수는 이 api를 통해서만 통신이 가능
  updateNotes = (data) => {
    if (!win) return
    win.postMessage(data, location.origin)
  }
  addEventListener('message', (event) => {
    if (event.source !== win) return
    if (event.data[0] === 'nextSlide') nextSlide()
  })
}

let slide = 1
function nextSlide() {
  slide += 1
  updateNotes(['setSlide', slide])
}
document.body.onclick = nextSlide
```

여전히 window간에 서로 참조는 필요하지만, 어느 것도 다른 창에서 현재 문서에 대한 참조를 유지 하지 않는다. 또한 메시지 전달 방식은 한 곳에서만 고정되도록 설계했는데 (`updateNotes`) 이는 창을 닫거나 탐색할 때 하나의 참조만 해제 하면 된다는 것을 의미한다. 위의 예제에서는, 오직 `showNotes`만 팝업창에 대한 참조를 유지하고, `pageHide`이벤트를 사용하여 참조가 정리되도록 한다.

#### 해결책: `noopener` 사용하기

팝업창이 열리긴 했지만, 페이지와 통신이 필요하지 않는 경우, 창에 대한 참조를 회피하고 싶을 수 있다. 이는 특히 기존 사이트와 완전히 다른 사이트를 여는 경우에 유용하다. 이러한 경우, `window.open()`에 [noopener option](https://developer.mozilla.org/en-US/docs/Web/API/Window/open#noopener)를 넣어서 마치 HTML링크의 [rel="noopener" 속성](https://web.dev/external-anchors-use-rel-noopener/) 과 동일하게 동작하게 만든다.

```javascript
window.open('https://example.com/share', null, 'noopener')
```

`noopener`를 사용하면 `window.open()`은 `null`를 리턴하여, 실수로 팝업창에 대한 참조를 방지할 수 있다. 팝업창 역시 `window.opener`가 `null` 이므로 부모창에 대한 참조를 막을 수 있다.
