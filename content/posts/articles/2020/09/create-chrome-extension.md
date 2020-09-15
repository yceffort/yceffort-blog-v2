---
title: ECMAScript 명세 읽어보기 (1)
tags:
  - chrome
published: true
date: 2020-09-25 21:56:02
description: '필요한 기능 하나 쯤 만들어서 사용해보자.'
category: chrome
template: post
---

크롬 익스텐션은 단순히 js, html, css로 이루어져있기 때문에, 몇가지 API를 추가한다면 크롬에서 사용할 수 있는 익스텐션을 만들 수 있다.

## Manifest.json 만들기

크롬 익스텐션용 `package.json`이라고 생각하면 좋을 것 같다.

```json
{
  "manifest_version": 2,
  "name": "Demo Extension",
  "version": "1.0.0",
  "description": "Sample description",
  "short_name": "Short Name",
  "permissions": ["activeTab", "declarativeContent", "storage", "<all_urls>"],
  "background": {
    "scripts": ["background.js"]
  },
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "css": ["background.css"],
      "js": ["contentscript.js"]
    }
  ],
  "browser_action": {
    "default_title": "Does a thing when you do a thing",
    "default_popup": "popup.html",
    "default_icon": {
      "16": "icons/icon16.png",
      "32": "icons/icon32.png"
    }
  },
  "icons": {
    "16": "icons/icon16.png",
    "32": "icons/icon32.png",
    "48": "icons/icon48.png",
    "128": "icons/icon128.png"
  }
}
```

여기에서 몇가지 대표적인 내용을 살펴보자.

- `manifest_version`: 그냥 2라고 생각하면 된다. 1 버전은 크롬 18 이후로 부터는 deprecated 되었다.
- `name`: 이름이다.
- `description`: 설명이다.
- `version`: 익스텐션의 버전이다.
- `short_name`: optional 필드
- `permission`: 어떤 권한이 필요한지를 나타낸다. 획득 가능한 권한들은 [여기](https://developer.chrome.com/extensions/declare_permissions)에 나와 있다. 대표적인 권한 들은 아래와 같다. 예를 들어 `content_scripts`로 현재 웹 페이지의 DOM을 읽어 올 수 있다.

`browser_action`을 사용하여 주소 바 옆에 작은 아이콘을 만들 수 있고, 이를 클릭 했을 떄 html이 나오게 할 수 있다.

```json
"browser_action": {
   "default_title": "Does a thing when you do a thing",
   "default_popup": "popup.html",
   "default_icon": {
     "16": "icons/icon16.png",
     "32": "icons/icon32.png"
   }
 },
```

앞서 `content_scripts`로 현재 DOM을 읽어올 수 있다고 했지만, 여기에서 사용할 수 있는 api는 한정적이다. 따라서 다양한 api를 사용하기 위해서는 `background`를 추가하여 작업해야 한다. `content_script`로 DOM을 읽어오고, 이를 `background.js`로 보내서 필요한 API 작업을 한다고 보면 된다.

## 흐름

`content_script`에서는 DOM과 관련된 처리를 하고, 그와 관련된 비즈니스 로직은 `background`에서 실행하면 된다.

`conten_script.js`

```javascript
document.ondblclick = function () {
  // 블록처리된 문자
  const selectedMessage = window.getSelection().toString()

  // 메시지를 보낸다.
  chrome.runtime.sendMessage({ word: selectedMessage }, async function (
    response,
  ) {
    console.log(response)
  })
}
```

`background.js`

```javascript
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  const { word } = request

  // do something..
  // async 처리를 잘못하는 것 같다.
  // promise. then()으로 처리하고,
  // 마지막에 꼭 return true를 해줘야 한다.

  return true
})
```
