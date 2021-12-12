---
title: 'nodejs의 메모리 제한'
tags:
  - javascript
  - nodejs
published: true
date: 2021-12-13 19:21:45
description: ''
---

## V8 가비지 콜렉션

힙은 메모리 할당이 필요한 곳이고, 이는 여러 `generational regions`로 나뉜다. 이 `region`