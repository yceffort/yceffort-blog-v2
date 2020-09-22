---
title: Referer와 Refeerer-Policy를 위한 가이드
tags:
  - javascript, webpack
published: true
date: 2020-09-16 17:42:06
description: '웹 어플리케이션에서 request를 받기 위한 최적의 Referer와 Referrer 정책'
category: javascript
template: post
---

```toc
tight: true,
from-heading: 2
to-heading: 3
```

## same-site와 same-origin은 다르다.

`Origin`

https://yceffort.kr:443

- `origin`: 은 `scheme` (`protocol`로도 알려진)와 `host name`, 그리고 `port`의 조합을 의미한다. 예를 들어 https://yceffort.kr:443
- `scheme`: `https://`
- `host name`: `yceffort.kr`
- `port`: 443
