---
title: 블로그 개편했습니다. 😎
tags:
  - diary
  - react
published: true
date: 2019-12-30 09:16:10
description: 주말에 집구석에 혼자 오래있을 일이 있어서, 생각난 김에 블로그를 개편했습니다. 이 전에는 hexo 기반으로 만들어진
  블로그를 작업했는데, hexo 생태계가 관리가 잘 안되고 있는 건지 플러그인이나 기능들이 제대로 동작을 안하더군요. wordpress ->
  ??? -> github pages -> hexo -> gatsby 까지 벌써 개편만 한 다섯번 쯤...
category: diary
slug: /2019/12/30/blog-renewal/
template: post
---
주말에 집구석에 혼자 오래있을 일이 있어서, 생각난 김에 블로그를 개편했습니다. 이 전에는 hexo 기반으로 만들어진 블로그를 작업했는데, hexo 생태계가 관리가 잘 안되고 있는 건지 플러그인이나 기능들이 제대로 동작을 안하더군요.

wordpress -> ??? -> github pages -> hexo -> gatsby 까지 벌써 개편만 한 다섯번 쯤 한거 같네요. 이제 그만 하겠습니다.

정적 사이트 생성기로 괜찮은게 뭐가 있나 알아보던 차에, 회사에서도 Gatsby를 쓰고 있길래 gatsby로 변경해보았습니다.

## 변경과정의 난관

- front-matter를 관리하는게 묘하게 달라서 파이썬 스크립트로 통일 작업을 좀 해야 했다.
- 태그들에 svg 아이콘이 달려 있는게 이뻐서 테마를 선택했는데, 몇몇 아이콘들은 직접 svg를 편집해서 작업해야 했는데 이게 정말 귀찮았다. 그리고 이번 기회에 svg에 대해서 공부하게 되었는데, 몇몇 아이콘들은 간지 때문에 그 크기가 너무 크다. 향후에 최적화가 필요한 부분
- 기존 md 파일 몇개가 렌더링이 안되었는데, 이것도 따로 확인을 해봐야 했다. 그러나 리액트 기반이라 수정하는데 어렵지는 않았다.

## 장점

- 여러가지 다양한 플러그인들이 많고 관리도 잘되고 있다.
- 리액트로 쓰여져 있어서 스스로 관리가 용이하고, 확장성도 더 늘어났다.
- 이전 보다 블로그 디자인이 마음에 든다.

## 단점

- dev에서 파일하나만 수정하는데, 그 파일만 수정하는게 아니라 query 전체가 돌고 있는 느낌이다. 이건 내가 발적화를 때문일까

```shell
info changed file at /Users/jayg/work/private/yceffort-blog/content/blog/2019/12/30/renewal-blog.md
success createPages - 0.204s
success createPages - 0.159s
success write out requires - 0.008s
success run queries - 5.169s - 125/125 24.18/s
success run queries - 0.058s - 3/3 51.45/s
success run queries - 0.023s - 1/1 44.27/s
[==                          ]   0.724 s 9/88 10% run queries
```

- 빌드 타임이 늘어났고, 빌드 후 결과 물도 사이즈가 커졌다. (90mb -> 352mb) 이것도 내 발적화 때문인 것으로 추정해본다
- hot reloading 이 되었다 안되었다 한다. 이건 내가 뭘 설정을 잘못 건든걸까?

이제 왠만한 기능은 다 구현했고, 몇가지만 더 작업하면 된다.

## 앞으로 남은 과제

- about 페이지 작성
- 구 blog github archive 처리 및 README.md 작성
- aloglia 기반 검색 component 개발
- code highlighter prismjs 의 media query 처리 (현재 pc화면에서 코드가 길어지면 사이드바가 찌그러진다.)
- 폰트 수정
- gitment 기반 댓글 component 개발
- github action으로 배포 연동
