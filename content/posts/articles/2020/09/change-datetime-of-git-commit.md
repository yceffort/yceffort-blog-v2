---
title: Git commit의 일시를 변경하기
tags:
  - git
published: true
date: 2020-09-15 10:56:58
description: '왜 바꿔야 하는지는 비밀'
category: git
template: post
---

```bash
GIT_COMMITTER_DATE="Tue Sep 15 2020 10:57:29 +0900" git commit --date="Tue Sep 15 2020 10:57:29 +0900"
```

주의: `git rebase`시 강제로 일시를 변경하기 때문에 꼬일 수 있음.

출처: https://stackoverflow.com/questions/454734/how-can-one-change-the-timestamp-of-an-old-commit-in-git
