---
title: 머지된 브랜치를 삭제하는 스크립트
tags:
  - git
published: true
date: 2020-01-02 08:15:56
description: remote에서 이미 master로 머지된 local/remote 브랜치를 삭제하는 스크립트 ```shell git
  fetch --all -p git branch --merged | grep -E -v "master|\*" | xargs -n 1 git
  branch -d git branch -vv | grep gone | sed | awk '{print ...
category: git
slug: /2020/01/delete-merged-branch/
template: post
---
remote에서 이미 master로 머지된 local/remote 브랜치를 삭제하는 스크립트

```shell
git fetch --all -p
git branch --merged | grep -E -v "master|\*" | xargs -n 1 git branch -d
git branch -vv | grep gone | sed | awk '{print $1}' | xargs -n 1 git branch -D
```

우교수님께 감사의 말씀을 🙇‍♂️
