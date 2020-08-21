---
title: Git Cheat Sheet
tags:
  - git
published: true
date: 2020-08-21 11:52:53
description: "이제 git도 GUI 대신에 커맨드를 활용해서 작업해보자."
category: git
template: post
---

여기저기 잘 만들어져 있는 Git Cheat Sheet를 모아서 한글로 번역해 보았다. 작성시 `[]`는 제거해야 한다.

```toc
tight: true,
from-heading: 2
to-heading: 2
```

## SETUP

- `git config --global user.name "[firstname lastname]"`: git에서 사용할 글로벌 이름을 설정한다.
- `git config --global user.email "[valid-email]"`: git에서 사용할 글로벌 이메일을 설정한다.
- `git config --global color.ui auto`: git 리뷰를 쉽게할 수 있도록 커맨드라인에 자동으로 색깔을 칠해준다.

## SETUP & INIT

- `git init`: git repository 초기화
- `git clone [url]`: URL을 통해서 git repository를 클론한다.

## STAGE & SNAPSHOT

- `git status`: 작업중인 디렉토리에서 변경된 파일 목록을 보여준다.
- `git add [file]`: 다음 커밋에 추가될 파일 (스테이징할)을 추가한다.
- `git reset [file]`: 작업중인 디렉토리에서 스테이징 된 파일을 다시 unstage 상태로 되돌린다.
- `git reset --hard [file]`: 스테이징 영역과 작업 디렉토리를 가장 최근 커밋과 일치하도록 리셋하고, 작업 디렉토리의 모든 변경사항을 엎어버린다.
- `git reset [commit]`: 현재 브랜치를 커밋ID 쪽으로 되돌리고, 모든 스테이징되어 있는 변경사항을 되돌리지만, 작업중인 내용은 되돌리지 않는다.
- `git reset --hard [commit]`: 스테이징영역과 작업중인 영역 모두를 리셋해 버린다. 커밋되지 않는 변경내역은 모두 날라가고, commit 이후의 내용도 모두 날라간다.
- `git diff`: 스테이징되지 않은 파일들 중에서 diff를 확인한다.
- `git diff --staged`: 스테이징된 파일들 중에서 diff를 확인한다.
- `git commit -m "[message]"`: 스테이징된 파일을 메시지와 함께 커밋한다.
- `git commit --amend`: 가장 마지막 커밋을 현재 스테이징되어 있는 내용가 마지막 커밋을 병합한다. 스테이징과 별도로 사용한다면, 단순히 커밋 메시지를 변경하는 용도로도 사용할 수 있다.

## BRANCH & MERGE

- `git branch`: 브랜치 목록을 보여준다. `*`이 떠있는 브랜치는 현재 활성화된 브랜치를 의미한다.
- `git branch [branch-name]`: 현재 커밋을 기준으로 새로운 브랜치를 만든다.
- `git checkout [branch-name]`: 다른 브랜치로 변경 한다음, 해당 내용을 작업중인 브랜치로 가져온다.
- `git merge [branch-name]`: 특정 브랜치의 작업 내용을 현재 브랜치와 병합한다.

## INSPECT & COMPARE

- `git log`: 현재 브랜치의 모든 커밋 히스토리를 보여준다.
- `git log [branchB]..[branchA]`: 브랜치A의 커밋중 브랜치B에 없는 히스토리를 보여준다.
- `git log --follow [file]`: 파일명 변경까지 포함해서 해당 파일의 커밋을 보여준다.
- `git diff [branchB]...[branchA]`: 브랜치A를 기준으로 브랜치B와 다른 내용을 보여준다.
- `git show [SHA]`: 사람이 읽을 수 있는 형태로 모든 오브젝트를 보여준다.

## TRACKING PATH CHANGES

- `git rm [file]`: 해당 파일을 삭제하고, 스테이지에서도 이를 제거한다.
- `git mv [existing-path] [new-path]`: 파일 위치를 변경하고 스테이지에 이를 기록한다.
- `git log --stat -M`: 경로가 이동한 모든 커밋 로그를 보여준다.

## IGNORING PATTERNS

```
logs/
*.notes
pattern*/
```

git이 무시하기를 원하는 파일들의 패턴을 `.gitignore`에 기록해 둔다.

- `git config --global core.excludesfile [file]`: 시스템 레벨에서 모든 레파지토리에서 무시할 파일을 설정한다.
- `git remote add [alias] [url]`: git URL을 별칭과 함께 추가한다.
- `git fetch [alias]`: Git remote에서 모든 브랜치를 패치한다.
- `git merge [alias]/[branch]`: 현재 브랜치에다가 리모트 브랜치의 최신내용을 병합한다.
- `git push [alias] [branch]`: 로컬 브랜치 커밋을 리모트 레파지토리의 브랜치에 전송한다.
- `git pull`: 리모트 브랜치에서 추적하고 있는 모든 커밋을 패치하고 병합하여 최신화 한다.

## REWRITE HISTORY

- `git rebase [branch]`: 현재 브랜치보다 앞서있는 모든 변경 내용 (커밋)을 땡겨와서 적용한다.
- `git reset --hard [commit]`: 스테이징 영역에 있는 것을 모두 클리어하고, 특정 커밋 버전으로 모든 작업내용을 덮어써버린다.

## TEMPORARY COMMITS

- `git stash`: 현재 수정되거나 스테이징되어 있는 변경사항을 모두 저장한다.
- `git stash list`: stack 순서로 되어 있는 모든 stash 목록을 보여준다.
- `git stash pop`: stash stack 최상단에 있는 변경사항을 적용한다.
- `git stash drop`: stash stack 최상단에 stash를 제거한다.



### References

- https://education.github.com/git-cheat-sheet-education.pdf
- https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet