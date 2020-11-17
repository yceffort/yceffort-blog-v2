---
title: Github을 아름답게 관리하기
date: 2019-12-18 11:31:17
published: true
tags:
  - git
description: "## Commit Message [좋은 git commit 메시지를 위한
  영어사전](https://blog.ull.im/engineering/2019/03/10/logs-on-git.html)  [좋은 git 커밋
  메시지를 작성하기 위한 7가지 약속](https://meetup.toast.com/posts/106)  ### 요약  Single
  Line  ..."
category: git
slug: /2019/12/18/lets-beautify-git/
template: post
---
## Commit Message

[좋은 git commit 메시지를 위한 영어사전](https://blog.ull.im/engineering/2019/03/10/logs-on-git.html)

[좋은 git 커밋 메시지를 작성하기 위한 7가지 약속](https://meetup.toast.com/posts/106)

### 요약

Single Line

```
[#issue number] :emoji: Commit Message
```

Multi Line

```
[#issue number] :emoji: Commit Message
- change detail1
- change detail2
```

- Single Line 과 동일하지만, Multi Line 으로 가면 두 번째 라인은 반드시 비워둘 것
- 세 번째 라인부터 Change 상세를 리스트 형식으로 기술

## Linear History in git

### 장점

1. **git bisect**
2. **possibility of submitting with history to another version control system like SVN**
3. **Documentation for the posterity**. A linear history is typically easier to follow. This is similar to how you want your code to be well structured and documented: whenever someone needs to deal with it later (code or history) it is very valuable to be able to quickly understand what is going on.
4. **Improving code review efficiency and effectiveness**. If a topic branch is divided into linear, logical steps, it is much easier to review it compared to reviewing a convoluted history or a squashed change-monolith (which can be overwhelming).
5. **When you need to modify the history at a later time**. For instance when reverting or cherry-picking a feature in whole or in part.
6. **Scalability.** Unless you strive to keep your history linear when your team grows larger (e.g. hundreds of contributors), your history can become very bloated with cross branch merges, and it can be hard for all the contributors to keep track of what is going on.

[출처](https://stackoverflow.com/questions/20348629/what-are-advantages-of-keeping-linear-history-in-git)

### Rebase

리베이스가 최고다

[출처](https://dev.to/maxwell_dev/the-git-rebase-introduction-i-wish-id-had)

간단히 요약하면, 내가 작업한 내용을 master의 최신 커밋 뒤에 이어서 붙이는 것이다.

우리의 목표

![git-rebase](https://git-scm.com/book/en/v2../../../images/perils-of-rebasing-5.png)

1. rebase 대상 브랜치 (보통은 master)를 checkout해서 pull
2. rebase 하려는 브랜치 (내가 작업한 브랜치)를 checkout해서 pull

![git-rebase](https://git-scm.com/book/en/v2../../../images/basic-rebase-1.png)
현재까지의 상태는 이럴 것이다.

3. `git rebase master` 를 때린다

![git-rebase](https://git-scm.com/book/en/v2../../../images/basic-rebase-3.png)

4. 컨플릭이 없다면 6번으로

5. 컨플릭이 있다면 컨플릭을 해결한 후에 `git rebase --continue`를 한다.

6. `git push origin <branch> --force`로 force push를 한다.

리베이스는 과거 커밋을 지우고 뒤에 이어 붙인 새로운 커밋을 만들기 때문에, 저장소의 커밋 히스토리를 다시 쓰게 된다.
