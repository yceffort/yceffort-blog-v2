---
title: '더 나은 Dockerfile 작성을 위한 best practice: 2022년 버전'
tags:
  - docker
published: true
date: 2022-02-05 23:24:24
description: '리액트 18 존버 하는 중'
---

https://docs.docker.com/develop/develop-images/dockerfile_best-practices/ 글을 번역하고 조금 이해가 안되는 부분은 개인적으로 내용을 추가 했습니다.

## Table of Contents

## Introduction

docker는 `Dockerfile`을 읽어서 자동으로 이미지를 빌드 한다. 이 텍스트 파일 내부에는 주어진 이미지에서 실행되야할 모든 명령어가 담겨 있다. `Dockerfile`을 작성하는 방법은 [여기](https://docs.docker.com/engine/reference/builder/)에 나와 있다.

도커 이미지는 Dockerfile의 명령어를 나타내는 읽기전용 레이어로 구성되어 있다. 이 레이어가 쌓이고, 각 레이어는 이전 레이어에서 변경된 내용을 담고 있다. 다음 파일을 살펴보자.

```Dockerfile
# syntax=docker/dockerfile:1
FROM ubuntu:18.04
COPY . /app
RUN make /app
CMD python /app/app.py
```

여기에서 각 명령어는 하나씩 레이어를 생성한다.

- `FROM`: `ubuntu:18.04` 도커 이미지로 부터 레이어를 생성한다.
- `COPY`: 도커 클라이언트의 현재 디렉토리에서 파일을 추가한다.
- `RUN`: `make` 명령어로 애플리케이션을 빌드
- `CMD`: 컨테이너 내부에서 실행해야할 커맨듣

이미지를 실행하고 컨테이너를 생성할 때, 기본적으로 주어저있는 레이어 위에 새로운 writable 레이어를 추가한다. 파일 추가, 수정, 삭제 등 실행중인 컨테이너에 대한 모든 변경사항이 이 writable 레이어 위에서 기록된다.

## 더 나은 Dockerfile을 위한 가이드라인과 제안

### 수명이 짧은 컨테이너 만들기

`Dockerfile`에 의해 정의된 이미지는 가능한 수명이 짧은 컨테이너를 생성해야 한다. 여기서 수명이 짧다 (ephemeral) 라는 것의 의미는, 컨테이너가 멈추고, 삭제되고 그리고 다시 빌드되고 재구축 되는 일련의 과정이 최소한의 구성과 설정으로 이루어져야 한다는 뜻이다.

### build context에 대한 이해

`docker build` 명령어를 실행했을 때, 현재 작업 디렉토리를 build context (이하 빌드 컨텍스트)라고 한다. 기본적으로 `Dockerfile`은 여기에 위치하는데, `-f` 플래그로 다른 곳에 위치한 파일도 지정할 수 있다. `Dockerfile`의 위치와 관계 없이, 현재 디렉토리에 있는 파일 및 디렉토리 내부의 재귀적으로 존재하는 모든 내용은 빌드 컨텍스트로 도커 데몬에 전송된다.

이미지를 만드는데 필요하지 않은 파일을 포함시키면 빌드 컨텍스트가 커지고, 이미지 크기도 커진다. 이미지 크기가 커지면 빌드에 걸리는 시간, push pull에 소요되는 시간, 컨테이너 런타임 크기 등 모든 것이 늘어난다. 이 빌드 컨텍스트의 크기를 보려면 `Dockerfile`을 작성할 때 다음과 같은 메시지를 확인해보자.

```
Sending build context to Docker daemon  187.8MB
```

### `stdin`을 활용한 `Dockerfile` pipe

도커는 원격 또는 리모트 빌드 컨텍스트를 `stdin` 명령어를 통해 이미지를 만들 수 있다. `Dockerfile`을 `stdin` 명령어로 파이핑 하는 것은 디스크에 `Dockerfile`을 쓰지 않고 일회성으로 일회성으로 빌드하거나, `Dockerfile`이 있지만 이후에 삭제될 수도 있는 상황에서 유용하다.

```
echo -e 'FROM busybox\nRUN echo "hello world"' | docker build -
```

또는

```
docker build -<<EOF
FROM busybox
RUN echo "hello world"
EOF
```

#### 빌드 컨텍스트를 전송하지 않고 `Dockerfile` `stdin`으로 이미지 빌드하기

추가된 파일을 빌드 컨텍스트로 보내지 않고, `Dockerfile` `stdin` 을 사용하여 이미지를 작성하려면 아래와 같이 하면 된다. `-`는 `PATH`의 위치를 가리키고, 도커가 디렉토리 대신 `stdin`에서 빌드 컨텍스트 (`Dockerfile`만 있는)를 읽도롤 명령을 내릴 수 있다.

```
docker build [OPTIONS] -
```

```
docker build -t myimage:latest -<<EOF
FROM busybox
RUN echo "hello world"
EOF
```

빌드 컨텍스트를 생략하면 `Dockerfile`이 이미지로 파일을 복사할 필요가 없고, 데몬으로 파일이 전송되지 않으므로 빌드 속도가 향샹 될 수 있다. (`stdin`으로 필요한 파일을 대신 넘겼으므로)

