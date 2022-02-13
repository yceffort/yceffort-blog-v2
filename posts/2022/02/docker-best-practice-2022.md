---
title: '더 나은 Dockerfile 작성을 위한 best practice - 2022년 버전'
tags:
  - docker
published: true
date: 2022-02-05 23:24:24
description: '갑자기 docker를 파는 이유는'
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

```shell
Sending build context to Docker daemon  187.8MB
```

### `stdin`을 활용한 `Dockerfile` pipe

도커는 원격 또는 리모트 빌드 컨텍스트를 `stdin` 명령어를 통해 이미지를 만들 수 있다. `Dockerfile`을 `stdin` 명령어로 파이핑 하는 것은 디스크에 `Dockerfile`을 쓰지 않고 일회성으로 일회성으로 빌드하거나, `Dockerfile`이 있지만 이후에 삭제될 수도 있는 상황에서 유용하다.

```shell
echo -e 'FROM busybox\nRUN echo "hello world"' | docker build -
```

또는

```shell
docker build -<<EOF
FROM busybox
RUN echo "hello world"
EOF
```

#### 빌드 컨텍스트를 전송하지 않고 `Dockerfile` `stdin`으로 이미지 빌드하기

추가된 파일을 빌드 컨텍스트로 보내지 않고, `Dockerfile` `stdin` 을 사용하여 이미지를 작성하려면 아래와 같이 하면 된다. `-`는 `PATH`의 위치를 가리키고, 도커가 디렉토리 대신 `stdin`에서 빌드 컨텍스트 (`Dockerfile`만 있는)를 읽도롤 명령을 내릴 수 있다.

```shell
docker build [OPTIONS] -
```

```shell
docker build -t myimage:latest -<<EOF
FROM busybox
RUN echo "hello world"
EOF
```

빌드 컨텍스트를 생략하면 `Dockerfile`이 이미지로 파일을 복사할 필요가 없고, 데몬으로 파일이 전송되지 않으므로 빌드 속도가 향샹 될 수 있다. (`stdin`으로 필요한 파일을 대신 넘겼으므로)

#### `stdin` `dockerfile`을 이용하여 로컬 빌드 컨텍스트에서 빌드하기

이 방법을 사용하면, 로컬 파일시스템의 파일을 사용하여, `stdin`의 `Dockerfile`파일을 통해 이미지를 빌드할 수 있다. `-f` `--file`로 특정 `Dockerfile`을 지정하고, `-`을 파일 이름으로 사용하여 `Docker`가 `stdin`에서 `Dockerfile`을 읽도록 지시한다.

```shell
docker build [OPTIONS] -f- PATH
```

```shell
# create a directory to work in
mkdir example
cd example

# create an example file
touch somefile.txt

# build an image using the current directory as context, and a Dockerfile passed through stdin
docker build -t myimage:latest -f- . <<EOF
FROM busybox
COPY somefile.txt ./
RUN cat /somefile.txt
EOF
```

#### `stdin` `dockerfile`을 이용하여 리모트 빌드 컨텍스트에서 빌드하기

이 방법을 사용하면 리모트 git 저장소의 파일을 사용하여, `stdin`의 `Dockerfile`파일을 통해 이미지를 빌드할 수 있다. `-f` `--file`로 특정 `Dockerfile`을 지정하고, `-`을 파일 이름으로 사용하여 `Docker`가 `stdin`에서 `Dockerfile`을 읽도록 지시한다.

```shell
docker build [OPTIONS] -f- PATH
```

```shell
docker build -t myimage:latest -f- https://github.com/docker-library/hello-world.git <<EOF
FROM busybox
COPY hello.c ./
EOF
```

### `.dockerignore`로 파일 제외하기

원본 소스 저장소를 건들지 않고 빌드와 관련 없는 파일을 제거하기 위해서는 `.dockerignore` 파일을 사용하면 된다. 이 파일은 `.gitignore` 파일과 유사한 패턴을 지원한다.

> https://docs.docker.com/engine/reference/builder/#dockerignore-file

### 멀티 스테이지 빌드 사용하기

https://docs.docker.com/develop/develop-images/multistage-build/

멀티 스테이지 빌드를 사용하면, 중간 레이어와 파일 수를 줄이는데 힘쓰지 않아도 최종 이미지 크기를 줄일 수 있다.

빌드 프로세스의 마지막 단계에서 실제로 사용되는 이미지가 작성되므로, 빌드 캐시를 활용하여 이미지 레이어를 최소화 할 수 있다.

예를 들어, 빌드에 여러 개의 레이어가 포함되어 있는 경우, 변경이 별로 없는 레이어 (빌드 캐시를 적극 활용할 수 있는 레이어)에서 자주 변경이 일어나는 레이어로 순서를 지정할 수 있다.

- 애플리케이션 빌드를 위해 필요한 툴 설치
- 라이브러리 의존성 설치 또는 업데이트
- 애플리케이션 생성

아래 Go 애플리케이션 예제를 살펴보자.

```Dockerfile
# syntax=docker/dockerfile:1
FROM golang:1.16-alpine AS build

# `docker build --no-cache .` 실행시 의존성 업데이트
RUN apk add --no-cache git
RUN go get github.com/golang/dep/cmd/dep

# Gopkg.toml Gopkg.lock 에 있는 프로젝트 의존성 나열
# 이러한 레이어는 GoPkg파일이 업데이트 되었을 때만 재 빌드 된다.
COPY Gopkg.lock Gopkg.toml /go/src/project/
WORKDIR /go/src/project/
# 라이브러리 의존성 설치
RUN dep ensure -vendor-only

# 전체 프로젝트를 복사하고 빌드
# 이 레이어는 프로젝트 디렉토리에 파일 변경이 있을 때만 다시 빌드됨
COPY . /go/src/project/
RUN go build -o /bin/project

# 이 결과물이 싱글 레이어 이미지에 들어감
FROM scratch
COPY --from=build /bin/project /bin/project
ENTRYPOINT ["/bin/project"]
CMD ["--help"]
```

### 불필요한 패키지를 설치하지 않기

복잡성, 의존성, 파일크기, 빌드시간을 줄이려면 '있으면 좋다' 라는 이유만으로 불필요한 파일이나 패키지를 추가하는 것은 좋지 않다. 예를 들어 데이터 베이스 이미지에 텍스트 에디터는 필요 없다.

### 애플리케이션 디커플링

각 컨테이너에는 하나의 관심사만 존재해야 한다. 애플리케이션을 여러 컨테이너로 분리하면 수평 확장이 용이해지고, 컨테이너를 재사용하기 쉬워진다. 예를 들어, 웹 애플리케이션은 웹, 데이터 베이스, 인메모리 캐시 등으로 분리하여 관리할 수 있다.

각 컨테이너를 하나의 프로세스로 제한하는 것은 좋은 규칙이지만, 쉽게 적용할 수는 없다. 예를 들어 [컨테이너 들만 프로세스를 생성할 수 있는 것은 아니고](https://docs.docker.com/engine/reference/run/#specify-an-init-process), 일부 프로그램 또한 프로세스를 자체적으로 생성할 수 도 있다.

컨테이너를 가능한 클린하고 모듈식으로 유지하기 위해 최선의 판단을 내리자. 컨테이너가 서로 종속적이라면, [Docker Container Network](https://docs.docker.com/network/)를 통해 컨테이너 끼리 통신하도록 할 수 있다.

### 레이어 수를 최소화 하기

옛날 버전의 도커에서는, 이미지에서 레이어 수를 최소화 하여 레이어 성능을 보장하는 것이 중요했다. 이러한 제한을 줄이기 위해 다음과 같은 기능이 추가되었다.

- `RUN` `COPY` `ADD` 만 레이어를 생성한다. 다른 명령어는 임시로 중간 이미지를 생성하며, 빌드 사이즈에 영향을 미치지 않는다.
- 가능하다면, 멀티 스테이지 빌드를 사용하고 필요한 아티팩트만 마지막 이미지에 복사하는 것이 좋다. 이렇게 하면 최종 이미지의 크기를 늘리지 않고도 중간 빌드 단계에 각종 도구와 디버그 정보를 포함 시킬 수 있다.

### 여러줄 인수를 정렬

가능하다면, 여러 줄 인수를 알파벳순서로 정렬하는 것이 좋다. 이렇게 하면 패키지 중복을 방지하고 목록을 쉽게 업데이트 할 수 있다. 또한 PR 검토 또한 용이해진다. `\` 앞에 공백을 추가하는 것도 도움이 된다.

```Dockerfile
RUN apt-get update && apt-get install -y \
  bzr \
  cvs \
  git \
  mercurial \
  subversion \
  && rm -rf /var/lib/apt/lists/*
```

### 빌드 캐시 활용

이미지를 빌드할때, 도커는 `Dockerfile` 내부에 지정되어 있는 순서대로 실행한다. 각 명령을 검토할 때 도커는 새로운 (중복) 이미지를 만들지 않고 캐시에서 재사용할 수 있는 이미지를 찾는다.

캐시를 전혀 사용하지 않으려면 `--no-cache=true`를 사용할 수 있다. 그러나 도커가 캐시를 활용할 수 있도록 하기 위해서는, 일치하는 이미지를 찾을 수 있는 경우와 없는 경우에 대해 이해하는 것이 중요하다. 도커가 따르는 기본적인 규칙은 아래와 같다.

- 이미 캐시에 있는 부모 이미지를 시작으로, 다음 명령어를 해당 기본 이미지에서 파생된 모든 하위 이미지와 비교하여 동일한 명령어를 사용하여 빌드되었는지 확인한다. 그렇지 않으면 캐시가 무효화 된다.
- 대부분의 경우 `Dockerfile`의 명령어를 하위 이미지 들과 비교하는 것으로 충분하다. 하지만, 어떤 명령어는 더 많은 검토가 필요하다.
- `ADD` `COPY`의 경우 이미지 파일 내용을 검사하고 각 파일에 대한 체크섬을 추가로 계산한다. 여기에서 파일의 마지막 수정 시간이나 엑세스 시간은 고려하지 않는다. 캐시 조회 중에 체크섬을 기존 이미지의 체크섬과 비교한다. 파일에서 내용이나 메타데이터의 변경이 있으면 캐시가 무효화 된다.
- `ADD` `COPY` 명령어 외에도 캐시 일치 여부를 확인하기 위해 컨테이너의 파일을 확인하지 않는다. 일례로 `RUN apt-get -y update`를 처리할때 컨테이너에서 업데이트된 파일을 검사하여 캐시와 치하는 경우가 존재하는지 확인하지 않는다. 이 경우는, 명령 문자열 자체만 일치하는지만 검토한다.

일단 캐시가 무효화되면 이후의 모든 Dockerfile 명령은 새로운 이미지를 생성하며, 캐시는 사용되지 않는다.
