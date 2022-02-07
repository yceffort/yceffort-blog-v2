---
title: 'Dockerfile 작성 가이드'
tags:
  - docker
published: true
date: 2022-02-07 18:04:31
description: '갑자기 docker를 파는 이유는 22'
---

[여기](/2022/02/docker-best-practice-2022) 에서 이어집니다.

하단 권장사항은 효율적이고 유지관리가 용이한 Dockerfile을 만드는데 도움이 되도록 제공되었다.

## Table of Contents

## FROM

https://docs.docker.com/engine/reference/builder/#from

가능하면, 현재 제공되고 있는 공식 이미지를 사용하는 것이 좋다. 알파인 이미지는 리눅스 배포 판 중에서 크키가 매우작고 (6mb) 엄격하게 관리되고 있기 때문에 사용을 추천한다.

## LABEL

https://docs.docker.com/config/labels-custom-metadata/

이미지에 레이블을 추가하여 프로젝트별 이미지 구성, 라이센스 정보 기록, 자동화 정보 등 기타 여러가지 정보를 기록할 수 있다. 각 레이블은 `LABEL`로 시작하고, 하나 이상의 키-값 쌍으로 추가하면 된다. 

공백이 있는 문자열은 따옴표로 묶거나 공백을 이스케이프 해야 한다. (`''` 도 마찬가지다.)

```Dockerfile
# Set one or more individual labels
LABEL com.example.version="0.0.1-beta"
LABEL vendor1="ACME Incorporated"
LABEL vendor2=ZENITH\ Incorporated
LABEL com.example.release-date="2015-02-12"
LABEL com.example.version.is-production=""
```

모든 이미지는 레이블을 하나 이상 가지고 있을 수 있다. Docker 1.10 이전 버전에서는, 추가적인 레이어가 생성되지 않도록 여러 레이블을 하나의 `LABEL`로 묶는 것이 권장되었다. 이제 더이상 필요하진 않지만 여전히 여러개를 결합하는 방식은 가능하다.

```Dockerfile
# Set multiple labels on one line
LABEL com.example.version="0.0.1-beta" com.example.release-date="2015-02-12"
```

```Dockerfile
# Set multiple labels at once, using line-continuation characters to break long lines
LABEL vendor=ACME\ Incorporated \
      com.example.is-beta= \
      com.example.is-production="" \
      com.example.version="0.0.1-beta" \
      com.example.release-date="2015-02-12"
```

- https://docs.docker.com/config/labels-custom-metadata/
- https://docs.docker.com/engine/reference/builder/#label

## RUN

https://docs.docker.com/engine/reference/builder/#run

길거나 복잡한 `RUN` 구문은 백슬래시를 활용하여 여러줄로 분할하는 것이 `Dockerfile` 관리에 좋다.

### `apt-get`

`RUN`에서 아마 가장 자주 사용되는 명령어는 `apt-get`일 것이다. `RUN apt-get`은 패키지를 설치하는 명령어이기 때문에 몇가지를 고려 해야 한다.

`RUN apt-get update` 와 `apt-get install`은 항상 같은 `RUN`구문 안에 있어야 한다. 

```Dockerfile
RUN apt-get update && apt-get install -y \
    package-bar \
    package-baz \
    package-foo  \
    && rm -rf /var/lib/apt/lists/*
```

`apt-get update`를 `RUN`구문에서 단독으로 쓰면 캐시 문제가 있을 수 있고, 이어지는 `apt-get install`가 실패할 가능성도 있다. 예를 들면

```Dockerfile
# syntax=docker/dockerfile:1
FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install -y curl
```

이미지가 빌드된 이후에, 모든 레이어가 도커 캐시안에 들어가게 된다. `apt-get install` 뒤에 구문을 추가했다고 가정해보자.

```Dockerfile
# syntax=docker/dockerfile:1
FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install -y curl nginx
```

도커는 이전 명령어와 수정된 명령어가 동일 할 때에만 이전단계의 캐시를 사용한다. 따라서 빌드가 캐신된 버전을 사용하기 때문에 `apt-get update`가 실행되지 않는다. 그러므로 이 빌드는 잠재적으로 오래된 버전의 `curl`과 `nginx` 패키지를 얻게되는 결과를 초래할 수 있다.

`RUN apt-get update && apt-get install -y` 를 수행하면, 도커파일이 더이상의 코딩이나 수동작업 없이 최신 패키지 버전을 설치할 수 있다. 이 기술을 `cache busting`이라고 한다. 패키지 버전을 지정하여 이 캐시버스팅을 수행할 수도 있다.

```Dockerfile
RUN apt-get update && apt-get install -y \
    package-bar \
    package-baz \
    package-foo=1.3.*
```

버전을 이런식으로 고정하면 캐시에 무엇이 있든 상관없이 특정 버전을 검색하도록 강제할 수 있다. 또한 이 기술을 사용하면 예기치 않은 필수 패키지의 버저닝으로 인한 장애를 줄일 수 있다.

아래는 모든 적절한 권장사항을 잘 수행한 예시다.

```Dockerfile
RUN apt-get update && apt-get install -y \
    aufs-tools \
    automake \
    build-essential \
    curl \
    dpkg-sig \
    libcap-dev \
    libsqlite3-dev \
    mercurial \
    reprepro \
    ruby1.9.1 \
    ruby1.9.1-dev \
    s3cmd=1.1.* \
 && rm -rf /var/lib/apt/lists/*
```

`s3cmd`는 `1.1.*` 버전을 사용하도록 했다. 이미지가 만약 이전 버전을 사용하고, 새로운 버전을 지정하면 `apt-get update`에 캐시 버스팅을 발생시키고 새로운 버전을 설치한다. 각 라인에 패키지를 나열하면 패키지가 중복되는 오류도 방지할 수 있다.

또한 `/var/lib/apt/lists`를 제거하여 캐시를 적절히 정리하면 캐시가 레이어에 저장되지 않기 때문에 이미지 용량을 줄일 수 있다. `RUN` 구문은 `apt-get update`와 함께 시작하므로, 패키지 캐시는 항상 `apt-get install` 전에 정리될 것이다.

> Debian Ubuntu에서는 자동으로 `apt-get clean`을 수행해주므로 이럴 필요가 없다.

### Pipe

몇 몇 `RUN` 커맨드는 `|`에 의존하여 동작할 수 있다. 예를 들어

```Dockerfile
RUN wget -O - https://some.site | wc -l > /number
```

도커는 `/bin/sh -c ` 커맨드를 사용하여, 이러한 명령어를 실행한다. 이 명령어는 마지막 작업의 종료코드만 확인하여 성공 실패 여부를 결정한다. 위의 예제에서 살펴보면, 이 빌드 단계는 `wget` 명령어가 실패하더라도, `wc -l` 명령어가 성공하면 새로운 이미지를 만들어 낼 것이다.

파이프의 어느 단계에서든 오류로 인해 명령이 실패하도록 하려면, `set -o pipefail &&`를 앞에 추가하면 된다.

```Dockerfile
RUN set -o pipefail && wget -O - https://some.site | wc -l > /number
```

> 모든 쉘이 `-o pipefail`을 제공하는 것은 아니므로, 아래와 같이 별도로 나눠서 실행해야 할 수도 있다.

```Dockerfile
RUN ["/bin/bash", "-c", "set -o pipefail && wget -O - https://some.site | wc -l > /number"]
```

## CMD

https://docs.docker.com/engine/reference/builder/#cmd

`CMD` 는 나열되어 있는 인수와 함께, 이미지에 포함되어 있는 소프트웨어를 실행하는데 사용된다. CMD는 거의 대부분 항상 `["실행 파일", "param1", "param2"...]` 와 같은 형태로 사용되어야 한다. 

## EXPOSE

## ENV

## ADD or COPY

## ENTRYPOINT

## VOLUME

## USER

## WORKDIR

## ONBUILD