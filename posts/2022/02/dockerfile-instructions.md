---
title: 'Dockerfile 작성 가이드'
tags:
  - docker
published: true
date: 2022-02-07 18:04:31
description: '갑자기 docker를 파는 이유는 22'
---

[여기](/2022/02/docker-best-practice-2022) 에서 이어집니다.

하단 권장사항은 효율적이고 유지관리가 용이한 `Dockerfile`을 만드는데 도움이 되도록 제공되었다.

## Table of Contents

## `FROM`

[https://docs.docker.com/engine/reference/builder/#from](https://docs.docker.com/engine/reference/builder/#from)

가능하면, 현재 제공되고 있는 공식 이미지를 사용하는 것이 좋다. 알파인 이미지는 리눅스 배포 판 중에서 크키가 매우작고 (6mb) 엄격하게 관리되고 있기 때문에 사용을 추천한다.

## `LABEL`

[https://docs.docker.com/config/labels-custom-metadata/](https://docs.docker.com/config/labels-custom-metadata/)

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

- [https://docs.docker.com/config/labels-custom-metadata/](https://docs.docker.com/config/labels-custom-metadata/)
- [https://docs.docker.com/engine/reference/builder/#label](https://docs.docker.com/engine/reference/builder/#label)

## `RUN`

[https://docs.docker.com/engine/reference/builder/#run](https://docs.docker.com/engine/reference/builder/#run)

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

### `Pipe`

몇 몇 `RUN` 커맨드는 `|`에 의존하여 동작할 수 있다. 예를 들어

```Dockerfile
RUN wget -O - https://some.site | wc -l > /number
```

도커는 `/bin/sh -c` 커맨드를 사용하여, 이러한 명령어를 실행한다. 이 명령어는 마지막 작업의 종료코드만 확인하여 성공 실패 여부를 결정한다. 위의 예제에서 살펴보면, 이 빌드 단계는 `wget` 명령어가 실패하더라도, `wc -l` 명령어가 성공하면 새로운 이미지를 만들어 낼 것이다.

파이프의 어느 단계에서든 오류로 인해 명령이 실패하도록 하려면, `set -o pipefail &&`를 앞에 추가하면 된다.

```Dockerfile
RUN set -o pipefail && wget -O - https://some.site | wc -l > /number
```

> 모든 쉘이 `-o pipefail`을 제공하는 것은 아니므로, 아래와 같이 별도로 나눠서 실행해야 할 수도 있다.

```Dockerfile
RUN ["/bin/bash", "-c", "set -o pipefail && wget -O - https://some.site | wc -l > /number"]
```

## `CMD`

[https://docs.docker.com/engine/reference/builder/#cmd](https://docs.docker.com/engine/reference/builder/#cmd)

`CMD` 는 나열되어 있는 인수와 함께, 이미지에 포함되어 있는 소프트웨어를 실행하는데 사용된다. CMD는 거의 대부분 항상 `["실행 파일", "param1", "param2"...]` 와 같은 형태로 사용되어야 한다.

대부분의 경우, `CMD`는 bash, paython, perl과 같은 대화형 셸이 필요하다. 예를들어 `CMD ["perl", "-de0"]`, CMD `["python"]`, or CMD `["php", "-a"]` 등이 있다. 이러한 형태를 사용하면, `docker run -it python`고과 같은 것을 실행하면 바로 셸로 진입할 수 있다.

## `EXPOSE`

[https://docs.docker.com/engine/reference/builder/#expose](https://docs.docker.com/engine/reference/builder/#expose)

`EXPOSE`는 컨테이너가 연결을 받는 포트를 나타낸다. 따라서 애플리케이션에서 공통으로 사용되는 기존 포트를 사용해야 한다. (아파치 `EXPOSE 80`, 몽고 디비 `EXPOSE 27017`과 같이)

외부에서 접근을 위해 `docker run`에 플래그를 사용하여 이 포트가 어떤 포트에 연결될지 지정할 수 있다.

## `ENV`

[https://docs.docker.com/engine/reference/builder/#env](https://docs.docker.com/engine/reference/builder/#env)

`ENV`를 사용하여 컨테이너가 설치하는 소프트웨어의 PATH 환경변수를 업데이트 할 수 있다. 예를 들어, `ENV PATH=/usr/local/nginx/bin:$PATH`는 `CMD ["nginx"]` 명령어가 실행될 수 있도록 해준다.

또한 Postgres의 `PGDATA`와 같이 컨테이너에 포함하려는 서비스와 관련된 필수 환경변수를 제공하는데 유용하다.

마지막으로, `ENV`는 일반적으로 사용되는 버전 번호를 설정하기 위해 사용할 수도 있다.

```Dockerfile
ENV PG_MAJOR=9.3
ENV PG_VERSION=9.3.4
RUN curl -SL https://example.com/postgres-$PG_VERSION.tar.xz | tar -xJC /usr/src/postgres && …
ENV PATH=/usr/local/postgres-$PG_MAJOR/bin:$PATH
```

프로그램에 상수값이 있는 것과 비슷하게, `ENV`를 사용하면 자동적으로 컨테이너 내부의 소프트웨어 버전을 지정하는 것도 가능하다.

각 `ENV` 라인은 `RUN`과 동일하게 새로운 중간 레이어를 생성한다. 즉, 이후 레이어에서 환경변수를 설정해제하더라도, 이 레이어에서 계속 유지되며 해당 값이 덤프될 수 있다.

```Dockerfile
# syntax=docker/dockerfile:1
FROM alpine
ENV ADMIN_USER="mark"
RUN echo $ADMIN_USER > ./mark
RUN unset ADMIN_USER
```

```shell
$ docker run --rm test sh -c 'echo $ADMIN_USER'

mark
```

이러한 사태를 방지하고, 환경 변수를 실제로 해지 하기 위해서는 `RUN`을 사용하여 변수를 단일 레이어에서 설정, 사용, 해제를 하면 된다. 이 명령어는 `;` `&&`을 사용하여 구분할 수 있다. 후자를 사용한다면, 명령이 실패한다면 도커 빌드도 실패한다.

```Dockerfile
# syntax=docker/dockerfile:1
FROM alpine
RUN export ADMIN_USER="mark" \
    && echo $ADMIN_USER > ./mark \
    && unset ADMIN_USER
CMD sh
```

## `ADD` or `COPY`

- [https://docs.docker.com/engine/reference/builder/#add](https://docs.docker.com/engine/reference/builder/#add)
- [https://docs.docker.com/engine/reference/builder/#copy](https://docs.docker.com/engine/reference/builder/#copy)

`ADD` `COPY` 두 명령어가 기능적으로 거의 유사하지만, `COPY`가 일반적으로 더 사용된다. 그 이유는 `ADD` 보다 더 순수하기 때문이다. `COPY`는 단순히 컨테이너에 있는 로컬 파일을 복사할 뿐이다. 반면 `ADD`는 몇가지 추가적이 있다. (로컬 전용 tar 파일 해제, 원격 URL 지원 등) 따라서 `ADD`는 `ADD rootfs.tar.xz /` 와 같은 상황에서 사용하는 것이 좋다.

컨텍스트에서 다른 파일을 사용하는 여러 단계가 `Dockerfile` 내부에 있을 경우, 한번에 하지말고 개별적으로 복사하는 것이 좋다. 이렇게 하면 각 단계의 빌드 캐시는 필요한 파일이 변경되었을 경우에만 무효화 (재실행)된다.

예를 들어

```Dockerfile
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt
COPY . /tmp/
```

`COPY . /tmp/` 를 앞에 두는 경우보다 캐시 무효화가 더 줄어든다.

이미지 크기는 중요한 문제이므로, 원격 URL에서 패키지를 가져올때는 `ADD`를 사용하는 것은 권장되지 않는다. 대신 `curl` `wget`을 사용해야 한다. 이렇게 하면 파일 압축을 해제한 후 더이상 필요 없는 파일을 삭제할 수 있으며, 이미지에 다른 레이어를 추가할 필요가 없다. 예를 들어, 아래와 같은 경우는 피해야 한다.

```Dockerfile
ADD https://example.com/big.tar.xz /usr/src/things/
RUN tar -xJf /usr/src/things/big.tar.xz -C /usr/src/things
RUN make -C /usr/src/things all
```

이 대신,

```Dockerfile
RUN mkdir -p /usr/src/things \
    && curl -SL https://example.com/big.tar.xz \
    | tar -xJC /usr/src/things \
    && make -C /usr/src/things all
```

자동 tar 파일 압축 해제 기능 등이 필요하지 않은 다른 항목 (파일, 디렉토리) 에는 `COPY`를 쓰자.

## `ENTRYPOINT`

[https://docs.docker.com/engine/reference/builder/#entrypoint](https://docs.docker.com/engine/reference/builder/#entrypoint)

`ENTRYPOINT`를 쓰는 가장 좋은 방법은 이미지의 메인 커맨드를 설정해두어, 해당 명령어를 기본으로 사용할 수 있게 하는 것이다.

```Dockerfile
ENTRYPOINT ["s3cmd"]
CMD ["--help"]
```

이렇게 하면 아래와 같이 실행했을 때 도움말을 볼 수 있다.

```shell
docker run s3cmd
```

혹은 파라미터를 바로 두어서 바로 커맨드를 실행할 수도 있다.

```shell
docker run s3cmd ls s3://mybucket
```

`ENTRYPOINT` 명령은 helper 스크립트와 함께 사용할 수 있으므로, 특정 tool 을 시작할때 위의 명령어와 유사한 방식으로 동작할 수도 있다.

예를 들어, [Postgres 공식 이미지](https://hub.docker.com/_/postgres/)는 다음 스크립트를 `ENTRYPOINT`로 사용한다.

```Dockerfile
#!/bin/bash
set -e

if [ "$1" = 'postgres' ]; then
    chown -R postgres "$PGDATA"

    if [ -z "$(ls -A "$PGDATA")" ]; then
        gosu postgres initdb
    fi

    exec gosu postgres "$@"
fi

exec "$@"
```

helper 스크립트는 컨테이너에 복사되고, 컨테이너 시작시 `ENTRYPOINT`에 의해 실행된다.

```Dockerfile
COPY
Learn more about the "COPY" Dockerfile command.
 ./docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["postgres"]
```

이 스크립트는 유저가 Postgres를 다양한 방식으로 상호작용할 수 있도록 해준다.

단순히 Postgres를 실행할수도 있고

```shell
 docker run postgres
```

서버에 파라미터를 전달하여 실행할 수도있고

```shell
 docker run postgres postgres --help
```

또한 Bash와 같은 완전히 다른 툴에서도 실행할 수 있다.

```shell
 docker run --rm -it postgres bash
```

## `VOLUME`

[https://docs.docker.com/engine/reference/builder/#volume](https://docs.docker.com/engine/reference/builder/#volume)

`VOLUME`은 도커 컨테이너에서 만든 데이터 저장소 영역, 설정 저장소, 또는 파일이나 폴더를 노출하는데 사용해야 한다. 이미지의 변경 가능한 부분 및 사용자가 수정가능한 부분에는 `VOLUME`을 사용하는 것이 좋다.

## `USER`

[https://docs.docker.com/engine/reference/builder/#user](https://docs.docker.com/engine/reference/builder/#user)

서비스를 실행하는데 별도로 권한이 필요 없다면, `USER` 를 사용하여 루트가 아닌 사용자로 변경해야 한다. `RUN groupadd -r postgres && useradd --no-log-init -r -g postgres postgres`와 같은 명령어로 유저나 그룹을 생성할 수 있다.

`sudo`는 문제를 일으킬 여지가 있으므로 사용하지 않는 것이 좋다. 그럼에도 `sudo`가 어쩔 수 없이 필요한 경우 [gosu](https://github.com/tianon/gosu)의 사용을 고려해보자.

마지막으로, 레이어와 복잡성을 줄이기 위해서는 너무 자주 `USER`를 사용하지 않는 것이 좋다.

## `WORKDIR`

[https://docs.docker.com/engine/reference/builder/#workdir](https://docs.docker.com/engine/reference/builder/#workdir)

명확성, 그리고 신뢰성을 위해 `WORKDIR`은 항상 절대 경로를 사용해야 한다. 읽기 어렵고, 유지보수도 어려운 `RUN cd … && do-something` 대신 `WORKDIR`을 사용하자.

## `ONBUILD`

[https://docs.docker.com/engine/reference/builder/#onbuild](https://docs.docker.com/engine/reference/builder/#onbuild)

`ONBUILD` 명령은 현재 `Dockerfile`의 빌드가 완료된 후 실행된다. `ONBUILD`는 현재 이미지에서 파생된 하위 이미지에서 실행된다. `ONBUILD` 명령은 상위 `Dockerfile`이 하위 `Dockerfile`에 제공하는 명령이라고 보면 된다.

도커는 하위 Dockerfile의 명령에 앞서 `ONBUILD`를 수행한다.

`ONBUILD`는 지정된 이미지에서 빌드할 이미지가 필요할 때 유용하다. 예를 들어, `Dockerfile`내에서 해당 언어로 소프트웨어를 필요로 하는 이미지가 있다면, `ONBUILD` 명령어가 유용하다.

`ONBUILD`로 빌드된 이미지에는 별도 태그가 있어야 한다. (`ruby:1.9-onbuild` `ruby:2.0-onbuild`)

`ONBUILD`에 `ADD` `COPY`를 넣을 때 주의하자. 새 빌드 컨텍스트에 이렇게 추가되는 리소스가 없을 경우 하위 "onbuild" 이미지가 실패할 것이다. 위에서 권장한대로 태그를 추가해서 구별하면, `Dockerfile` 작성자가 이를 선택할 수 있으므로 이러한 문제를 예방할 수 있다.
