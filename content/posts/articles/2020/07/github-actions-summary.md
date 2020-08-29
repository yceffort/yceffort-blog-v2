---
title: Github actions 요약
tags:
  - github
published: true
date: 2020-07-23 10:13:11
description: "# Github action ## Github action 은 무엇인가?  github actions은 사용자 정의
  소프트웨어 개발 라이프 사이클 워크 플로우를 github 레파지토리에 직접 만들수 있도록 도와주는 도구다.  > GitHub Actions
  enables you to create custom software development life c..."
category: github
slug: /2020/07/github-actions-summary/
template: post
---
# Github action

## Github action 은 무엇인가?

github actions은 사용자 정의 소프트웨어 개발 라이프 사이클 워크 플로우를 github 레파지토리에 직접 만들수 있도록 도와주는 도구다.

> GitHub Actions enables you to create custom software development life cycle (SDLC) workflows directly in your GitHub repository.

github actions을 활용하여 코드를 저장하고 협업하는 공간에서 동시에 소프트웨어 개발 워크 플로우를 자동화 할 수 있다.

## 핵심 개념

- action: 작업을 생성하기 위한 단계로, 개별 step을 결합한 단위다. 액션은 워크 플로우 블록을 만드는 가장 작은 포터블 단위다. 직접 만들거나, 깃헙 커뮤니티에 공개되어 있는 것을 쓰거나, 퍼플릭 액션을 커스터마이징해서 쓸 수 있따. 워크플로우에서 액션을 사용하기 위해서는, step이 포함되어 있어야 한다.
- artifact: 코드를 빌드하거나 테스트할 때 만들어지는 파일들을 의미한다. 예를 들어, 아티팩트는 바이너리나 패키지 파일, 테스트 결과, 로그 파일, 스크린 샷등을 포함할 수 있다. 아티팩트는 생성된 워크플로우와 연결되며, 다른작업에서 사용하거나 배포에 이용할 수 있다.
- Continuous integration (CI): 소프트웨어 개발 관습상 공유하고 있는 레파지토리에 종종 작은 단위의 코드로 기여한다. 깃헙 액션을 사용하면, 사용자정의 CI 워크 플로우를 만들어서 빌드와 코드 테스트를 자동화 할 수 있다. 저장소에서, 워크 플로우에 있는 각 액션을 활용하여 코드 변화, 로그 등을 확인할 수 있다. CI는 코드 변화에 따른 버그 등을 빠르게 감지할 수 있다.
- Continuous deployment (CD): 새로 만든 코드가 CI 테스트를 통화가면, 코드는 자동으로 프로덕션에 배포 될 수 있다. 깃헙 액션을 활용하면, 사용자정의 CD 워크플로우를 만들어 클라우드, self-hosted 서비스 또는 플랫폼 등등에 자동으로 배포할 수 있다. CD는 배포 과정과 테스트를 자동화하여 개발자의 시간을 절약해주며, 안정적인 코드 변화를 고객에게 제공할 수 있다.
- Event: 워크플로우 실행이 트리거하는 특정 활동을 의미한다. 예를 들어, 액티비티는 누군가 깃헙 저장소에 코드를 푸시하거나 이슈를 만들거나, PR을 만들 때 발생할 수 있다. 또한 레파지토리에 웹훅을 연결하여 외부 이벤트와 연결 할 수도 있다.
- Github-hosted runner: 깃헙은 리눅스, 윈도우, 맥OS 러너를 호스팅한다. 잡은 완전히 새로운 가상 환경에서 실행되는데, 이 가상환경에는 일반적으로 사용되는 소프트웨어 들이 설치되어 있다. 깃헙은 github-hosted runner의 유지보수 및 업그레이드를 담당하며, 사용자가 임의로 커스터마이징 할 수는 없다.
- Job: 같은 러너에서 실행되는 일련의 단계. 워크플로 파일에서 작업을 실행하는 방법에 대한 규칙을 정의할 수 있다. 작업 (Job)은 이전 작업의 상태에 따라 동시에 병렬로 실행하거나, 순차적으로 실행할 수 있다. 예를 들어 워크플로는 빌드 작업의 상태에 따라 테스트 작업이 달라지는 빌드 및 테스트 작업 두개를 순차적으로 실행할 수 있다. 예를 들어 빌드 작업이 실패하면, 테스트 작업은 수행되지 않을 것이다. github 호스트 러너의 경우 워크플로우의 각 작업은 가상 환경의 새로운 인스턴스에서 실행된다.
- Runner: Github actions runner 애플리케이션이 설치된 모든 시스템. Github이 만든 runner를 사용하거나, 직접 만든 runner를 사용해도 된다. runner는 실행 가능한 작업을 기다린다. runner가 작업을 선택하면, 작업의 액션을 실행하고, 진행상황을 보고하며, 로그를 남기고, 마지막 결과를 깃헙에 리턴한다. 러너는 한번에 하나의 작업만 실행할 수 있다.
- Self-hosted runner: 사용자가 관리하고 유지하는 self-hosted runner 애플리케이션이 설치되어 있는 머신. 이는 깃헙이 제공하는 러너에 비해 더 많은 관리 포인트를 요구한다. 이를 활용하면 작업이 실행될 하드웨어를 커스터마이징 할 수 있다. 
- Step: 명령어와 액션을 실행하는 개별 태스크. 작업은 한개 이상의 step으로 이루어져 있다. 한 작업내에서 각각의 step은 같은 러너안에서 실행되며, 파일시스템을 활용하여 작업으에 필요한 정보를 공유할 수 있다.
- Virtual Environment: Github hosted runner의 가상환경에는 가상 머신 설정, 운영채제, 소프트웨어 등이 포함되어 있다.
- Workflow: 사용자가 레파지토리에서 직접 빌드, 테스트, 패키지, 릴리즈, 배포 등을 할 수 있는 자동화 프로세스를 의미한다. 워크플로우는 한개이상의 잡으로 이루어져 있고, 스케쥴링되거나 특정 이벤트에 의해 실행 될 수 있다.
- Workflow file: 하나이상의 작업에 대한 워크플로우 설정이 들어 있는 YAML 파일. 이 파일은 루트 디렉토리의 `.github/workflows` 폴더에 있어야 한다.
- Workflow run: 미리 설정한 이벤트에 의해 실행된 워크플로우 인스턴스. 여기에서 작업, 액션, 로그, 각 워크플로우의 실행항태를 볼 수 있다.

## 워크플로우 파일 살펴보기

위에서 언급했던 것처럼, 이 파일은 `./github/workflows`에 정의 되어 있어야 한다.

```yaml
name: Greet Everyone
# 이 워크플로우는 코드가 푸쉬되면 발동한다.
on: [push]

jobs:
  build:
    # 작업 명칭
    name: Greeting
    # 이 작업은 우분투에서 실행된다.
    runs-on: ubuntu-latest
    steps:
      # 이 작업은 hello-world-javascript-action: https://github.com/actions/hello-world-javascript-action 의 예제다.
      - name: Hello world
        uses: actions/hello-world-javascript-action@v1
        with:
          who-to-greet: 'Mona the Octocat'
        id: hello      
      # 여기에서는 이전 스텝에서 부터 얼마나 걸렸는지를 나타낸다.
      - name: Echo the greeting's time
        run: echo 'The time was ${{ steps.hello.outputs.time }}.'
```

## check-out action 활용하기

워크플로우에서 사용할 수 있는 스탠다드 액션 중에, checkout action은 아래와 같은 액션이 있다면 반드시 이전에 실행되어야 한다.

- 저장소 코드를 빌드하거나, 세트스하거나, CI에 활용하기 위하여 저장소 코드의 사본이 필요한 경우
- 워크플로우에 동일하나 저장소에 정의된 작업이 하나 이상 있을 경우

checkout-cation을 사용하기 위해서는, 다음 스텝을 포함하면 된다.

```
- uses: actions/checkout@v2
```

## 워크플로우의 작업 유형을 선택하기

프로젝트의 요구사항에 맞추어 사용할 수 있는 두가지 유형의 액션이 존재한다.

- Docker container 액션
- 자바스크립트 액션

액션을 선택하는데 앞서서 이미 공개 저장소나 도커 허브에 나와 있는 다양한 액션을 살펴보기를 권장한다.

## Nodejs CI 예제

```yaml
name: Node.js CI

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest

    # 어떤 노드버전을 사용할지 나타낸다. x는 해당 버전의 가장 최신버전을 의미한다. (wildcard)
    strategy:
      matrix:
        node-version: [8.x, 10.x, 12.x]

    steps:
    - uses: actions/checkout@v2
    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v1
      with:
        # strategy.matrix를 지정하지 않고, 단일 노드버전을 명시하여 하나만 설치할 수도 있다.
        node-version: ${{ matrix.node-version }}
    - run: npm install
    - run: npm run build --if-present
    - run: npm test
      env:
        CI: true
```

### 디펜던시 캐싱하기

유니크 키를 활용하여, 현재 디펜던시를 캐싱하고 `cache` 액션을 추가하여 이 캐싱한 디펜던시를 재 활용할 수 있다.

```yaml
steps:
- uses: actions/checkout@v2
- name: Use Node.js
  uses: actions/setup-node@v1
  with:
    node-version: '12.x'
- name: Cache Node.js modules
  uses: actions/cache@v2
  with:
    # npm cache files are stored in `~/.npm` on Linux/macOS
    path: ~/.npm 
    # 캐시키
    key: ${{ runner.OS }}-node-${{ hashFiles('**/package-lock.json') }}
    # 캐시키로 찾지 못했을 경우 다시 시도해볼 키
    restore-keys: |
      ${{ runner.OS }}-node-
      ${{ runner.OS }}-
- name: Install dependencies
  run: npm ci
```