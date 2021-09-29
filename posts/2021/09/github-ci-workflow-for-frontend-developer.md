---
title: '프론트엔드 프로젝트를 위한 github action workflow'
tags:
  - frontend
  - github
  - "CI/CD"
published: true
date: 2021-09-28 21:21:56
description: '사랑해요 Github'
---

## Introduction

프론트엔드 엔지니어로 일을 하다 보면, 당연히 많은 오픈소스와 다양한 도구에 도움을 받고 의존하게 된다. VS Code 를 비롯해서 여러가지가 있지만, 최근에 내가 가장 도움을 많이 받은 도구 중 하나는 Github Action 이다. 이 글에서는 내 개인 프로젝트와 직장에서 아용해 온 github action 을 정리하고, 이러한 Workflow ci 워크플로우를 활용하여 프론트엔드 팀의 CI/CD 파이프라인에 도움을 주는 방법을 살펴보자.

## 좋은 Github CI workflow는 무엇일까

- cost saving: github actions은 [빌드 시간 만큼 비용을 청구](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions)하므로 빌드 시간을 최소한으로 낮춰야 한다.
- efficient: workflow는 가능한 빨리 수행 되어서 성공 또는 실패 여부를 확인할 수 있어야 한다.
- well-architected: 각 모든 step에는 모두 목적이 있으며, 쓸데 없는 step이 없어야 한다.

## Workflow 가 해야 하는 일

- lint
- formatting
- type checking
- unit test
- build
- e2e test (다양한 브라우저)

물론 여유가 있다면 이러한 작업을 별도의 workflow에서 실행하는 것이 가장 간단한 방법이다. 그러나 한 작업이 실패한다면, 다른 작업은 진행할 필요가 없음에도 (모든 테스트가 통과하는게 의미 있기 때문에) 다른 작업을 중단하는 것은 불가능하다.

정리하자면, 이러한 방식의 워크플로우는 병렬로 실행되기 때문에 서로 상호 작용할 방법이 없다. 즉, 다른 워크 플로우의 실패를 다른 워크플로우의 중단으로 트리거할 수 없다. 

따라서 좋은 방법은 모든 워크플로우를 하나로 결합하는 것이다. 독립적인 워크플로우 였던 모든 태스를 하나의 워크 플로우로 통합한다면 이러한 문제를 해결할 수 있다.

```yaml
jobs:
  lint-format:
    runs-on: ubuntu-latest
    strategy:
      matrix:
      node: [14, 16]
    steps:
      - name: Checkout Commit
      uses: actions/checkout@v2
      - name: Use Node.js ${{ matrix.node }}
      uses: actions/setup-node@v1
      with:
        node-version: ${{ matrix.node }}
      - name: Run lint
      run: |
        npm run lint
      - name: Run prettier
      run: |
        npm run prettier
```

이 job은 원하는 작업을 순차적으로 하거나 또는 병렬로 실행할 수 있다. github에서는 [`needs`라고 하는 키워드를 제공하여](https://docs.github.com/en/actions/learn-github-actions/workflow-syntax-for-github-actions#jobsjob_idneeds) 하나 또는 여러개의 작업을 dependencies로 설정할 수 있으므로, 하나의 작업이 성공적으로 끝나기 전까지는 달느 작업이 시작되지 않게 할 수 있다. 이러한 방법을 활용하면, 빠르게 workflow를 실패하게 만들 수 있고 비싼 작업을 여러번 반복하지 않아도 된다.

```yaml
# 타입체크와 유닛테스트가 병렬로 발생한다
# 빌드는 앞선 두 가지 작업이 성공적으로 발생했을 때만 수행
jobs:
  type-check:
    runs-on: ubuntu-latest
    strategy:
      matrix:
      node: [14, 16]
    steps:
      - name: Checkout Commit
      uses: actions/checkout@v2
      - name: Use Node.js ${{ matrix.node }}
      uses: actions/setup-node@v1
      with:
        node-version: ${{ matrix.node }}
      - name: Check types
      run: |
        tsc -p tsconfig.json --noEmit
  unit-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
      node: [14, 16]
    steps:
      - name: Checkout Commit
      uses: actions/checkout@v2
      - name: Use Node.js ${{ matrix.node }}
      uses: actions/setup-node@v1
      with:
        node-version: ${{ matrix.node }}
      - name: Run test
      run: |
        npm run test
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
      node: [14, 16]
    needs: [type-check, unit-test]
    steps:
      - name: Checkout Commit
      uses: actions/checkout@v2
      - name: Use Node.js ${{ matrix.node }}
      uses: actions/setup-node@v1
      with:
        node-version: ${{ matrix.node }}
      - name: Run build
      run: |
        npm run build
```

무엇을 병렬로 실행할지는 프로젝트의 필요에 따라 달라진다. 예를 들어 유닛테스트와 타입체크는 병렬로 진행하곤 한다. 이 두가지 단계는 빠르게 수행할 수 있고, 비용이 적게 들기 떄문에 다른 테스트와 의존해서 실행될 필요가 없다. 따라서 위 작업이 수행된 후에 빌드 작업이 포함되어도 늦지 않다.

모든 워크플로우를 하나로 잘 결합하고, 병렬화할 작업 또는 순차적으로 실행할 작업을 신중하게 선택하여, CI 파이프라인의 작동방식과 각 단계간의 의존성에 대한 가시성을 높일 수 있다.

## 작업 결과 공유하기

모든 CI 단계를 결합했다면, 이제 다음 과제는 CI 과정에서 나온 결과물을 공유하여 CI를 최대한 효율적으로 작동하도록 하는 것이다. github action에서 사용할 수있는 방법은 두가지가 있다.

1. [actions/cache](https://github.com/actions/cache)를 활용한 레버리지 캐싱
2. [actions/upload-artifact](https://github.com/actions/upload-artifact)와 [download-artifact](https://github.com/actions/download-artifact)를 활용한 artifact 관리

첫번째 만으로도 이미 훌륭(?)하지만 npm install과 같이 반복적이고 시간이 지나도 크게 변하지 않는 출력을 가진 작업에만 사용할 수 있다.

> 참고하기: https://docs.github.com/en/actions/advanced-guides/caching-dependencies-to-speed-up-workflows#example-using-the-cache-action

```yaml
jobs:
  # 이 jobs은 이름에서 알 수 있는 것 처럼, 이전 워크플로우 실행에서 캐시되고 변경이 일어나지 않는 경우 npm dependencies를 설치하고 캐시한다.
  install-cache:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [12]
    steps:
      - name: Checkout Commit
        uses: actions/checkout@v2
      - name: Use Node.js ${{ matrix.node }}
        uses: actions/setup-node@v1
        with:
          node-version: ${{ matrix.node }}
      - name: Cache npm dependencies
        uses: actions/cache@v2
        id: cache-dependencies
        with:
          path: node_modules
          key: ${{ runner.os }}-npm-${{ hashFiles('**/package-lock.json') }}
          restore-keys: |
            ${{ runner.os }}-npm-
      - name: Install Dependencies
        # 캐시가 있다면 해당 스텝을 넘어가고, 그렇지 않다면 설치
        if: steps.cache-dependencies.outputs.cache-hit != 'true'
        run: |
          npm ci

  # 이전에 캐시로 체크했던 의존성을 사용
  type-check:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node: [12]
    needs: install-cache
    steps:
      - name: Checkout Commit
        uses: actions/checkout@v2
      - name: Use Node.js ${{ matrix.node }}
        uses: actions/setup-node@v1
        with:
          node-version: ${{ matrix.node }}
      # 여기에서도 actions/cache를 사용하지만, 여기에서는 의존성을 되살리는 용도로만 사용
      # 워크플로우 전단계에서 이미 설치하거나 캐시를 불러왔을 것이기 때문에 install을 하지 않음
      # Here we use actions/cache again but this time only to restore the dependencies
      - name: Restore npm dependencies
        uses: actions/cache@v2
        id: cache-dependencies
        with:
          path: node_modules
          key: ${{ runner.os }}-npm-${{ hashFiles('**/package-lock.json') }}
          restore-keys: |
            ${{ runner.os }}-npm-
      - name: Check types
        run: |
          tsc -p tsconfig.json --noEmit
```

여기에서 artifacts를 사용하면 더 향상시킬 수 있다. 

예를 들어, 파이어폭스와 크롬에서 각각 e2e 테스트를 하는 작업이 있다고 가정해보자. 이 경우 빌드를 두번하게 되어  github action에 과금 부담이 증가할 수 있으므로 두번 이상 빌드하지 않는 것이 좋다. 이를 해결하기 위해서는 e2e 테스트를 실행하기전에 빌드 작업을 수행한다음, 이 빌드 결과물을 가지고 두군데에서 공유해서 사용하는 것이다.

이를 위해 사용하는 것이 `actions/upload-artifact` 와 `actions/download-artifact` 다.

- 빌드가 성공적으로 끝나면, `actions/upload-artifact` 로 빌드 결과물을 업로드
- 해당 빌드가 필요한 job에서 `actions/download-artifact`를 사용

이 방법은 동일한 워크플로우 실행중에 업로드된 워크플로우의 아티팩트만 다운로드 할 수 있다. 즉 여러개의 개별 action사이에서는 불가능한 방법이다.

```yaml
jobs:
  build:
    # ...
    steps:
      # ...
      - name: Run build
        run: |
          npm run build
      # This step in the build job will upload the build output generated by the previous step
      # 이전 스텝에서 만든 빌드 결과물을 업로드
      - name: Upload build artifacts
        uses: actions/upload-artifact@v2
        with:          
          # 빌드 결과물에 이름을 부여
          name: build-output
          # 업로드할 결과물 path
          path: .next
  e2e-tests-chrome:
    # ...
    needs: build
    steps:
      ...
      # 이전에 업로드 했던 빌드 결과물을 다운로드
      - name: Download build artifacts
        uses: actions/download-artifact@v2
        with:
          name: build-output
          # 아티팩트를 어느 위치에 둘 것인지 설정
          path: .next
      - name: Run cypress
        uses: cypress-io/github-action@v2.10.1
        with:
          start: next start
          browser: chrome
  e2e-tests-firefox:
    # ...
    needs: build
    steps:
      ...
      # 반복
      - name: Download build artifacts
        uses: actions/download-artifact@v2
        with:
          name: build-output
          path: .next
      - name: Run cypress
        uses: cypress-io/github-action@v2.10.1
        with:
          start: next start
          browser: firefox
```

> 물론, artifacts를 업로드해서 저장하는 것도 매월 과금에 추가된다. 따라서 업로드 하기 전에 얼마나 과금이 될지 미리 고민을 해보는 것이 좋다. https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions#included-storage-and-minutes

> `retention-days` 옵션을 사용하면, 시간이 경과한 아티팩트를 자동으로 삭제할 수 있다.

## 반복 작업 삭제하기

코드를 PR 올리기로 결정하고, 푸쉬를 하고 PR을 열었다고 가정해보자. PR로 인해 트리거된 워크플로우가 실행될 것이다. 그러나 잠깐 사이에 무언가 빠트린 코드가 생각나 다시 커밋후 푸시를 해보자. 이 경우에는 또다른 워크플로우가 실행 될 것이다.

기본적으로는 실행중인 이전 워크플로우를 중단할 수는 없다. 워크플로우가 완료될 때 까지 계속 실행되므로 과금에 낭비가 일어날 것이다.

이를 해결 하기 위해 github에서 비교적 최근에 [concurrency](https://github.blog/changelog/2021-04-19-github-actions-limit-workflow-run-or-job-concurrency/)라는 개념을 도입했다.

`concurrency`를 사용하면 워크플로우나 job에 대해서 하나의 concurrency group을 만들 수 있다. 이렇게 하면 한 그룹에서 실행중인 워크플로우가 있을 경우, 'pending' 상태로 표시된다. 그리고 새 워크플로우가 대기열에 추가될 떄 마다 그룹에서 진행중인 워크플로우를 취소하도록 명령을 내릴 수 있다.

```yaml
name: CI

on:
  pull_request:
    branches:
      - main

concurrency:
  # 그룹을 pr의 head_ref로 정의
  group: ${{ github.head_ref }}  
  # 해당 pr에서 새로운 워크플로우가 실행될 경우, 이전에 워크플로우가 있다면 이전 워크플로우를 취소하도록 한다.
  cancel-in-progress: true

jobs:
  install-cache:
  # ...
```

워크플로우 레벨에서 이 작업을 수행하면, 새로운 변경사항을 커밋하여 새로운 워크플로우가 실행 될 때 이전 워크플로우를 취소 시킬 수 있으므로 시간과 비용을 절약할 수 있다.

> 더 많은 `concurrency` 예제 살펴보기: https://docs.github.com/en/actions/learn-github-actions/workflow-syntax-for-github-actions#concurrency

