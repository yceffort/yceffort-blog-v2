---
title: Github 액션으로 스케쥴링 작업하기
tags:
  - github
published: true
date: 2020-07-16 04:55:31
description:
  Github actions가 나오면서 cron job을 실행하기가 더 편해졌습니다. 굳이 내 컴퓨터를 24시간 돌리고
  있을 필요도 없고, 비싼 돈 주며 어디 이상한 compute를 쓸 필요도 없어졌습니다. [물론 공짜로 쓸 수 있는 Cron
  서비스](https://www.easycron.com/)도 있지만 아무래도 github 과 연동할 수 있다는 점이 큰...
category: github
slug: /2020/07/cron-job-with-github-actions/
template: post
---

Github actions가 나오면서 cron job을 실행하기가 더 편해졌습니다. 굳이 내 컴퓨터를 24시간 돌리고 있을 필요도 없고, 비싼 돈 주며 어디 이상한 compute를 쓸 필요도 없어졌습니다. [물론 공짜로 쓸 수 있는 Cron 서비스](https://www.easycron.com/)도 있지만 아무래도 github 과 연동할 수 있다는 점이 큰 장점인 거 같네요.

## cron 설정

```yaml
name: cron

on:
  schedule:
    # 실제 스케쥴 작업이 시작될 cron을 등록하면 됩니다.
    # 크론은 https://crontab.guru/ 여기서 확인하면 좋을 것 같습니다.
    # 이 크론은 평일 5시 (한국시간 14시)에 실행됩니다.
    - cron: '0 5 * * 1-5'

jobs:
  cron:
    runs-on: ubuntu-latest
    # 빌드 매트릭스는 여러가지 환경에서 실행될 수 있게 끔 도움을 줍니다.
    # 어차피 저는 테스트가 필요한 것이 아니고, 일반적인 node환경 만 필요하므로 이렇게만 설정해두겠습니다.
    # https://docs.github.com/en/actions/configuring-and-managing-workflows/configuring-a-workflow#configuring-a-build-matrix
    strategy:
      matrix:
        node-version: [12.x]

    # 현재 레파지토리를 체크아웃합니다.
    # https://github.com/actions/checkout
    steps:
      - uses: actions/checkout@v2

      # Nodejs를 셋업합니다.
      - name: Use Node.js ${{ matrix.node-version }}
        uses: actions/setup-node@v2.1.0
        with:
          node-version: ${{ matrix.node-version }}

      # 캐시된 노드모듈이 있다면 그것을 쓰도록 합니다.
      # https://docs.github.com/en/actions/configuring-and-managing-workflows/caching-dependencies-to-speed-up-workflows#using-the-cache-action 를 참고했습니다.
      # key: key를 활용해서 cache를 만들어 저장합니다.
      # path: 캐시될 파일의 위치입니다.
      # OS와 package-lock.json을 기준으로 node_modules의 캐시를 만듭니다.
      - name: Cache node modules
        uses: actions/cache@v2.0.0
        env:
          cache-name: cache-node-modules
        with:
          path: node_modules
          key: ${{ runner.OS }}-build-${{ hashFiles('package-lock.json') }}
          restore-keys: |
            ${{ runner.OS }}-build-${{ env.cache-name }}-
            ${{ runner.OS }}-build-

      # 일반적인 ci를 실행합니다.
      - name: CI
        run: |
          npm ci

      # cron job을 실행합니다.
      - name: Run Cron
        run: |
          npm run something
```
