---
title: 'nodejs의 성능 분석은 어떻게 할까?'
tags:
  - javascript
  - nodejs
published: false
date: 2023-06-15 22:32:58
description: '이론도 중요하지만 🤔'
---

## Table of Contents

## 서론

아마 대다수의 프론트엔드 개발자, 아니 개발자라면 누구나 성능이 중요하다는 사실에 대해 공감하고 있을 것이다. 그러나 성능 분석은 꽤나 까다롭고, 기능 추가와 다르게 드라마틱한 향상도 기대하기 어렵기 때문에 많은 개발자가 꺼리기도 한다. 특히 프론트엔드, 즉 웹 영역은 크롬 개발자 도구가 주는 강력한 도구에 의지해 왠만한 내용의 분석은 손쉽게 할 수 있지만, nodejs, 특히 서버와 같이 영속적으로 실행되는 서비스가 아닌 스크립트의 경우에는 성능 분석을 등한시하는 경우가 많다.

회사에서 크고 작은 일을 벌리면서 성능 분석에 대한 중요성을 많이 깨닫고, 또 좌충우돌을 겪어오면서 전문가 까지는 아니지만 어느정도 인사이트를 얻을 수 있는 기회가 있었어서 그 이야기를 미약하게 나마 다뤄볼까 한다.

## 오늘의 분석 대상

분석을 하기 위해서는 먼저 분석의 대상이 될 스크립트가 필요하다. 심도 있는 내용을 다루고 싶다면 직접 스크립트를 작성하면서 하나하나 알아보는 것도 좋겠지만, 그러기엔 너무 귀찮기 때문에 이번 실험의 대상으로는 요즘 내가 제일 자주 쓰는 패키지 관리자인 pnpm에 다뤄 볼 까 한다.

pnpm은 요즘 많이 힘든 시기를 겪고 있는 우크라이나의 개발자 [Zoltan Kochan](https://github.com/zkochan) 가 만든 npm 보다 진보한 nodejs 패키지 매니저로, 빠르고 디스크 효율적인 nodejs 패키지 매니저를 지향하고 있다.

사실 '빠르다' 라는 것은 여러가지 측면이 있을 수 있지만, pnpm이 추구하는 빠름은 `npm`이 실행하는 작업 자체를 빠르게 한다기보다는 node_modules를 보다 효율적으로 관리하는데에 있다. `pnpm`을 써봤다 하더라도 `pnpm`자체가 nodejs로 작성되어 있다는 사실까지 알아채는 사람은 많이 없을 것이다.

https://github.com/pnpm/pnpm/blob/main/pnpm/src/main.ts

저장소를 방문하여 살펴보면, 여타 다른 nodejs 프로젝트 처럼 타입스크립트로 작성된 모노레포로 구성되어 있는 것을 볼 수 있다. 즉 슬프게도(?) 완전히 다른 언어로 작성하여 같은 작업을 더욱 빠르게 하고 있는 SWC, turbopack 들 과는 다르게 여전히 느린 nodejs 환경 기반으로 운영되고 있다는 것이다. 이는 경쟁 package manger인 yarn도 동일하다.

이야기가 조금 샜지만 pnpm을 대상으로 작성하게 된 것은, 요즘 왠지 pnpm이 조금 느려진 것 같다는 생각을 하게 되면서다. pnpm 코드 자체에는 크게 관심이 없었지만, 무언가 조금 느려진 것이 아닌가 하는 의심을 하면서 부터 조금씩 성능에 대해 파해쳐 봐야겠다는 생각을 하기 시작했다.

## `--prof`

nodejs 에는 크롬 개발자 도구를 직접 붙이는 것 외에도 내장 프로파일러 도구가 존재하는데, 그것이 바로 `--prof`다. `node --prof`를 실행하면 스크립트를 직접 분석할 수 있다. `--prof`로 nodejs 스크립트를 실행하면 `*.log`이라는 파일이 생성되는데, 이 파일을 기반으로 성능을 확인할 수 있다.

```bash
> node --prof ./node_modules/pnpm/dist/pnpm.cjs run echo
```

```text
isolate-0x150078000-38143-v8.log

v8-version,10,2,154,26,-node.26,0
v8-platform,macos,macos
shared-library,/Users/yceffort/.nvm/versions/node/v18.16.0/bin/node,0x102e60000,0x104186294,48611328
shared-library,/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation,0x19d5acb00,0x19d79c0a0,487899136
shared-library,/usr/lib/libobjc.A.dylib,0x19d1aa000,0x19d1de7e0,487899136
shared-library,/System/Library/PrivateFrameworks/CoreServicesInternal.framework/Versions/A/CoreServicesInternal,0x1a06560e0,0x1a068cec0,487899136
shared-library,/usr/lib/liboah.dylib,0x1a8ea6c9c,0x1a8eabda0,487899136
shared-library,/usr/lib/libfakelink.dylib,0x1a8eda4d0,0x1a8edbb80,487899136
shared-library,/usr/lib/libicucore.A.dylib,0x1a008ec3c,0x1a02d93f0,487899136
shared-library,/usr/lib/libSystem.B.dylib,0x1a8ed74c8,0x1a8ed7aec,487899136
shared-library,/System/Library/PrivateFrameworks/SoftLinking.framework/Versions/A/SoftLinking,0x1a8edcbbc,0x1a8edce50,487899136
shared-library,/usr/lib/libc++abi.dylib,0x19d4f41d8,0x19d507e58,487899136
shared-library,/usr/lib/libc++.1.dylib,0x19d466c40,0x19d4c7694,487899136
// ...
```

그러나 이 파일만으로는 시각적인 정보를 얻기 쉽지 않다. 그래서 이 파일을 기반으로 또다른 명령어를 실행한다.

```bash
> node --prof-process --preprocess -j isolate*.log > v8.json
```

이 명령어는 앞서 `node --prof`로 생성된 로그 파일을 하나로 모아서 json 파일을 만드는 역할을 한다.

```json
{
  "code": [
    {
      "name": "/Users/yceffort/.nvm/versions/node/v18.16.0/bin/node",
      "type": "SHARED_LIB"
    },
    {
      "name": "T node::AsyncResource::AsyncResource(v8::Isolate*, v8::Local<v8::Object>, char const*, double)",
      "type": "CPP"
    }
  ]
}
```

그러나 이 파일 역시 만만치 않은 크기를 자랑한다. 이 json 파일을 바탕으로 시각적인 정보를 확인할 수 있는 도구가 존재하는데, 바로 [flamebearer](https://mapbox.github.io/flamebearer/) 다. 이 도구를 활용하면 앞서 만든 `json`파일을 기반으로 flame graph를 생성할 수 있다.

![flamebearer](./images/flamebearer1.png)

![flamebearer](./images/flamebearer2.png)

이렇게하면 이 `node`가 실행되기 위해 어떠한 과정을 거쳤고, 또 각 모듈이 스크립트 실행에 미치는 영향도를 파악할 수 있게 된다. 이러한 정보를 바탕으로, 스크립트 실행과정에서 불필요한 작업이 발생하지는 않았는지, 또 너무 오래 걸리거나 많은 비용이 드는 작업은 없었는지 확인 할 수 있다.

## Import Graph Visualizer

![](./images/import-graph-visualizer.png)
