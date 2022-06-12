---
title: 'Vercel에서 배포가 안됐던 이야기'
tags:
  - vercel
  - next
  - nodejs
published: true
date: 2021-05-17 20:11:42
description: 'Vercel 고객센터랑 싸운썰 푼다.txt'
---

지난달 (2021년 4월) 에는 포스팅이 좀 뜸했었다. 지난 달에 ~~빌어먹을~~ 프로젝트 하나가 끝난게 핑계라면 핑계겠지만, 사실 그것보다 더 큰 문제가 있었다.

갑자기 어느 순간 부터 배포가 안되기 시작했다.

![error](./images/vercel1.png)

```bash
Error: The Serverless Function "[year]/[...slugs]" is 102.99mb which exceeds the maximum size limit of 50mb. Learn More: https://vercel.link/serverless-function-size
```

이유인 즉슨, `[...slugs].js`가 [aws serverless function에서 허용하는 50mb를 초과한다는 것이었다.](https://vercel.link/serverless-function-size) 근데 갑자기 이런 에러가 나는게 이상했다. 나는 저 함수에 뭔가 엄청난 걸 추가한 적이 없는데? 그래서 마지막으로 성공한 커밋을 한번 다시 배포해보았다.

![success](./images/vercel2.png)

![fail](./images/vercel3.png)

어라? 근데 같은 커밋 sha 에서도 배포가 되지 않는 것이었다. 이상했다. 내가 아는 상식선에서는 같은 커밋은 곧 같은 결과를 만들어야 하는대데? 그 때부터 vercel 고객센터와 싸움이 시작되었다.

첫번째로 제시한 해결책은 next canary 버전 설치와 node 12에서 node 14로 버전업이었다. 사실 이건 그냥 전자제품 껐다 켜보셨나요 수준의 솔루션이라고 생각했다. 이미 나는 next 최신버전을 항상 사용중이기 때문에, canary와 별차이가 없었고 (github까지 가서 확인했다.) node는 죄가 없다고 생각했기 때문이다.

당연히 고쳐지지 않았고, CS 팀에서는 내 빌드 결과를 보여달라고 했다.

```bash
Page                                                                                                          Size     First Load JS
┌ ● /                                                                                                         2.28 kB        90.3 kB
├   /_app                                                                                                     0 B            82.2 kB
├ ○ /404                                                                                                      2.59 kB        84.8 kB
├ ○ /about                                                                                                    3.98 kB        86.2 kB
├ ● /blogs/2018/[...slugs]                                                                                    2.92 kB        96.6 kB
├   ├ /blogs/2018/2018/10/26/central-bank-issued-digital-currencies-why-governments-may-or-may-not-need-them
├   ├ /blogs/2018/2018/12/17/step-by-step-machine-learning-05
├   ├ /blogs/2018/2018/12/15/ibm-blockchain-maersk-shipping-struggling
├   └ [+183 more paths]
├ ● /blogs/2019/[...slugs]                                                                                    2.92 kB        96.6 kB
├   ├ /blogs/2019/2019/12/30/blog-renewal
├   ├ /blogs/2019/2019/09/06/javascript-event-loop
├   ├ /blogs/2019/2019/12/23/tensorflowjs-03-linear_regression
├   └ [+66 more paths]
├ ● /blogs/2020/[...slugs]                                                                                    2.92 kB        96.6 kB
├   ├ /blogs/2020/2020/12/preview-ES2021
├   ├ /blogs/2020/2020/12/javascrpt-async-await-in-map-and-reduce
├   ├ /blogs/2020/2020/12/partitioning-cache
├   └ [+142 more paths]
├ ● /blogs/2021/[...slugs]                                                                                    2.92 kB        96.6 kB
├   ├ /blogs/2021/2021/04/nodejs-multithreading-worker-threads
├   ├ /blogs/2021/2021/04/blog-4.0-update
├   ├ /blogs/2021/2021/03/Lighthouse-CI-with-github-actions
├   └ [+23 more paths]
├ λ /generate-screenshot                                                                                      1.2 kB         83.4 kB
├   └ css/e33eabf74e0f0bf8472d.css                                                                            715 B
├ ● /pages/[id]                                                                                               2.37 kB        90.4 kB
├   ├ /pages/1
├   ├ /pages/2
├   ├ /pages/3
├   └ [+82 more paths]
├ ● /tags                                                                                                     2.04 kB        84.2 kB
└ ● /tags/[tag]/pages/[id]                                                                                    2.45 kB        90.5 kB
    ├ /tags/javascript/pages/1
    ├ /tags/javascript/pages/2
    ├ /tags/javascript/pages/3
    └ [+137 more paths]
+ First Load JS shared by all                                                                                 82.2 kB
  ├ chunks/247.886d94.js                                                                                      5.1 kB
  ├ chunks/288.d236ff.js                                                                                      9.12 kB
  ├ chunks/597.82ffaf.js                                                                                      13.3 kB
  ├ chunks/733.36e935.js                                                                                      6.2 kB
  ├ chunks/framework.8d065a.js                                                                                42 kB
  ├ chunks/main.7713a1.js                                                                                     168 B
  ├ chunks/pages/_app.64b1f5.js                                                                               5.28 kB
  ├ chunks/webpack.86b2b5.js                                                                                  993 B
  └ css/d1cedbca7f6cf6142911.css                                                                              6.07 kB

λ  (Server)  server-side renders at runtime (uses getInitialProps or getServerSideProps)
○  (Static)  automatically rendered as static HTML (uses no initial props)
●  (SSG)     automatically generated as static HTML + JSON (uses getStaticProps)
   (ISR)     incremental static regeneration (uses revalidate in getStaticProps)
```

최초에 `[...slug].js`가 문제라고 했기 때문에, 나는 `[...slug].js]`를 년도 별로 쪼개서 빌드를 해보았는데, 세개를 다 쪼개 봐도 다합쳐서 50mb가 되지 않았다. 그러나 역시나 vercel에서는 빌드가 되지 않았다. 그리고 CS 담당자가 내 github을 포크해서 빌드해보았는데, 역시나 같은 문제가 나고 있었다.

이 단계에서부터 CS 팀 응답이 뜸해지기 시작했다. (pro계정이지만 그보다 더 높은 티어의 고객의 문제를 상담하느라 지연되고 있다고 했다.) 그리고 이 때부터 굉장히 화가나서 (일도 많았고) digital ocean으로 갈아탈 준비를 마쳐두었다. digital ocean이 프로젝트당 20달러를 받고 있어서 굉장히 비쌌지만, (vercel은 계정 당 20달러) vercel에 비해 기능도 많고 복잡했다. 어차피 공짜로 100$가 주어진 김에, 그 돈으로 넘어가려고 설정을 다 마쳐두었다.

```yaml
domains:
  - domain: yceffort.kr
    type: PRIMARY
    zone: yceffort.kr
name: yceffort-blog-v-2
region: sgp
services:
  - build_command: npm run build
    environment_slug: node-js
    github:
      branch: main
      deploy_on_push: true
      repo: yceffort/yceffort-blog-v2
    http_port: 3000
    instance_count: 1
    instance_size_slug: professional-s
    name: yceffort-blog-v-2
    routes:
      - path: /
    run_command: npm start
```

그러다가 한 일주일이 흐른 뒤에 내가 원하던 답이 왔다.

> There are quite a few images included in the new deployment that amounted to the increased size that caused the failure. What the engineers suggest is that you do the following Instead of gathering the sizes of images in getStaticProps which is causing these files to be included, you could move these to a pre-build script that outputs just the sizes to a JSON file and use this instead.

https://github.com/yceffort/yceffort-blog-v2/blob/0fdc55ba753aaa5f41ecebb7e0b9215af67accdb/src/utils/Markdown.ts#L123-L135

```javascript
const imageURL = `/${imgPath}/${imageNode.url.slice(imageIndex)}`

const dimensions = sizeOf(imagePath)

imageNode.type = 'jsx'
imageNode.value = `<Image
  alt={\`${imageNode.alt}\`}
  src={\`${imageURL}\`}
  width={${dimensions.width}}
  height={${dimensions.height}}
/>`
```

이 무렵에 블로그의 대대적인 개편이 있었고, 마크다운을 jsx로 말아주는 과정에서 jsx에 이미지의 정확한 사이즈의 계산하도록 [image-size](https://github.com/image-size/image-size)를 사용하고 있었다. 이 코드는 `getStaticProps`에서 미리 static한 파일을 만드는 과정에서 사용되고 있었고, 이 과정에서 이 코드가 계속해서 실행되면서 이미지 크기만큼 `[...slug].js`의 크기가 커져가고 있었던 것이었다.

그래서 그 과정에서 이미지 크기를 계산하는 대신에, 빌드 이전에 `public/`아래에 있는 모든 이미지의 크기를 다 계산해둔 다음 해당 이미지들의 사이즈 정보가 담긴 파일을 `json`으로 떨궈두고, 여기에 있는 이미지 크기만 가져다 쓰도록 변경했다.

근데 왜 이전 빌드에서는 정상적으로 성공했던 것일까? 그 원인은 vercel 빌드시스템의 버그 때문이었다.

> Since the previous commit that was successful, we have fixed tracing for webpack 5 which would explain why you didn't previously hit the serverless function size since node-file-trace wasn't correctly capturing dependencies with webpack 5 which it is now.

- https://github.com/vercel/nft/pull/186
- https://github.com/vercel/nft/issues/185
- https://github.com/vercel/next.js/issues/23894
- https://github.com/vercel/next.js/issues/23668

추가로 모든 이미지들에 대해서 최적화 작업을 진행했다. [이 포스트](/2021/05/compress-all-images-in-directory)가 나온건 그 때문이었다.

## 마치며

어쨌거나, vercel의 잘못이었기 때문에 (내 잘못도 없는 건 아니지만) vercel 측에서는 지난달 요금을 모두 환불해주고, 나도 next의 팬으로서 digital ocean으로 넘어갔던 블로그를 다시 vercel로 돌려놓았다. 나의 부족한 영어실력 때문인지, 아니면 바빴던 vercel CS 팀의 문제인지는 모르겠지만 문제의 난이도에 비해 해결하는데 너무 오랜 시간이 걸렸다 (거의 3주 가량)

그래도, 외국에 있는 CS 팀과 이야기 해본건 좋은 경험이었다. 그리고 vercel의 [node-file-trace](https://github.com/vercel/nft)에 대해 알게된 것도 좋았다. 이제 여기가 어떻게 돌아가고 있는지 조금이나마 상상해볼 수 있었다. 더불어, digital ocean도 경험해볼 수 있어서 좋았다. 잘 쓸 것 같지는 않지만 =서도.
