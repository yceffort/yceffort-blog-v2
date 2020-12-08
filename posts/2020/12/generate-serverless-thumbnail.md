---
title: '서버리스로 블로그 포스트 썸네일 생성하기'
tags:
  - javascript
  - firebase
  - cloudinary
  - puppeteer
published: true
date: 2020-12-08 23:26:40
description: '어차피 나만 볼거임 ㅋㅅㅋ'
---

사이트를 카카오톡, 페이스북 등 SNS에서 공유할 때 이른바 썸네일이 보이게 하기 위해서는 [open graph tool](https://ogp.me/)를 사용해야 한다. 트위터의 경우에는 자체 정의된 `twitter:*` 시리즈의 무언가를 이용해야 한다. 내 모든 블로그 글에 이쁜 대문 이미지가 있으면 좋겠지만, 모든 것에 현실적으로 이미지를 만드는 것은 불가능하고 귀찮다. 그래서 주어진 이미지에 블로그의 메타 정보를 쓰는 방식으로 정적 이미지를 만든 다음, 이 이미지를 공유용 이미지로 서빙하는 방법에 대해서 고민해 보았다.

## 구성

og tag image 주소 요청 > 해당 주소로 cloudinary에 이미지가 있는지 확인

- 있다면 그 주소로 리다이렉트
- 없다면 > 블로그에 이미지 형태로 준비되어 있는 페이지 방문 > 해당 페이지 스크린샷 > 스크린샷 한 이미지를 cloudinary에 업로드 > 해당 이미지 주소로 리다이렉트

## 1. 정적인 이미지를 만들 페이지 구성하기

`/generate-screenshot`이라는 이름으로 페이지를 하나 만들고, 거기에 정적으로 생성할 이미지를 일단 웹페이지 형태로 만들어보았다.

https://yceffort.kr/generate-screenshot?tags=javascript&title=%EC%9E%90%EB%B0%94%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8%20%ED%95%A8%EC%88%98%EC%9D%98%20%EC%84%B1%EB%8A%A5%20%EC%B8%A1%EC%A0%95%ED%95%98%EA%B8%B0&url=https%3A%2F%2Fyceffort.kr%2F2020%2F12%2Fmeasuring-performance-of-javascript-functions

https://github.com/yceffort/yceffort-blog-v2/blob/master/pages/generate-screenshot.tsx

여기저기 글을 본 결과 최적의 사이즈는 `1200x630`으로 알려져 있으며, 해당 사이즈에 맞게 페이지를 구성하면 된다.

## 2. 해당 페이지를 스크린샷 찍기

이제 해당 페이지를 방문해서 스크린샷을 찍어야 한다. 첫 번째로 시도한 것은 [nextjs에 api를 활용하여 vercel에서 시도하는 것](https://nextjs.org/docs/api-routes/introduction)이었다. 그러나 결과적으로 이 시도는 실패했는데, 일단 스크린샷을 찍기 위해서는 puppetter의 headless chrome instance를 띄워야 하는데 이 메모리가 생각보다 많이 들었다. 그리고 별개의 폰트도 설치해야 하는데 그 과정까지 vercel에서 할 수 없었으므로, [firebase functions](https://firebase.google.com/docs/functions)을 활용하기로 했다.

시작하는 방법은 [여기](https://firebase.google.com/docs/functions/get-started)에 잘 나와있다. 심지어 한글로 되어 있다.

```javascript
exports.screenshot = functions.https.onRequest(async (req, res) => {
  const query = req.query
  const title = encodeURI(query.)
  const firebaseTitle = title.replace(/\//gi, '-')
  const screenshotRef = db.collection('screenshot')

  const exist = await screenshotRef.doc(firebaseTitle).get()

  if (exist.exists) {
    return res.redirect(exist.data().url)
  }

  try {
    const postUrl = `http://yceffort.kr/generate-screenshot?${queryString.stringify(
      query,
    )}`
    const screenshot = await takeScreenshot(postUrl)
    const uploadedImage = await putImage(title, screenshot)
    screenshotRef.doc(firebaseTitle).set({
      url: uploadedImage,
    })
    res.redirect(uploadedImage)
  } catch (e) {
    console.error(e)
    res.json({ error: e.toString() })
  }
})
```

```javascript
const takeScreenshot = async function (url) {
  const chromiumPath = await chromium.executablePath

  const browser = await chromium.puppeteer.launch({
    executablePath: chromiumPath,
    args: chromium.args,
    defaultViewport: chromium.defaultViewport,
    headless: chromium.headless,
  })

  const page = await browser.newPage()
  await page.setViewport({ height: 630, width: 1200 })
  await page.goto(url)
  const buffer = await page.screenshot({ encoding: 'base64' })
  await browser.close()
  return `data:image/png;base64,${buffer}`
}
```

여기서 겪은 삽질을 몇가지 소개해본다.

### 1) puppeteer는 무겁다

puppeteer를 npm install 로 설치해보면 꽤 시간이 걸린다는 것을 알 수 있다. 그래서 `puppeteer`를 `puppeteer-core`만 설치하고, cloud function에서 쓸 수 있는 다른 chromium 브라우저를 알아 봐야 한다. 그래서 https://github.com/alixaxel/chrome-aws-lambda 를 설치했다. 그리고 `iltorb` 도 함께 설치해 주어야 한다.

### 2) react metatag의 query escape

원래는 이 주소를 넘겼다.

https://us-central1-yceffort.cloudfunctions.net/screenshot?slug=2020%2F12%2Fmeasuring-performance-of-javascript-functions&tags=javascript&title=%EC%9E%90%EB%B0%94%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8%20%ED%95%A8%EC%88%98%EC%9D%98%20%EC%84%B1%EB%8A%A5%20%EC%B8%A1%EC%A0%95%ED%95%98%EA%B8%B0&url=https%3A%2F%2Fyceffort.kr%2F2020%2F12%2Fmeasuring-performance-of-javascript-functions

그러나 소스 보기로 해당 주소를 보면 아래와 같이 되어 있었다.

```html
<meta
  property="og:image"
  content="https://us-central1-yceffort.cloudfunctions.net/screenshot?slug=2020%2F12%2Fmeasuring-performance-of-javascript-functions&amp;tags=javascript&amp;title=%EC%9E%90%EB%B0%94%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8%20%ED%95%A8%EC%88%98%EC%9D%98%20%EC%84%B1%EB%8A%A5%20%EC%B8%A1%EC%A0%95%ED%95%98%EA%B8%B0&amp;url=https%3A%2F%2Fyceffort.kr%2F2020%2F12%2Fmeasuring-performance-of-javascript-functions"
/>
```

`&`가 `&amp;`로 escape 처리 되어 있는 것이다.

- https://github.com/facebook/react/issues/13838
- https://github.com/vercel/next.js/issues/2006

두가지 선택이 있었는데, query param 구조로 되어 있는 주소를 path variable 로 모두 바꾸거나, 혹은 받는 쪽에서 처리를 하는 것이다. path를 다 바꾸기는 넘 귀찮아서 아래와 같은 처리를 추가해주었다.

```javascript
const query = Object.keys(context.query).map(
    (key) => (query[key.replace(/amp;/, '')] = context.query[key]),
  ) as any
```

### 3) 느린 속도

스크린샷을 찍고, 이미지를 업로드 해서 내려주는 최초의 과정은 느릴 수 밖에 없다.그러나 문제는 두번째 과정 이후 부터 있었다. 이미 이미지가 생성되었는지 확인하기 위해 cloudinary에 get 요청을 날리는데, 이 과정 또한 쓸데 없이 오래 걸렸다. 그래서 cloudinary에 찔러서 확인하는대신, 한번 생성된 이미지는 key와 value 형태로 주소를 firebase에 저장해 두어 더 빠르게 내려주었다.

저장

```javascript
const uploadedImage = await putImage(title, screenshot)
screenshotRef.doc(firebaseTitle).set({
  url: uploadedImage,
})
```

불러오기

```javascript
const exist = await screenshotRef.doc(firebaseTitle).get()

if (exist.exists) {
  return res.redirect(exist.data().url)
}
```

그렇다고 해서 속도문제가 완전히 해결된 것은 아니었다. 첫단계에서는 여전히 생성속도가 느리고, 주소에서 느껴지겠지만, 미국 동부를 거쳐서 왔다리 갔다리 해야 하기 때문에 여전히 좀 답답한면이 있다.

## 3. 메타 태그에 심기

```html
<meta
  property="og:image"
  content="https://us-central1-yceffort.cloudfunctions.net/screenshot?slug=2020%2F12%2Fmeasuring-performance-of-javascript-functions&amp;tags=javascript&amp;title=%EC%9E%90%EB%B0%94%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8%20%ED%95%A8%EC%88%98%EC%9D%98%20%EC%84%B1%EB%8A%A5%20%EC%B8%A1%EC%A0%95%ED%95%98%EA%B8%B0&amp;url=https%3A%2F%2Fyceffort.kr%2F2020%2F12%2Fmeasuring-performance-of-javascript-functions"
/>
```

```html
<meta
  name="twitter:image"
  content="https://us-central1-yceffort.cloudfunctions.net/screenshot?slug=2020%2F12%2Fmeasuring-performance-of-javascript-functions&amp;tags=javascript&amp;title=%EC%9E%90%EB%B0%94%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8%20%ED%95%A8%EC%88%98%EC%9D%98%20%EC%84%B1%EB%8A%A5%20%EC%B8%A1%EC%A0%95%ED%95%98%EA%B8%B0&amp;url=https%3A%2F%2Fyceffort.kr%2F2020%2F12%2Fmeasuring-performance-of-javascript-functions"
/>
```

## 결과

![preview1](./images/metadata-preview1.png)

![preview2](./images/metadata-preview2.png)

## 문제점

- 여전히 좀 느리다. 당연히, 초기 생성단계에서는 느릴 수 밖에 없다. 이것을 어떻게 해결할 것인가가 관건이다. 글이 올라간 뒤에, github action으로 트리거 해서 생성할 것인가? 혹은 배포 단계에 이를 포함할 것인가?
- firebase functions, firebase storage, 거기에 cloudinary까지 사용하고 있다. 코로나 시대에 줄어든 용돈으로, 과연 여기까지 커버할 수 있을 것인가? vercel은 언제 또 나에게 pro 버전으로 내 지갑을 재차 노릴 것인가?

참으로 무시무시한 일이 아닐 수 없다.
