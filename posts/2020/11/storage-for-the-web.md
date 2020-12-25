---
title: '웹에서 사용 가능한 스토리지 살펴보기'
tags:
  - web
  - browser
published: true
date: 2020-11-23 23:13:53
description: 'PWA에서 가장 적절한 것은 무엇일까'
---

인터넷 연결은 시시때때로 끊길 수 있고 불안정하므로, PWA에서는 오프라인 환경과 신뢰할 수 있는 성능을 안정적으로 제공하는 것이 필수다. 또한 완벽히 온라인 환경이 제공된다 하더라도, 캐싱과 다른 스토리지 기술을 적절히 활용한다면 사용자의 경험을 향상 시킬 수 있다. 정적 애플리케이션 리소스(HTML, Javascript, CSS)와 데이터 (사용자 데이터, 뉴스 기사 등)를 캐싱하는 방법이 꽤 있다. 어떤 것이 가장 좋은 해결책이며, 이들은 얼마나 저장할 수 있을까?

## 무엇을 사용해야 하는가?

- 네트워크 리소스나 파일 기반의 콘텐츠가 필수적일 때는, [Cache Storage API](https://developer.mozilla.org/en-US/docs/Web/API/CacheStorage)를 사용하는 것이 좋다.
- 다른 데이터의 경우에는, [IndexedDB](https://developer.mozilla.org/ko/docs/Web/API/IndexedDB_API) 를 사용하는 것이 좋다.

이 두 방식은 모두 모던 브라우저에서 지원한다 (IE 제외). 그리고 둘다 비동기로 이루어지며 메인스레드를 블로킹하지 않는다. 또한 `window`, 웹 워커, 서비스 워커에서 접근 가능 하기 때문에 어디서든 사용하기 쉽다.

## 다른 스토리지는?

물론 이 외에도 브라우저에서 사용할 수 있는 다른 스토리지가 존재한다. 그러나 이들은 사용에 제한이 있으며, 성능적인 문제 또한 존재한다.

- [SessionStorage](https://developer.mozilla.org/ko/docs/Web/API/Window/sessionStorage): 데이터가 탭과 연결되어 있기 때문에, 탭의 라이프타임과 함께 한다. 따라서 세션과 관련된 작은 양의 데이터를 저장하는데 유용하다. (IndexedDB의 키 등) 또한 동기로 작동하며 메인스레드를 블로킹하기 때문에 사용에 주의가 필요하다. 5MB 까지의 데이터만 저장가능하며, 문자열만 가능하다. 또한 탭에 묶여 있기 때문에, 웹워커나 서비스워커에서는 사용이 불가능하다.
- [LocalStorage](https://developer.mozilla.org/ko/docs/Web/API/Window/localStorage): 마찬가지로 메인스레드를 블로킹하고 동기로 작동한다. 마찬가지로 5MB까지 가능하며, 문자열 데이터만 저장가능하다. 웹 워커와 서비스워커에서는 사용이 불가능하다.
- [Cookies](https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies): 쿠키는 쿠키 나름대로의 사용처가 있기 때문에, 데이터 저장용도로 사용해서는 안된다. 쿠키는 모든 HTTP 요청에 함께 보내지기 때문에, 큰 데이터를 저장했다가는 모든 HTTP 요청에 함께 날라가게 되므로 신 사이즈가 커지게 된다. 또한 이들은 동기로 작동하며, 웹워커에서는 접근이 불가능하다. 위 두개와 마찬가지로, 문자열 데이터만 저장 가능하다.
- [File System API](https://developer.mozilla.org/en-US/docs/Web/API/File_and_Directory_Entries_API/Introduction): 샌드박스 (제한된 영역의) 파일시스템에 파일을 읽고 쓸 수 있도록 도와준다. 비동기로 작동되는 반면 [크로미윰에서만 지원되기 때문에](https://caniuse.com/filesystem) 널리 사용하기는 어렵다.
- [File System Access API](https://web.dev/file-system-access/): 사용자가 로컬 파일 시스템에 파일을 읽고 쓰기 쉽게 만들어주는 API다. 따라서 사용자에게 권한을 획득하는 것이 필수 이며, 또한 이 권한은 세션을 넘어가면 유지 되지 않는다.
- WEB SQL: IndexedDB의 등장과 함께 사라진 기능으로, 사용해서는 안된다.
- [Application Cache](https://developer.mozilla.org/ko/docs/Web/HTML/Using_the_application_cache): 이 또한 deprecated 되었다. 브라우저에서 지원이 중단될 예정이며, 서비스워커와 Cache API로 대체 해야 한다.

## 얼마나 저장할 수 있는가?

몇 백 메가바이트, 그리고 잠재적으로 수백 기가바이트 이상이 될 수도 있다. 브라우저 별로 다를 수 있지만, 사용가능한 스토리지의 크기는 대개 장치에서 사용 가능한 스토리지의 크기에 따라서 결정된다.

- 크롬에서는 원래 전체 디스크 공간의 60%까지 허용해주었지만, 이제는 80% 까지 가능하다. StorageManger API를 이용해서 사용가능한 스토리지의 크기를 점검할 수 있다. 다른 크로미윰 기반 브라우저의 경우 더 많은 스토리지를 사용할 수도 있다. [여기](https://github.com/GoogleChrome/web.dev/pull/3896)를 참고
- IE 10 이상의 브라우저에서는 250mb까지 가능하며, 10mb이상의 스토리지를 사용할 경우 사용자에게 메시지를 띄운다.
- 파이어폭스는 여유 디스크공간의 50%까지 사용하게 해준다. `eTLD+1` (effective Top Level Domain) 의 경우에는 최대 2GB까지 가능하다. StorageManager API를 통해서 얼마나 사용가능한지 확인할 수 있다.
- 사파리는 1gb까지 지원하는 것으로 보인다. 최대 스토리지에 도달하면, 사용자에게 메시지를 띄우고 200mb 를 추가로 사용할 수 있게 해준다. (이에 대한 정확한 공식문서는 찾지 못했으므로, 약간의 차이가 있을수 있다.)

과거 스토리지 용량 최대치에 근접하게 되면, 브라우저는 사용자에게 메시지를 띄워서 권한을 획득한 후, 추가로 스토리지용량을 제공했다. 그러나 요즘 모든 브라우저는 사용자에게 특별히 메시지를 띄우지 않고 최대 사용량 까지 사용하게 해준다. 사파리의 경우는 조금 다르다. 앞서 이야기 한것처럼 추가 할당량 사용여부를 유저에게 묻고 허락할 경우 사용하게 해주지만, 그 이상은 불가능 한 것으로 보인다.

## 사용가능한 스토리지 용량 확인하는 방법

[Storage Manager API](https://caniuse.com/mdn-api_storagemanager)를 통애서 확인 가능하다.

```javascript
if (navigator.storage && navigator.storage.estimate) {
  const quota = await navigator.storage.estimate()
  const percentageUsed = (quota.usage / quota.quota) * 100
  const remaining = quota.quota - quota.usage

  console.table(quota)
}
```

| index        | value        | caches    | indexedDB | serviceWorkerRegistrations |
| ------------ | ------------ | --------- | --------- | -------------------------- |
| quota        | 299977904946 |           |           |                            |
| usage        | 621244075    |           |           |                            |
| usageDetails |              | 620952837 | 256156    | 35082                      |

Storage Manager API가 모든 브라우저에서 사용가능한 것은 아니므로, feature detect를 꼭 걸어줘야 한다. 또한 quota를 넘는 경우 에러가 발생할 수도 있으므로, `try.. catch`로 이를 잡아 주는 처리를 해야 한다.

[예제 사이트 확인해보기](https://storage-quota.glitch.me/)

## 사용량 초과시 대처

중요한 점은 코드를 작성할 시에 `QuotaExceededError`와 같은 에러에 대해 염두해 두어야 한다는 것이다. IndexedDB 와 Cache API 모두 사용량 초과시에 `DOMError`나 `QuotaExceededError`를 던진다.

### IndexedDB

데이터 사용량을 초과한다면, `IndexedDB`에 쓰려는 시도는 모두 실패한다. 트랜색션의 `onabort()`가 호출된다. 여기에는 `DOMException`이 포함된다. 에러의 `name`을 확인하면 `QuotaExceededError`가 보일 것이다.

```javascript
const transaction = idb.transaction(['entries'], 'readwrite')
transaction.onabort = function (event) {
  const error = event.target.error // DOMException
  if (error.name == 'QuotaExceededError') {
    // Fallback code goes here
  }
}
```

### Cache API

```javascript
try {
  const cache = await caches.open('my-cache')
  await cache.add(new Request('/sample1.jpg'))
} catch (err) {
  if (error.name === 'QuotaExceededError') {
    // Fallback code goes here
  }
}
```

## Eviction

> 최대 용량 초과로 인해 데이터가 지워지는 것을 의미한다.

웹 스토리지는 `Best Effort`와 `Persistent` 두개의 버켓으로 구분할 수 있다. `Best Effort`란 스토리지가 사용자를 방해하지 않고 브라우저에 의해 정리될 수 있다는 것을 의미하는데, 이는 중요한 데이터에 적합하지 않다는 것을 의미한다. `Persistent` 스토리지는 저장가용량이 낮더라도 자동으로 삭제 되지는 않는다. 사용자가 수동으로 데이터를 삭제 해야 한다.

기본적으로, 사이트의 데이터는 `Best Effort`로 분류되어 사이트가 별도로 [persistent storage를 요청](https://web.dev/persistent-storage/)하지 않는다면, 가용량이 낮아지게 되면 자동으로 데이터를 삭제하게 된다.

`best effort`내에서 eviction 정책은 아래와 같다.

- 크로미윰 기반 브라우저는 브라우저의 저장공간이 부족할 때 데이터를 제거하며, 브라우저의 저장공간이 초과되지 않을 때 까지 가장 오래된 데이터부터 삭제하기 시작한다.
- IE 10이상에서는 데이터를 제거되지는 않지만, 더이상 데이터를 쓰는 것이 불가능해진다.
- 파이어 폭스는 크로미윰과 마찬가지로 저장공간이 초과되지 않을 때까지 오래된 데이터부터 삭제한다.
- 사파리의 경우 이전 버전에서는 데이터를 제거하지 않았는데, 최근에는 스토리지에 대해 7일짜리 제한을 두기 시작했다.

> iOS, iPadOS 13.4, 맥의 safari 13.1 부터, Cache API, indexed DB, storage 등의 쓰기 스토리지에 대해서 7일간의 제한을 걸어두기 시작했다. 이는 사용자가 사이트에서 인터랙션을 하지 않을 경우, 사파리가 7일 이후에는 캐시에서 모든 데이터를 제거한다는 것을 의미한다. 이 정책은 홈스크린에 설치된 PWA에는 적용되지 않는다. 자세한 내용은 [여기](https://webkit.org/blog/10218/full-third-party-cookie-blocking-and-more/)를 참조!

## 왜 indexedDB 래퍼를 사용해야 할까?

IndexedDB는 저수준 API로, 아주 작은 양의 데이터를 저장하는데 사용한다 할지라도 처음 설치에 있어 많은 심혈을 기울여야 한다. 다른 모든 promise 기반 API와는 다르게, 이벤트 베이스로 작동된다. [idb](https://github.com/jakearchibald/idb)를 사용하면 일부 강력한 기능을 사용할수 없지만 트랜잭션과 스키바 버전과 같은 복잡한 기능을 사용하지 않더라도 promise 기반의 indexeddb를 사용할 수 있게 해준다.

출처: https://web.dev/storage-for-the-web/
