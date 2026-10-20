# PWA 전용 오프라인 저장 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 오프라인 저장을 설치된 PWA에서만 하게 바꾸고, PWA 전용 "저장한 글" 페이지와 최근 글 자동 저장을 추가한다.

**Architecture:** 서비스 워커(`apps/blog/public/sw.js`)는 요청을 가로챌 때 정적 자산과 폰트 외에는 캐시에 쓰지 않고 읽기(오프라인 폴백)만 한다. 글과 이미지 저장은 페이지 쪽 TS 모듈(`apps/blog/src/utils/offline/`)이 standalone일 때만 window의 `caches`에 직접 쓴다. 메타데이터(자동 꼬리표, ✕ 기억, 설정)는 localStorage에 두고, 목록의 기준은 실제 `pages-v4` 캐시 내용이다.

**Tech Stack:** Next.js 16 App Router, React 19, StyleX, Cache Storage API, vitest 5 (설정 파일 없음, `@/` 별칭이 안 풀리므로 모듈과 테스트는 상대 경로 import).

**Spec:** `docs/superpowers/specs/2026-10-19-pwa-saved-posts-design.md`

## Global Constraints

- 캐시 이름은 `sw.js`와 같은 `pages-v4`, `images-v4`. `CACHE_VERSION`은 올리지 않는다. 새 캐시 이름을 만들지 않는다(워커 activate가 `ALL_CACHES` 밖의 캐시를 지운다).
- 페이지 캐시 상한 200, 이미지 캐시 상한 300, 오래된 것부터 정리하고 `/offline`은 정리 대상에서 뺀다.
- 자동 저장 선택지는 0(끄기)/5/10/20, 기본 5. 본문 이미지 스위치 기본 켜짐(2026-10-20 사용자 지시). 같은 스위치로 `<video poster>`와 같은 오리진 `.html` iframe 문서(`pages-v4`)와 그 문서가 참조하는 이미지도 저장한다. 크기 상한 없음(2026-10-20 사용자 결정).
- 저장 크기는 저장 시점에 재서 메타(`sizes`)에 기록한다(HTML + 같은 오리진 데모 문서 + 문서 이미지 + 글 이미지, 공유 자산도 글마다 셈). 외부 opaque 이미지는 바이트 대신 장수로 표시. 행, 요약 줄, 설정 아래 "한 편당 평균, 20편이면 약" 추정 표시. 크기 기록이 없는 자동 저장분은 다음 동기화 때 다시 받아 잰다(2026-10-20 사용자 결정).
- 저장은 standalone(`matchMedia('(display-mode: standalone)').matches || navigator.standalone === true`)일 때만.
- 브라우저 독자에게 글 HTML이나 이미지를 위한 추가 요청이 나가면 안 된다.
- 웹 푸시(`push`, `notificationclick` 핸들러, 등록)는 건드리지 않는다.
- 문서, 주석, 커밋 메시지에 em dash, en dash, 가운뎃점을 쓰지 않는다.
- 커밋은 용찬님이 요청했을 때만 한다. 메시지는 기존 컨벤션(gitmoji + 영어 한 줄, 예: `🔒 Block unpublished decks in production`).

## Review Focus

1. 오프라인 상태로 PWA를 켜서 최근 글 목록 요청이 실패하면 자동 저장분을 하나도 지우지 않아야 한다. (Task 2 테스트 `keeps auto entries when the recent list fetch fails`)
2. 데이터 절약 모드(`saveData`)에서는 자동 저장 요청이 0건이고 기존 자동분도 지우지 않아야 한다. (Task 2 `does nothing for auto entries when saveData is on`)
3. v4가 남긴 옛 캐시 엔트리(메타데이터 없음, 키가 `Request`)가 "읽음"으로 보이고, 제목은 HTML의 첫 `<title>`이어야 한다. 글 HTML에는 SVG의 `<title>Mail icon</title>`이 뒤에 또 있다. (Task 2 `lists legacy entries as read with the first title`)
4. 리다이렉트된 응답(`/`가 쿠키 때문에 `/en`으로 가는 경우, 번역 없는 영문 글의 308)은 저장하지 않아야 한다. 저장하면 내비게이션에 redirected 응답을 주게 돼 오프라인에서 열리지 않는다. (Task 2 `does not store redirected responses`)
5. ✕로 지운 글은 최근 N편 안에 있어도 다시 받지 않고, 전체 삭제 뒤에는 다시 받는다. (Task 2 `skips excluded posts` / `clearAllSaved resets exclusions`)

---

### Task 1: 최근 글 JSON 엔드포인트

**Files:**

- Create: `apps/blog/src/utils/recentPosts.ts`
- Create: `apps/blog/src/app/api/recent-posts/route.ts`
- Create: `apps/blog/src/app/api/recent-posts/en/route.ts`

**Interfaces:**

- Produces: `GET /api/recent-posts`, `GET /api/recent-posts/en` → `{"posts": [{"path": string, "title": string}]}`. 최신순 20개. `path`는 한국어 `/2026/10/web-vitals-report-timing`, 영어 `/en/2026/10/web-vitals-report-timing`. `title`은 `stripTitleEmphasis` 적용.
- Produces: `export type RecentPost = {path: string; title: string}` (Task 2가 같은 모양을 자체 정의해도 된다. 서버 모듈을 클라이언트가 import하지 않게 한다)

- [ ] **Step 1: `buildRecentPostsResponse(locale: Locale): Promise<Response>`를 `recentPosts.ts`에 구현**

`apps/blog/src/utils/feed.ts`의 `buildFeedResponse`와 같은 방식(`getAllPosts(locale)`, `pathPrefixOf(locale)`, `post.fields.slug`). 요청 시점 IO(`new Date()`, searchParams) 금지. `NextResponse.json`.

- [ ] **Step 2: 두 라우트를 `apps/blog/src/app/api/llms-full/route.ts`, `.../en/route.ts`와 같은 모양으로 작성**

- [ ] **Step 3: 빌드해서 정적 라우트인지 확인**

Run: `cd /Users/yceffort/private/blog && pnpm build:blog 2>&1 | grep recent-posts`
Expected: 두 경로 모두 정적(○) 표시.

- [ ] **Step 4: 응답 확인**

Run: `cd apps/blog && pnpm start` (포트 3000이 다른 프로세스에 잡혀 있지 않은지 `lsof -iTCP:3000 -sTCP:LISTEN`으로 먼저 확인) 후 `curl -s localhost:3000/api/recent-posts | node -e "const p=JSON.parse(require('fs').readFileSync(0)).posts;console.log(p.length,p[0])"`
Expected: `20 { path: '/2026/10/...', title: '...' }`, 첫 항목이 가장 최근 발행 글이고 제목에 `<em>`이 없다. `/api/recent-posts/en`은 `path`가 `/en/`으로 시작한다.

- [ ] **Step 5: 커밋 (요청 시)** `✨ Add recent posts JSON for offline auto-save`

### Task 2: 오프라인 저장 모듈

**Files:**

- Create: `apps/blog/src/utils/offline/paths.ts`
- Create: `apps/blog/src/utils/offline/meta.ts`
- Create: `apps/blog/src/utils/offline/cache.ts`
- Create: `apps/blog/src/utils/offline/sync.ts`
- Test: `apps/blog/src/utils/offline/__tests__/fakeCaches.ts` (테스트용 CacheStorage 대역)
- Test: `apps/blog/src/utils/offline/__tests__/cache.test.ts`
- Test: `apps/blog/src/utils/offline/__tests__/sync.test.ts`
- Modify: `apps/blog/package.json` (scripts에 `"test:offline": "vitest run src/utils/offline"`)

**Interfaces:**

- Consumes: Task 1의 `/api/recent-posts`, `/api/recent-posts/en`.
- Produces (Task 3, 4가 사용):
  - `paths.ts`: `isPostPath(pathname: string): boolean` (`/^(\/en)?\/\d{4}\/\d{2}\/[^/.]+$/`), `localeOfPath(pathname: string): 'ko' | 'en'`, `homePathOf(locale)`(`/`, `/en`), `savedPathOf(locale)`(`/saved`, `/en/saved`).
  - `meta.ts`: `type SavedMeta = {count: 0 | 5 | 10 | 20; images: boolean; locale: 'ko' | 'en'; auto: string[]; excluded: string[]; lastSync: number | null; announced: boolean}`, `readMeta(storage: Storage): SavedMeta`(키 `offline-saved-meta`, 없거나 파싱 실패면 기본값 `{count: 5, images: false, locale: 'ko', auto: [], excluded: [], lastSync: null, announced: false}`), `writeMeta(storage: Storage, meta: SavedMeta): void`. localStorage 접근은 try/catch.
  - `type OfflineDeps = {caches: CacheStorage; fetch: typeof fetch; storage: Storage; origin: string; saveData: boolean; now: () => number}`는 `cache.ts`에 둔다. `browserDeps(): OfflineDeps`는 `sync.ts`에 둔다(`saveData`는 `navigator.connection?.saveData === true`). `sync.ts`가 `cache.ts`를 import하고 반대 방향 import는 없다.
  - `cache.ts`: `extractImageUrls(html: string): string[]`, `savePage(path: string, options: {images: boolean}, deps: OfflineDeps): Promise<'new' | 'updated' | 'failed'>`, `deletePage(path: string, deps): Promise<void>`, `listSavedPosts(deps): Promise<Array<{path: string; title: string}>>`, `clearSavedPosts(deps): Promise<void>`.
  - `sync.ts`: `recordRead(pathname: string, deps): Promise<boolean>`(새로 저장됐으면 true), `syncRecent(deps): Promise<number>`(이번에 새로 받은 자동분 수), `syncOnLaunch(pathname: string, deps): Promise<{savedCurrent: boolean; announce: number | null}>`(`announce`는 첫 자동 저장 안내를 띄울 때 그 편수), `deleteSaved(path: string, deps): Promise<void>`, `clearAllSaved(deps): Promise<void>`, `updateSettings(patch: Partial<Pick<SavedMeta, 'count' | 'images'>>, deps): SavedMeta`.

- [ ] **Step 1: `fakeCaches.ts` 작성**

`createFakeCaches(): CacheStorage`. `open(name)`은 같은 이름이면 같은 인스턴스. 인스턴스는 `match(req)`, `put(req, res)`, `delete(req)`, `keys()`만 구현하고 키는 `new Request(req).url`로 비교한다(query 포함). `put`은 기존 키를 지우고 맨 뒤에 넣어 실제 Cache API처럼 삽입 순서를 유지한다. `keys()`는 `Request[]`. `caches.delete(name)`도 구현. 응답은 `res.clone()`을 저장하고 `match`도 clone을 돌려준다.

- [ ] **Step 2: `cache.test.ts` 작성 (실패 확인용)**

```ts
// deps 헬퍼: origin 'https://yceffort.kr', fetch는 경로별 Response를 돌려주는 vi.fn
it('extracts one srcSet variant per source, closest to 1080', () => {
  const html =
    '<img srcSet="/_next/image?url=%2Fa.png&amp;w=640&amp;q=75 640w, /_next/image?url=%2Fa.png&amp;w=1080&amp;q=75 1080w" src="/_next/image?url=%2Fa.png&amp;w=3840&amp;q=75">'
  expect(extractImageUrls(html)).toEqual([
    '/_next/image?url=%2Fa.png&w=1080&q=75',
  ])
})
it('stores a post as new, then updated', async () => {
  expect(await savePage('/2026/10/a', {images: false}, deps)).toBe('new')
  expect(await savePage('/2026/10/a', {images: false}, deps)).toBe('updated')
  expect(
    await (
      await deps.caches.open('pages-v4')
    ).match('https://yceffort.kr/2026/10/a'),
  ).toBeDefined()
})
it('does not store redirected responses', async () => {
  // fetch가 redirected: true인 Response를 돌려준다 (Object.defineProperty로 redirected 설정)
  expect(await savePage('/', {images: false}, deps)).toBe('failed')
})
it('does not store non-ok responses', async () => {
  /* 404 → 'failed', 캐시 비어 있음 */
})
it('saves images only when images is true', async () => {
  /* images:false면 images-v4 비어 있음, true면 HTML의 img가 저장됨 */
})
it('keeps /offline when trimming pages over 200', async () => {
  // pages-v4에 /offline + 글 200개를 넣은 뒤 savePage 1회 → keys 201개(/offline 포함), 가장 오래된 글이 빠짐
})
it('lists legacy entries as read with the first title', async () => {
  // pages-v4에 new Request('https://yceffort.kr/2026/09/old')로 HTML '<title>옛 글</title>...<svg><title>Mail icon</title></svg>' 저장
  expect(await listSavedPosts(deps)).toEqual([
    {path: '/2026/09/old', title: '옛 글'},
  ])
})
it('lists only post paths, newest saved first', async () => {
  // '/', '/en', '/offline', '/saved', '/pages/2', '/tags/react', '/2026/10/a?x=1', '/2026/10/a', '/2026/10/b' 순서로 저장
  expect((await listSavedPosts(deps)).map((p) => p.path)).toEqual([
    '/2026/10/b',
    '/2026/10/a',
  ])
})
```

- [ ] **Step 3: 실패 확인**

Run: `cd apps/blog && pnpm test:offline`
Expected: FAIL (모듈 없음)

- [ ] **Step 4: `paths.ts`, `meta.ts`, `cache.ts` 구현**

`extractImageUrls`는 `sw.js`의 같은 이름 함수를 타입만 붙여 옮긴다(동작 변경 금지, `i` 플래그 포함). `savePage`는 `sw.js`의 `savePageHTML` + `saveImagesFromHTML` + `putWithTrim`의 동작을 따른다: 키는 `new URL(path, deps.origin).href`, `!response.ok || response.redirected`면 `'failed'`, 이미지는 최대 50장이고 외부(`https://`)는 `{mode: 'no-cors'}`, `ok || type === 'opaque'`만 저장, 이미 캐시에 있으면 건너뛴다. 개별 이미지 실패는 무시. `listSavedPosts`는 `keys()` 역순에서 `isPostPath(new URL(key.url).pathname)`이고 query가 없는 것만, 제목은 `/<title>([^<]*)<\/title>/`의 첫 매치(없으면 경로).

- [ ] **Step 5: `sync.test.ts` 작성 (실패 확인용)**

```ts
// 최근 목록 fetch는 '/api/recent-posts'에 a..f 6편(/2026/10/a 가 최신)을 돌려준다
it('saves the newest count posts as auto', async () => {
  expect(await syncRecent(deps)).toBe(5)
  expect(readMeta(deps.storage).auto).toEqual([
    '/2026/10/a',
    '/2026/10/b',
    '/2026/10/c',
    '/2026/10/d',
    '/2026/10/e',
  ])
})
it('does not tag an already read post as auto', async () => {
  /* recordRead('/2026/10/a') 후 syncRecent → auto에 a 없음, 받은 수 4 */
})
it('removes auto entries pushed out of the newest count', async () => {
  // auto=['/2026/09/z'] 이고 z가 캐시에 있음 → syncRecent 후 z가 캐시와 auto에서 빠짐
})
it('keeps read entries pushed out of the newest count', async () => {
  /* 읽은 '/2026/09/z'는 남음 */
})
it('removes all auto entries when count is 0 without fetching the list', async () => {})
it('keeps auto entries when the recent list fetch fails', async () => {
  /* fetch reject → auto와 캐시 그대로, 반환 0 */
})
it('does nothing for auto entries when saveData is on', async () => {
  /* fetch 호출 0회, auto 그대로 */
})
it('removes auto entries of the previous locale after a locale switch', async () => {
  // ko 자동분이 있는 상태에서 syncOnLaunch('/en/2026/10/a-en') → '/api/recent-posts/en'을 요청하고 ko 자동분은 지워짐
})
it('turns an auto post into read when opened', async () => {
  /* recordRead 후 auto에서 빠지고 캐시에는 남음 */
})
it('skips excluded posts', async () => {
  /* deleteSaved('/2026/10/a') 후 syncRecent → a 다시 안 받음, 받은 수 0(b..e는 이미 있음) */
})
it('clearAllSaved resets exclusions', async () => {
  /* clearAllSaved 후 syncRecent → a 포함 5편 다시 받음, images-v4도 비워짐 */
})
it('announces only the first auto save', async () => {
  // 첫 syncOnLaunch → announce 5, 다음 syncOnLaunch → announce null
})
it('saves home and saved page on launch but does not list them', async () => {
  // syncOnLaunch('/') 후 pages-v4에 '/'와 '/saved'가 있고 listSavedPosts에는 없음
})
```

- [ ] **Step 6: 실패 확인** Run: `pnpm test:offline` Expected: sync 테스트 FAIL

- [ ] **Step 7: `sync.ts` 구현**

`syncRecent` 알고리즘 (스펙의 자동 저장 규칙. 실패하면 아무것도 지우지 않는 것이 핵심):

```ts
const meta = readMeta(deps.storage)
if (deps.saveData) return 0
let target: string[] = []
if (meta.count > 0) {
  const list = await fetchRecent(meta.locale, deps) // 실패하면 null
  if (!list) return 0
  target = list
    .slice(0, meta.count)
    .map((p) => p.path)
    .filter((p) => !meta.excluded.includes(p))
}
for (const path of meta.auto.filter((p) => !target.includes(p)))
  await deletePage(path, deps)
// target 중 캐시에 없는 것만 savePage(path, {images: meta.images}) → 'new'면 auto에 추가
// meta.auto = (target ∩ 기존 auto) ∪ 새로 받은 것, lastSync = deps.now(), writeMeta
```

`syncOnLaunch`: `meta.locale = localeOfPath(pathname)` 저장 → `recordRead(pathname)` → `savePage(homePathOf(locale), {images: true})`, `savePage(savedPathOf(locale), {images: false})` → `syncRecent` → 받은 수가 1 이상이고 `!meta.announced`면 `announced = true`로 쓰고 `announce`에 그 수. `deleteSaved`는 캐시 삭제 + `excluded`에 추가 + `auto`에서 제거. `clearAllSaved`는 `clearSavedPosts`(목록에 보이는 글 엔트리 삭제 + `images-v4` 캐시 통째 삭제) 후 `auto`, `excluded` 비우기.

- [ ] **Step 8: 통과 확인** Run: `pnpm test:offline` Expected: 전부 PASS

- [ ] **Step 9: 타입체크와 린트** Run: `cd /Users/yceffort/private/blog && cd apps/blog && npx tsc --noEmit --incremental false && cd ../.. && pnpm lint` Expected: 새 오류 0건

- [ ] **Step 10: 커밋 (요청 시)** `✨ Add PWA offline saving module`

### Task 3: 서비스 워커를 읽기 전용으로 바꾸기

**Files:**

- Modify: `apps/blog/public/sw.js`
- Modify: `apps/blog/src/app/offline/page.tsx` (문구)

**Interfaces:**

- Produces: 워커는 `pages-v4`, `images-v4`, `rsc-v4`를 읽기만 한다. 페이지로 `postMessage`를 보내지 않는다.

- [ ] **Step 1: 쓰기 경로 제거**

`PRECACHE_URLS = [OFFLINE_URL]`, install은 `cache.addAll(PRECACHE_URLS)` 후 `skipWaiting`만. `networkFirst`는 성공 응답을 저장하지 않는다(실패 시 캐시 폴백, 내비게이션이면 `/offline`은 그대로). `handleNavigation`은 `networkFirst` 결과만 반환. `handleRSC`는 `savePageHTML`과 RSC 캐시 쓰기를 빼고, 실패 시 캐시 폴백과 503은 유지. `handleImage`는 쓰기를 빼고 캐시 읽기, 네트워크, `matchImageVariant` 폴백은 유지. `cacheFirst`(정적 자산, 폰트)는 그대로.

- [ ] **Step 2: 이 변경으로 안 쓰이게 된 것만 삭제**

`savePageHTML`, `saveImagesFromHTML`, `extractImageUrls`(Task 2로 이동), `isPrefetchRequest`, `PRECACHE_HREFS`와 `putWithTrim`의 그 필터, `MAX_ENTRIES`의 pages/images/rsc 항목. 관련 주석 중 사실이 아니게 된 것만 고친다.

- [ ] **Step 3: `/offline` 문구 교체**

`인터넷 연결이 끊어졌습니다. 이전에 방문한 페이지는 오프라인에서도 읽을 수 있습니다.` → `인터넷 연결이 끊어졌습니다. 블로그를 앱으로 설치하면 읽은 글과 최근 글을 오프라인에서도 읽을 수 있습니다.` (구현 후 용찬님 확인)

- [ ] **Step 4: 문법 확인** Run: `node --check apps/blog/public/sw.js && grep -nE "savePageHTML|saveImagesFromHTML|postMessage" apps/blog/public/sw.js` Expected: 출력 없음(grep 0건)

- [ ] **Step 5: 커밋 (요청 시)** `🔥 Stop offline page saving in the service worker fetch path`

### Task 4: PWA 실행 시 저장과 토스트

**Files:**

- Create: `apps/blog/src/hooks/useIsStandalone.ts`
- Modify: `apps/blog/src/components/pwa/ServiceWorkerRegistration.tsx`

**Interfaces:**

- Consumes: Task 2의 `browserDeps`, `syncOnLaunch`, `recordRead`.
- Produces: `useIsStandalone(): boolean` (`useSyncExternalStore`, 서버 스냅샷 `false`, `display-mode: standalone` 미디어 쿼리의 change 구독, iOS `navigator.standalone`). Task 5가 사용.

- [ ] **Step 1: `useIsStandalone` 구현**

- [ ] **Step 2: `ServiceWorkerRegistration` 수정**

워커 등록은 그대로. `message` 리스너와 `startMessages()`는 워커가 더 이상 메시지를 안 보내므로 제거. 마운트 시 standalone이면 `syncOnLaunch(location.pathname, browserDeps())`, 이후 `usePathname()` 변화마다 `recordRead`. 토스트 상태를 경로 대신 문구 문자열로 바꾼다. 문구(정확히): 읽은 글 첫 저장 `✓ 오프라인에 저장됨` / `✓ Saved for offline`(기존), 첫 자동 저장 `✓ 최근 글 ${n}편을 오프라인에 저장했습니다` / `✓ Saved ${n} recent posts for offline`. 같은 실행에서 둘 다면 자동 저장 문구 하나만. 언어는 현재 경로 기준. 표시 시간과 트랜지션은 기존 값(2500ms, 300ms) 유지.

- [ ] **Step 3: 타입체크** Run: `cd apps/blog && npx tsc --noEmit --incremental false` Expected: 오류 0건

- [ ] **Step 4: 커밋 (요청 시)** `✨ Save posts only in the installed PWA`

### Task 5: 저장한 글 페이지와 메뉴

**Files:**

- Create: `apps/blog/src/components/pwa/SavedPostsView.tsx`
- Create: `apps/blog/src/app/saved/page.tsx`, `apps/blog/src/app/en/saved/page.tsx`
- Modify: `apps/blog/src/proxy.ts` (`EN_PREFIXES`에 `'saved'`)
- Modify: `apps/blog/src/components/layout/LayoutWrapper.tsx` (`HeaderNav`), `apps/blog/src/components/layout/MobileNav.tsx`

**Interfaces:**

- Consumes: Task 2의 `listSavedPosts`, `readMeta`, `updateSettings`, `syncRecent`, `deleteSaved`, `clearAllSaved`, `browserDeps`. Task 4의 `useIsStandalone`.
- Produces: `SavedPostsView({locale}: {locale: 'ko' | 'en'})`.

- [ ] **Step 1: `SavedPostsView` 구현 (시안 A 배치)**

마운트 후 로컬 데이터만으로 그린다. standalone이 아니면 안내 문구만. 설정 변경은 `updateSettings` 후 바로 `syncRecent`를 돌리고 목록을 다시 읽는다. 전체 삭제는 `window.confirm`으로 한 번 묻는다(내 판단). 스타일은 `apps/blog/src/app/offline/page.tsx`의 StyleX 패턴. 문구(ko / en):

- 제목 `저장한 글` / `Saved posts`
- 요약 `${n}편 저장됨, 마지막 갱신 ${시각}` / `${n} saved, last updated ${time}` (`toLocaleString(locale, {dateStyle: 'short', timeStyle: 'short'})`, 갱신 기록이 없으면 앞부분만)
- 설정 `최근 글 자동 저장`, `끄기 / 5편 / 10편 / 20편`, `본문 이미지도 저장` / `Auto-save recent posts`, `Off / 5 / 10 / 20`, `Also save images in posts`
- 꼬리표 `자동`, `읽음` / `Auto`, `Read`; 삭제 버튼 aria-label `삭제` / `Delete`
- `전체 삭제` / `Delete all`, 확인 `저장한 글을 모두 지울까요?` / `Delete all saved posts?`
- 빈 목록 `저장한 글이 없습니다` / `No saved posts yet`
- 브라우저 안내 `앱으로 설치하면 글을 오프라인에 저장해 둘 수 있습니다.` / `Install this site as an app to save posts for offline reading.`

- [ ] **Step 2: 두 페이지 작성**

`export const metadata = {title: '저장한 글 - yceffort', robots: {index: false}}`(영문 `Saved posts - yceffort`), 본문은 `<SavedPostsView locale=... />`.

- [ ] **Step 3: proxy와 메뉴**

`EN_PREFIXES`에 `'saved'`. `HeaderNav`, `MobileNav`에서 `useIsStandalone()`이 true면 메뉴 끝에 `{label: 'Saved', path:`${pathPrefix}/saved`}`를 붙인다. 두 곳의 기존 영문 매핑 차이는 고치지 않는다.

- [ ] **Step 4: 빌드와 404 확인**

Run: `pnpm build:blog` 후 `pnpm start`, `curl -s -o /dev/null -w '%{http_code}\n' localhost:3000/saved localhost:3000/en/saved`
Expected: `200` 두 번.

- [ ] **Step 5: 커밋 (요청 시)** `✨ Add PWA-only saved posts page`

### Task 6: 실제 브라우저 검증 (수동 절차)

**Files:** 없음 (검증만)

로컬 prod 서버(`pnpm build:blog`, `pnpm start`)에서 chrome-devtools MCP로 확인한다. claude-in-chrome은 로컬 서버에 못 붙는다. 시작 전에 옛 워커를 해제하고 캐시를 비운다(로컬 SW 캐시 때문에 옛 빌드가 보일 수 있음). 오프라인 확인은 DevTools 오프라인 에뮬레이션이 아니라 서버 프로세스를 죽여서 한다(리스너 PID를 `lsof`로 확인).

- [ ] **Step 1: 브라우저 탭**: 홈 → 글 클릭 → 다른 글 하드 내비게이션. Expected: 글 HTML을 두 번 받는 요청 없음, `caches.open('pages-v4')`의 키에 `/offline`만, `images-v4` 비어 있음, 토스트 없음, 메뉴에 `Saved` 없음, `/saved`는 브라우저 안내 문구.
- [ ] **Step 2: standalone**: CDP `Emulation.setEmulatedMedia`의 `display-mode` 에뮬레이션이 되는지 먼저 확인한다(확인 안 된 사항). 안 되면 localhost에서 PWA를 실제로 설치해 앱 창에서 확인한다. Expected: 실행 직후 `pages-v4`에 `/`, `/saved`, 최근 5편, 메뉴에 `Saved`, 첫 자동 저장 토스트 1회, `/saved`에 5편이 `자동`.
- [ ] **Step 3: 오프라인**: 서버를 죽이고 앱에서 `/saved` → 자동 저장 글 → 다른 저장 글(소프트 내비게이션) 순서로 연다. Expected: 모두 열리고, 저장 안 된 글은 `/offline`.
- [ ] **Step 4: 설정**: 10편으로 바꾸면 5편이 더 받아지고, 끄기로 바꾸면 `자동` 항목이 사라지고 `읽음`은 남는다. ✕한 자동 글은 새로고침 뒤에도 돌아오지 않는다.
- [ ] **Step 5: 웹 푸시**: 브라우저 탭에서 글 끝 알림 버튼이 지금처럼 구독 상태를 보여 준다.
