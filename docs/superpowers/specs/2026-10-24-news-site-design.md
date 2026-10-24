# news.yceffort.kr 설계

작성일 2026-10-24. brain 저장소의 뉴스레터 봇을 블로그 모노레포로 옮겨 news.yceffort.kr 로 공개하는 설계다. 사이트 뼈대(`apps/news`)는 이미 만들어져 로컬에서 확인했고, 이 문서는 그 뒤에 남은 일인 봇 이전, 수집 기준 개정, 자동 배포를 다룬다.

## 의도

매일 아침 내가 관심 있는 글을 골라 핵심을 요약해 주는 뉴스를 한 곳(블로그 도메인)에서 본다. 지금은 brain 저장소(비공개, Obsidian 볼트)에서 봇이 매일 캡처 노트와 읽기 목록을 만들지만, Obsidian을 거의 열지 않아 결과를 보지 않는다. 읽는 곳을 웹으로 옮기고, 요약은 링크 모음이 아니라 원문을 안 열어도 논지가 잡히는 깊이로 올린다.

## 목표

- 일간 호: 매일 09:00 KST에 `apps/news/content/daily/YYYY-MM-DD.md` 를 생성해 커밋하고, Vercel이 news.yceffort.kr 에 배포한다.
- 주간 호: 월요일 09:30 KST에 지난 7일의 일간 호를 재료로 `apps/news/content/weekly/YYYY-Www.md` 를 생성한다.
- 선정 기준은 세 기둥을 같은 무게로 둔다. 기술 심화, AI 에세이, 보안 패치와 릴리스. 각 기둥이 매일 자리를 보장받는다.
- 요약 깊이는 항목마다 훅 1문장, 요점 4~5개(각 2~4문장, 숫자와 사례와 저자의 단서 포함), "왜 볼 만한가" 2~3문장이다. 이미 `apps/news/content/daily/2026-10-23.md` 에 이 깊이의 표본이 있다.
- 봇 코드는 `services/news` 한 곳에만 있다. brain의 봇과 워크플로우는 news가 안정되면 지운다.
- 사이트에 RSS(`/feed.xml`)를 더한다.

## 비목표 (이번에 하지 않는다)

- 이메일 발송, 웹 푸시 알림, 영문판, OG 이미지 생성, 검색, 댓글.
- brain 볼트 노트(`clippings/`, MOC)를 news로 옮기는 일. 볼트는 그대로 두고 손대지 않는다.
- 사이트 디자인 변경. 이미 확인한 모양을 유지한다.

## 반목표 (이렇게 되면 실패다)

- brain 볼트의 노트나 내부 정보가 공개 저장소에 들어간다.
- brain과 news의 봇이 동시에 돌아 Claude 구독 사용량을 두 배로 쓴다.
- 매일 들어오는 콘텐츠 커밋이 blog와 research의 Vercel 재빌드를 유발한다.
- 한 기둥이 호를 독식한다. 지금 brain 봇은 캡처의 65%가 AI 에세이다.
- 봇이 쓴 마크다운이 lefthook의 markdownlint나 oxfmt 규칙에 걸려 사람이 손봐야 한다.

## 제약

- 저장소 yceffort/yceffort-blog-v2 는 PUBLIC 이다.
- Claude는 API 키가 아니라 구독 OAuth 토큰(`CLAUDE_CODE_OAUTH_TOKEN`, 블로그 저장소에 이미 등록됨)으로 `anthropics/claude-code-action@v1` 을 통해 쓴다. `FIRECRAWL_API_KEY` 는 선택이고 블로그 저장소에는 없다. 없어도 파이프라인은 동작한다.
- 워크플로우 실행 제한 30분. brain에서 일간은 6~8분이었다.
- 모노레포는 pnpm 12, Node 24, oxlint, oxfmt, rust-markdownlint를 쓴다. 생성된 마크다운은 `.markdownlint.jsonc` 규칙(목록은 `-`, 제목과 목록 앞뒤 빈 줄, 강조는 `_`와 `**`)과 oxfmt 포맷을 만족해야 한다.
- 글은 한국어 "~입니다" 공손체. em dash, en dash, 가운뎃점을 쓰지 않는다. 한 문단은 한 줄로 쓴다.
- 커밋 메시지 컨벤션은 `@naverpay/commit-helper` 를 따르지만, 봇 커밋은 brain과 같이 `news: YYYY-MM-DD`, `news weekly: YYYY-Www` 접두사를 쓴다. 잘못된 호는 `git revert` 로 지운다.

## 이미 정한 것

### 사이트 (`apps/news`, 완료)

- Next.js 16, StyleX, research와 같은 헤더/테마/푸터. 단일 열 46rem.
- 홈은 최신 일간 호 전문 + 지난 호 7개 + 주간 3개. `/daily`, `/daily/[date]`, `/weekly`, `/weekly/[week]`.
- 콘텐츠는 frontmatter(`date`, `count`, 주간은 `period`)가 있는 마크다운 하나다. h2 제목이 절의 종류를 정한다. "오늘 하나만 본다면", "읽을 만한 것", "보안과 릴리스"(기존 이름 "짧게 알아두기"도 같은 종류로 인식한다), "수집 상태", 주간은 "이번 주"와 자유 제목의 흐름 절. h3 하나가 항목 하나다. 항목 안의 문단은 순서로 역할이 정해진다. 출처 표기(코드로 시작), 훅, 요점 목록, 왜 볼 만한가.
- 마크다운 파서는 unified(remark). `<` 같은 문자가 본문에 들어와도 깨지지 않게 MDX 파서(markdown-rs)를 쓰지 않는다.

### 봇 위치와 구조 (`services/news`)

- brain의 `bot/` 을 `services/news` 로 옮긴다. 수집기(`src/collect.ts` 와 하위), 테스트 8개, 프롬프트 2개, `state/seen-urls.json`. `pnpm-workspace.yaml` 에 `services/*` 를 추가한다.
- `state/seen-urls.json` 은 brain의 현재 파일을 그대로 가져온다. 지난 60일 동안 이미 실은 글을 다시 고르지 않기 위해서다.
- 두 단계 구조는 유지한다. 수집기는 LLM 없이 후보 JSON을 만들고, 큐레이터(claude-code-action)가 프롬프트를 읽어 마크다운 한 파일을 쓴다. 판단이 필요 없는 일은 에이전트에게 맡기지 않는다.
- 큐레이터가 만드는 파일은 `apps/news/content/daily/<날짜>.md` 하나다. 볼트 노트, MOC, `[[ ]]` 링크는 없다. 출처는 원문 URL 링크다.

### 수집 기준 개정

brain 봇 55일치 결과와 2026-10-24 dry-run을 근거로 한다. 봇 캡처 83편 중 AI 에세이 54편, 성능/브라우저 17편, 프레임워크 내부와 번들러 0편, K8s와 관측 0편이었다. 내가 올해 블로그에 쓴 글의 태그는 web-performance 28, browser 14, nextjs 13, backend 13, ai 12, react 10, devops 9, bundler 8, kubernetes 6 이다. dry-run에서는 본문 확보 23건 중 개인 블로그가 2건, 55일간 한 번도 채택된 적 없는 Sentry와 Smashing Magazine이 4건이었다.

세 기둥의 정의.

1. **기술 심화**: 동작 원리를 파는 글. React와 Next.js 내부, 번들러와 컴파일러(Turbopack, oxc, Rolldown, SWC, V8), 웹 성능 실측(핵심 웹 지표), 브라우저 엔진과 CSS 동작, Node 런타임, 컨테이너와 쿠버네티스, 관측(OpenTelemetry, Grafana), npm 공급망. 소개글과 튜토리얼은 아니다.
2. **AI 에세이**: AI가 개발자의 일과 판단을 어떻게 바꾸는가에 대한 논증, 에이전트 시대의 검증과 규율, 학습과 전문성의 실증 연구, 설계 철학과 장인성. 지금 brain 프롬프트의 기준 그대로다.
3. **보안 패치와 릴리스**: Node.js, Next.js, 브라우저, 주요 도구의 보안 패치와 릴리스. 호의 "보안과 릴리스" 절에 한 줄씩 싣되, 보안 패치는 맨 위에 심각도와 "내가 할 일"을 붙인다. 릴리스는 그 아래다.

제외 목록은 유지한다. Python 생태계, 오픈소스 재단과 거버넌스 소식, 게임과 우주와 하드웨어와 정치, 데이터 엔지니어링과 모델 벤치마크, 새 LLM 출시 자체, 회사 홍보와 채용과 컨퍼런스 안내, "N가지 팁" 류.

소스 변경.

- 추가(피드 생존을 2026-10-24에 확인함): Pragmatic Engineer, Kent Beck(Tidy First), The Engineering Manager, antirez, charity.wtf, ferd.ca, matklad, Hillel Wayne, Geoffrey Litt, Grafana Labs, OpenTelemetry, Kubernetes Blog, Platformatic, csswizardry, infrequently.org(Alex Russell), Philip Walton. 앞의 9개는 내가 brain에 손으로 캡처한 출처이고(Pragmatic Engineer 9편, Kent Beck 5편), 뒤의 7개는 블로그 주제를 덮는다.
- 추가 시도: oxc, Rolldown, Socket, Barry Pollard. 흔한 피드 주소가 404였다. 다른 주소를 찾아 되면 넣고 안 되면 뺀다.
- 제거: AWS Blog, Netflix Tech, Sentry, Smashing Magazine, OpenAI, Hugging Face, Tailwind, Svelte, Vue, Remix, Biome. 55일간 본문 항목으로 채택된 적이 없고 OpenAI와 Hugging Face는 7일 수집량이 각각 1,318건과 890건으로 칸만 차지했다.
- 유지: 그 밖의 전부. 특히 릴리스 피드, Node.js Blog, Vercel Blog, pnpm, WebKit은 "보안과 릴리스" 절의 재료다.

쇼트리스트 몫(`shortlist.ts`). 지금은 커뮤니티(HN, Lobsters) 10칸, 트렌딩 3칸, 나머지 17칸은 순위 없이 수집 순서대로 소스당 2개다. 바꾼다.

- 커뮤니티 6칸(소스당 3), 트렌딩 2칸, 개인 블로그 8칸(소스당 2, 최신순), 나머지(벤더, 프레임워크, 표준, 릴리스, 뉴스레터) 14칸(소스당 2, 최신순). 합계 30. 정확한 숫자는 dry-run 결과를 보고 빌더가 조정해도 된다. 지켜야 할 것은 개인 블로그가 몫을 보장받고 나머지가 최신순으로 정렬되는 것이다.
- `test/sources.test.ts` 와 `test/shortlist.test.ts` 의 개수와 기대값을 함께 고친다.

### 프롬프트 (`services/news/prompts/daily.md`, `weekly.md`)

- brain의 `daily.md` 를 바탕으로 고친다. 볼트 전용 부분(capture 노트 스키마, 태그 어휘, `[[ ]]`, `clippings/` 중복 확인, MOC)을 지우고, 출력은 위 콘텐츠 형식의 마크다운 한 파일로 바꾼다.
- 독자 소개는 유지하되 "실제로 저장하는 글의 결" 단락을 세 기둥 정의로 바꾼다. 고르는 수는 5~8편이고 기술 심화와 AI 에세이가 각각 2편 이상이어야 한다. 한쪽에 좋은 글이 없으면 그쪽을 줄이되 다른 쪽으로 채우지 않는다.
- 요약 깊이 규칙을 넣는다. 훅 1문장. 요점 4~5개, 각 2~4문장, 원문의 숫자와 사례와 비교 대상을 그대로 싣고 저자가 단 단서(자체 보고, 근거 없음, 시리즈 지도 글)를 빠뜨리지 않는다. "왜 볼 만한가" 2~3문장은 이 독자에게 왜 닿는지와 출처의 한계를 적는다. 원문에 없는 말을 지어내지 않는다. 좋은 훅과 나쁜 훅의 예시는 brain 프롬프트의 것을 그대로 쓴다.
- 중복은 `state/seen-urls.json` 이 1차로 막고, 큐레이터는 `apps/news/content/daily/` 의 최근 14일 파일에서 같은 URL을 `grep` 해 2차로 막는다.
- `weekly.md` 는 `clippings/` 대신 지난 7일의 `apps/news/content/daily/*.md` 를 읽는다. 흐름을 묶는 규칙은 그대로다. 글 링크는 원문 URL이고, 그 글이 실린 호는 `/daily/YYYY-MM-DD` 상대 링크로 잇는다. "MOC에 올린 것" 절은 "이번 주의 두 편"으로 바꾸고 MOC 수정 단계는 없다.
- 두 프롬프트 모두 마지막에 생성 파일을 `pnpm exec oxfmt <파일>` 과 `pnpm exec rust-markdownlint --fix --no-globs <파일>` 로 정리하도록 지시하거나, 워크플로우가 커밋 전에 같은 명령을 돌린다. 빌더가 한쪽을 고른다. 워크플로우 쪽이 더 확실하다.

### 워크플로우 (`.github/workflows/news-daily.yml`, `news-weekly.yml`)

- brain의 `digest-daily.yml`, `digest-weekly.yml` 을 바탕으로 한다. 일간은 `0 0 * * *` UTC, 주간은 `30 0 * * 1` UTC. 같은 concurrency 그룹 `news` 로 직렬화한다. `workflow_dispatch` 를 둔다.
- **main 푸시 권한.** main은 필수 상태 체크 4개(`Build (24.x)`, `Lint & Prettier (24.x)`, `Vercel – blog`, `Vercel – research`)로 보호돼 있고 관리자만 우회한다(2026-10-24 확인). 워크플로우 기본 `GITHUB_TOKEN` 은 우회가 안 되므로, 저장소 소유자의 fine-grained PAT(이 저장소 하나, Contents 읽기/쓰기, 만료 1년)를 `NEWS_BOT_TOKEN` 시크릿으로 등록해 푸시에 쓴다. PR과 자동 병합 경로는 쓰지 않는다. 그 경로는 매일 blog와 research의 Vercel 체크를 요구해서 "뉴스 커밋이 blog와 research를 재빌드하지 않는다"는 반목표와 충돌한다.
- **에이전트와 푸시를 다른 job으로 가른다.** 큐레이터는 신뢰할 수 없는 외부 본문을 읽으며 공개 저장소에서 돈다. 본문에 심은 프롬프트 인젝션이 main 변조로 이어지지 않도록, `curate` job은 `permissions: contents: read` 만 갖고 PAT를 받지 않으며, 생성한 마크다운을 artifact로 넘긴다. `publish` job이 artifact를 받아 포맷과 린트를 돌리고 PAT로 커밋, 푸시한다. 포맷과 린트는 프롬프트가 아니라 여기서 한다(이 결정으로 "빌더에게 맡기는 것"의 해당 항목은 닫힌다).
- 큐레이터의 도구는 `--allowedTools "Read,Glob,Grep,Write(apps/news/content/**),Edit(apps/news/content/**),Bash(date:*)"` 로 좁힌다. brain의 `Bash` 전체 허용을 가져오지 않는다. 프롬프트에서 `grep` 이 필요한 곳은 Grep 도구로, 날짜는 `date -u` 로 충분하다.
- 일간 `curate`: checkout, pnpm 12와 Node 24 설치(캐시는 루트 `pnpm-lock.yaml`), 봇 패키지와 그 의존성만 설치(`--filter`), `collect`, claude-code-action(`prompt` 는 "`services/news/prompts/daily.md` 를 읽고 그대로 수행한다", `show_full_output: 'true'`), `apps/news/content/daily` 와 `services/news/state` 를 artifact로 업로드. 일간 `publish`: checkout(PAT), artifact 다운로드, `pnpm exec oxfmt <파일>` 과 `pnpm exec rust-markdownlint --fix --no-globs <파일>`(루트 devDependency가 필요하므로 설치 범위를 맞춘다), 변경 없으면 종료, 커밋(`git add apps/news/content services/news/state`), rebase 재시도 3회 푸시.
- 주간: 수집기 없이 같은 두 job 구조. 커밋 대상은 `apps/news/content/weekly`.
- 커밋 작성자는 `news-bot`, 이메일은 `root@yceffort.kr`. 메시지는 `news: YYYY-MM-DD`, `news weekly: YYYY-Www`.
- 처음 올리는 워크플로우 파일은 `schedule` 을 뺀 `workflow_dispatch` 전용으로 둔다. 수동 실행이 성공하면 `schedule` 을 넣는 커밋을 따로 한다. 워크플로우가 main에 오르는 순간 크론이 켜지는 것을 피하기 위해서다.
- 블로그 저장소의 다른 워크플로우(`notify-push.yaml` 등)는 `apps/blog/posts/**` 경로 필터가 있어 뉴스 커밋에 반응하지 않는다. `ci.yaml` 은 main 이 아닌 브랜치 push에서만 돈다. PAT 푸시는 `GITHUB_TOKEN` 푸시와 달리 다른 워크플로우를 깨우므로, 빌더는 뉴스 커밋에 반응하는 워크플로우가 없는지 한 번 더 확인한다.

### 배포 (Vercel)

- Vercel 프로젝트 `news` 를 만들고 Root Directory 를 `apps/news` 로, 도메인을 news.yceffort.kr 로 둔다. blog, research 프로젝트와 같은 GitHub 연결이다.
- 반목표를 지키기 위해 blog와 research 프로젝트의 Ignored Build Step 에 `apps/news/**` 와 `services/news/**` 만 바뀐 커밋은 건너뛰는 조건을 넣는다. `services/news/state/seen-urls.json` 이 매일 바뀌므로 `apps/news/**` 만으로는 부족하다. news 프로젝트는 반대로 `apps/news`, `packages/shared`, 루트 설정이 바뀔 때만 빌드한다. 정확한 명령은 빌더가 정한다. 지금 두 프로젝트에 Ignored Build Step 이 설정돼 있는지는 확인하지 못했다.
- 봇은 PAT로 보호 규칙을 우회해 직접 푸시하므로, 빌드를 건너뛴 커밋의 Vercel 체크 상태가 어떻게 보고되든 푸시에는 영향이 없다.
- 프로젝트 생성과 도메인 연결은 사람이 하거나 Vercel MCP로 한다. 이 문서의 승인과 별도로 그 시점에 다시 확인받는다. 워크플로우를 main에 올리기 전에 끝나 있어야 한다.

### brain 정리

- 두 봇이 하루라도 겹쳐 돌지 않게 한다. news 일간을 처음 수동 실행하기 직전에 brain의 두 워크플로우를 `gh workflow disable` 로 끄고, 그 시점의 `bot/state/seen-urls.json` 을 `services/news/state/` 로 복사해 커밋한다. 이보다 먼저 복사해 두면 겹치는 날 brain이 고른 글이 news에 다시 실린다.
- news의 일간이 수동 1회와 정기 1회, 주간이 1회 성공한 뒤 brain의 `.github/workflows/digest-*.yml` 과 `bot/` 을 지우고 `CLAUDE.md` 와 `bot/README.md` 의 봇 설명을 정리한다. 이 삭제는 별도로 확인받는다.
- brain은 사람이 직접 capture하는 Obsidian 볼트로 남는다. 2-Mac 동기화 규약도 그대로다.

## 빌더에게 맡기는 것

- `services/news` 패키지 이름, 디렉터리 안 배치, vitest와 tsx 버전을 모노레포 카탈로그에 맞추는 일.
- 쇼트리스트 몫의 정확한 숫자와 정렬 구현.
- oxc, Rolldown, Socket, Barry Pollard의 대체 피드 주소 탐색.
- `/feed.xml` 구현(일간과 주간을 한 피드에 넣고 제목은 호 제목, 본문은 호 전문).
- Ignored Build Step 명령.
- 일간 호가 하나도 없는 날(후보 0건)의 처리. brain과 같이 파일을 만들지 않고 보고만 한다.

## 순서와 확인 지점

1. 코드를 만든다. `services/news`, 프롬프트, `/feed.xml`, 워크플로우 2개(`workflow_dispatch` 전용). 이 단계는 로컬 커밋까지만 하고 푸시하지 않는다.
2. **확인 지점 1 (용찬님).** `NEWS_BOT_TOKEN` 시크릿 등록. Vercel에 news 프로젝트 생성과 도메인 연결, blog와 research의 Ignored Build Step 설정. Vercel 작업은 요청하시면 Vercel MCP로 대신한다.
3. brain 워크플로우를 끄고 `seen-urls.json` 을 복사한 뒤 main에 푸시한다.
4. `workflow_dispatch` 로 일간을 한 번 돌린다.
5. **확인 지점 2 (용찬님).** 첫 호를 보고 괜찮으면 `schedule` 을 넣는 커밋을 올린다.
6. 정기 일간 1회와 주간 1회가 성공하면 brain의 봇 코드와 워크플로우 삭제를 확인받고 지운다.

## 검증

1. `services/news` 에서 `pnpm test` 와 `pnpm typecheck` 통과. 소스 개수 테스트가 새 목록과 맞아야 한다.
2. `pnpm collect --dry-run` 을 평일에 한 번 돌려 후보 구성을 본다. 개인 블로그 6건 이상, 기술 심화 후보와 AI 에세이 후보가 모두 있어야 한다.
3. 로컬에서 큐레이터를 한 번 돌려(`claude -p` 로 `daily.md` 실행) 생성 파일이 markdownlint와 oxfmt를 통과하고 사이트에서 세 절이 의도대로 렌더되는지 본다.
4. 첫 `workflow_dispatch` 실행에서 커밋, Vercel 배포, news.yceffort.kr 응답을 확인한다. 같은 커밋으로 blog와 research가 재빌드되지 않았는지 Vercel 배포 목록에서 확인한다. `curate` job의 로그에서 에이전트가 `apps/news/content` 밖을 건드리려 한 흔적이 없는지 본다.
5. 다음 월요일 주간 실행 결과를 확인한 뒤 brain 정리로 넘어간다.
