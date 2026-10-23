// GA4 성과 리포트용 데이터 조회 스크립트.
// 사용법: node .claude/skills/ga4-report/scripts/fetch-ga4.cjs [--days 28] [--trend 91] [--out ./ga4-data.json]
// 자격증명은 apps/blog/.env.local의 GA4_PROPERTY_ID, GOOGLE_APPLICATION_CREDENTIALS_JSON을 사용한다.
const {createRequire} = require('module')
const fs = require('fs')
const path = require('path')

const repoRoot = path.resolve(__dirname, '../../../..')
const blogDir = path.join(repoRoot, 'apps/blog')
const blogRequire = createRequire(path.join(blogDir, 'package.json'))

// @next/env는 pnpm 스토어에만 있으므로 버전 무관하게 탐색한다.
const pnpmDir = path.join(repoRoot, 'node_modules/.pnpm')
const nextEnvDir = fs
  .readdirSync(pnpmDir)
  .find((d) => d.startsWith('@next+env@'))
if (!nextEnvDir) {
  console.error(
    '@next/env를 찾을 수 없습니다. pnpm install 후 다시 실행하세요.',
  )
  process.exit(1)
}
const {loadEnvConfig} = require(
  path.join(pnpmDir, nextEnvDir, 'node_modules/@next/env'),
)
loadEnvConfig(blogDir, true, {info: () => {}, error: console.error})

const {BetaAnalyticsDataClient} = blogRequire('@google-analytics/data')

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`)
  return i >= 0 ? process.argv[i + 1] : fallback
}
const DAYS = Number(arg('days', 28))
const TREND = Number(arg('trend', 91))
const WEEKS = Number(arg('weeks', 4))
const TREND_WEEKS = Number(arg('trend-weeks', 26))
const OUT = arg('out', path.join(process.cwd(), 'ga4-data.json'))

// 고정 독자 지표는 ISO 주(월요일 시작) 단위로 끊는다. 28일 창은 경계만 옮겨도 증감률이 흔들린다.
function ymd(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}
function addDays(d, n) {
  const x = new Date(d)
  x.setDate(x.getDate() + n)
  return x
}
const thisMonday = addDays(new Date(), -((new Date().getDay() + 6) % 7))
thisMonday.setHours(0, 0, 0, 0)
const weekRange = {
  current: {
    startDate: ymd(addDays(thisMonday, -7 * WEEKS)),
    endDate: ymd(addDays(thisMonday, -1)),
  },
  previous: {
    startDate: ymd(addDays(thisMonday, -14 * WEEKS)),
    endDate: ymd(addDays(thisMonday, -7 * WEEKS - 1)),
  },
  trend: {
    startDate: ymd(addDays(thisMonday, -7 * TREND_WEEKS)),
    endDate: 'today',
  },
}
// research.yceffort.kr도 같은 측정 ID를 쓰므로 블로그 호스트로 한정한다 (발표 연습 같은 저자 방문이 research에 몰린다).
const blogHost = {
  filter: {fieldName: 'hostName', stringFilter: {value: 'yceffort.kr'}},
}
const krBlog = {
  andGroup: {
    expressions: [
      {filter: {fieldName: 'country', stringFilter: {value: 'South Korea'}}},
      blogHost,
    ],
  },
}
// 다음 주까지 끝난 최근 여덟 주. 이번 주와 지난주에 처음 온 사람은 아직 다음 주가 없다.
const weeklyCohorts = Array.from({length: 8}, (_, i) => {
  const start = addDays(thisMonday, -7 * (i + 2))
  return {
    name: ymd(start),
    dimension: 'firstSessionDate',
    dateRange: {startDate: ymd(start), endDate: ymd(addDays(start, 6))},
  }
})
const humanOnly = {
  andGroup: {
    expressions: [
      {filter: {fieldName: 'eventName', stringFilter: {value: 'human_visit'}}},
      blogHost,
    ],
  },
}

// dotenv가 큰따옴표 값의 \n을 실제 개행으로 확장하는 경우 재시도 (apps/blog/src/utils/analytics.ts와 동일)
function parseCredentials(credentialsJson) {
  try {
    return JSON.parse(credentialsJson)
  } catch {
    return JSON.parse(credentialsJson.replace(/\r?\n/g, '\\n'))
  }
}

const propertyId = process.env.GA4_PROPERTY_ID
const credentialsJson = process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON
if (!propertyId || !credentialsJson) {
  console.error(
    'GA4_PROPERTY_ID 또는 GOOGLE_APPLICATION_CREDENTIALS_JSON이 없습니다. apps/blog/.env.local을 확인하세요.',
  )
  process.exit(1)
}
const client = new BetaAnalyticsDataClient({
  credentials: parseCredentials(credentialsJson),
})
const property = `properties/${propertyId}`

// GA4는 속성당 동시 요청 한도(10)가 있어 한꺼번에 보내면 RESOURCE_EXHAUSTED가 난다. 4개씩만 보내고 걸리면 한 번 쉬었다 다시 보낸다.
const MAX_CONCURRENT = 4
let active = 0
const waiting = []
async function runReport(req) {
  if (active >= MAX_CONCURRENT) await new Promise((r) => waiting.push(r))
  active++
  try {
    for (let attempt = 0; ; attempt++) {
      try {
        return await client.runReport({property, ...req})
      } catch (e) {
        if (e.code !== 8 || attempt >= 2) throw e
        await new Promise((r) => setTimeout(r, 3000 * (attempt + 1)))
      }
    }
  } finally {
    active--
    const next = waiting.shift()
    if (next) next()
  }
}

async function run(name, req) {
  const [res] = await runReport(req)
  const rows = (res.rows || []).map((r) => ({
    d: (r.dimensionValues || []).map((v) => v.value),
    m: (r.metricValues || []).map((v) => v.value),
  }))
  return {name, rows}
}

// 최근 2개월(YYYY/MM) 경로 프리픽스 → 신규 글 성과 조회용. 영문판(/en/…)도 함께 센다.
function recentMonthPrefixes() {
  const now = new Date()
  const prefixes = []
  for (const offset of [0, 1]) {
    const d = new Date(now.getFullYear(), now.getMonth() - offset, 1)
    const ym = `/${d.getFullYear()}/${String(d.getMonth() + 1).padStart(2, '0')}`
    prefixes.push(ym, `/en${ym}`)
  }
  return prefixes
}

async function main() {
  const range = [{startDate: `${DAYS}daysAgo`, endDate: 'today'}]
  const reports = [
    run('daily', {
      dateRanges: [{startDate: `${TREND}daysAgo`, endDate: 'today'}],
      dimensions: [{name: 'date'}],
      metrics: [
        {name: 'activeUsers'},
        {name: 'sessions'},
        {name: 'screenPageViews'},
      ],
      orderBys: [{dimension: {dimensionName: 'date'}}],
      limit: TREND + 5,
    }),
    run('totalsCompare', {
      dateRanges: [
        {startDate: `${DAYS}daysAgo`, endDate: 'today', name: 'current'},
        {
          startDate: `${DAYS * 2}daysAgo`,
          endDate: `${DAYS + 1}daysAgo`,
          name: 'previous',
        },
      ],
      metrics: [
        {name: 'activeUsers'},
        {name: 'sessions'},
        {name: 'screenPageViews'},
        {name: 'engagementRate'},
        {name: 'averageSessionDuration'},
        {name: 'newUsers'},
      ],
    }),
    run('topPosts', {
      dateRanges: range,
      dimensions: [{name: 'pagePath'}],
      metrics: [
        {name: 'screenPageViews'},
        {name: 'activeUsers'},
        {name: 'userEngagementDuration'},
      ],
      // 영문판(/en/20…)도 같이 센다. 빼면 검색 유입 상위가 통째로 누락된다.
      dimensionFilter: {
        orGroup: {
          expressions: ['/20', '/en/20'].map((v) => ({
            filter: {
              fieldName: 'pagePath',
              stringFilter: {matchType: 'BEGINS_WITH', value: v},
            },
          })),
        },
      },
      orderBys: [{metric: {metricName: 'screenPageViews'}, desc: true}],
      limit: 40,
    }),
    run('channels', {
      dateRanges: range,
      dimensions: [{name: 'sessionDefaultChannelGroup'}],
      metrics: [{name: 'sessions'}, {name: 'activeUsers'}],
      orderBys: [{metric: {metricName: 'sessions'}, desc: true}],
      limit: 15,
    }),
    run('sources', {
      dateRanges: range,
      dimensions: [{name: 'sessionSource'}],
      metrics: [{name: 'sessions'}],
      orderBys: [{metric: {metricName: 'sessions'}, desc: true}],
      limit: 15,
    }),
    // 레퍼럴 유입만 (source/medium). 채널 그룹의 Referral 내역 확인용
    run('referrals', {
      dateRanges: range,
      dimensions: [{name: 'sessionSource'}, {name: 'sessionMedium'}],
      metrics: [{name: 'sessions'}, {name: 'activeUsers'}],
      orderBys: [{metric: {metricName: 'sessions'}, desc: true}],
      limit: 40,
    }),
    // 실제 레퍼러 URL (이벤트 스코프). 어느 페이지에서 링크를 타고 왔는지
    run('referrerUrls', {
      dateRanges: range,
      dimensions: [{name: 'pageReferrer'}],
      metrics: [{name: 'screenPageViews'}, {name: 'activeUsers'}],
      orderBys: [{metric: {metricName: 'screenPageViews'}, desc: true}],
      limit: 40,
    }),
    // 채널 → 랜딩 글 교차. 사용자 흐름 다이어그램의 뼈대
    run('channelLanding', {
      dateRanges: range,
      dimensions: [
        {name: 'sessionDefaultChannelGroup'},
        {name: 'landingPagePlusQueryString'},
      ],
      metrics: [{name: 'sessions'}, {name: 'activeUsers'}],
      orderBys: [{metric: {metricName: 'sessions'}, desc: true}],
      limit: 150,
    }),
    // 소스 → 랜딩 글 교차. "어디서 어느 글로" 표
    run('sourceLanding', {
      dateRanges: range,
      dimensions: [
        {name: 'sessionSource'},
        {name: 'landingPagePlusQueryString'},
      ],
      metrics: [{name: 'sessions'}],
      orderBys: [{metric: {metricName: 'sessions'}, desc: true}],
      limit: 150,
    }),
    // 랜딩 글별 세션 깊이·참여. 흐름의 마지막 단계(더 읽었나 / 한 장만 보고 떠났나)
    run('landingDepth', {
      dateRanges: range,
      dimensions: [{name: 'landingPagePlusQueryString'}],
      metrics: [
        {name: 'sessions'},
        {name: 'engagedSessions'},
        {name: 'screenPageViews'},
        {name: 'bounceRate'},
      ],
      orderBys: [{metric: {metricName: 'sessions'}, desc: true}],
      limit: 40,
    }),
    run('countries', {
      dateRanges: range,
      dimensions: [{name: 'country'}],
      metrics: [{name: 'activeUsers'}, {name: 'sessions'}],
      orderBys: [{metric: {metricName: 'activeUsers'}, desc: true}],
      limit: 10,
    }),
    run('devices', {
      dateRanges: range,
      dimensions: [{name: 'deviceCategory'}],
      metrics: [{name: 'activeUsers'}],
      orderBys: [{metric: {metricName: 'activeUsers'}, desc: true}],
      limit: 5,
    }),
    run('newReturning', {
      dateRanges: range,
      dimensions: [{name: 'newVsReturning'}],
      metrics: [{name: 'activeUsers'}],
      limit: 5,
    }),
    run('recentPosts', {
      dateRanges: range,
      dimensions: [{name: 'pagePath'}],
      metrics: [{name: 'screenPageViews'}, {name: 'activeUsers'}],
      dimensionFilter: {
        orGroup: {
          expressions: recentMonthPrefixes().map((p) => ({
            filter: {
              fieldName: 'pagePath',
              stringFilter: {matchType: 'BEGINS_WITH', value: p},
            },
          })),
        },
      },
      orderBys: [{metric: {metricName: 'screenPageViews'}, desc: true}],
      limit: 40,
    }),
    // 글 하단 도선의 노출 대비 클릭. InternalNavTracker가 보내는 커스텀 이벤트로,
    // "스크롤이 거기까지 안 간 것"과 "보고도 안 누른 것"을 나눠서 본다.
    run('navBlocks', {
      dateRanges: range,
      dimensions: [{name: 'customEvent:nav_block'}, {name: 'eventName'}],
      metrics: [{name: 'eventCount'}, {name: 'activeUsers'}],
      dimensionFilter: {
        filter: {
          fieldName: 'eventName',
          inListFilter: {
            values: ['internal_nav_view', 'internal_nav_click'],
          },
        },
      },
      orderBys: [{metric: {metricName: 'eventCount'}, desc: true}],
      limit: 30,
    }),
    // 도선이 어느 글에서 노출·클릭됐는지. 글별로 도선을 손볼 때 우선순위가 된다.
    run('navBlockPages', {
      dateRanges: range,
      dimensions: [
        {name: 'pagePath'},
        {name: 'customEvent:nav_block'},
        {name: 'eventName'},
      ],
      metrics: [{name: 'eventCount'}],
      dimensionFilter: {
        filter: {
          fieldName: 'eventName',
          inListFilter: {
            values: ['internal_nav_view', 'internal_nav_click'],
          },
        },
      },
      orderBys: [{metric: {metricName: 'eventCount'}, desc: true}],
      limit: 40,
    }),
    // BotTracker가 직접 분류한 봇/사람. 국가별 1:1 휴리스틱보다 정확하다.
    run('botVisits', {
      dateRanges: range,
      dimensions: [{name: 'eventName'}],
      metrics: [
        {name: 'eventCount'},
        {name: 'activeUsers'},
        {name: 'sessions'},
      ],
      dimensionFilter: {
        filter: {
          fieldName: 'eventName',
          inListFilter: {values: ['human_visit', 'bot_visit']},
        },
      },
      limit: 5,
    }),
  ]

  // ── 고정 독자 ──
  // 사람 판별(human_visit) 기준 주별 신규/재방문. 저자 본인 방문은 세션에서는 크지만 사용자로는 몇 명이다.
  reports.push(
    // 진단 부록 KPI. totalsCompare와 같은 지표를 ISO 주 경계로 다시 뽑는다.
    run('totalsWeeks', {
      dateRanges: ['current', 'previous'].map((name) => ({
        ...weekRange[name],
        name,
      })),
      metrics: [
        {name: 'activeUsers'},
        {name: 'sessions'},
        {name: 'screenPageViews'},
        {name: 'engagementRate'},
        {name: 'averageSessionDuration'},
      ],
    }),
    run('weeklyReaders', {
      dateRanges: [weekRange.trend],
      dimensions: [{name: 'isoYearIsoWeek'}, {name: 'newVsReturning'}],
      metrics: [{name: 'activeUsers'}],
      dimensionFilter: humanOnly,
      orderBys: [{dimension: {dimensionName: 'isoYearIsoWeek'}}],
      limit: TREND_WEEKS * 3 + 10,
    }),
    // 처음 온 주별 코호트의 N주 뒤 재방문. 크롤러가 분모를 키우므로 한국으로 한정한다.
    run('cohorts', {
      dimensions: [{name: 'cohort'}, {name: 'cohortNthWeek'}],
      metrics: [{name: 'cohortActiveUsers'}],
      dimensionFilter: krBlog,
      cohortSpec: {
        cohorts: weeklyCohorts,
        cohortsRange: {granularity: 'WEEKLY', endOffset: 4},
      },
    }),
    // 같은 여덟 코호트를 처음 들어온 경로로 나눈 다음 주 재방문. 경로별 막대와 비중 비교에 쓴다.
    ...[
      ['cohortsBySource', 'firstUserSource'],
      ['cohortsByChannel', 'firstUserDefaultChannelGroup'],
    ].map(([name, dimension]) =>
      run(name, {
        dimensions: [
          {name: 'cohort'},
          {name: 'cohortNthWeek'},
          {name: dimension},
        ],
        metrics: [{name: 'cohortActiveUsers'}],
        dimensionFilter: krBlog,
        cohortSpec: {
          cohorts: weeklyCohorts,
          cohortsRange: {granularity: 'WEEKLY', endOffset: 1},
        },
        limit: 500,
      }),
    ),
    // 글별 신규/재방문 독자. "고정 독자가 읽은 글"까지만 말해 준다.
    run('postReaders', {
      dateRanges: [weekRange.current],
      dimensions: [{name: 'pagePath'}, {name: 'newVsReturning'}],
      metrics: [
        {name: 'activeUsers'},
        {name: 'userEngagementDuration'},
        {name: 'screenPageViews'},
      ],
      dimensionFilter: {
        filter: {
          fieldName: 'pagePath',
          stringFilter: {matchType: 'PARTIAL_REGEXP', value: '^/(en/)?20'},
        },
      },
      orderBys: [{metric: {metricName: 'activeUsers'}, desc: true}],
      limit: 250,
    }),
    // 처음 유입 경로별 재방문 독자 (현재/직전 4주)
    ...['current', 'previous'].map((k) =>
      run(`firstTouch_${k}`, {
        dateRanges: [weekRange[k]],
        dimensions: [{name: 'firstUserSource'}, {name: 'newVsReturning'}],
        metrics: [{name: 'activeUsers'}],
        dimensionFilter: humanOnly,
        orderBys: [{metric: {metricName: 'activeUsers'}, desc: true}],
        limit: 60,
      }),
    ),
    // 도선 주별 노출/클릭과 본 사람/누른 사람. 미뤄 둔 판정과 시리즈 이어 읽기에 쓴다.
    run('weeklyNav', {
      dateRanges: [weekRange.trend],
      dimensions: [
        {name: 'isoYearIsoWeek'},
        {name: 'customEvent:nav_block'},
        {name: 'eventName'},
      ],
      metrics: [{name: 'eventCount'}, {name: 'activeUsers'}],
      dimensionFilter: {
        filter: {
          fieldName: 'eventName',
          inListFilter: {values: ['internal_nav_view', 'internal_nav_click']},
        },
      },
      orderBys: [{dimension: {dimensionName: 'isoYearIsoWeek'}}],
      limit: 300,
    }),
  )

  const out = {
    meta: {
      generatedAt: new Date().toISOString(),
      days: DAYS,
      trend: TREND,
      weeks: WEEKS,
      trendWeeks: TREND_WEEKS,
      weekRange,
    },
  }
  for (const r of await Promise.all(reports)) out[r.name] = r.rows
  fs.writeFileSync(OUT, JSON.stringify(out, null, 1))
  console.log(`wrote ${OUT}`)
  console.log(
    Object.keys(out)
      .filter((k) => k !== 'meta')
      .map((k) => `${k}:${out[k].length}`)
      .join(' '),
  )
}

main().catch((e) => {
  console.error(e.message)
  process.exit(1)
})
