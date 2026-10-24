/**
 * 새 글/새 시리즈가 main에 추가되면 웹 푸시 구독자 전원에게 알림을 보낸다.
 *
 * 사용법: node apps/blog/scripts/notify-push.mjs <추가된 md 파일 경로...>
 * 필요 환경변수: KV_REST_API_URL, KV_REST_API_TOKEN, VAPID_PUBLIC_KEY, VAPID_PRIVATE_KEY
 */
import {readFile, readdir} from 'node:fs/promises'
import {join} from 'node:path'
import {fileURLToPath} from 'node:url'

import webpush from 'web-push'

const SUBSCRIPTIONS_KEY = 'push:subscriptions'
// 알림을 눌러 들어온 세션을 GA4에서 Direct와 가르기 위한 꼬리표
const PUSH_UTM = 'utm_source=push&utm_medium=notification'

const {
  KV_REST_API_URL,
  KV_REST_API_TOKEN,
  VAPID_PUBLIC_KEY,
  VAPID_PRIVATE_KEY,
} = process.env

if (
  !KV_REST_API_URL ||
  !KV_REST_API_TOKEN ||
  !VAPID_PUBLIC_KEY ||
  !VAPID_PRIVATE_KEY
) {
  console.error('missing required env vars')
  process.exit(1)
}

webpush.setVapidDetails(
  `mailto:root@yceffort.kr`,
  VAPID_PUBLIC_KEY,
  VAPID_PRIVATE_KEY,
)

async function redis(command) {
  const response = await fetch(KV_REST_API_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${KV_REST_API_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(command),
  })
  if (!response.ok) {
    throw new Error(`redis request failed: ${response.status}`)
  }
  const {result} = await response.json()
  return result
}

function parseFrontmatter(markdown) {
  const match = markdown.match(/^---\n([\s\S]*?)\n---/)
  if (!match) {
    return {}
  }
  const fields = {}
  for (const line of match[1].split('\n')) {
    const kv = line.match(/^(\w+):\s*(.+)$/)
    if (kv) {
      fields[kv[1]] = kv[2].replace(/^['"]|['"]$/g, '')
    }
  }
  return fields
}

const POSTS_DIR = fileURLToPath(new URL('../posts', import.meta.url))

// 시리즈 리드미에는 published 필드가 없어서, draft 글과 함께 커밋된 시리즈가
// 그대로 알림으로 나간다. 공개된 글이 하나라도 있는 시리즈만 알린다.
async function seriesHasPublishedPost(seriesName) {
  const entries = await readdir(POSTS_DIR, {recursive: true})
  for (const entry of entries) {
    if (!entry.endsWith('.md') || entry.endsWith('.en.md')) {
      continue
    }
    const fm = parseFrontmatter(await readFile(join(POSTS_DIR, entry), 'utf8'))
    if (fm.series === seriesName && fm.published !== 'false') {
      return true
    }
  }
  return false
}

// posts/{y}/{m}/{slug}.md → 새 글, series/{slug}.md → 새 시리즈
async function toNotification(filePath) {
  const post = filePath.match(/posts\/(\d{4})\/(\d{2})\/([^/]+)\.md$/)
  const series = filePath.match(/series\/([^/]+)\.md$/)
  if (!post && !series) {
    return null
  }
  const fm = parseFrontmatter(await readFile(filePath, 'utf8'))
  if (fm.published === 'false') {
    return null
  }
  if (post) {
    return {
      title: '새 글이 올라왔어요',
      // 알림 본문은 일반 텍스트라 제목의 <em> 강조를 걷어낸다
      body: fm.title?.replace(/<\/?em>/g, '') || post[3],
      url: `/${post[1]}/${post[2]}/${post[3]}?${PUSH_UTM}`,
    }
  }
  if (!(await seriesHasPublishedPost(fm.name))) {
    return null
  }
  return {
    title: '새 시리즈가 시작됐어요',
    body: fm.name || series[1],
    url: `/series/${series[1]}?${PUSH_UTM}`,
  }
}

const files = process.argv.slice(2).filter((f) => !f.endsWith('.en.md'))
const notifications = (
  await Promise.all(files.map((f) => toNotification(f)))
).filter(Boolean)

if (notifications.length === 0) {
  console.log('no notifications to send')
  process.exit(0)
}

const subscriptions = (await redis(['HVALS', SUBSCRIPTIONS_KEY])) || []
console.log(
  `sending ${notifications.length} notification(s) to ${subscriptions.length} subscriber(s)`,
)

let sent = 0
let removed = 0
for (const notification of notifications) {
  const payload = JSON.stringify(notification)
  await Promise.all(
    subscriptions.map(async (raw) => {
      const subscription = JSON.parse(raw)
      try {
        await webpush.sendNotification(subscription, payload)
        sent += 1
      } catch (error) {
        // 만료되거나 해지된 구독은 정리한다
        if (error.statusCode === 404 || error.statusCode === 410) {
          await redis(['HDEL', SUBSCRIPTIONS_KEY, subscription.endpoint])
          removed += 1
        } else {
          console.error(`send failed (${error.statusCode}): ${error.message}`)
        }
      }
    }),
  )
}

console.log(`done: ${sent} sent, ${removed} expired subscriptions removed`)
