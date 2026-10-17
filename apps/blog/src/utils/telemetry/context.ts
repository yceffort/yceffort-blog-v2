import type {Attributes} from '@opentelemetry/api'

import {routeOf} from './routes'

/**
 * 페이지 로드 하나에 한 번 정하는 맥락. 모두 span 속성으로 싣고,
 * 값의 종류가 몇 개로 정해진 것(`labels`)만 메트릭 라벨로도 쓴다.
 */
export interface PageContext {
  attributes: Attributes
  labels: Attributes
}

export type DeviceClass = 'mobile' | 'tablet' | 'desktop'
export type BrowserEngine = 'chromium' | 'webkit' | 'gecko' | 'other'
export type ReferrerType =
  | 'direct'
  | 'internal'
  | 'search'
  | 'social'
  | 'ai'
  | 'other'

export function langOf(pathname: string): 'ko' | 'en' {
  return /^\/en(\/|$)/.test(pathname) ? 'en' : 'ko'
}

/** 화면 폭으로 나눈다. 블로그 레이아웃이 바뀌는 지점과 같다. */
export function deviceClassOf(width: number): DeviceClass {
  if (width < 768) return 'mobile'
  if (width < 1024) return 'tablet'
  return 'desktop'
}

/** iOS 의 브라우저는 이름과 상관없이 WebKit 이라 먼저 가른다. */
export function browserEngineOf(userAgent: string): BrowserEngine {
  if (/\b(CriOS|FxiOS|EdgiOS)\//.test(userAgent)) return 'webkit'
  if (/\bFirefox\//.test(userAgent)) return 'gecko'
  if (/\b(Chrome|Chromium|Edg)\//.test(userAgent)) return 'chromium'
  if (/\bAppleWebKit\//.test(userAgent)) return 'webkit'
  return 'other'
}

const AI_HOSTS =
  /(^|\.)(chatgpt\.com|openai\.com|perplexity\.ai|claude\.ai|gemini\.google\.com|copilot\.microsoft\.com)$/
const SEARCH_HOSTS =
  /(^|\.)(google\.[a-z.]+|bing\.com|naver\.com|daum\.net|duckduckgo\.com|yahoo\.[a-z.]+|baidu\.com|yandex\.[a-z.]+|ecosia\.org)$/
const SOCIAL_HOSTS =
  /(^|\.)(t\.co|x\.com|twitter\.com|facebook\.com|linkedin\.com|threads\.net|threads\.com|reddit\.com|news\.ycombinator\.com|bsky\.app)$/

/** 유입 경로. referrer 는 origin 만 오는 경우가 많아 host 로만 판단한다. */
export function referrerTypeOf(
  referrer: string,
  currentHost: string,
): ReferrerType {
  if (!referrer) return 'direct'

  let host: string
  try {
    host = new URL(referrer).hostname
  } catch {
    return 'other'
  }

  const site = currentHost.replace(/^www\./, '')
  if (host === site || host.endsWith(`.${site}`)) return 'internal'
  // gemini.google.com 이 검색보다 먼저 걸리게 AI 를 먼저 본다.
  if (AI_HOSTS.test(host)) return 'ai'
  if (SEARCH_HOSTS.test(host)) return 'search'
  if (SOCIAL_HOSTS.test(host)) return 'social'
  return 'other'
}

type NavigatorExtras = Navigator & {
  connection?: {
    effectiveType?: string
    rtt?: number
    downlink?: number
    saveData?: boolean
  }
  deviceMemory?: number
}

/** 브라우저에서 지금 페이지의 맥락을 읽는다. 없는 값은 `unknown` 이나 -1 로 채운다. */
export function readPageContext(): PageContext {
  const nav = navigator as NavigatorExtras
  const [navigation] = performance.getEntriesByType(
    'navigation',
  )
  const {pathname} = window.location
  const width = window.innerWidth

  const labels = {
    'page.route': routeOf(pathname),
    'page.lang': langOf(pathname),
    'device.class': deviceClassOf(width),
    'browser.engine': browserEngineOf(nav.userAgent),
  }

  return {
    labels,
    attributes: {
      ...labels,
      'page.path': pathname.slice(0, 200),
      'page.referrer_type': referrerTypeOf(
        document.referrer,
        window.location.hostname,
      ),
      // 내비게이션 자체가 서비스 워커를 거쳤는지. 보고 시점의 controller 와 달리 첫 방문에도 맞다.
      'page.sw_controlled': (navigation?.workerStart ?? 0) > 0 ? 'yes' : 'no',
      'viewport.width': width,
      'network.effective_type': nav.connection?.effectiveType ?? 'unknown',
      'network.rtt': nav.connection?.rtt ?? -1,
      'network.downlink': nav.connection?.downlink ?? -1,
      'network.save_data': nav.connection?.saveData ?? false,
      'device.memory': nav.deviceMemory ?? -1,
      'device.cores': nav.hardwareConcurrency ?? -1,
    },
  }
}
