import {afterEach, describe, expect, it, vi} from 'vitest'

import {
  browserEngineOf,
  deviceClassOf,
  langOf,
  readPageContext,
  referrerTypeOf,
} from '../context'

describe('langOf', () => {
  it.each([
    ['/', 'ko'],
    ['/2026/10/post', 'ko'],
    ['/en', 'en'],
    ['/en/2026/10/post', 'en'],
    ['/english', 'ko'],
  ])('%s 는 %s', (pathname, lang) => {
    expect(langOf(pathname)).toBe(lang)
  })
})

describe('deviceClassOf', () => {
  it.each([
    [375, 'mobile'],
    [767, 'mobile'],
    [768, 'tablet'],
    [1023, 'tablet'],
    [1024, 'desktop'],
  ])('폭 %i 는 %s', (width, device) => {
    expect(deviceClassOf(width)).toBe(device)
  })
})

describe('browserEngineOf', () => {
  it.each([
    [
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:159.0) Gecko/20100101 Firefox/159.0',
      'gecko',
    ],
    [
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36',
      'chromium',
    ],
    [
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36 Edg/156.0.0.0',
      'chromium',
    ],
    [
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Safari/605.1.15',
      'webkit',
    ],
    // iOS 의 Chrome 과 Firefox 는 WebKit 이다.
    [
      'Mozilla/5.0 (iPhone; CPU iPhone OS 26_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/156.0.0.0 Mobile/15E148 Safari/604.1',
      'webkit',
    ],
    [
      'Mozilla/5.0 (iPhone; CPU iPhone OS 26_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) FxiOS/159.0 Mobile/15E148 Safari/605.1.15',
      'webkit',
    ],
    ['curl/8.17.0', 'other'],
  ])('%s', (userAgent, engine) => {
    expect(browserEngineOf(userAgent)).toBe(engine)
  })
})

describe('referrerTypeOf', () => {
  it.each([
    ['', 'direct'],
    ['https://yceffort.kr/2026/09/post', 'internal'],
    ['https://research.yceffort.kr/', 'internal'],
    ['https://www.google.com/', 'search'],
    ['https://www.google.co.kr/', 'search'],
    ['https://search.naver.com/', 'search'],
    ['https://gemini.google.com/', 'ai'],
    ['https://chatgpt.com/', 'ai'],
    ['https://t.co/abc', 'social'],
    ['https://news.ycombinator.com/', 'social'],
    ['https://github.com/yceffort/blog', 'other'],
    ['not a url', 'other'],
  ])('%s 는 %s', (referrer, type) => {
    expect(referrerTypeOf(referrer, 'yceffort.kr')).toBe(type)
  })
})

describe('readPageContext', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('글 경로처럼 값이 끝없이 늘어나는 것은 메트릭 라벨에 넣지 않는다', () => {
    vi.stubGlobal('window', {
      location: {pathname: '/en/2026/10/post', hostname: 'yceffort.kr'},
      innerWidth: 390,
    })
    vi.stubGlobal('document', {referrer: 'https://www.google.com/'})
    vi.stubGlobal('navigator', {
      userAgent: 'Mozilla/5.0 (iPhone) AppleWebKit/605.1.15 CriOS/156.0',
      hardwareConcurrency: 6,
    })
    vi.stubGlobal('performance', {
      getEntriesByType: () => [{workerStart: 12}],
    })

    const {labels, attributes} = readPageContext()

    expect(labels).toEqual({
      'page.route': 'post',
      'page.lang': 'en',
      'device.class': 'mobile',
      'browser.engine': 'webkit',
    })
    expect(attributes).toMatchObject({
      ...labels,
      'page.path': '/en/2026/10/post',
      'page.referrer_type': 'search',
      'page.sw_controlled': 'yes',
      'network.effective_type': 'unknown',
      'device.cores': 6,
    })
  })
})
