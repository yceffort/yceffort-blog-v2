import {afterEach, describe, expect, it, vi} from 'vitest'

import {
  browserEngineOf,
  deviceClassOf,
  isAutomatedVisit,
  langOf,
  pageContextFor,
  readDeviceContext,
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

describe('pageContextFor', () => {
  afterEach(() => vi.unstubAllGlobals())

  function stubBrowser() {
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
  }

  it('글 경로처럼 값이 끝없이 늘어나는 것은 메트릭 라벨에 넣지 않는다', () => {
    stubBrowser()
    const device = readDeviceContext()
    const {labels, attributes} = pageContextFor(
      {key: 'hard', kind: 'hard', startTime: 0, pathname: '/en/2026/10/post'},
      device,
    )

    expect(labels).toEqual({
      'page.route': 'post',
      'page.lang': 'en',
      'device.class': 'mobile',
      'browser.engine': 'webkit',
    })
    expect(attributes).toMatchObject({
      ...labels,
      'page.path': '/en/2026/10/post',
      'page.navigation': 'hard',
      'page.referrer_type': 'search',
      'page.sw_controlled': 'yes',
      'network.effective_type': 'unknown',
      'device.cores': 6,
    })
  })

  it('SPA 이동은 그 이동의 경로로 화면을 정하고 유입은 internal 이다', () => {
    stubBrowser()
    const {labels, attributes} = pageContextFor(
      {key: 'soft:3', kind: 'soft', startTime: 4000, pathname: '/tags/react'},
      readDeviceContext(),
    )

    expect(labels['page.route']).toBe('tag')
    expect(labels['page.lang']).toBe('ko')
    expect(attributes['page.referrer_type']).toBe('internal')
    expect(attributes).not.toHaveProperty('page.sw_controlled')
  })
})

describe('isAutomatedVisit', () => {
  it.each([
    [
      'Mozilla/5.0 (Linux; Android 11) AppleWebKit/537.36 Chrome/156.0.0.0 Mobile Safari/537.36 Chrome-Lighthouse',
      false,
      true,
    ],
    [
      'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 HeadlessChrome/156.0.0.0 Safari/537.36',
      false,
      true,
    ],
    [
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36',
      true,
      true,
    ],
    [
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36',
      false,
      false,
    ],
    [
      'Mozilla/5.0 (iPhone; CPU iPhone OS 26_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.1 Mobile/15E148 Safari/604.1',
      false,
      false,
    ],
  ])('%s (webdriver %s) 는 %s', (userAgent, webdriver, automated) => {
    expect(isAutomatedVisit(userAgent, webdriver)).toBe(automated)
  })
})
