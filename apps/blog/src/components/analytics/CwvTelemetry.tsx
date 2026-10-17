'use client'

import {useEffect} from 'react'

import {SiteConfig} from '@/config'
import {routeOf} from '@/utils/telemetry/routes'

/**
 * CWV 를 OpenTelemetry 로 infra/otel 수집 서버에 보낸다. 페이지 로드(하드 내비게이션) 하나가 trace 하나다.
 * OTel SDK 는 첫 화면에 필요 없어서 따로 받는다. web-vitals 는 buffered 항목을 읽어 늦게 붙어도 LCP, CLS 를 놓치지 않는다.
 * 다만 SDK 를 받기 전의 104ms 미만 입력은 Event Timing 버퍼에 남지 않아 INP 후보에서 빠진다.
 */
export function CwvTelemetry() {
  useEffect(() => {
    void import('@/utils/telemetry/setup').then(({startTelemetry}) =>
      startTelemetry({
        collectorUrl: SiteConfig.otelCollectorUrl,
        route: routeOf(window.location.pathname),
      }),
    )
  }, [])

  return null
}
