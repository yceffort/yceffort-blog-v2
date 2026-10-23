// 로딩 기록 그래픽의 문구. 경로가 /en으로 시작하면 영어를 쓴다.
import type {BarKind, PhaseKey} from '@/components/about/collectTrace'
import {useLocale} from '@/hooks/useLocale'

const ko = {
  loading: '이 탭의 로딩 기록을 읽는 중',
  reading: '기록을 읽는 중',
  caption:
    '방금 이 브라우저가 이 탭을 불러온 실제 기록입니다. 측정값은 브라우저 밖으로 보내지 않습니다.',
  replay: '다시 재생',
  hint: '드래그해서 돌리고, 두 손가락으로 벌려 확대하고 옮깁니다. 막대를 누르면 자세히 볼 수 있습니다',
  hintMouse:
    '드래그해서 돌리고, Ctrl+휠로 확대하고, Shift+드래그로 옮깁니다. 더블클릭하면 처음 구도로 돌아오고, 막대를 누르면 자세히 볼 수 있습니다',
  selfLabel: '이 그래픽의 코드',
  selfNote: '이 그래픽을 그리는 코드',
  mainThread: '메인 스레드를 막은 작업',
  longTask: '긴 작업',
  gap: (ms: number) => `⋯ ${ms}ms 생략 ⋯`,
  resources: (n: number) => `리소스 ${n}개`,
  hidden: (n: number) => `이후 요청 ${n}개 생략`,
  noWebgl: '이 브라우저에서는 WebGL을 쓸 수 없어 그래픽을 그리지 못했습니다.',
  timing: (start: number, length: number) =>
    `시작 ${start}ms, 길이 ${length}ms`,
  cached: '캐시 또는 서비스 워커(네트워크 전송 없음)',
  unknownSize: '다른 출처라 크기를 알 수 없음',
  transferred: (size: string) => `전송 ${size}`,
  close: '닫기',
  detailOf: (label: string) => `${label} 상세`,
  noPhases:
    '다른 출처라 단계별 시간을 알 수 없습니다(Timing-Allow-Origin 없음).',
  yes: '예',
  no: '아니요',
  row: {
    kind: '종류',
    start: '시작',
    length: '길이',
    transfer: '전송',
    decoded: '압축 해제 전후',
    protocol: '프로토콜',
    initiator: '요청 주체',
    blocking: '렌더링 차단',
  },
  kind: {
    document: '문서',
    script: '스크립트',
    css: '스타일',
    font: '폰트',
    image: '이미지',
    other: '기타',
  } satisfies Record<BarKind, string>,
  phase: {
    stalled: '대기',
    dns: 'DNS',
    connect: '연결',
    tls: 'TLS',
    ttfb: '첫 바이트 대기',
    download: '다운로드',
  } satisfies Record<PhaseKey, string>,
}

export type TraceText = typeof ko

const en: TraceText = {
  loading: 'Reading how this tab loaded',
  reading: 'Reading the record',
  caption:
    'This is the actual record of your browser loading this tab just now. The measurements never leave your browser.',
  replay: 'Replay',
  hint: 'Drag to rotate, pinch to zoom, two-finger drag to pan. Tap a bar for details.',
  hintMouse:
    'Drag to rotate, Ctrl+wheel to zoom, Shift+drag to pan. Double-click to reset. Click a bar for details.',
  selfLabel: 'Code for this graphic',
  selfNote: 'the code drawing this graphic',
  mainThread: 'Tasks that blocked the main thread',
  longTask: 'Long task',
  gap: (ms) => `⋯ ${ms}ms skipped ⋯`,
  resources: (n) => `${n} resources`,
  hidden: (n) => `${n} later requests skipped`,
  noWebgl:
    'WebGL is not available in this browser, so the graphic could not be drawn.',
  timing: (start, length) => `Start ${start}ms, duration ${length}ms`,
  cached: 'Cache or service worker (no network transfer)',
  unknownSize: 'Size unknown (cross-origin)',
  transferred: (size) => `Transferred ${size}`,
  close: 'Close',
  detailOf: (label) => `${label} details`,
  noPhases:
    'Phase timings are unavailable for this cross-origin request (no Timing-Allow-Origin).',
  yes: 'Yes',
  no: 'No',
  row: {
    kind: 'Type',
    start: 'Start',
    length: 'Duration',
    transfer: 'Transfer',
    decoded: 'Encoded → decoded',
    protocol: 'Protocol',
    initiator: 'Initiator',
    blocking: 'Render-blocking',
  },
  kind: {
    document: 'Document',
    script: 'Script',
    css: 'Stylesheet',
    font: 'Font',
    image: 'Image',
    other: 'Other',
  },
  phase: {
    stalled: 'Stalled',
    dns: 'DNS',
    connect: 'Connect',
    tls: 'TLS',
    ttfb: 'Waiting (TTFB)',
    download: 'Download',
  },
}

export function useTraceText(): TraceText {
  return useLocale().locale === 'en' ? en : ko
}
