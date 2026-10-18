import type {Locale} from '@/utils/postPaths'

interface OpenSourceProject {
  name: string
  tags: string[]
  href: string
  category: Record<Locale, string>
  description: Record<Locale, string>
}

export const openSourceProjects: OpenSourceProject[] = [
  {
    name: '@yceffort/coldpath',
    tags: ['rust', 'source-map', 'v8-coverage'],
    category: {ko: '개발 도구', en: 'Developer tool'},
    href: 'https://github.com/yceffort/coldpath',
    description: {
      ko: 'JavaScript 번들의 바이트를 소스맵과 V8 커버리지로 원본 파일까지 추적하는 Rust CLI입니다. 처음 실행되는 코드와 상호작용할 때만 실행되는 코드를 나누고, 번들 트리맵과 풀 리퀘스트 사이의 비교 보고서를 만듭니다. 분석은 Node.js나 브라우저 없이 오프라인으로 동작합니다.',
      en: 'A Rust CLI that traces JavaScript bundle bytes back to their original sources with source maps and V8 coverage. It separates code that runs on load from code that runs only on interaction, and produces bundle treemaps and pull request comparisons. Analysis runs offline, without Node.js or a browser.',
    },
  },
  {
    name: '@yceffort/number-flow',
    tags: ['typescript', 'javascript', 'react', 'web-component'],
    category: {ko: '포크와 확장', en: 'Fork and extension'},
    href: 'https://github.com/yceffort/number-flow',
    description: {
      ko: 'barvian/number-flow를 포크해 구형 브라우저와 웹뷰에서도 숫자 애니메이션이 동작하도록 확장했습니다. 기존 API를 유지하면서 requestAnimationFrame 기반 폴백 엔진을 구현하고 실제 구형 브라우저로 검증했습니다.',
      en: 'A fork of barvian/number-flow that makes its number animations work in older browsers and webviews. It keeps the original API, adds a fallback engine built on requestAnimationFrame, and was verified on real legacy browsers.',
    },
  },
  {
    name: '@yceffort/storage-inspector',
    tags: ['typescript', 'lit', 'web-component'],
    category: {ko: '웹뷰 디버깅', en: 'Webview debugging'},
    href: 'https://github.com/yceffort/storage-inspector',
    description: {
      ko: '웹뷰 안에서 localStorage와 sessionStorage를 조회하고 편집하는 Lit 웹 컴포넌트입니다. 키별 설명과 타입을 스키마로 정의하고, 값 검증과 편집을 지원하도록 만들었습니다.',
      en: 'A Lit web component for inspecting and editing localStorage and sessionStorage inside a webview. Each key gets a description and type from a schema, and values are validated as you edit them.',
    },
  },
  {
    name: '@yceffort/rust-markdownlint',
    tags: ['rust', 'rayon'],
    category: {ko: '개발 도구', en: 'Developer tool'},
    href: 'https://github.com/yceffort/rust-markdownlint',
    description: {
      ko: 'markdownlint-cli2 호환을 목표로 만든 Rust 기반 마크다운 린터입니다. 기존 도구와의 진단, 자동 수정 결과를 대조하고, 직접 운영하는 저장소에 적용했습니다.',
      en: 'A Markdown linter written in Rust that aims to be compatible with markdownlint-cli2. Its diagnostics and auto-fixes were checked against the original tool, and it runs on the repositories I maintain.',
    },
  },
  {
    name: 'yc.clipboard',
    tags: ['swift', 'swiftui', 'appkit'],
    category: {ko: 'macOS 앱', en: 'macOS app'},
    href: 'https://github.com/yceffort/yc-clipboard',
    description: {
      ko: '텍스트, 이미지, 파일 등 클립보드 이력을 검색하고 다시 사용하는 macOS 메뉴 막대 앱입니다. 항목 고정과 미리보기, 이전 앱으로 붙여넣기를 지원합니다.',
      en: 'A macOS menu bar app for searching and reusing your clipboard history, including text, images, and files. It supports pinning, previews, and pasting back into the previous app.',
    },
  },
]
