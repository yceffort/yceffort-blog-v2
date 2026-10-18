import type {Locale} from '@/utils/postPaths'

interface Job {
  company: string
  role: string
  period: string
  description: string
  contributions: string[]
  stack: string
  link?: {href: string; label: string}
}

interface Entry {
  period: string
  title: string
  // '/'로 시작하면 사이트 안 링크
  href?: string
  description: string
  honor?: string
}

interface Heading {
  title: string
  // 한국어 제목 옆에 붙는 영어 부제. 영어 페이지에서는 제목과 같아 생략한다.
  sub?: string
}

export interface ResumeContent {
  eyebrow: string
  title: string
  lead: string
  status: string
  expertise: {term: string; detail: string}[]
  navLabel: string
  sections: {id: string; label: string}[]
  experience: Heading & {note: string; jobs: Job[]}
  publications: Heading & {
    books: {title: string; role: string; subject: string; href: string}[]
    publisherLink: string
    writing: {eyebrow: string; title: string; href: string; note: string}
  }
  openSource: Heading
  activities: Heading & {entries: Entry[]}
  education: Heading & {entries: Entry[]}
}

const ko: ResumeContent = {
  eyebrow: 'EXPERIENCE & CONTRIBUTIONS',
  title: '서비스 개발부터 팀이 일하는 기반까지.',
  lead: '2014년부터 다양한 도메인의 소프트웨어를 만들고 운영해 왔습니다. 프론트엔드 팀을 이끌고, 여러 서비스가 함께 쓰는 라이브러리와 도구를 만들었습니다. 그 과정에서 얻은 경험을 책과 글, 오픈소스로 나눕니다.',
  status: '프론트엔드 엔지니어, 재직 중',
  expertise: [
    {term: '서비스 개발', detail: 'React, TypeScript, Next.js'},
    {term: '운영과 성능', detail: 'Node.js, Kubernetes, 웹 성능'},
    {term: '팀의 개발 기반', detail: '공통 라이브러리, CLI, 코드 리뷰'},
  ],
  navLabel: '이력서 목차',
  sections: [
    {id: 'experience', label: '경력'},
    {id: 'publications', label: '저술과 번역'},
    {id: 'open-source', label: '오픈소스'},
    {id: 'activities', label: '발표와 활동'},
    {id: 'education', label: '학력'},
  ],
  experience: {
    title: '경력',
    sub: 'Experience',
    note: '이전에 근무한 곳에서의 경험입니다.',
    jobs: [
      {
        company: '네이버 파이낸셜',
        role: 'Frontend Engineer, 내자산&회원FE 리더 → 카드FE 리더/공통개발TF 리더',
        period: '2020.09 ~ 2026.03',
        description:
          '최대 14명 규모의 프론트엔드 팀을 이끌며 금융 서비스 개발과 프론트엔드 조직의 공통 기반 구축',
        contributions: [
          'CRA 단일 서비스를 Koa와 Next.js 기반 20여 개 SSR 서비스로 분리하는 전환을 주도해, 장애가 서비스 단위로 격리되고 서비스마다 더 자주 배포할 수 있는 구조로 개선',
          '20여 개 서비스가 함께 쓰는 공통 패키지(@mydata, @financial, @financial-auth, @pie) 개발을 이끌고 레지스트리 배포 체계 마련',
          '번들 크기 최적화를 주도해 모든 페이지가 공유하는 _app 청크 345kB(51%), 110개 페이지 합계 1.17MB를 줄이고 초기 로딩 평균을 2.6초에서 1.4초로 단축',
          'NaverPayDev 오픈소스 공개 체계를 만들고 팀의 공통 패키지 공개와 운영을 주도',
          'lodash를 대체하는 유틸리티 라이브러리 hidash 개발',
          '마이데이터, 회원 인증, 카드탭 등 모바일 금융 서비스 개발',
          '상태 관리 라이브러리, 디자인 시스템, 프로젝트 생성 도구, Node.js CLI, 업무용 MCP 개발과 운영',
          '팀의 코드 리뷰와 멘토링을 주도하고 인턴 채용 면접과 채용 과제, 실습 과제 출제 및 운영',
        ],
        stack:
          'React, TypeScript, Next.js, Node.js, Kubernetes, pnpm, Turborepo',
        link: {
          href: 'https://github.com/orgs/NaverPayDev/repositories',
          label: '공개 패키지 보기',
        },
      },
      {
        company: '트리플',
        role: 'Frontend Engineer',
        period: '2019.05 ~ 2020.08',
        description: '여행의 탐색부터 예약과 결제까지 이어지는 웹 서비스 개발',
        contributions: [
          '글로벌 도시 검색과 항공권 판매 서비스 개발과 운영',
          '결제 서비스와 공통 컴포넌트를 JavaScript에서 TypeScript로 전환',
          '판매자와 사용자 간 1:1 문의 채팅 서비스 개발',
        ],
        stack: 'React, TypeScript, Next.js, Koa, Docker, Amazon ECS',
      },
      {
        company: '카카오',
        role: 'Software Engineer',
        period: '2016.09 ~ 2019.05',
        description: '카카오 같이가치의 프론트엔드와 서버 개발, 운영',
        contributions: [
          'AngularJS 기반 레거시 애플리케이션을 Angular로 전환',
          '관리자 도구를 개발하고 Ruby on Rails 기반 서버를 함께 운영',
        ],
        stack: 'Angular, TypeScript, Ruby on Rails, MySQL, Redis',
      },
      {
        company: '삼성SDS',
        role: 'Software Engineer',
        period: '2014.03 ~ 2016.06',
        description: '글로벌 의료 솔루션 개발과 신기술 개념 검증',
        contributions: [
          '해외 병원용 EHR 솔루션과 간호 병동 서비스 개발',
          'iOS 비콘 기반 실내 내비게이션과 금융 고객사 블록체인 PoC 개발',
          'Python을 활용한 데이터 정합성과 유효성 검증',
        ],
        stack: 'Delphi, Oracle DB, Spring, JavaScript, Python',
      },
    ],
  },
  publications: {
    title: '저술과 번역',
    sub: 'Publications',
    books: [
      {
        title: '프런트엔드 성능 최적화 Deep Dive',
        role: '단독 저자',
        subject: '네트워크부터 브라우저와 프레임워크까지, 성능을 이해하는 원리',
        href: 'https://wikibook.co.kr/frontend-optimization/',
      },
      {
        title: 'npm Deep Dive',
        role: '공동 저자',
        subject: '모듈 시스템, 패키지 관리와 JavaScript 생태계',
        href: 'https://wikibook.co.kr/npm-deep-dive/',
      },
      {
        title: '모던 리액트 Deep Dive',
        role: '단독 저자',
        subject: 'React의 동작 원리와 웹 애플리케이션 개발',
        href: 'https://wikibook.co.kr/react-deep-dive/',
      },
      {
        title: '리액트 인터뷰 가이드',
        role: '번역',
        subject: '질문과 답으로 살펴보는 React 개발 지식',
        href: 'https://wikibook.co.kr/react-interview-guide/',
      },
    ],
    publisherLink: '출판사에서 보기',
    writing: {
      eyebrow: 'WRITING NOW',
      title: '남은 판단은 누가 배우는가',
      href: '/2026/09/who-learns-to-judge-beta-reader',
      note: 'AI 시대의 개발자, 판단과 학습에 관한 에세이(가제, 집필 중)',
    },
  },
  openSource: {title: '오픈소스', sub: 'Open source'},
  activities: {
    title: '발표와 활동',
    sub: 'Beyond work',
    entries: [
      {
        period: '2024',
        title: 'DAN24 컨퍼런스 발표',
        description: '웹 서비스 번들 사이즈 최적화를 주제로 발표했습니다.',
      },
      {
        period: '2021.01 ~ 2021.12',
        title: 'SSAFY 멘토',
        description:
          '삼성 청년 소프트웨어 아카데미에서 프로그래밍 질의응답과 칼럼 작성으로 청년 개발자들의 학습을 도왔습니다.',
      },
      {
        period: '2018 ~ 현재',
        title: '기술 블로그 yceffort.kr',
        href: '/',
        description:
          '블로그를 직접 개발하고 운영하며 프레임워크 내부 동작, 성능과 운영, 개발 도구, AI와 함께 일하는 경험을 기록합니다.',
      },
      {
        period: '2016.03 ~ 현재',
        title: '사단법인 점프 이사',
        href: 'https://jumpsp.org/',
        description:
          'Google Cloud Platform을 활용한 업무 자동화와 프로그래밍 교육에 참여하고 있습니다.',
      },
    ],
  },
  education: {
    title: '학력',
    sub: 'Education',
    entries: [
      {
        period: '2018.03 ~ 2020.02',
        title: '한국과학기술원 (KAIST)',
        description: '기술경영전문대학원 석사',
        honor: 'Highest Honor, GPA 4.23 / 4.3',
      },
      {
        period: '2007.03 ~ 2014.02',
        title: '동국대학교',
        description: '국제통상학 전공, 영어통번역학 복수전공 학사',
      },
    ],
  },
}

const en: ResumeContent = {
  eyebrow: 'EXPERIENCE AND CONTRIBUTIONS',
  title: 'From building services to the ground a team works on.',
  lead: 'I have built and run software across many domains since 2014. I have led a frontend team and built libraries and tools shared by many services. I pass on what I learned through books, writing, and open source.',
  status: 'Frontend engineer, currently employed',
  expertise: [
    {term: 'Building services', detail: 'React, TypeScript, Next.js'},
    {
      term: 'Operations and performance',
      detail: 'Node.js, Kubernetes, web performance',
    },
    {
      term: 'Team foundations',
      detail: 'Shared libraries, CLIs, code review',
    },
  ],
  navLabel: 'Resume contents',
  sections: [
    {id: 'experience', label: 'Experience'},
    {id: 'publications', label: 'Publications'},
    {id: 'open-source', label: 'Open source'},
    {id: 'activities', label: 'Beyond work'},
    {id: 'education', label: 'Education'},
  ],
  experience: {
    title: 'Experience',
    note: 'Experience from the places I have worked.',
    jobs: [
      {
        company: 'NAVER Financial',
        role: 'Frontend Engineer, Assets & Membership FE Lead → Card FE Lead / Shared Development TF Lead',
        period: '2020.09 to 2026.03',
        description:
          'Led a frontend team of up to 14 people, building financial services and the shared foundation of the frontend organization',
        contributions: [
          'Led the move from a single CRA app to 20-plus SSR services on Koa and Next.js, so a failure stays inside one service and each service can deploy on its own, more often',
          'Led the shared packages used by those 20-plus services (@mydata, @financial, @financial-auth, @pie) and set up how they are published to the registry',
          'Led bundle size work that cut the _app chunk shared by every page by 345 kB (51%) and 110 pages by 1.17 MB in total, bringing the average initial load from 2.6 s to 1.4 s',
          "Started the NaverPayDev open source program and led publishing and maintaining the team's shared packages",
          'Built hidash, a lodash replacement',
          'Built mobile financial services including MyData, member authentication, and the card tab',
          'Built and maintained a state management library, the design system, a project scaffolding tool, Node.js CLIs, and MCP servers for everyday productivity',
          'Led code review and mentoring on the team, ran intern hiring interviews, and created and ran hiring assignments and intern practice assignments',
        ],
        stack:
          'React, TypeScript, Next.js, Node.js, Kubernetes, pnpm, Turborepo',
        link: {
          href: 'https://github.com/orgs/NaverPayDev/repositories',
          label: 'See the public packages',
        },
      },
      {
        company: 'Triple',
        role: 'Frontend Engineer',
        period: '2019.05 to 2020.08',
        description:
          'Built a travel web service that runs from discovery through booking and payment',
        contributions: [
          'Built and operated global city search and flight ticket sales',
          'Migrated the payment service and shared components from JavaScript to TypeScript',
          'Built a one-to-one chat service between sellers and customers',
        ],
        stack: 'React, TypeScript, Next.js, Koa, Docker, Amazon ECS',
      },
      {
        company: 'Kakao',
        role: 'Software Engineer',
        period: '2016.09 to 2019.05',
        description:
          'Built and operated the frontend and server of Kakao Together, a donation platform',
        contributions: [
          'Migrated a legacy AngularJS application to Angular',
          'Built admin tools and operated the Ruby on Rails server alongside them',
        ],
        stack: 'Angular, TypeScript, Ruby on Rails, MySQL, Redis',
      },
      {
        company: 'Samsung SDS',
        role: 'Software Engineer',
        period: '2014.03 to 2016.06',
        description:
          'Built global healthcare solutions and proofs of concept for new technology',
        contributions: [
          'Built an EHR solution for overseas hospitals and a nursing ward service',
          'Built iOS beacon based indoor navigation and a blockchain PoC for a financial client',
          'Validated data consistency and integrity with Python',
        ],
        stack: 'Delphi, Oracle DB, Spring, JavaScript, Python',
      },
    ],
  },
  publications: {
    title: 'Publications',
    books: [
      {
        title: 'Frontend Performance Optimization Deep Dive',
        role: 'Author',
        subject:
          'From the network to the browser and frameworks, the principles behind performance',
        href: 'https://wikibook.co.kr/frontend-optimization/',
      },
      {
        title: 'npm Deep Dive',
        role: 'Co-author',
        subject:
          'Module systems, package management, and the JavaScript ecosystem',
        href: 'https://wikibook.co.kr/npm-deep-dive/',
      },
      {
        title: 'Modern React Deep Dive',
        role: 'Author',
        subject: 'How React works and how to build web applications with it',
        href: 'https://wikibook.co.kr/react-deep-dive/',
      },
      {
        title: 'React Interview Guide',
        role: 'Translator',
        subject: 'React knowledge through questions and answers',
        href: 'https://wikibook.co.kr/react-interview-guide/',
      },
    ],
    publisherLink: 'Publisher page (Korean)',
    writing: {
      eyebrow: 'WRITING NOW',
      title: 'Who Learns the Judgment That Remains',
      href: '/2026/09/who-learns-to-judge-beta-reader',
      note: 'An essay on developers, judgment, and learning in the age of AI. Working title, in progress (Korean).',
    },
  },
  openSource: {title: 'Open source'},
  activities: {
    title: 'Beyond work',
    entries: [
      {
        period: '2024',
        title: 'Talk at DAN24',
        description: 'Spoke about optimizing the bundle size of web services.',
      },
      {
        period: '2021.01 to 2021.12',
        title: 'SSAFY mentor',
        description:
          'Supported young developers at the Samsung Software Academy For Youth by answering programming questions and writing columns.',
      },
      {
        period: '2018 to present',
        title: 'Tech blog yceffort.kr',
        href: '/en',
        description:
          'I build and run this blog myself, and write about framework internals, performance and operations, developer tools, and working with AI.',
      },
      {
        period: '2016.03 to present',
        title: 'Board member, JUMP (nonprofit)',
        href: 'https://jumpsp.org/',
        description:
          'I help with work automation on Google Cloud Platform and with programming education.',
      },
    ],
  },
  education: {
    title: 'Education',
    entries: [
      {
        period: '2018.03 to 2020.02',
        title: 'KAIST',
        description: 'M.S., Graduate School of Management of Technology',
        honor: 'Highest Honors, GPA 4.23 / 4.3',
      },
      {
        period: '2007.03 to 2014.02',
        title: 'Dongguk University',
        description:
          'B.A. in International Trade, double major in English Interpretation and Translation',
      },
    ],
  },
}

export const resumeContent: Record<Locale, ResumeContent> = {ko, en}
