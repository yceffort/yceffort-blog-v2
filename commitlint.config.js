module.exports = {
  parserPreset: {
    parserOpts: {
      headerPattern: /(?:(\[\w+\]\s))?(?:(:\w+:)\s)(.+)/,
      headerCorrespondence: ['issue', 'type', 'subject'],
    },
  },
  plugins: [
    {
      rules: {
        'header-match-team-pattern': (parsed) => {
          const { type, subject } = parsed
          if (!type || !subject) {
            return [false, "header must be in format ':gitmoji: subject'"]
          }
          return [true, '']
        },
        'gitmoji-type-enum': (parsed, _when, expectedValue) => {
          const { type } = parsed
          if (type && !expectedValue.includes(type)) {
            return [
              false,
              `type must be one of ${expectedValue}. see https://gitmoji.dev`,
            ]
          }
          return [true, '']
        },
      },
    },
  ],
  rules: {
    'header-match-team-pattern': [2, 'always'],
    'gitmoji-type-enum': [
      2,
      'always',
      [
        ':bug:', // 버그
        ':sparkles:', // 신규
        ':recycle:', // 리팩토링
        ':fire:', // 삭제
        ':memo:', // 블로그 글
        ':boom:', // breaking changes
        ':lipstick:', // 마크업, 스타일
        ':green_heart:', // ci 수정
        ':wrench:', // 환경 변경
        ':truck:', // 이름경로 변경
        ':rocket:', // 배포
        ':label:', // 타입
        ':package:', // 패키지 신규 설치
      ],
    ],
  },
}
