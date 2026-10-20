// 표지용 deck-cover 클래스. 공유 미리보기처럼 작게 보여도 제목이 읽히도록 글자를 키운다.
// 색은 각 테마의 변수를 그대로 써서 invert와 함께 쓰면 지금 색을 유지한다
export const deckCoverCss = `
section.deck-cover {
  --marp-slide-padding: 64px 96px;
  justify-content: center;
  word-break: keep-all;
}
section.deck-cover::after {
  display: none;
}
section.deck-cover h1 {
  margin: 0 0 32px;
  padding: 0;
  border: 0;
  font-size: 76px;
  font-weight: 700;
  line-height: 1.2;
  letter-spacing: -0.01em;
  text-wrap: balance;
}
section.deck-cover h1 code {
  font-size: 0.8em;
  white-space: nowrap;
}
section.deck-cover h1::before {
  content: '';
  display: block;
  width: 88px;
  height: 8px;
  margin-bottom: 40px;
  border-radius: 4px;
  background: var(--fgColor-accent);
}
section.deck-cover h2 {
  margin: 0 0 24px;
  padding: 0;
  border: 0;
  font-size: 40px;
  color: var(--fgColor-default);
}
section.deck-cover p {
  margin: 0 0 12px;
  font-size: 36px;
  line-height: 1.45;
  color: var(--fgColor-default);
}
section.deck-cover blockquote {
  margin: 0 0 12px;
  font-size: 26px;
  color: var(--fgColor-muted);
}
section.deck-cover > p:last-child {
  margin-top: 36px;
  font-size: 26px;
  color: var(--fgColor-muted);
}
`
