// Stage layouts for the FEConf SDK talk. Content and speaker notes stay in Markdown.
export const feconfTheme = `
/* @theme feconf */
@import 'kakao';

section {
  --marp-slide-padding: 100px 72px 76px;
  --bgColor-default: #f7f7f2;
  --bgColor-muted: #eeeee7;
  --bgColor-neutral-muted: #e7e7de;
  --fgColor-default: #20211f;
  --fgColor-muted: #62645d;
  --fgColor-accent: #20211f;
  --borderColor-default: #d8d9ce;
  --borderColor-muted: #dfe0d7;
  --paginate-color: #62645d;
  --ink: #20211f;
  --h1-color: #20211f;
  --heading-strong-color: #20211f;
  background: var(--bgColor-default);
  color: var(--fgColor-default);
  font-size: 27px;
  font-weight: 400;
  line-height: 1.5;
  letter-spacing: -0.035em;
  justify-content: center;
}

section::before {
  top: 48px;
  left: 72px;
  right: 72px;
  height: 1px;
  background: var(--borderColor-default);
}
section::after {
  bottom: 28px;
  right: 72px;
  font-size: 15px;
  letter-spacing: 0;
}
section header {
  top: 23px;
  left: 72px;
  right: 320px;
  color: var(--fgColor-muted);
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.12em;
}
section footer {
  top: 23px;
  right: 72px;
  bottom: auto;
  left: auto;
  color: var(--fgColor-muted);
  font-size: 14px;
  font-weight: 500;
  letter-spacing: 0;
}
section h1, section h2, section h3 {
  color: var(--fgColor-default);
  letter-spacing: -0.055em;
  text-wrap: balance;
}
section h1 { font-size: 68px; line-height: 1.2; font-weight: 700; }
section h2 {
  font-size: 40px;
  font-weight: 650;
  line-height: 1.25;
  padding: 0;
  margin: 0 0 30px;
  border: 0;
}
section h3 { font-size: 29px; font-weight: 550; }
section p { margin: 0 0 18px; }
section strong {
  color: inherit;
  background: none;
  padding: 0;
  font-weight: 550;
}
section > p strong, section table strong { font-weight: 500; }
section a {
  color: inherit;
  text-decoration-color: var(--borderColor-default);
  text-decoration-thickness: 2px;
  text-underline-offset: 5px;
}
section ul, section ol { padding-left: 1.2em; }
section li { margin: 12px 0; }
section li::marker { color: inherit; }
section code { background: var(--bgColor-muted); border-radius: 3px; }
section pre {
  border: 1px solid #363830;
  border-radius: 5px;
  background: #20231f;
  padding: 22px 26px;
  margin: 0 0 24px;
  font-size: 21px;
  line-height: 1.65;
  letter-spacing: -0.03em;
}
section pre code { line-height: inherit; }
/* Explanatory slides stay quiet; color is opt-in for turning points. */
section blockquote {
  background: none;
  color: var(--fgColor-default);
  border: 0;
  border-radius: 0;
  padding: 0;
  font-size: inherit;
  margin: 12px 0 18px;
}
section blockquote p:last-child { margin-bottom: 0; }
section blockquote strong { font-weight: 600; }
section table {
  display: table;
  width: 100%;
  font-size: 23px;
  line-height: 1.4;
  border: 0;
  border-radius: 0;
  margin: 0 0 24px;
}
section table th {
  background: transparent;
  color: var(--fgColor-muted);
  padding: 10px 16px;
  border-bottom: 2px solid var(--ink);
  font-size: 18px;
  font-weight: 500;
}
section table td { padding: 12px 16px; border-color: var(--borderColor-default); }
section table tr, section table tr:nth-child(2n) { background: transparent; }
section table tbody tr:last-child td { border-bottom: 1px solid var(--borderColor-default); }
section table code { background: transparent; padding: 0; }
section p > img:only-child { border: 0; border-radius: 3px; }

/* Fine print is explicit emphasis, so prose never shrinks by position. */
section > p:has(> em:only-child) {
  color: var(--fgColor-muted);
  font-size: 17px;
  line-height: 1.5;
  letter-spacing: -0.015em;
  margin: 10px 0 0;
}
section > p > em:only-child { font-style: normal; }

section.dark, section.cover, section.verdict, section.closing {
  --bgColor-default: #20231f;
  --fgColor-default: #f7f7ef;
  --fgColor-muted: #b9bdb0;
  --borderColor-default: #51564b;
  --paginate-color: #b9bdb0;
  --h1-color: #f7f7ef;
  --heading-strong-color: #f7f7ef;
}
section.dark strong, section.verdict strong { color: var(--yellow); }

/* Opening: the result is a typographic object, not a background decoration. */
section.cover { justify-content: center; }
section.cover h1 {
  width: 715px;
  font-size: 76px;
  font-weight: 750;
  line-height: 1.2;
  margin: -28px 0 24px;
}
section.cover h2 {
  width: 640px;
  font-size: 28px;
  line-height: 1.5;
  color: #d2d5c9;
  font-weight: 500;
  margin: 0;
}
section.cover > p {
  position: absolute;
  bottom: 105px;
  font-size: 22px;
  color: #d2d5c9;
}
section.cover > p strong { color: white; margin-right: 14px; }
section.cover blockquote {
  background: var(--yellow);
  color: var(--ink);
  position: absolute;
  right: 0;
  top: 155px;
  width: 420px;
  height: 354px;
  box-sizing: border-box;
  padding: 54px 32px;
  transform: none;
  border-radius: 0;
  font-size: 19px;
  letter-spacing: -0.02em;
}
section.cover blockquote strong {
  display: block;
  font-size: 130px;
  line-height: 1.15;
  letter-spacing: -0.085em;
  font-weight: 850;
  margin-bottom: 22px;
  white-space: nowrap;
}

/* Chapter cards: a consistent oversized chapter numeral. */
section.chapter { background: var(--yellow); }
section.chapter h1 { width: 885px; font-size: 72px; margin-bottom: 28px; }
section.chapter h2 { font-size: 31px; font-weight: 550; width: 840px; }
section.chapter > p { width: 840px; font-size: 26px; }
section.chapter blockquote {
  position: absolute;
  right: 72px;
  bottom: 76px;
  padding: 0;
  margin: 0;
  font-size: 156px;
  line-height: 1;
  letter-spacing: -0.08em;
  font-weight: 800;
  background: none;
}
section.chapter::before { background: #a99b21; }
section.chapter header, section.chapter footer { color: #635c19; }

/* Two-column reasoning and numeric contrasts. */
section.columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  column-gap: 32px;
  align-content: center;
}
section.columns > h2, section.columns > p { grid-column: 1 / -1; }
section.columns > blockquote {
  align-self: stretch;
  padding: 30px;
  margin: 0;
  font-size: 25px;
  line-height: 1.5;
  background: #e9eae1;
}
/* Keep spacing outside the cards: Marp resets the last content item's margin. */
section.columns > blockquote + p { margin-top: 26px; }
section.columns blockquote h3 { margin: 0 0 18px; font-size: 32px; }
section.columns blockquote p { margin: 0 0 14px; }
section.columns blockquote p:last-child { margin-bottom: 0; }
section.columns blockquote strong { font-weight: 500; }
section.contrast blockquote strong {
  display: block;
  font-size: 98px;
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -0.07em;
  margin: 10px 0 28px;
}
section.contrast blockquote { font-size: 25px; }
section.contrast blockquote:first-of-type { background: var(--yellow); }

section.profile h2 { margin-bottom: 35px; }
section.profile > h3 { font-size: 64px; margin: 0 0 4px; }
section.profile > ul { font-size: 22px; margin: 8px 0 28px; }
section.profile blockquote { margin-bottom: 14px; font-size: 28px; }
section.code-focus pre { font-size: 32px; padding: 38px 34px; margin: 8px 0 30px; }

section.signal {
  display: grid;
  grid-template-columns: 380px 1fr;
  column-gap: 44px;
  align-content: center;
}
section.signal > h2, section.signal > p { grid-column: 1 / -1; }
section.signal blockquote {
  background: var(--yellow);
  display: flex;
  flex-direction: column;
  justify-content: center;
  margin: 0 0 28px;
  padding: 28px;
  font-size: 24px;
}
section.signal blockquote strong { font-size: 78px; font-weight: 800; line-height: 1.1; display: block; letter-spacing: -0.07em; }
section.signal table { align-self: center; font-size: 22px; }

section.evidence h2 { font-size: 38px; margin-bottom: 16px; }
section.evidence p:has(img) { margin-bottom: 12px; }
section.evidence img { height: 390px; width: auto; max-width: 100%; object-fit: contain; }
section.evidence > p { font-size: 23px; margin-bottom: 8px; }

/* Process diagrams are semantic lists, readable in raw Markdown and search. */
section.process > ol, section.recovery > ul {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 26px;
  padding: 0;
  margin: 0 0 28px;
  list-style: none;
  counter-reset: step;
}
section.process > ol > li, section.recovery > ul > li {
  position: relative;
  background: #e9eae1;
  margin: 0;
  padding: 24px;
  font-size: 23px;
  min-height: 136px;
  counter-increment: step;
}
section.process > ol > li::before {
  content: '0' counter(step);
  display: block;
  color: #66695f;
  font-family: 'JetBrains Mono', monospace;
  font-size: 18px;
  margin-bottom: 16px;
}
section.process > ol > li:not(:last-child)::after, section.recovery > ul > li:not(:last-child)::after {
  content: '→';
  position: absolute;
  right: -24px;
  top: 44%;
  font-size: 23px;
}
section.process > ol > li strong, section.recovery > ul > li strong { display: block; margin-bottom: 8px; }
section.recovery { font-size: 24px; }
section.recovery h2 { margin-bottom: 24px; }
section.recovery > ul { margin-bottom: 18px; }
section.recovery > ul > li { min-height: 110px; padding: 22px; }
section.recovery blockquote { font-size: 24px; margin: 0 0 18px; }
section.recovery > p { margin-bottom: 14px; }
section.recovery > ul > li:not(:last-child)::after { content: '+'; }
section.process pre { font-size: 22px; }
section:is(.process, .recovery, .flow, .takeaways) li strong + br { display: none; }

section.compact { font-size: 24px; }
section.compact h2 { font-size: 40px; margin-bottom: 24px; }
section.compact pre { font-size: 20px; padding: 16px 24px; margin-bottom: 20px; }
section.compact table { font-size: 22px; }
section.compact td { padding: 11px 16px; }
section.compact > p { margin-bottom: 16px; }
section.assembly pre { font-size: 21px; line-height: 1.5; }
section.assembly blockquote { font-size: 25px; }

/* Results: full data remains visible, with a proportional gzip bar in each row. */
section.results { --marp-slide-padding: 100px 72px 70px; }
section.results h2 { margin-bottom: 12px; }
section.results blockquote {
  background: none;
  padding: 0;
  margin: 0 0 18px;
  display: flex;
  align-items: baseline;
  gap: 26px;
  font-size: 30px;
}
section.results blockquote p { margin: 0; }
section.results blockquote strong { font-size: 92px; font-weight: 800; line-height: 1; letter-spacing: -0.07em; }
section.results table { font-size: 22px; margin-bottom: 8px; }
section.results table th { font-size: 17px; padding-top: 8px; padding-bottom: 8px; }
section.results table td { padding-top: 14px; padding-bottom: 14px; }
section.results tbody tr:first-child { background: var(--yellow); }
section.results tbody tr:first-child td:nth-child(3) strong { font-weight: 700; }
section.results tbody td:nth-child(3) {
  background-image: linear-gradient(var(--ink), var(--ink));
  background-repeat: no-repeat;
  background-origin: content-box;
  background-position: left bottom;
  background-size: var(--bar-width) 3px;
  padding-bottom: 19px;
}
section.results tbody tr:nth-child(1) { --bar-width: 31.09%; }
section.results tbody tr:nth-child(2) { --bar-width: 49%; }
section.results tbody tr:nth-child(3) { --bar-width: 69.55%; }
section.results tbody tr:nth-child(4) { --bar-width: 100%; }
section.results > p { font-size: 19px; margin: 8px 0 0; }

section.verdict { text-align: left; }
section.verdict h1 { font-size: 65px; margin: 0 0 20px; }
section.verdict h2 { font-size: 30px; font-weight: 500; color: #c8ccbf; }
section.verdict blockquote { background: none; color: var(--yellow); padding: 0; margin: 0 0 20px; }
section.verdict blockquote strong { font-size: 150px; font-weight: 800; line-height: 1; letter-spacing: -0.07em; }
section.verdict > p { font-size: 27px; border-top: 1px solid #51564b; padding-top: 24px; margin-top: 10px; }

section.mutation > blockquote { text-align: center; font-size: 52px; font-family: 'JetBrains Mono', monospace; padding: 22px; }
section.mutation > blockquote code { font-size: inherit; background: none; }
section.mutation pre { font-size: 21px; line-height: 1.5; }

section.flow { font-size: 24px; }
section.flow h2 { margin-bottom: 24px; }
section.flow > ol {
  counter-reset: flow;
  list-style: none;
  padding: 0;
  margin: 0 0 22px;
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 24px;
}
section.flow > ol > li {
  counter-increment: flow;
  position: relative;
  margin: 0;
  padding: 24px;
  background: #e9eae1;
  font-size: 23px;
  min-height: 142px;
}
section.flow > ol > li::before { content: '0' counter(flow); display: block; font-size: 16px; color: #63675a; margin-bottom: 14px; }
section.flow > ol > li:not(:last-child)::after { content: '→'; position: absolute; right: -23px; top: 44%; }
section.flow > ol > li:last-child { background: #dde0d4; }
section.flow > ol strong { display: block; margin-bottom: 12px; font-size: 27px; }
section.flow blockquote { font-size: 24px; margin: 0 0 18px; }

/* Harness: two SDKs fed the same input, with the verdict spanning both columns. */
section.harness {
  display: grid;
  grid-template-columns: 1fr 1fr;
  column-gap: 32px;
  align-content: center;
}
section.harness > h2 { grid-column: 1 / -1; margin-bottom: 26px; }
section.harness > blockquote {
  margin: 0;
  padding: 26px 30px;
  background: #e9eae1;
  font-size: 24px;
}
section.harness > blockquote h3 { margin: 0 0 12px; font-size: 30px; }
section.harness > blockquote p { margin: 0; }
section.harness > p {
  grid-column: 1 / -1;
  margin: 0;
  text-align: center;
  font-size: 25px;
  padding: 24px 0 20px;
}
section.harness > p::before { content: '↓'; display: block; font-size: 30px; line-height: 1; margin-bottom: 14px; color: #63675a; }
section.harness > blockquote:last-of-type {
  grid-column: 1 / -1;
  background: var(--yellow);
  text-align: center;
  font-size: 25px;
}

/* Lanes: two actors on one timeline. Numbering runs across both lanes in event order. */
section.lanes {
  display: grid;
  grid-template-columns: 1fr 1fr;
  column-gap: 32px;
  align-content: center;
  counter-reset: tick;
}
section.lanes > h2 { grid-column: 1 / -1; margin-bottom: 22px; }
section.lanes > blockquote {
  margin: 0;
  padding: 24px 28px;
  background: #e9eae1;
  font-size: 23px;
  align-self: start;
}
section.lanes > blockquote h3 { margin: 0 0 4px; font-size: 28px; }
section.lanes > blockquote:nth-of-type(-n + 2) > p { margin: 0 0 16px; font-size: 20px; color: #63675a; }
/* Numbering comes from the list's own start attribute, so the right lane continues the left. */
section.lanes > blockquote ol { list-style: none; padding: 0; margin: 0; }
section.lanes > blockquote li { position: relative; padding-left: 38px; margin: 0 0 11px; }
section.lanes > blockquote li:last-child { margin-bottom: 0; }
section.lanes > blockquote li::before {
  content: counter(list-item);
  position: absolute;
  left: 0;
  top: 1px;
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: var(--yellow);
  font-size: 16px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}
section.lanes > blockquote:last-of-type {
  grid-column: 1 / -1;
  margin: 26px 0 0;
  padding: 20px 28px;
  background: var(--yellow);
  font-size: 24px;
}
section.lanes > blockquote:last-of-type p { margin: 0; font-size: 24px; color: inherit; }
section.lanes > p { grid-column: 1 / -1; margin: 18px 0 0; font-size: 22px; }

section.limits blockquote { font-size: 25px; }
section.takeaways > ol { list-style: none; counter-reset: lesson; padding: 0; margin: 0; }
section.takeaways > ol > li {
  counter-increment: lesson;
  position: relative;
  padding: 18px 0 22px 94px;
  margin: 0;
  border-bottom: 1px solid var(--borderColor-default);
  font-size: 23px;
}
section.takeaways > ol > li::before { content: '0' counter(lesson); position: absolute; left: 0; top: 20px; font-size: 38px; font-weight: 500; letter-spacing: -0.05em; }
section.takeaways > ol strong { display: block; font-size: 29px; margin-bottom: 8px; }
section.takeaways.four-items > ol > li { padding-top: 12px; padding-bottom: 12px; }
section.takeaways.four-items > ol > li::before { top: 14px; }
section.takeaways > blockquote { margin-top: 36px; font-size: 29px; }

section.closing h1 { font-size: 76px; margin-bottom: 36px; }
section.closing h2 { color: var(--fgColor-default); font-size: 39px; font-weight: 500; margin: 0 0 30px; }
section.closing > p { font-size: 28px; margin-bottom: 14px; }
section.closing blockquote { background: none; color: var(--yellow); padding: 22px 0 0; border-top: 1px solid #51564b; margin-top: 24px; font-size: 28px; }

/* Trace: 두 카드에 같은 파일이 나오면 그 파일을 같은 색으로 묶는다. */
section.trace > blockquote { line-height: 1.8; }
section.trace > blockquote code { background: none; padding: 0; }
section.trace > blockquote strong code { background: var(--yellow); padding: 2px 6px; }
section.trace > pre { grid-column: 1 / -1; margin-top: 26px; }

section.resources h1 { font-size: 66px; margin: 0 0 4px; }
section.resources h2 { margin: 0 0 30px; }
section.resources > p { font-size: 25px; margin: 0 0 30px; }
section.resources > ul { list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 24px 40px; }
section.resources > ul > li { font-size: 21px; padding: 20px 0 0; margin: 0; border-top: 2px solid var(--ink); }
section.resources > ul > li a { display: block; font-size: 25px; font-weight: 500; margin-bottom: 10px; }
section.resources > ul > li a + br { display: none; }

/* Closing outcomes: five reductions above the related work. */
section.outcomes { --marp-slide-padding: 100px 72px 76px; }
section.outcomes h1 { font-size: 40px; font-weight: 650; line-height: 1.25; margin-bottom: 10px; }
section.outcomes > p { font-size: 22px; margin-bottom: 24px; }
section.outcomes table { table-layout: fixed; text-align: left; margin-bottom: 30px; }
section.outcomes table th { text-align: left; font-size: 19px; padding: 10px 12px; }
section.outcomes table td { padding: 16px 12px; font-size: 17px; }
section.outcomes > p:has(> em:only-child) { margin-bottom: 28px; }
section.outcomes table strong { display: block; font-size: 46px; font-weight: 600; line-height: 1.2; letter-spacing: -0.06em; margin-bottom: 4px; white-space: nowrap; }
section.outcomes h2 { font-size: 26px; font-weight: 500; margin-bottom: 16px; }
section.outcomes > ul { gap: 18px 40px; }
section.outcomes > ul > li { padding-top: 12px; font-size: 17px; border-top-width: 1px; }
section.outcomes > ul > li a { font-size: 22px; margin-bottom: 7px; }
section.outcomes > ul > li a + br { display: none; }

section.hidden-slide { --marp-slide-padding: 88px 72px 78px; font-size: 22px; }
section.hidden-slide h2 { font-size: 36px; margin-bottom: 24px; }
section.hidden-slide pre { font-size: 18px; line-height: 1.5; padding: 16px 22px; margin-bottom: 18px; }
section.hidden-slide table { font-size: 21px; margin-bottom: 18px; }
section.hidden-slide td { padding: 10px 14px; }
section.hidden-slide p { margin-bottom: 12px; }

/* 본문 장은 위에서부터 채운다. 가운데 정렬이면 내용 양에 따라 제목이 장마다 오르내린다. */
section:not(.cover):not(.chapter):not(.closing):not(.profile) { justify-content: flex-start; align-content: start; }
`
