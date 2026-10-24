// FEConf: an engineering field report. Diagrams and data remain semantic Markdown.
// Fonts are self-hosted and collected by the existing offline deck downloader.
export const feconfTheme = `
/* @theme feconf */
@import 'default';

@font-face {
  font-family: 'IBM Plex Sans KR'; font-style: normal; font-weight: 400; font-display: swap;
  src: url('/fonts/ibm-plex/IBMPlexSansKR-Regular.woff2') format('woff2');
}
@font-face {
  font-family: 'IBM Plex Sans KR'; font-style: normal; font-weight: 600; font-display: swap;
  src: url('/fonts/ibm-plex/IBMPlexSansKR-SemiBold.woff2') format('woff2');
}
@font-face {
  font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 400; font-display: swap;
  src: url('/fonts/ibm-plex/IBMPlexMono-Regular.woff2') format('woff2');
}
@font-face {
  font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 600; font-display: swap;
  src: url('/fonts/ibm-plex/IBMPlexMono-SemiBold.woff2') format('woff2');
}

section {
  --marp-slide-padding: 96px 68px 64px;
  --paper: #fff;
  --ink: #171717;
  --muted: #626262;
  --rule: #d4d4d4;
  --yellow: #fee500;
  --blue: #2457e6;
  --code-surface: #eef4ff;
  --code-inline: #e5eeff;
  --mono: 'IBM Plex Mono', 'IBM Plex Sans KR', monospace;
  --bgColor-default: var(--paper);
  --bgColor-muted: var(--code-surface);
  --bgColor-neutral-muted: var(--code-inline);
  --fgColor-default: var(--ink);
  --fgColor-muted: var(--muted);
  --fgColor-accent: var(--ink);
  --borderColor-default: var(--rule);
  --borderColor-muted: var(--rule);
  --h1-color: var(--ink);
  --heading-strong-color: var(--ink);
  --paginate-color: var(--muted);
  color-scheme: light;
  background: var(--paper);
  color: var(--ink);
  font-family: 'IBM Plex Sans KR', sans-serif;
  font-size: 28px;
  font-weight: 400;
  line-height: 1.5;
  letter-spacing: 0;
  word-break: keep-all;
  justify-content: flex-start;
  align-content: start;
  padding: var(--marp-slide-padding);
}
section::before {
  content: '';
  position: absolute;
  top: 38px; left: 68px;
  width: 13px; height: 13px;
  background: var(--yellow);
  border: 1px solid var(--ink);
}
section header {
  top: 34px; left: 94px; right: 68px;
  font-family: var(--mono);
  font-size: 12px; font-weight: 400;
  letter-spacing: 0.08em;
  color: var(--muted);
}
section footer {
  left: 68px; right: auto; bottom: 25px; top: auto;
  font-size: 12px; color: var(--muted);
}
section::after {
  bottom: 22px; right: 68px;
  font-family: var(--mono); font-size: 14px;
  letter-spacing: 0;
}
section h1, section h2, section h3 {
  color: var(--ink);
  border: 0; padding: 0;
  letter-spacing: -0.025em;
  text-wrap: balance;
}
section h1 { font-size: 68px; font-weight: 600; line-height: 1.2; margin: 0 0 28px; }
section h2 { font-size: 38px; font-weight: 600; line-height: 1.3; margin: 0 0 32px; }
section h3 { font-size: 27px; font-weight: 600; line-height: 1.4; margin: 0 0 16px; }
section p { margin: 0 0 20px; }
section strong { color: inherit; font-weight: 600; }
section a { color: inherit; text-decoration: underline; text-decoration-color: #a3a3a3; text-underline-offset: 5px; }
section ul, section ol { padding-left: 1.25em; margin: 0 0 24px; }
section li { margin: 10px 0; }
section li::marker { color: var(--muted); }
section blockquote {
  color: var(--ink); background: none;
  border: 0; border-left: 4px solid var(--yellow);
  border-radius: 0; padding: 0 0 0 22px;
  margin: 8px 0 24px;
  font-size: 28px;
}
section blockquote p:last-child { margin-bottom: 0; }
section code {
  font-family: var(--mono); font-size: 0.88em;
  background: var(--code-inline); color: inherit;
  padding: 2px 5px; border-radius: 0;
  letter-spacing: 0;
}
section pre {
  --color-prettylights-syntax-comment: #5d6b82;
  --color-prettylights-syntax-constant: #154ca3;
  --color-prettylights-syntax-entity: #5d3778;
  --color-prettylights-syntax-keyword: #154ca3;
  --color-prettylights-syntax-string: #27603d;
  --color-prettylights-syntax-variable: #171717;
  --color-prettylights-syntax-storage-modifier-import: #154ca3;
  background: var(--code-surface); color: var(--ink);
  border: 0; border-top: 2px solid var(--blue); border-radius: 0;
  padding: 22px 26px; margin: 0 0 24px;
  font-family: var(--mono); font-size: 24px; line-height: 1.5;
  letter-spacing: 0;
}
section pre code { background: none; padding: 0; font-size: inherit; line-height: inherit; }
section table {
  display: table; width: 100%;
  border: 0; border-collapse: collapse; border-radius: 0;
  font-size: 24px; line-height: 1.45;
  margin: 0 0 26px;
}
section table tr, section table tr:nth-child(2n) { background: none; border: 0; }
section table th {
  border: 0; border-bottom: 2px solid var(--ink);
  color: var(--muted); font-size: 18px; font-weight: 400;
  padding: 10px 12px; background: none;
}
section table td { border: 0; border-bottom: 1px solid var(--rule); padding: 15px 12px; }
section table code { background: none; padding: 0; }
section p > img:only-child { display: block; max-width: 100%; margin: 0 auto; border: 0; border-radius: 0; }
section > p:has(> em:only-child) {
  font-size: 16px; line-height: 1.5; color: var(--muted);
  margin: 12px 0 0;
}
section > p > em:only-child { font-style: normal; }

/* Cover: a disassembled module drawing, followed by its measured result. */
section.cover {
  justify-content: center; align-content: center;
  background: linear-gradient(90deg, var(--yellow) 58%, var(--paper) 58%);
}
section.cover::before, section.chapter::before { background: var(--ink); }
section.cover h1 { font-size: 70px; width: 670px; margin: -30px 0 28px; line-height: 1.25; }
section.cover h2 { font-size: 29px; font-weight: 400; line-height: 1.5; width: 640px; margin: 0; }
section.cover > p:has(img) { position: absolute; width: 430px; right: 50px; top: 119px; margin: 0; }
section.cover > blockquote {
  position: absolute; left: 800px; top: 505px; width: 400px;
  border: 0; border-top: 2px solid var(--ink); padding: 18px 0 0; margin: 0;
  display: flex; align-items: baseline; gap: 18px; font-size: 17px;
}
section.cover > blockquote p { margin: 0; }
section.cover > blockquote strong { color: var(--blue); font-family: var(--mono); font-size: 58px; font-weight: 600; letter-spacing: -0.06em; }
section.cover > p:last-of-type { position: absolute; bottom: 100px; left: 68px; font-size: 20px; margin: 0; }
section.cover > p:last-of-type strong { margin-right: 12px; }

/* Chapter pages form one four-stage assembly route. */
section.chapter {
  --marp-slide-padding: 190px 68px 145px 256px;
  --paper: var(--yellow); --muted: #575022; --rule: #b7a500;
  justify-content: center; align-content: center;
}
section.chapter h1 { font-size: 62px; line-height: 1.3; margin-bottom: 26px; }
section.chapter h2 { font-size: 27px; font-weight: 400; line-height: 1.5; margin: 0; max-width: 850px; }
section.chapter > blockquote {
  position: absolute; left: 68px; top: 211px;
  border: 0; padding: 0; margin: 0;
  font-family: var(--mono); font-size: 104px; line-height: 1; font-weight: 400;
  letter-spacing: -0.08em;
}
section.chapter > ol {
  position: absolute; left: 68px; right: 68px; bottom: 85px;
  list-style: none; display: grid; grid-template-columns: repeat(4, 1fr);
  padding: 0; margin: 0; gap: 0;
}
section.chapter > ol > li { border-top: 1px solid var(--rule); padding-top: 16px; margin: 0; font-size: 18px; color: var(--muted); }
section.chapter > ol > li:has(strong) { border-top: 4px solid var(--blue); padding-top: 13px; color: var(--ink); }
section.chapter > ol strong { background: var(--blue); color: #fff; padding: 3px 9px; }

section.profile { --marp-slide-padding: 130px 68px 80px 320px; justify-content: center; align-content: center; }
section.profile > p:has(img) { position: absolute; left: 68px; top: 218px; width: 190px; margin: 0; border-bottom: 8px solid var(--yellow); }
section.profile h2 { font-size: 30px; font-weight: 400; margin-bottom: 45px; }
section.profile h3 { font-size: 72px; margin: 0 0 12px; }
section.profile > p { font-size: 24px; }
section.profile > ul { font-size: 22px; line-height: 1.7; margin: 24px 0; }
section.profile > ul > li { white-space: nowrap; }
section.profile > p:has(> em:only-child) { font-size: 18px; }

section.home { --marp-slide-padding: 96px 68px 64px 400px; }
section.home > p:has(img) { position: absolute; left: 68px; top: 96px; margin: 0; line-height: 0; }
section.home > p:has(img) img { height: 560px; border: 1px solid var(--rule); border-radius: 24px; }
section.home > ul { list-style: none; padding: 0; margin: 12px 0 0; }
section.home > ul > li { border-top: 2px solid var(--blue); padding-top: 14px; margin: 0 0 24px; font-size: 23px; color: var(--muted); }
section.home > ul > li strong { display: block; font-size: 27px; color: var(--ink); margin-bottom: 4px; }
section.home > ul > li strong + br { display: none; }
section.home > p { font-size: 25px; }
section.home > p strong { background: var(--yellow); padding: 0 6px; }

/* Open columns use rules and whitespace instead of filled cards. */
section.columns { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 0 52px; align-content: start; }
section.columns > h2, section.columns > p, section.columns > pre { grid-column: 1 / -1; }
section.columns > blockquote { border: 0; border-top: 2px solid var(--ink); padding: 23px 0 0; margin: 0 0 26px; font-size: 25px; }
section.columns > blockquote h3 { font-size: 29px; margin-bottom: 20px; }
section.columns > blockquote p { margin-bottom: 18px; }
section.columns > blockquote p:last-child { margin-bottom: 0; }
section.columns > blockquote + p { margin-top: 10px; }

/* A shared package boundary makes the unwanted import visible. */
section.dependency > blockquote { border: 1px solid var(--ink); padding: 24px 30px 28px; margin: 0 0 25px; }
section.dependency > blockquote h3 { font-family: var(--mono); font-size: 23px; margin: 0 0 23px; }
section.dependency blockquote > ul { list-style: none; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 64px; margin: 0; padding: 0; }
section.dependency blockquote li { position: relative; border-top: 2px solid var(--blue); padding-top: 20px; margin: 0; font-size: 24px; }
section.dependency blockquote li:first-child::after { content: '→'; color: var(--blue); position: absolute; right: -44px; top: 15px; font-size: 30px; }
section.dependency blockquote li strong { display: block; font-size: 28px; margin-bottom: 12px; }
section.dependency blockquote li:last-child strong { background: var(--yellow); width: fit-content; padding: 0 6px; }
section.dependency blockquote li strong + br { display: none; }
section.dependency > p { font-size: 26px; }

/* Each comparison bar has its own scenario's original = 100. */
section.contrast > blockquote { position: relative; padding-top: 20px; padding-bottom: 79px; margin-bottom: 28px; }
section.contrast > blockquote h3 { font-size: 26px; margin-bottom: 18px; }
section.contrast > blockquote strong { display: block; font-family: var(--mono); font-size: 66px; font-weight: 400; line-height: 1.15; letter-spacing: -0.05em; margin: 0 0 10px; }
section.contrast > blockquote::before, section.contrast > blockquote::after { content: ''; position: absolute; left: 0; height: 15px; }
section.contrast > blockquote::before { width: 100%; bottom: 39px; background: #d8d8d8; }
section.contrast > blockquote::after { width: 26%; bottom: 16px; background: var(--blue); }
section.contrast > blockquote:last-of-type { border-top: 5px solid var(--yellow); padding-top: 17px; }
section.contrast > blockquote:last-of-type::after { width: 99%; background: var(--yellow); border: 1px solid var(--ink); box-sizing: border-box; }
section.contrast > blockquote:last-of-type strong { font-weight: 600; }
section.contrast > p { font-size: 24px; }
section.contrast > p:has(> em:only-child) { font-size: 16px; }

section.signal { display: grid; grid-template-columns: 340px 1fr; column-gap: 55px; align-content: start; }
section.signal > h2, section.signal > p { grid-column: 1 / -1; }
section.signal > blockquote { border: 0; margin: 30px 0 40px; padding: 0; font-size: 24px; }
section.signal > blockquote strong { display: block; font-family: var(--mono); font-size: 98px; line-height: 1.2; letter-spacing: -0.07em; font-weight: 400; }
section.signal table { align-self: center; font-size: 25px; }
section.signal th { font-size: 17px; }
section.signal td:nth-child(n+2) { font-family: var(--mono); }

section.compact { font-size: 25px; }
section.compact h2 { margin-bottom: 26px; }
section.compact > p { margin-bottom: 18px; }
section.compact pre { font-size: 23px; padding: 20px 24px; margin-bottom: 22px; }
section.compact blockquote { font-size: 25px; margin-bottom: 22px; }
section.compact table { font-size: 23px; margin-bottom: 22px; }
section.compact td { padding: 12px; }
section.code-focus { justify-content: center; align-content: center; }
section.code-focus h2 { margin-bottom: 42px; }
section.code-focus pre { font-size: 35px; padding: 32px; margin-bottom: 30px; }
section.code-focus blockquote { font-size: 30px; }
section.assembly { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 0 36px; align-content: start; }
section.assembly > h2, section.assembly > p { grid-column: 1 / -1; }
section.assembly > h3 { grid-row: 2; font-family: var(--mono); font-size: 21px; }
section.assembly > pre { grid-row: 3; font-size: 21px; padding: 22px 20px; }
section.assembly > pre:last-of-type { border-top-color: var(--yellow); border-top-width: 4px; padding-top: 20px; }
section.entry-map > blockquote { border-top: 0; padding-top: 0; }
section.entry-map > blockquote:first-of-type { border-left: 4px solid var(--yellow); padding-left: 24px; }
section.entry-map pre { background: none; border: 0; padding: 0; font-size: 23px; line-height: 1.8; }
section.entry-map > blockquote p { font-size: 23px; }

section.harness pre { font-size: 21px; }
section.harness > p:last-of-type { font-size: 22px; }

/* Module map: one square per source module, with the legend table beside it. */
section.module-map { display: grid; grid-template-columns: minmax(0, 1fr) 360px; column-gap: 44px; align-content: start; }
section.module-map > h2, section.module-map > p:not(:has(img)) { grid-column: 1 / -1; }
section.module-map > p:has(img) { margin: 0; }
section.module-map > p:has(img) img { width: 100%; }
section.module-map table { font-size: 21px; margin: 0; }
section.module-map th { font-size: 16px; padding: 6px 10px; }
section.module-map td { padding: 6px 10px; }
section.module-map th:nth-child(n+2), section.module-map td:nth-child(n+2) { font-family: var(--mono); text-align: right; }
section.module-map > p:has(> strong:only-child) { margin-top: 22px; }
section.module-map > p:has(> em:only-child) { font-size: 16px; color: var(--muted); }

/* Processes are open timelines: the rules connect the actual event nodes. */
section.flow > ol, section.recovery > ul { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 32px; list-style: none; padding: 0; margin: 30px 0 32px; counter-reset: step; }
section.flow > ol > li, section.recovery > ul > li { position: relative; border-top: 2px solid var(--blue); padding: 32px 12px 0 0; margin: 0; font-size: 24px; counter-increment: step; }
section.flow > ol > li::before, section.recovery > ul > li::before { content: '0' counter(step); color: var(--blue); position: absolute; left: 0; top: -19px; background: var(--paper); padding-right: 14px; font-family: var(--mono); font-size: 21px; }
section.flow > ol > li:not(:last-child)::after { content: '›'; color: var(--blue); position: absolute; top: -24px; right: -23px; font-family: var(--mono); font-size: 28px; background: var(--paper); }
section.flow > ol > li strong, section.recovery > ul > li strong { display: block; font-size: 27px; margin: 0 0 17px; }
section.flow > ol > li strong + br, section.recovery > ul > li strong + br { display: none; }
section.flow > ol > li:last-child strong { width: fit-content; background: var(--yellow); padding: 0 5px; }
section.flow > ol > li:last-child strong code { background: none; }
section.flow > p, section.flow > blockquote { font-size: 25px; }
section.recovery > ul > li::before { content: '+'; }
section.recovery > ul > li:first-child::before { content: '↳'; }
section.recovery > blockquote { margin-top: 20px; }

/* Results keep all raw/gzip values, with a shared zero-based gzip scale. */
section.results { display: grid; grid-template-columns: 280px 1fr; column-gap: 45px; align-content: start; }
section.results > h2, section.results > p { grid-column: 1 / -1; }
section.results > blockquote { align-self: center; border: 0; padding: 0; margin: 0 0 20px; font-size: 23px; }
section.results > blockquote strong { color: var(--blue); font-family: var(--mono); font-size: 82px; font-weight: 400; letter-spacing: -0.08em; line-height: 1.3; }
section.results table { table-layout: fixed; font-size: 22px; margin: 14px 0 28px; }
section.results th:first-child { width: 42%; }
section.results th:nth-child(2) { width: 14%; }
section.results td { padding: 12px 10px 20px; }
section.results td:nth-child(n+2) { font-family: var(--mono); }
section.results tbody td:nth-child(3) { background-image: linear-gradient(var(--bar-color, #171717), var(--bar-color, #171717)); background-repeat: no-repeat; background-origin: content-box; background-position: left bottom; background-size: var(--bar-width) 8px; padding-bottom: 25px; }
section.results tbody tr:nth-child(1) { --bar-width: 31%; --bar-color: var(--blue); }
section.results tbody tr:nth-child(2) { --bar-width: 49%; }
section.results tbody tr:nth-child(3) { --bar-width: 70%; }
section.results tbody tr:nth-child(4) { --bar-width: 100%; --bar-color: #969696; }
section.results tbody tr:first-child td:first-child { border-left: 4px solid var(--yellow); }
section.results > p { font-size: 23px; margin: 0 0 12px; }
section.results > p:has(> em:only-child) { font-size: 15px; }

section.mutation > blockquote { border: 0; border-top: 2px solid var(--ink); border-bottom: 2px solid var(--ink); padding: 22px 0; font-family: var(--mono); font-size: 44px; margin: 0 0 24px; }
section.mutation > blockquote code { background: none; font-size: inherit; }
section.mutation > p:last-of-type { font-size: 32px; }
section.mutation > p:last-of-type strong { background: var(--yellow); padding: 0 6px; }
section.timeline table { font-size: 24px; }
section.timeline table td:first-child { width: 70px; font-family: var(--mono); }
section.timeline table td:nth-child(2) { width: 300px; }
section.timeline table tr:nth-child(3) strong, section.timeline table tr:nth-child(4) strong { background: var(--yellow); }

section.resources > ul { list-style: none; counter-reset: resource; padding: 0; margin: 0; }
section.resources > ul > li { counter-increment: resource; position: relative; padding: 13px 0 14px 64px; border-top: 1px solid var(--rule); margin: 0; font-size: 19px; color: var(--muted); }
section.resources > ul > li::before { content: '0' counter(resource); position: absolute; left: 0; top: 17px; font-family: var(--mono); font-size: 21px; color: var(--ink); }
section.resources li a { color: var(--ink); display: block; width: fit-content; font-size: 25px; margin-bottom: 4px; text-decoration: none; }
section.resources li a + br { display: none; }
section.outcomes { justify-content: center; align-content: center; }
section.outcomes h1 { font-size: 42px; margin-bottom: 24px; }
section.outcomes table { table-layout: fixed; margin: 40px 0; }
section.outcomes th { font-size: 21px; }
section.outcomes td { padding: 30px 10px; }
section.outcomes table strong { font-family: var(--mono); font-size: 39px; font-weight: 400; white-space: nowrap; letter-spacing: -0.06em; }
section.outcomes > p:has(> em:only-child) { margin-bottom: 28px; }
section.takeaways > ol { list-style: none; counter-reset: lesson; padding: 0; margin: 0; }
section.takeaways > ol > li { counter-increment: lesson; position: relative; padding: 20px 0 23px 82px; border-top: 1px solid var(--rule); margin: 0; font-size: 24px; }
section.takeaways > ol > li::before { content: '0' counter(lesson); position: absolute; left: 0; top: 22px; font-family: var(--mono); font-size: 28px; }
section.takeaways > ol strong { display: block; font-size: 29px; margin-bottom: 9px; font-weight: 600; }
section.takeaways > ol strong + br { display: none; }
section.takeaways > blockquote { margin: 30px 0 0; font-size: 25px; }
section.hiring {
  --paper: #171717; --ink: #fff; --muted: #a3a3a3; --rule: #404040;
  justify-content: center; align-content: center;
}
section.hiring::before { border-color: var(--yellow); }
section.hiring h2 { font-size: 46px; margin-bottom: 40px; }
section.hiring > ol { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 36px; list-style: none; padding: 0; margin: 0 0 40px; counter-reset: lesson; }
section.hiring > ol > li { counter-increment: lesson; border-top: 4px solid var(--yellow); padding-top: 18px; margin: 0; font-size: 20px; color: var(--muted); }
section.hiring > ol > li::before { content: '0' counter(lesson); display: block; margin-bottom: 14px; font-family: var(--mono); font-size: 18px; color: var(--yellow); }
section.hiring > ol strong { display: block; margin-bottom: 12px; font-size: 27px; color: var(--ink); }
section.hiring > ol strong + br { display: none; }
section.hiring > blockquote { background: var(--yellow); color: #171717; border: 0; padding: 20px 26px; margin: 0; font-size: 23px; }
section.hiring > blockquote a { font-family: var(--mono); text-decoration-color: currentColor; }
section.closing { justify-content: center; align-content: center; }
section.closing h1 { width: 800px; font-size: 100px; font-weight: 400; margin-bottom: 32px; }
section.closing h2 { width: 800px; font-size: 34px; font-weight: 400; margin-bottom: 38px; }
section.closing > p { font-size: 23px; }
section.closing > p:last-of-type { margin-top: 38px; color: var(--muted); }
section.closing > ol { position: absolute; right: 68px; top: 220px; list-style: none; margin: 0; padding: 0; width: 220px; }
section.closing > ol > li { border-top: 1px solid var(--rule); padding: 14px 0; margin: 0; font-size: 21px; }
section.closing > ol strong { background: var(--yellow); padding: 0 4px; }

/* Backup material uses the same hierarchy, with room for reference tables. */
section.hidden-slide { --marp-slide-padding: 90px 68px 64px; font-size: 23px; }
section.hidden-slide h2 { font-size: 34px; margin-bottom: 25px; }
section.hidden-slide > p { margin-bottom: 16px; }
section.hidden-slide pre { font-size: 19px; line-height: 1.5; padding: 18px 22px; margin-bottom: 20px; }
section.hidden-slide table { font-size: 21px; }
section.hidden-slide td { padding: 11px 12px; }
section.hidden-slide blockquote { font-size: 23px; }
section.hidden-slide.flow > ol { margin: 22px 0 24px; }
section.hidden-slide.flow > ol > li { font-size: 22px; }
section.hidden-slide.flow > ol strong { font-size: 25px; }
/* Dense examples keep the normal text size; reclaim vertical spacing instead. */
section.event-test pre { padding-top: 16px; padding-bottom: 16px; margin-bottom: 18px; }
section.event-test > p { margin-bottom: 14px; }
section.hidden-slide.chunk-evidence h2 { margin-bottom: 16px; }
section.hidden-slide.chunk-evidence > p { font-size: 22px; margin-bottom: 10px; }
section.hidden-slide.chunk-evidence pre { padding: 12px 22px; margin-bottom: 12px; }
section.hidden-slide.chunk-evidence table { margin-bottom: 14px; }
section.hidden-slide.chunk-evidence th { padding-top: 8px; padding-bottom: 8px; }
section.hidden-slide.chunk-evidence td { padding: 7px 12px; }
section.hidden-slide.chunk-evidence > p:has(> em:only-child) { font-size: 16px; margin-top: 8px; }
section.trace > blockquote { font-size: 23px; line-height: 1.8; }
section.trace blockquote code { background: none; padding: 0; }
section.trace blockquote strong code { background: var(--yellow); }
`
