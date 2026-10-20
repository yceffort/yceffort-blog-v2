// Ink: an essay on paper. One modern sans (SUIT) on cool white, one violet accent,
// no diagrams. Built for a Zoom screen share, so nothing relies on thin strokes,
// pale greys, or motion. Fonts are self-hosted like the FEConf theme.
export const inkTheme = `
/* @theme ink */
@import 'default';

@font-face {
  font-family: 'SUIT Variable'; font-style: normal; font-weight: 100 900; font-display: swap;
  src: url('/fonts/suit/SUIT-Variable.woff2') format('woff2');
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
  --marp-slide-padding: 96px 76px 76px;
  --paper: #fbfbfd;
  --paper-deep: #eceef6;
  --ink: #121826;
  --muted: #5b6475;
  --rule: #d3d7e2;
  --accent: #5b3df5;
  --accent-soft: #a797ff;
  --muted-light: #aab2c6;
  --display: 'SUIT Variable', SUIT, sans-serif;
  --sans: 'SUIT Variable', SUIT, sans-serif;
  --mono: 'IBM Plex Mono', 'SUIT Variable', monospace;
  --bgColor-default: var(--paper);
  --bgColor-muted: var(--paper-deep);
  --bgColor-neutral-muted: var(--paper-deep);
  --fgColor-default: var(--ink);
  --fgColor-muted: var(--muted);
  --fgColor-accent: var(--accent);
  --borderColor-default: var(--rule);
  --borderColor-muted: var(--rule);
  --h1-color: var(--ink);
  --heading-strong-color: var(--ink);
  --paginate-color: var(--muted);
  color-scheme: light;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 28px;
  font-weight: 400;
  line-height: 1.6;
  letter-spacing: -0.01em;
  word-break: keep-all;
  justify-content: flex-start;
  align-content: start;
  padding: var(--marp-slide-padding);
}
/* A short vermilion rule under the running head marks every page. */
section::before {
  content: '';
  position: absolute;
  top: 70px; left: 76px;
  width: 40px; height: 3px;
  background: var(--accent);
}
section header {
  top: 40px; left: 76px; right: 76px;
  font-family: var(--display);
  font-size: 15px; font-weight: 500;
  letter-spacing: 0.04em;
  color: var(--muted);
}
section footer {
  left: 76px; right: auto; bottom: 34px; top: auto;
  font-family: var(--display);
  font-size: 15px; font-weight: 500;
  color: var(--muted);
}
section::after {
  bottom: 31px; right: 76px;
  font-family: var(--display); font-size: 16px; font-weight: 500;
  letter-spacing: 0;
}
section h1, section h2, section h3 {
  font-family: var(--display);
  color: var(--ink);
  border: 0; padding: 0;
  letter-spacing: -0.025em;
  text-wrap: balance;
}
section h1 { font-size: 60px; font-weight: 800; line-height: 1.3; margin: 0 0 28px; }
section h2 { font-size: 40px; font-weight: 700; line-height: 1.35; margin: 0 0 30px; }
section h3 { font-size: 28px; font-weight: 700; line-height: 1.45; margin: 0 0 16px; }
section p { margin: 0 0 22px; }
section strong {
  color: inherit; font-weight: 600;
  text-decoration: underline;
  text-decoration-color: var(--accent);
  text-decoration-thickness: 3px;
  text-underline-offset: 7px;
}
section em { font-style: normal; color: var(--accent); }
section a { color: inherit; text-decoration: underline; text-decoration-color: var(--rule); text-underline-offset: 6px; }
section ul, section ol { padding-left: 1.2em; margin: 0 0 24px; }
section li { margin: 12px 0; }
section li::marker { color: var(--accent); }
section blockquote {
  color: var(--ink); background: none;
  border: 0; border-left: 4px solid var(--accent);
  border-radius: 0; padding: 4px 0 4px 28px;
  margin: 10px 0 28px;
  font-family: var(--display);
  font-size: 32px; font-weight: 500; line-height: 1.6;
}
section blockquote p { margin: 0 0 14px; }
section blockquote p:last-child { margin-bottom: 0; }
section blockquote strong { font-weight: 700; }
section code {
  font-family: var(--mono); font-size: 0.88em;
  background: var(--paper-deep); color: inherit;
  padding: 2px 6px; border-radius: 0;
}
section pre {
  background: var(--paper-deep); color: var(--ink);
  border: 0; border-top: 3px solid var(--ink); border-radius: 0;
  padding: 22px 26px; margin: 0 0 24px;
  font-family: var(--mono); font-size: 24px; line-height: 1.5;
}
section pre code { background: none; padding: 0; font-size: inherit; line-height: inherit; }
section pre [class^='hljs-'], section pre [class*=' hljs-'] { color: inherit; background: none; font-style: normal; }
section pre .hljs-comment { color: var(--muted); }
section pre .hljs-string, section pre .hljs-number, section pre .hljs-literal { color: var(--accent); }
section pre .hljs-keyword { font-weight: 600; }
section table {
  display: table; width: 100%;
  border: 0; border-collapse: collapse; border-radius: 0;
  font-size: 25px; line-height: 1.45;
  margin: 0 0 26px;
}
section table tr, section table tr:nth-child(2n) { background: none; border: 0; }
section table th {
  border: 0; border-bottom: 2px solid var(--ink);
  font-family: var(--display); color: var(--muted);
  font-size: 18px; font-weight: 500;
  padding: 10px 12px; background: none; text-align: left;
}
section table td { border: 0; border-bottom: 1px solid var(--rule); padding: 15px 12px; vertical-align: top; }
section table strong { text-decoration: none; color: var(--accent); }
section p > img:only-child { display: block; max-width: 100%; margin: 0 auto; border: 0; border-radius: 0; }
/* A paragraph made of a single em is a footnote or caption. */
section > p:has(> em:only-child) {
  font-size: 17px; line-height: 1.55; color: var(--muted);
  margin: 14px 0 0;
}
section > p > em:only-child { color: var(--muted); }

/* Dark openers: the cover, every chapter divider, and the closing page flip to ink. */
section.cover, section.chapter, section.closing {
  background: var(--ink); color: var(--paper);
  --paginate-color: var(--muted-light);
}
section.cover header, section.chapter header, section.closing header,
section.cover footer, section.chapter footer, section.closing footer { color: var(--muted-light); }
section.cover::before, section.chapter::before, section.closing::before { background: var(--accent-soft); }
section.cover h1, section.cover h2, section.cover p,
section.chapter h1, section.chapter h2,
section.closing h1, section.closing h2, section.closing p { color: var(--paper); }
section.cover a, section.closing a { text-decoration-color: var(--muted-light); }

/* Cover: a title set like a book jacket, with the running head replaced by the event. */
section.cover { justify-content: center; align-content: center; }
section.cover::before { top: 150px; width: 120px; height: 6px; }
section.cover h1 { font-size: 72px; line-height: 1.3; width: 960px; margin: 36px 0 28px; }
section.cover h2 { font-family: var(--sans); font-size: 28px; font-weight: 400; line-height: 1.6; color: var(--muted-light); width: 820px; margin: 0 0 56px; }
section.cover > p { font-size: 22px; margin: 0; }
section.cover > p strong { text-decoration: none; margin-right: 12px; }

/* Profile: name and affiliation beside the blog mark. */
section.profile { --marp-slide-padding: 140px 76px 80px 330px; justify-content: center; align-content: center; }
section.profile > p:has(img) { position: absolute; left: 76px; top: 228px; width: 180px; margin: 0; border-bottom: 6px solid var(--accent); padding-bottom: 14px; }
section.profile h2 { font-size: 30px; font-weight: 500; margin-bottom: 40px; }
section.profile h3 { font-size: 64px; font-weight: 700; margin: 0 0 10px; }
section.profile > p { font-size: 23px; margin-bottom: 14px; }
section.profile > ul { font-size: 21px; line-height: 1.6; margin: 20px 0; }
section.profile > ul > li { margin: 6px 0; }
section.profile > p:has(> em:only-child) { font-size: 18px; }

/* Chapter: the five-part table of contents, current part in ink. */
section.chapter { display: grid; grid-template-columns: minmax(0, 1fr) 360px; column-gap: 70px; align-content: center; }
section.chapter > blockquote { grid-column: 1; grid-row: 1; border: 0; padding: 0; margin: 0 0 12px; font-size: 96px; font-weight: 800; line-height: 1; color: var(--accent-soft); }
section.chapter h1 { grid-column: 1; grid-row: 2; font-size: 54px; line-height: 1.3; margin: 0 0 24px; }
section.chapter h2 { grid-column: 1; grid-row: 3; font-family: var(--sans); font-size: 26px; font-weight: 400; line-height: 1.6; color: var(--muted-light); margin: 0; }
section.chapter > ol { grid-column: 2; grid-row: 1 / 4; align-self: center; list-style: none; padding: 0; margin: 0; counter-reset: part; }
section.chapter > ol > li { counter-increment: part; position: relative; border-top: 1px solid #3a4258; padding: 14px 0 14px 46px; margin: 0; font-size: 20px; line-height: 1.5; color: var(--muted-light); }
section.chapter > ol > li::before { content: counter(part, decimal-leading-zero); position: absolute; left: 0; top: 15px; font-family: var(--display); font-size: 17px; color: var(--muted-light); }
section.chapter > ol > li:has(strong) { color: var(--paper); border-top: 3px solid var(--accent-soft); padding-top: 12px; }
section.chapter > ol > li:has(strong)::before { color: var(--accent-soft); top: 13px; }
section.chapter > ol strong { text-decoration: none; font-weight: 600; }

/* Quote: one sentence, set large, with its source beneath. */
section.quote { justify-content: center; align-content: center; }
section.quote > blockquote { border: 0; padding: 0; margin: 0 0 36px; font-size: 46px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.55; width: 1040px; }
section.quote > blockquote::before { content: ''; display: block; width: 48px; height: 5px; background: var(--accent); margin: 0 0 34px; }
section.quote > p { font-size: 22px; color: var(--muted); margin: 0; }
section.quote > p strong { text-decoration: none; color: var(--ink); }

/* Scene: a told episode. The dialogue line is set heavier with a large opening mark. */
section.scene h2 { margin-bottom: 26px; }
section.scene > blockquote { position: relative; border: 0; padding: 0 0 0 52px; margin: 6px 0 30px; font-size: 34px; font-weight: 600; }
section.scene > blockquote::before { content: '“'; position: absolute; left: 0; top: -6px; font-size: 72px; line-height: 1; color: var(--accent); }
section.scene > p { font-size: 27px; }

/* Columns: two framed statements side by side. The right one carries the accent. */
section.columns { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 0 56px; align-content: start; }
section.columns > h2, section.columns > p { grid-column: 1 / -1; }
section.columns > blockquote { border: 0; border-top: 3px solid var(--ink); padding: 22px 0 0; margin: 0 0 24px; font-family: var(--sans); font-size: 25px; font-weight: 400; line-height: 1.6; }
section.columns > blockquote h3 { font-size: 30px; margin-bottom: 18px; }
section.columns > blockquote p { margin-bottom: 16px; }
section.columns > blockquote:last-of-type { border-top-color: var(--accent); }
section.columns > blockquote:last-of-type h3 { color: var(--accent); }
section.columns > blockquote + p { margin-top: 8px; }

/* Ledger: a table that ends on the one row that mattered. */
section.ledger table { font-size: 24px; margin-bottom: 22px; }
section.ledger td { padding: 11px 12px; }
section.ledger td:last-child { font-family: var(--display); font-weight: 600; }
section.ledger tbody tr:last-child td { border-bottom: 3px solid var(--accent); }

/* Loop: the learning path written as one line of steps. */
section.loop > ol { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 18px; list-style: none; padding: 0; margin: 58px 0 36px; counter-reset: step; }
section.loop > ol > li { counter-increment: step; position: relative; border-top: 3px solid var(--ink); padding: 16px 4px 0 0; margin: 0; font-family: var(--display); font-size: 23px; font-weight: 600; line-height: 1.4; }
section.loop > ol > li::before { content: counter(step); position: absolute; top: -34px; left: 0; font-size: 17px; color: var(--muted); }
section.loop > ol > li:nth-child(4) { border-top-color: var(--accent); }
section.loop > ol > li:nth-child(4) strong { text-decoration: none; color: var(--accent); }
section.loop > p { font-size: 26px; }

/* Closing: the question, set alone. */
section.closing { justify-content: center; align-content: center; }
section.closing h1 { font-size: 80px; font-weight: 700; line-height: 1.3; width: 1000px; margin: 0 0 44px; }
section.closing h2 { font-family: var(--sans); font-size: 28px; font-weight: 400; color: var(--muted-light); margin: 0 0 40px; }
section.closing > p { font-size: 23px; margin-bottom: 10px; }
section.closing > p strong { text-decoration: none; }
section.closing > p:last-of-type { margin-top: 30px; color: var(--muted-light); font-size: 20px; }

/* Compact: a dense page keeps the type family and loses only size. */
section.compact { font-size: 25px; }
section.compact h2 { font-size: 36px; margin-bottom: 24px; }
section.compact > p { margin-bottom: 18px; }
section.compact blockquote { font-size: 28px; margin-bottom: 22px; }
section.compact table { font-size: 22px; margin-bottom: 20px; }
section.compact td { padding: 10px 10px; }
section.compact th { font-size: 17px; }
section.compact pre { font-size: 21px; padding: 18px 22px; margin-bottom: 20px; }

/* Backup material keeps the hierarchy at a smaller size. */
section.hidden-slide { font-size: 24px; }
section.hidden-slide h2 { font-size: 34px; margin-bottom: 18px; }
section.hidden-slide table { font-size: 19px; line-height: 1.4; }
section.hidden-slide td { padding: 8px 10px; }
section.hidden-slide th { font-size: 16px; padding: 6px 10px; }
`
