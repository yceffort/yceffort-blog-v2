---
title: 'Core Web Vitals, 무엇을 왜 어떻게 수집할까'
marp: true
paginate: true
theme: default
tags:
  - web-performance
  - browser
  - observability
date: 2026-10-19
description: 'LCP, INP, CLS가 무엇을 재는지에서 시작해 왜 실제 사용자에게서 재야 하는지, 브라우저와 web-vitals가 값을 어떻게 만들고 언제 넘겨주는지까지 따라간다. 값이 서버에 도착하지 않는 다섯 가지 경우를 직접 측정한 수치로 확인한다.'
published: false
---

<style>
@font-face { font-family: 'IBM Plex Sans KR'; font-style: normal; font-weight: 400; font-display: swap; src: url('/fonts/ibm-plex/IBMPlexSansKR-Regular.woff2') format('woff2'); }
@font-face { font-family: 'IBM Plex Sans KR'; font-style: normal; font-weight: 600; font-display: swap; src: url('/fonts/ibm-plex/IBMPlexSansKR-SemiBold.woff2') format('woff2'); }
@font-face { font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 400; font-display: swap; src: url('/fonts/ibm-plex/IBMPlexMono-Regular.woff2') format('woff2'); }
@font-face { font-family: 'IBM Plex Mono'; font-style: normal; font-weight: 600; font-display: swap; src: url('/fonts/ibm-plex/IBMPlexMono-SemiBold.woff2') format('woff2'); }

section {
  --bg: #f6f7fb; --card: #ffffff; --ink: #141821; --muted: #5d6476; --line: #e1e4ee; --soft: #eceff6;
  --accent: #3b5bfd; --accent-soft: #e8ecff; --night: #0f1320;
  --good: #0c8558; --good-bg: #e0f4ea; --warn: #a96500; --warn-bg: #fdf0d6; --poor: #d3352a; --poor-bg: #fde4e1;
  --lcp: #3b5bfd; --inp: #ec6327; --cls: #9643dc;
  --sans: 'IBM Plex Sans KR', 'Apple SD Gothic Neo', sans-serif;
  --mono: 'IBM Plex Mono', 'IBM Plex Sans KR', monospace;
  display: flex; flex-direction: column;
  background: var(--bg); color: var(--ink);
  font-family: var(--sans); font-size: 27px; line-height: 1.55; letter-spacing: -0.01em; word-break: keep-all;
}
section { --marp-slide-padding: 168px 72px 70px; justify-content: center; }
section > h2 { position: absolute; top: 72px; left: 72px; right: 72px; margin: 0; }
section.ask { --marp-slide-padding: 206px 72px 70px; }
section.ask > h2 { top: 112px; }
section.cover, section.chapter, section.statement { --marp-slide-padding: 76px 72px 60px; }
section.statement > h2 { position: static; }
section > header { top: 30px; left: 72px; right: 72px; font-size: 16px; font-weight: 600; color: var(--accent); letter-spacing: 0.06em; }
section::after { right: 40px; bottom: 24px; font-size: 15px; color: var(--muted); font-family: var(--mono); }
h1, h2, h3 { font-family: var(--sans); color: var(--ink); border: 0; padding: 0; margin: 0; font-weight: 600; letter-spacing: -0.03em; line-height: 1.25; }
h2 { font-size: 42px; margin-bottom: 28px; }
h3 { font-size: 25px; margin-bottom: 6px; }
p { margin: 0 0 14px; }
ul { margin: 0 0 14px; padding-left: 1.1em; }
li { margin: 4px 0; }
li::marker { color: var(--accent); }
strong { font-weight: 600; color: inherit; background: linear-gradient(transparent 60%, #ffe27a 60%); }
b { font-weight: 600; }
code { font-family: var(--mono); font-size: 0.84em; background: var(--soft); color: var(--ink); border-radius: 6px; padding: 0.06em 0.34em; }
pre { margin: 0 0 16px; padding: 18px 24px; background: var(--card); border: 1px solid var(--line); border-radius: 16px; font-size: 22px; line-height: 1.5; box-shadow: 0 14px 34px -28px rgba(20, 24, 33, 0.45); }
pre code { background: none; padding: 0; font-size: inherit; border-radius: 0; }
table { display: table; width: 100%; border-collapse: separate; border-spacing: 0; margin: 0 0 16px; font-size: 23px; line-height: 1.4; background: var(--card); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
table th, table td { border: 0; padding: 10px 16px; text-align: left; background: transparent; }
table th { background: var(--soft); color: var(--muted); font-weight: 600; font-size: 18px; border-bottom: 1px solid var(--line); }
table td { border-bottom: 1px solid var(--line); }
table tr, table tr:nth-child(2n) { background: transparent; }
table tr:last-child td { border-bottom: 0; }
table td strong { background: var(--poor-bg); color: var(--poor); padding: 0.04em 0.36em; border-radius: 6px; }
blockquote { margin: 0 0 22px; padding: 18px 26px; border: 0; border-left: 6px solid var(--accent); background: var(--accent-soft); border-radius: 0 14px 14px 0; color: var(--ink); font-size: 30px; font-weight: 600; }
blockquote p { margin: 0; }
blockquote strong { background: none; color: var(--accent); }

section.cover, section.chapter { background: var(--night); color: #e8ebf5; justify-content: center; }
section.cover::after, section.chapter::after { color: #6b7591; }
section.cover h1 { font-size: 64px; color: #fff; line-height: 1.2; margin-bottom: 26px; letter-spacing: -0.035em; }
section.cover p { color: #a8afc4; font-size: 27px; }
section.chapter h1 { font-size: 76px; color: #fff; margin-bottom: 18px; }
section.chapter p { color: #a8afc4; font-size: 28px; }
section.cover strong, section.chapter strong { background: none; color: #8ea2ff; }
.stripes { display: flex; gap: 10px; margin-bottom: 44px; }
.stripes i { width: 92px; height: 10px; border-radius: 5px; }
.stripes .lcp { background: var(--lcp); } .stripes .inp { background: var(--inp); } .stripes .cls { background: var(--cls); }
.byline { margin-top: 36px; font-family: var(--mono); color: #8ea2ff; font-size: 22px; }
.num { font-family: var(--mono); font-size: 22px; font-weight: 600; letter-spacing: 0.14em; color: #8ea2ff; margin-bottom: 14px; }
section.statement { justify-content: center; }
section.statement h2 { font-size: 54px; line-height: 1.3; margin-bottom: 36px; }
section.statement h2 strong { background: linear-gradient(transparent 62%, #ffe27a 62%); }
section.quiz { background: var(--accent-soft); }

.k { font-size: 16px; font-weight: 600; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 6px; }
.src { position: absolute; left: 72px; right: 130px; bottom: 24px; font-size: 15px; color: var(--muted); line-height: 1.45; }
.cols { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 30px; align-items: start; margin-bottom: 22px; }
.cols:last-child { margin-bottom: 0; }
.cols.wide-left { grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr); }
.cols > div > :last-child { margin-bottom: 0; }
.small-code pre { font-size: 17px; line-height: 1.45; }
.tight table { font-size: 20px; }
.tight table th, .tight table td { padding: 7px 16px; }

.cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; margin: 0 0 18px; }
.cards.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.cards.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.cards.five { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 18px 22px; box-shadow: 0 14px 34px -28px rgba(20, 24, 33, 0.45); font-size: 23px; line-height: 1.5; }
.card > :last-child { margin-bottom: 0; }
.card + .card { margin-top: 14px; }
.cards > .card + .card { margin-top: 0; }
.card p { margin: 0 0 6px; }
.card ul { margin: 0; }
.card .big { font-size: 60px; font-weight: 600; letter-spacing: -0.04em; line-height: 1.1; margin: 2px 0 6px; }
.card .no { font-family: var(--mono); font-weight: 600; color: var(--accent); font-size: 20px; margin-bottom: 22px; }
.card .ask { font-size: 22px; font-weight: 600; color: var(--muted); margin-bottom: 2px; }
.card .fine { font-size: 17px; color: var(--muted); }
.card.lcp { border-top: 6px solid var(--lcp); } .card.lcp .big { color: var(--lcp); }
.card.inp { border-top: 6px solid var(--inp); } .card.inp .big { color: var(--inp); }
.card.cls { border-top: 6px solid var(--cls); } .card.cls .big { color: var(--cls); }
.card.good { background: var(--good-bg); border-color: transparent; box-shadow: none; } .card.good .big { color: var(--good); }
.card.bad { background: var(--poor-bg); border-color: transparent; box-shadow: none; } .card.bad .big { color: var(--poor); }
.card.mute { background: var(--soft); border-color: transparent; box-shadow: none; }
.card.mute .big { color: var(--muted); }
.card.hl { background: var(--accent-soft); border-color: var(--accent); box-shadow: none; }
.cards.five .card { text-align: center; padding: 16px 12px; }
.c-lcp { color: var(--lcp); } .c-inp { color: var(--inp); } .c-cls { color: var(--cls); }

.tag { display: inline-block; padding: 1px 12px; border-radius: 999px; font-size: 18px; font-weight: 600; line-height: 1.6; color: #fff; background: var(--muted); }
.tag.lcp { background: var(--lcp); } .tag.inp { background: var(--inp); } .tag.cls { background: var(--cls); }
.tag.ghost { background: var(--soft); color: var(--muted); }
.tag.hidden { background: var(--ink); }

.q { align-self: flex-start; display: inline-flex; align-items: center; gap: 10px; margin: 0; position: absolute; top: 62px; left: 72px; padding: 5px 18px 5px 5px; border-radius: 999px; background: var(--card); border: 1px solid var(--line); color: var(--muted); font-size: 20px; font-weight: 600; }
.q em, .asks em { font-style: normal; display: inline-grid; place-items: center; flex: none; width: 30px; height: 30px; border-radius: 50%; background: var(--accent); color: #fff; font-size: 16px; font-weight: 600; }

.flow { display: flex; gap: 34px; margin: 4px 0 22px; align-items: stretch; }
.flow > div { flex: 1; position: relative; background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 16px 20px; font-size: 22px; line-height: 1.4; }
.flow > div:not(:last-child)::after { content: ''; position: absolute; top: 50%; right: -23px; width: 11px; height: 11px; border-top: 3px solid #a2a8b8; border-right: 3px solid #a2a8b8; transform: translateY(-50%) rotate(45deg); }
.flow > div > b { display: block; font-size: 24px; margin-bottom: 2px; }
.flow > div > small { display: block; margin-top: 6px; color: var(--muted); font-size: 18px; }
.flow > div.hl { background: var(--accent-soft); border-color: var(--accent); }
.flow > div.warn { background: var(--warn-bg); border-color: var(--warn); }
.flow > div.ok { background: var(--good-bg); border-color: var(--good); }
.flow.lcp > div { border-top: 5px solid var(--lcp); }
.flow.inp > div { border-top: 5px solid var(--inp); }

.scale { display: grid; grid-template-columns: 120px repeat(3, minmax(0, 1fr)); gap: 8px; align-items: center; margin-bottom: 10px; }
.scale > b { font-size: 30px; }
.scale > span { padding: 13px 20px; border-radius: 12px; font-weight: 600; font-size: 25px; }
.scale > .g { background: var(--good-bg); color: var(--good); }
.scale > .w { background: var(--warn-bg); color: var(--warn); }
.scale > .p { background: var(--poor-bg); color: var(--poor); }
.scale.head > span { padding: 0 20px; font-size: 16px; color: var(--muted); letter-spacing: 0.06em; }
.scale.minor { margin-top: 2px; }
.scale.minor > b { font-size: 22px; color: var(--muted); }
.scale.minor > span { font-size: 19px; padding: 7px 20px; font-weight: 400; }

.tl { display: flex; position: relative; margin: 10px 0 24px; }
.tl::before { content: ''; position: absolute; left: 5%; right: 5%; top: 66px; height: 4px; border-radius: 2px; background: var(--line); }
.tl > div { flex: 1; position: relative; text-align: center; padding: 0 4px; font-size: 20px; line-height: 1.35; }
.tl > div > b { display: flex; align-items: flex-end; justify-content: center; height: 54px; font-size: 21px; }
.tl > div > i { display: block; position: relative; z-index: 1; width: 20px; height: 20px; margin: 4px auto 12px; border-radius: 50%; background: var(--card); border: 4px solid #a2a8b8; box-sizing: border-box; }
.tl > div.end > i { background: var(--ink); border-color: var(--ink); }
.tl > div.hl > i { background: var(--accent); border-color: var(--accent); }
.tl > div > small { display: block; margin-top: 4px; color: var(--muted); font-size: 17px; }
.tl .tag { margin: 2px; }

.screens { display: flex; align-items: center; gap: 26px; margin: 4px 0 6px; }
.shot { flex: none; text-align: center; }
.shot p { margin: 10px 0 0; font-size: 20px; color: var(--muted); }
.shot p b { display: block; font-family: var(--mono); font-size: 22px; color: var(--ink); }
.screen { position: relative; width: 210px; height: 280px; background: var(--card); border: 2px solid var(--line); border-radius: 16px; overflow: hidden; }
.screen > i { position: absolute; left: 16px; right: 16px; border-radius: 6px; background: var(--soft); }
.screen .t1 { top: 20px; height: 34px; right: 50px; }
.screen .t2 { top: 66px; height: 12px; }
.screen .t3 { top: 86px; height: 12px; right: 60px; }
.screen .img { top: 114px; height: 146px; }
.screen .empty { background: none; border: 2px dashed var(--line); }
.screen .on { background: #dfe5ff; box-shadow: 0 0 0 4px var(--lcp); }
.go { flex: none; width: 20px; height: 20px; margin: 0 8px 48px 0; border-top: 4px solid #a2a8b8; border-right: 4px solid #a2a8b8; transform: rotate(45deg); }
.say { flex: 1; padding-left: 18px; font-size: 26px; }
.say p { margin-bottom: 20px; }

.clsbox { display: flex; gap: 26px; align-items: center; }
.clsbox .screen { width: 200px; height: 290px; }
.clsbox .union { top: 0; left: 0; right: 0; height: 75%; border-radius: 0; background: #f2e7fc; }
.clsbox .before { top: 8px; left: 14px; right: 14px; height: calc(50% - 10px); background: none; border: 3px dashed #c29be8; }
.clsbox .after { top: calc(25% + 6px); left: 14px; right: 14px; height: calc(50% - 10px); background: rgba(150, 67, 220, 0.3); border: 3px solid var(--cls); }
.legend p { margin: 0 0 10px; font-size: 21px; }
.legend .eq { margin-top: 16px; font-family: var(--mono); font-size: 21px; }
.legend .eq b { color: var(--cls); }
.sw { display: inline-block; width: 18px; height: 18px; border-radius: 4px; margin-right: 8px; vertical-align: -2px; }
.sw.u { background: #f2e7fc; border: 1px solid #c29be8; }
.sw.a { background: rgba(150, 67, 220, 0.3); border: 2px solid var(--cls); }

.p75 { position: relative; height: 62px; margin: 46px 0 46px; border-radius: 12px; background: linear-gradient(90deg, var(--good-bg), var(--warn-bg) 72%, var(--poor-bg)); }
.p75 .fill { position: absolute; left: 0; top: 0; bottom: 0; width: 75%; display: flex; align-items: center; padding-left: 22px; border-right: 4px solid var(--ink); font-weight: 600; font-size: 21px; }
.p75 .mark { position: absolute; left: 75%; top: -38px; transform: translateX(-50%); font-family: var(--mono); font-weight: 600; font-size: 22px; }
.p75::before, .p75::after { position: absolute; top: 70px; font-size: 16px; color: var(--muted); }
.p75::before { content: '빠른 로드'; left: 0; }
.p75::after { content: '느린 로드'; right: 0; }

.asks { display: grid; gap: 14px; justify-items: start; }
.asks > div { display: flex; align-items: center; gap: 16px; padding: 13px 26px 13px 14px; background: var(--card); border: 1px solid var(--line); border-radius: 24px 24px 24px 6px; font-size: 28px; box-shadow: 0 14px 34px -28px rgba(20, 24, 33, 0.45); }

.steps { display: grid; gap: 10px; margin: 0 0 18px; }
.steps > div { display: grid; grid-template-columns: 30px 250px 100px minmax(0, 1fr); gap: 14px; align-items: center; padding: 13px 20px; background: var(--card); border: 1px solid var(--line); border-radius: 14px; font-size: 22px; }
.steps > div.hl { background: var(--accent-soft); border-color: var(--accent); }
.steps em { font-style: normal; font-family: var(--mono); font-weight: 600; color: var(--muted); }
.steps code { background: none; padding: 0; font-size: 24px; font-weight: 600; }
.steps .tag { justify-self: start; }

.bars { display: grid; gap: 10px; margin: 6px 0 20px; }
.bar { display: grid; grid-template-columns: 100px minmax(0, 1fr) 72px; gap: 12px; align-items: center; font-size: 21px; }
.bar > span { color: var(--muted); }
.bar > i { display: block; height: 32px; border-radius: 8px; background: var(--poor-bg); overflow: hidden; }
.bar > i > i { display: block; height: 100%; border-radius: 8px; background: var(--good); }
.bar > em { font-style: normal; font-family: var(--mono); font-weight: 600; text-align: right; }
.w0 { width: 0; } .w10 { width: 10%; } .w20 { width: 20%; } .w45 { width: 45%; } .w80 { width: 80%; } .w100 { width: 100%; }

.ratio { display: grid; grid-template-columns: 4fr 1fr 4fr 1fr; gap: 4px; margin: 0 0 16px; }
.ratio > span { display: flex; align-items: center; justify-content: center; height: 38px; border-radius: 8px; color: #fff; font-size: 17px; font-weight: 600; white-space: nowrap; }
.ratio .r1, .ph1 { --ph: #6f86ff; } .ratio .r2, .ph2 { --ph: #e8931f; } .ratio .r3, .ph3 { --ph: #3b5bfd; } .ratio .r4, .ph4 { --ph: #e2622b; }
.ratio > span { background: var(--ph); }
.phases .card { border-top: 6px solid var(--ph); padding: 16px 18px; font-size: 20px; }
.phases .card h3 { font-size: 22px; }
.phases .card .k { color: var(--ink); }
.gantt { display: grid; gap: 12px; margin: 0 0 24px; }
.gantt > div { display: grid; grid-template-columns: 96px minmax(0, 1fr) 250px; gap: 16px; align-items: center; font-size: 21px; }
.gantt > div > b { font-size: 25px; }
.gantt > div > i { display: block; height: 30px; border-left: 3px solid var(--ink); }
.gantt > div > i > i { display: block; height: 100%; border-radius: 0 8px 8px 0; background: #b7bfd4; }
.gantt > div > i > i.lcp { background: var(--lcp); }
.gantt > div > span { color: var(--muted); }

.qa { display: grid; gap: 10px; }
.qa > div { display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 18px; align-items: center; }
.qa > div > span:first-child { padding: 9px 18px; border-radius: 18px 18px 18px 5px; background: var(--card); border: 1px solid var(--line); color: var(--muted); font-size: 21px; font-weight: 600; }
.qa > div > span:last-child { font-size: 24px; }
</style>

<!-- _class: cover -->
<!-- _paginate: false -->

<div class="stripes"><i class="lcp"></i><i class="inp"></i><i class="cls"></i></div>

# Core Web Vitals,<br>무엇을 왜 어떻게 수집할까

사용자가 **페이지를 떠날 때** 확정되는 값

<div class="byline">@yceffort</div>

<!--
대상: 프론트엔드 1~3년차. LCP, CLS 이름은 들어 봤거나 Lighthouse 점수를 본 적은 있지만, 실제 사용자 브라우저에서 값을 모아 본 적은 없는 사람.
목표는 세 가지다. 세 지표가 무엇을 측정하는지 설명할 수 있다. 왜 실제 사용자 환경에서 측정해야 하는지 설명할 수 있다. 수집한 값이 서버에 도착하지 않는 경우를 원칙 하나로 설명할 수 있다.
4장의 수치는 블로그 시리즈 "Core Web Vitals 직접 수집하기" 1, 2편에서 직접 잰 값이다. 45분 안팎.
-->

---

## 오늘 순서

<div class="cards four">
<div class="card"><div class="no">01</div>

### 무엇을 재나

LCP, INP, CLS와 p75

</div>
<div class="card"><div class="no">02</div>

### 왜 재나

매출과 검색, Lighthouse로는 알 수 없는 것

</div>
<div class="card"><div class="no">03</div>

### 어떻게 재나

브라우저의 성능 기록과 web-vitals

</div>
<div class="card hl"><div class="no">04</div>

### 수집할 때 생기는 질문

값이 서버에 도착하지 않는 다섯 가지 경우

</div>
</div>

---

## 이 덱의 측정 환경

<div class="cards four">
<div class="card"><div class="k">web-vitals</div><div class="big">6.2.3</div><p>2026-10-19 기준 npm 최신</p></div>
<div class="card"><div class="k">Next.js</div><div class="big">16.3.5</div><p>web-vitals 4.2.1 내장</p></div>
<div class="card"><div class="k">Chromium</div><div class="big">156</div><p>Playwright 1.64.1. 일부는 Chrome Beta 157로 측정</p></div>
<div class="card"><div class="k">브라우저 지원표</div><div class="big">8.1.6</div><p>MDN browser-compat-data</p></div>
</div>

<div class="card mute">

라이브러리 버전에 따라 결과가 달라진다. 4장의 일부 내용은 **web-vitals 다음 배포에서 바뀌며**, 해당 위치에 표시했다.

</div>

---

## 먼저 알아 둘 용어

<div class="cards">
<div class="card"><h3>페이지 로드</h3><p>문서 하나를 새로 여는 것. 지표는 페이지 로드마다 따로 측정된다</p></div>
<div class="card"><h3>성능 기록</h3><p>"언제 무엇을 그렸는지", "클릭 처리에 몇 ms 걸렸는지" 같은 브라우저의 기록</p></div>
<div class="card"><h3>숨겨진 상태</h3><p>탭 전환, 앱 전환, 페이지 이동으로 화면에서 사라진 상태. <code>visibilityState</code>가 <code>hidden</code></p></div>
<div class="card"><h3>p75</h3><p>값을 작은 순서로 정렬했을 때 75% 위치의 값. 100건이면 75번째 값</p></div>
<div class="card"><h3>필드 데이터</h3><p>실제 사용자 브라우저에서 측정한 값</p></div>
<div class="card"><h3>랩 데이터</h3><p>정해진 환경에서 도구로 측정한 값. Lighthouse가 대표적이다</p></div>
</div>

---

<!-- _class: chapter -->
<!-- _paginate: false -->

<div class="num">1장</div>

# 무엇을 재나

사용자 경험을 세 가지 숫자로

---

<!-- header: '1장 무엇을 재나' -->

## 사용자가 느끼는 세 가지 질문

<div class="cards">
<div class="card lcp"><div class="ask">"떴나?"</div><div class="big">LCP</div><div class="k">Largest Contentful Paint</div><p>화면에서 가장 큰 텍스트 블록이나 이미지가 그려진 시점</p></div>
<div class="card inp"><div class="ask">"눌렀는데 반응하나?"</div><div class="big">INP</div><div class="k">Interaction to Next Paint</div><p>클릭, 탭, 키 입력부터 다음 화면이 그려지기까지</p></div>
<div class="card cls"><div class="ask">"읽다가 밀리지 않나?"</div><div class="big">CLS</div><div class="k">Cumulative Layout Shift</div><p>화면 요소가 예고 없이 움직인 정도. 단위 없는 점수</p></div>
</div>

이 세 지표가 **Core Web Vitals**다. 각각 로딩, 반응성, 시각적 안정성을 측정한다.

---

## 보조 지표: TTFB와 FCP

<div class="gantt">
<div><b>TTFB</b><i><i class="w20"></i></i><span>응답 첫 바이트 도착</span></div>
<div><b>FCP</b><i><i class="w45"></i></i><span>처음 무언가 그려짐</span></div>
<div><b class="c-lcp">LCP</b><i><i class="w80 lcp"></i></i><span>가장 큰 요소가 그려짐</span></div>
</div>

<div class="cards two">
<div class="card"><div class="k">TIME TO FIRST BYTE</div><h3>요청부터 응답 첫 바이트까지</h3><p>리다이렉트, 서비스 워커 시작, DNS, 연결과 TLS, 서버가 첫 바이트를 보내기까지를 모두 더한 시간</p><p class="fine">원인 후보: 서버, CDN, 리다이렉트</p></div>
<div class="card"><div class="k">FIRST CONTENTFUL PAINT</div><h3>처음 무언가 그려진 시점</h3><p>텍스트, 이미지(배경 이미지 포함), svg, 흰색이 아닌 canvas 중 하나. 빈 화면이 끝났다는 신호</p><p class="fine">원인 후보: 렌더링을 막는 CSS와 JS, 늦게 로드되는 웹 폰트</p></div>
</div>

<div class="src">세 지표 모두 탐색 시작부터 측정한다(막대 길이는 예시). TTFB와 FCP가 Core Web Vitals에 들지 않은 이유는 다음 장. web.dev/articles/ttfb, web.dev/articles/fcp</div>

<!--
web.dev TTFB: "TTFB is a metric that measures the time between starting navigating to a page and when the first byte of a response begins to arrive." "TTFB is the sum of the following request phases: Redirect time, Service worker startup time (if applicable), DNS lookup, Connection and TLS negotiation, Request, up until the point at which the first byte of the response has arrived."
"Because TTFB isn't a Core Web Vitals metric, it's not absolutely necessary that sites meet the "good" TTFB threshold, provided that it doesn't impede their ability to score well on the metrics that matter."
web.dev FCP: "measures the time from when the user first navigated to the page to when any part of the page's content is rendered on the screen. For this metric, "content" refers to text, images (including background images), <svg> elements, or non-white <canvas> elements."
브라우저 기록으로는 TTFB가 navigation 항목의 responseStart, FCP가 paint 항목의 first-contentful-paint다(3장 첫 장의 표).
"원인 후보"의 늦게 로드되는 웹 폰트는 글자가 폰트를 기다리느라 그려지지 않는 경우다. 이 덱에서 따로 재지는 않았다.
-->

---

## TTFB와 FCP는 왜 Core Web Vitals가 아닐까

<div class="k">CORE WEB VITALS의 조건 (web.dev)</div>
<div class="cards">
<div class="card mute"><h3>① 서로 다른 측면</h3><p>사용자 경험의 한 측면을 지표 하나가 맡는다</p></div>
<div class="card mute"><h3>② 필드에서 측정</h3><p>실제 사용자 브라우저에서 측정할 수 있다</p></div>
<div class="card mute"><h3>③ 사용자가 느끼는 결과</h3><p>사용자에게 중요한 결과를 반영한다</p></div>
</div>

<div class="cards two">
<div class="card"><div class="k">FCP, ③에 맞지 않는다</div><h3>로딩의 시작만 잡는다</h3><p>스플래시 화면이나 로딩 표시만 떠도 FCP가 된다. 주 콘텐츠가 보였는지는 알 수 없다</p></div>
<div class="card"><div class="k">TTFB, ③에 맞지 않는다</div><h3>화면에 보이는 결과가 아니다</h3><p>서버와 연결의 응답 속도를 재는 기반 지표다. 첫 바이트가 빨라도 SPA처럼 내용은 늦게 그려질 수 있다</p></div>
</div>

<div class="card hl">① 로딩이라는 측면은 <b>LCP</b>가 대표한다. TTFB는 느린 서버 응답을, FCP는 렌더링을 막는 리소스를 찾을 때 쓴다.</div>

<div class="src">web.dev가 제외 이유를 직접 밝히지는 않아, CWV 조건과 지표 문서의 설명을 맞춰 정리했다. 출처: web.dev의 vitals, lcp, ttfb 문서</div>

<!--
CWV 조건, web.dev/articles/vitals: "Each of the Core Web Vitals represents a distinct facet of the user experience, is measurable in the field, and reflects the real-world experience of a critical user-centric outcome." 같은 문서는 현재 세트가 다루는 세 측면을 loading, interactivity, visual stability로 든다.
같은 문서의 "Other Web Vitals": "the metrics Time to First Byte (TTFB) and First Contentful Paint (FCP) are both vital aspects of the loading experience, and are both useful in diagnosing issues with LCP (slow server response times or render-blocking resources, respectively)."
FCP, web.dev/articles/lcp: "user-centric performance metrics like First Contentful Paint (FCP) only capture the very beginning of the loading experience. If a page shows a splash screen or displays a loading indicator, this moment isn't very relevant to the user."
TTFB, web.dev/articles/ttfb: "a foundational metric for measuring connection setup time and web server responsiveness". "Because TTFB isn't a Core Web Vitals metric, it's not absolutely necessary that sites meet the "good" TTFB threshold, provided that it doesn't impede their ability to score well on the metrics that matter." 이어서 SPA처럼 첫 마크업은 빨리 보내도 JS가 내용을 채워야 하는 경우를 예로 든다.
②는 두 지표 모두 충족한다. 같은 문서에서 "필드 측정이 안 되고 사용자 결과를 반영하지 않아서" 빠졌다고 명시한 것은 TBT이고, TTFB와 FCP 이야기가 아니다.
-->

---

## good과 poor를 가르는 기준값

<div class="scale head"><b></b><span>GOOD</span><span>개선 필요</span><span>POOR</span></div>
<div class="scale"><b class="c-lcp">LCP</b><span class="g">2.5초 이하</span><span class="w">4초 이하</span><span class="p">4초 초과</span></div>
<div class="scale"><b class="c-inp">INP</b><span class="g">200ms 이하</span><span class="w">500ms 이하</span><span class="p">500ms 초과</span></div>
<div class="scale"><b class="c-cls">CLS</b><span class="g">0.1 이하</span><span class="w">0.25 이하</span><span class="p">0.25 초과</span></div>
<div class="scale minor"><b>FCP</b><span class="g">1.8초 이하</span><span class="w">3초 이하</span><span class="p">3초 초과</span></div>
<div class="scale minor"><b>TTFB</b><span class="g">0.8초 이하</span><span class="w">1.8초 이하</span><span class="p">1.8초 초과</span></div>

<div class="src">기준값은 web-vitals가 내보내는 LCPThresholds 등의 상수와 같다. 판정은 p75로 한다(1장 마지막 장).</div>

---

## LCP 후보는 계속 바뀐다

<div class="screens">
<div class="shot"><div class="screen"><i class="t1 on"></i><i class="t2"></i><i class="t3"></i><i class="img empty"></i></div><p><b>28ms</b>후보: 제목 (H1)</p></div>
<div class="go"></div>
<div class="shot"><div class="screen"><i class="t1"></i><i class="t2"></i><i class="t3"></i><i class="img on"></i></div><p><b>1052ms</b>후보: 이미지 (IMG)</p></div>
<div class="say">

더 큰 요소가 그려지면 **후보가 바뀐다**

탭, 스크롤, 키 입력이 들어오면 브라우저는 후보 기록을 멈춘다

그래서 **언제 값을 확정할지** 정해야 한다. 4장에서 다룬다

</div>
</div>

<!--
재현 환경 demo-web의 ?lcp=200&img=800에서 잰 값이다. 제목(크기 6156)이 먼저 그려지고, 0.8초 늦게 온 이미지(크기 129600)가 더 커서 후보가 바뀌었다. 3장 첫 장의 코드를 콘솔에 붙이면 직접 볼 수 있다.
후보 기록을 멈추는 조건은 web.dev LCP 문서: "The browser will stop reporting new entries as soon as the user interacts with the page (via a tap, scroll, or keypress)".
-->

---

## LCP를 네 구간으로 나누면 고칠 곳이 보인다

<div class="k">LCP 요소가 이미지일 때, web.dev 권장 비율</div>
<div class="ratio"><span class="r1">① 약 40%</span><span class="r2">② 10%↓</span><span class="r3">③ 약 40%</span><span class="r4">④ 10%↓</span></div>

<div class="cards four phases">
<div class="card ph1"><div class="k">① TTFB</div><h3>HTML 첫 바이트까지</h3><p>이동을 시작해 HTML 첫 바이트를 받기까지</p><p class="fine">길어지는 경우: 느린 서버, CDN 캐시 미스, 리다이렉트</p></div>
<div class="card ph2"><div class="k">② 리소스 요청 지연</div><h3>요청 시작까지</h3><p>브라우저가 HTML에서 LCP 이미지를 발견해 요청하기까지</p><p class="fine">길어지는 경우: JS로 넣는 이미지, lazy 로딩, CSS 배경</p></div>
<div class="card ph3"><div class="k">③ 리소스 다운로드</div><h3>내려받는 시간</h3><p>요청한 이미지가 모두 도착하기까지</p><p class="fine">길어지는 경우: 큰 이미지, 캐시 없음, 동시에 받는 다른 요청</p></div>
<div class="card ph4"><div class="k">④ 요소 렌더링 지연</div><h3>그려지기까지</h3><p>도착한 이미지가 화면에 실제로 그려지기까지</p><p class="fine">길어지는 경우: 렌더링을 막는 CSS와 JS, 바쁜 메인 스레드</p></div>
</div>

<div class="card hl">네 구간의 <b>합이 곧 LCP</b>다. "지연"이 붙은 ②와 ④는 0에 가까울수록 좋다. LCP가 텍스트면 ②와 ③은 0이다.</div>

<div class="src">비율은 지침이다. LCP가 꾸준히 2.5초 안이면 비율은 크게 상관없다(web.dev/articles/optimize-lcp). web-vitals/attribution으로 네 구간 값을 받을 수 있다.</div>

<!--
web.dev/articles/optimize-lcp(2025-03-31 갱신)의 정의:
TTFB "The time from when the user initiates loading the page until the browser receives the first byte of the HTML document response."
Resource load delay "The time between TTFB and when the browser starts loading the LCP resource. If the LCP element doesn't require a resource load to render (for example, if the element is a text node rendered with a system font), this time is 0."
Resource load duration "The duration of time it takes to load the LCP resource itself."
Element render delay "The time between when the LCP resource finishes loading and the LCP element rendering fully."
"There is no overlap or gap between them. Collectively, they add up to the full LCP time."
권장 비율 표: TTFB ~40%, Resource load delay <10%, Resource load duration ~40%, Element render delay <10%. "Note that these time breakdowns are guidelines, not strict rules."
web-vitals attribution 필드 이름은 timeToFirstByte, resourceLoadDelay, resourceLoadDuration, elementRenderDelay(v6.2.3 src/attribution/onLCP.ts).
발표할 때는 "HTML이 오고, 브라우저가 HTML을 읽다가 img를 발견해 요청하고, 이미지가 다 오고, 화면에 그린다"는 순서로 한 번 따라가면 된다.
-->

---

## INP: 상호작용 하나를 세 구간으로 나눈다

<div class="flow inp">
<div><b>입력 지연</b>핸들러가 시작되기까지<small>원인 후보: 메인 스레드가 다른 작업 중</small></div>
<div><b>처리 시간</b>이벤트 핸들러 실행<small>원인 후보: 무거운 핸들러</small></div>
<div><b>다음 화면까지</b>스타일, 레이아웃, 그리기<small>원인 후보: 많은 DOM 변경</small></div>
</div>

<div class="cards two">
<div class="card mute"><div class="k">예를 들어</div><p>핸들러가 50ms 만에 끝나더라도, 그 전에 300ms짜리 작업이 실행 중이었다면</p></div>
<div class="card bad"><div class="k">INP</div><div class="big">350ms 이상</div></div>
</div>

<!-- attribution 필드는 inputDelay, processingDuration, presentationDelay다. Long Animation Frame을 지원하는 브라우저에서는 어떤 스크립트가 시간을 썼는지도 나온다. -->

---

## INP는 "거의 가장 느린" 상호작용

<div class="cards">
<div class="card mute"><div class="k">예전 FID (2024년 3월 12일까지)</div><div class="big">첫 입력</div><p>첫 상호작용의 입력 지연만 측정했다</p></div>
<div class="card inp"><div class="k">INP, 상호작용 50번 미만</div><div class="big">1위</div><p>가장 느린 것</p></div>
<div class="card inp"><div class="k">INP, 50번 이상 100번 미만</div><div class="big">2위</div><p>두 번째로 느린 것</p></div>
</div>

페이지에 머무는 동안의 **모든 상호작용**을 대상으로 한다. 오래 머문 페이지에서 한 번 튄 값이 INP가 되지 않도록, 상호작용 50번마다 가장 느린 것부터 하나씩 제외한다.

<!--
web-vitals 6.2.3 InteractionManager: 가장 긴 상호작용 10개를 들고 있다가 Math.floor(상호작용 수 / 50)번째를 고른다.
FID 교체: web.dev/blog/inp-cwv-march-12 "will replace First Input Delay on March 12", web.dev/articles/inp "FID only measured the input delay of the first interaction on a page."
web-vitals는 40ms 미만의 이벤트는 관찰하지 않는다(durationThreshold 기본 40). 단 첫 입력(first-input)은 길이와 상관없이 받는다.
-->

---

## CLS: 레이아웃 이동을 점수로 매긴다

<div class="cols">
<div>

<div class="clsbox"><div class="screen"><i class="union"></i><i class="before"></i><i class="after"></i></div><div class="legend"><p><span class="sw u"></span>영향 영역 <b>0.75</b></p><p><span class="sw a"></span>이동 거리 <b>0.25</b></p><p class="eq">0.75 × 0.25<br>= <b>0.1875</b></p></div></div>

</div>
<div>

### 이동 한 번의 점수

**영향 비율 × 거리 비율.** 화면 절반을 차지한 요소가 화면 높이의 25%만큼 내려가면 0.1875로, 이 한 번만으로 "개선 필요" 구간이다

### 이동이 여러 번이면

- 간격이 1초 미만인 이동을 한 묶음으로 본다(최대 5초)
- 합이 가장 큰 묶음의 점수가 CLS다
- 클릭, 탭, 키 입력 뒤 500ms 안의 이동은 제외한다

</div>
</div>

<!-- 공식, 예시, 세션 창, 500ms 규칙 모두 web.dev/articles/cls. 영향 비율은 밀리기 전과 후의 영역을 합친 것이 화면에서 차지하는 비율, 거리 비율은 가장 많이 움직인 거리를 화면의 긴 변으로 나눈 값이다. 예시는 세로가 긴 화면 기준. 스크롤, 드래그는 "입력 직후" 예외에 들지 않는다. -->

---

## 사용자 한 명이 아니라 p75로 판정한다

<div class="p75"><div class="fill">페이지 로드의 75%</div><div class="mark">p75</div></div>

<div class="cards two">
<div class="card"><h3>왜 75인가</h3><p>방문 대부분이 목표 성능을 경험했다고 볼 수 있고, 이상값 몇 개에 크게 흔들리지 않는다</p></div>
<div class="card"><h3>그래서 수집할 때도</h3><p>분포를 보존해야 한다. 평균만 저장하면 p75를 다시 계산할 수 없다</p></div>
</div>

<div class="src">같은 페이지라도 사용자마다 기기, 네트워크, 캐시 상태가 다르다. p75가 good이면 페이지 로드의 75% 이상이 good이라는 뜻이다. 근거: web.dev "Defining the Core Web Vitals metrics thresholds"</div>

<!-- "The 75th percentile was chosen based on two criteria. First, the percentile should ensure that a majority of visits to a page or site experienced the target level of performance. Second, the value at the chosen percentile shouldn't be overly impacted by outliers." -->

---

<!-- _class: chapter -->
<!-- _header: '' -->
<!-- _paginate: false -->

<div class="num">2장</div>

# 왜 재나

매출과 검색, 그리고 개발 환경과 다른 실제 사용자

---

<!-- header: '2장 왜 재나' -->

## 성능 개선 뒤 매출과 참여가 늘어난 사례

<div class="cards">
<div class="card lcp"><div class="k">VODAFONE</div><div class="big">+8%</div><p><b>판매</b><br>LCP 31% 개선</p><p class="fine">랜딩 페이지 A/B 테스트</p></div>
<div class="card inp"><div class="k">REDBUS</div><div class="big">+7%</div><p><b>판매</b><br>검색 페이지 INP 870ms대 → 350ms대 등</p><p class="fine">기간과 산정 방법은 공개하지 않음</p></div>
<div class="card cls"><div class="k">YAHOO! JAPAN 뉴스</div><div class="big">+15%</div><p><b>세션당 페이지뷰</b><br>CLS 0.2 개선</p><p class="fine">개선 전후 비교. 원문도 "상관관계"라고 표현</p></div>
</div>

<div class="card mute">

각 회사가 직접 발표한 수치다. 인과관계를 보여 준 것은 A/B 테스트를 한 Vodafone뿐이다. 우리 서비스에서도 같으리라는 보장은 없으니 **우리 서비스의 값을 직접 측정해야 한다.**

</div>

<!--
Vodafone: web.dev/case-studies/vodafone (2021-03-17) "a 31% improvement in LCP led to 8% more sales". 50:50 A/B, 하루 약 10만 클릭. 개선판에는 서버 렌더링과 이미지 최적화가 들어갔다. "Note that DCL actually increased 15%."
redBus: web.dev/case-studies/redbus-inp (2023-05-10 갱신) "The result was a 7% overall increase in sales." 입력 필드의 리듀서 호출을 줄여 그 페이지의 INP 72% 개선, 검색 결과를 30개에서 10개씩 불러오게 바꿔 870~900ms를 350~370ms로.
Yahoo! JAPAN 뉴스: web.dev/case-studies/yahoo-japan-news (2021-03-09) "they correlated the site's improved Cumulative Layout Shift (CLS) score with a 15% increase in page views per session and 13% increase in session duration."
-->

---

## 검색 순위에도 반영된다. 다만 가장 중요한 요소는 아니다

<div class="cards two">
<div class="card"><div class="k">GOOGLE SEARCH CENTRAL</div><h3>"Core Web Vitals는 순위 시스템에서 사용된다"</h3></div>
<div class="card"><div class="k">GOOGLE SEARCH CENTRAL</div><h3>"경험이 좋지 않더라도 가장 관련 있는 콘텐츠를 보여 주려 한다"</h3><p>유용한 콘텐츠가 많은 검색어에서는 좋은 페이지 경험이 도움이 된다</p></div>
</div>

<div class="card hl">

검색에서는 콘텐츠의 관련성이 먼저다. CWV는 순위보다 **사용자 경험을 확인하려고** 측정한다.

</div>

<!--
developers.google.com/search/docs/appearance/page-experience
"Core Web Vitals are used by our ranking systems. We recommend site owners achieve good Core Web Vitals for success with Search and to ensure a great user experience generally."
"Google Search always seeks to show the most relevant content, even if the page experience is sub-par. But for many queries, there is lots of helpful content available. Having a great page experience can contribute to success in Search, in such cases."
"There is no single signal."
-->

---

## Lighthouse 점수로는 왜 부족한가

<div class="cards two">
<div class="card"><div class="k">랩 데이터</div><h3>Lighthouse, DevTools</h3>

- 기기와 네트워크 조건이 하나로 고정
- 사용자 입력이 없어 **INP를 측정할 수 없다**
- 페이지 로드 구간만 측정

</div>
<div class="card"><div class="k">필드 데이터</div><h3>실제 사용자 브라우저</h3>

- 사용자마다 기기와 네트워크가 다르다
- 실제 클릭, 탭, 키 입력이 들어온다
- 페이지를 떠날 때까지 측정

</div>
</div>

<div class="flow">
<div class="hl"><b>필드</b>나빠진 지표를 찾는다</div>
<div><b>랩</b>재현하고 고친다</div>
<div class="hl"><b>필드</b>나아졌는지 확인한다</div>
</div>

<div class="src">랩 도구에서는 TBT(Total Blocking Time, 로드 중 메인 스레드가 막힌 시간)로 INP 문제를 가늠한다. INP를 대신하지는 못한다.</div>

<!--
web.dev/articles/inp: lab tools "won't report a page's INP because they only observe the loading of a page without any interactions".
web.dev/articles/lab-and-field-data-differences: "The Total Blocking Time (TBT) lab metric is intended to help diagnose issues with INP because it quantifies how much the main thread is blocked during page load."
Lighthouse에도 클릭을 재현하는 timespan 모드가 있지만, 그때 INP는 내가 한 클릭에 대한 값이다.
-->

---

## PageSpeed Insights의 필드 데이터만으로는 부족한 이유

<div class="cards four">
<div class="card"><div class="k">브라우저</div><h3>Chrome만</h3><p>데스크톱과 Android. iOS의 Chrome, Android 앱 웹뷰, Edge는 제외된다</p></div>
<div class="card"><div class="k">페이지</div><h3>방문이 많은 공개 페이지만</h3><p>검색 엔진이 색인할 수 있고 방문자 수가 기준 이상이어야 한다. 기준은 비공개</p></div>
<div class="card"><div class="k">기간</div><h3>최근 28일</h3><p>28일치를 합산한 값. 매일 갱신된다</p></div>
<div class="card"><div class="k">분석 기준</div><h3>우리 기준으로 못 나눈다</h3><p>배포 버전, 화면, 실험군별로 볼 수 없다</p></div>
</div>

PageSpeed Insights 상단의 필드 데이터는 **CrUX**(Chrome 사용자 경험 보고서)에서 가져온다. 출발점으로는 좋지만 위와 같은 한계가 있다.

<!--
developer.chrome.com/docs/crux/methodology: "There are a few notable exceptions that don't provide data to the CrUX dataset: Chrome on iOS. Android apps using WebView. Other Chromium browsers (for example Microsoft Edge)." "A page must be publicly discoverable ... using the same indexability criteria as search engines." "An exact number is not disclosed."
사용자도 사용 통계 보고와 방문 기록 동기화를 켠 사람만 들어간다.
developer.chrome.com/docs/crux/api: "28-day rolling average", "updated daily around 04:00 UTC". PageSpeed Insights 문서: "over the previous 28-day collection period".
회사 서비스가 앱 웹뷰 안에서 열린다면, PageSpeed Insights에 나오는 숫자는 같은 URL을 Chrome으로 직접 연 방문의 값이다. 웹뷰 사용자의 값이 아니다.
-->

---

## 그래서 직접 수집한다

<div class="flow">
<div><b>사용자 브라우저</b>실제 기기, 실제 클릭</div>
<div><b>web-vitals</b>성능 기록을 지표로</div>
<div><b>우리 서버</b>페이지를 떠날 때 전송</div>
<div class="hl"><b>대시보드</b>화면별, 버전별 p75</div>
</div>

<div class="cards">
<div class="card"><h3>빨리 알 수 있다</h3><p>배포 후 나빠진 지표를 다음 날 확인한다. 28일 평균을 기다릴 필요가 없다</p></div>
<div class="card"><h3>원하는 기준으로 나눠 본다</h3><p>화면, 기기, 배포 버전, 실험군</p></div>
<div class="card"><h3>원인 정보도 함께 받는다</h3><p>LCP 요소가 무엇이었는지, 어떤 클릭이 느렸는지</p></div>
</div>

<div class="card bad">대신 값이 <b>서버에 도착하지 않은 경우</b>도 직접 챙겨야 한다. 4장에서 다룬다.</div>

<!-- 흔히 RUM(Real User Monitoring)이라고 부른다. CrUX에 없는 방문(앱 웹뷰, iOS, 방문이 적은 페이지)도 볼 수 있다. 단 브라우저 지원 범위 안에서다(3장 마지막 장). -->

---

<!-- _class: chapter -->
<!-- _header: '' -->
<!-- _paginate: false -->

<div class="num">3장</div>

# 어떻게 재나

브라우저의 성능 기록과 web-vitals

---

<!-- header: '3장 어떻게 재나' -->

## 브라우저가 성능 기록을 남긴다

```js
new PerformanceObserver((list) => {
  for (const e of list.getEntries()) {
    console.log(Math.round(e.startTime), e.element?.tagName)
  }
}).observe({type: 'largest-contentful-paint', buffered: true})
```

| 기록 종류 | `navigation` | `paint` | `largest-contentful-paint` | `layout-shift` | `event`, `first-input` |
| --------- | ------------ | ------- | -------------------------- | -------------- | ---------------------- |
| 지표      | TTFB         | FCP     | LCP                        | CLS            | INP                    |

<div class="src">DevTools 콘솔에 붙여 넣으면 LCP 후보가 바뀔 때마다 출력된다. buffered: true를 주면 스크립트가 늦게 실행돼도 그 전에 쌓인 기록을 받을 수 있다.</div>

---

## 성능 기록은 브라우저 엔진이 남긴다

<div class="flow">
<div><b>브라우저 엔진</b>네트워크, 렌더링, 입력을 처리하다가</div>
<div><b>시각을 기록</b>그 순간을 <code>timeOrigin</code> 기준 ms로</div>
<div><b>Performance Timeline</b>기록을 버퍼에 보관</div>
<div class="hl"><b>PerformanceObserver</b>JS가 구독해서 받는다</div>
</div>

<div class="tight">

| 기록 종류                       | 정의한 명세               | 표준화 단계            |
| ------------------------------- | ------------------------- | ---------------------- |
| 공통 틀 (`PerformanceObserver`) | Performance Timeline      | W3C 후보 권고안 초안   |
| `navigation`                    | Navigation Timing Level 2 | W3C 작업 초안          |
| `paint`                         | Paint Timing              | W3C 작업 초안          |
| `largest-contentful-paint`      | Largest Contentful Paint  | W3C 작업 초안          |
| `event`                         | Event Timing              | W3C 작업 초안          |
| `layout-shift`                  | Layout Instability        | **WICG 커뮤니티 초안** |

</div>

<div class="src">JS가 시간을 재는 것이 아니라 엔진이 남긴 기록을 받아 본다. 그래서 스크립트가 늦게 실행돼도 기록된 시각은 그대로다. 표준화 단계는 2026-10-19 w3.org/TR, wicg.github.io 기준</div>

<!--
W3C 표준 트랙은 작업 초안(Working Draft) → 후보 권고안(Candidate Recommendation) → 권고안(Recommendation) 순서로 올라간다. 대부분은 W3C Web Performance 워킹 그룹 명세지만 아직 작업 초안 단계다.
Performance Timeline: W3C Candidate Recommendation Draft(TR 2025-05-21). Navigation Timing Level 2, Paint Timing, Largest Contentful Paint, Event Timing: W3C Working Draft.
Layout Instability API(wicg.github.io/layout-instability): Draft Community Group Report(2025-12-17). WICG는 W3C의 커뮤니티 그룹이라 이 문서는 표준 트랙에 들어가 있지 않다. 구현도 Chromium 계열뿐이다(BCD 8.1.6에서 Firefox, Safari 미지원). CLS가 Chromium에서만 잡히는 것과 같은 이야기다.
-->

---

## 계산 규칙은 web-vitals에 맡긴다

<div class="cols">
<div class="small-code">

```js
import {onCLS, onINP, onLCP} from 'web-vitals'

onLCP(console.log)
onINP(console.log)
onCLS(console.log)
```

LCP 확정, INP 선정, CLS 묶음 계산, 숨겨진 뒤의 그리기 제외, 뒤로 가기 복원 처리까지 Google의 web-vitals 라이브러리가 맡는다.

</div>
<div>

| 콜백이 받는 값   | 뜻                                          |
| ---------------- | ------------------------------------------- |
| `name`, `value`  | 지표 이름과 값                              |
| `rating`         | `good`, `needs-improvement`, `poor`         |
| `id`             | 페이지 로드마다 새로 생성. 보고를 묶는 기준 |
| `navigationType` | `back-forward-cache` 등                     |

</div>
</div>

<div class="src">web-vitals/attribution에서 불러오면 원인 정보(LCP 요소, 느린 구간)가 함께 전달된다.</div>

---

## 직접 해 보기: 로그는 언제 출력될까

<div class="tl">
<div><b>페이지를 연다</b><i></i><span class="tag ghost">출력 없음</span></div>
<div class="hl"><b>아무 곳이나 클릭</b><i></i><span class="tag lcp">LCP</span><small>조금 뒤</small></div>
<div class="end"><b>다른 탭에 다녀오기</b><i></i><span class="tag cls">CLS</span><span class="tag inp">INP</span></div>
<div class="end"><b>다시 클릭, 탭 전환</b><i></i><span class="tag ghost">값이 바뀐 지표만 다시</span></div>
</div>

<div class="card hl">

값은 페이지를 열자마자 나오지 않는다. **사용자가 무언가 하거나 화면을 떠날 때** 나온다. 4장의 내용은 대부분 여기서 출발한다.

</div>

<!-- 앞 장의 세 줄을 넣고 reportAllChanges 없이 쓴 경우다. 밀린 게 없으면 CLS는 0으로 찍힌다. 클릭 뒤 LCP가 찍히기까지는 브라우저가 한가해질 때까지, 길면 1초 기다린다(4장 질문 5). -->

---

## 브라우저마다 받을 수 있는 지표가 다르다

| 지표에 필요한 기록             | Chromium 계열 | Firefox  | Safari   |
| ------------------------------ | ------------- | -------- | -------- |
| LCP (`LargestContentfulPaint`) | 77            | 122      | 26.2     |
| INP (`interactionId`)          | 96            | 144      | 26.2     |
| CLS (`LayoutShift`)            | 77            | **없음** | **없음** |

<div class="cards">
<div class="card"><h3>iOS는 Safari 기준</h3><p>iOS의 Chrome, Firefox, 앱 웹뷰(WKWebView)도 같은 WebKit 엔진이다. 26.2 이전에는 TTFB와 FCP만 수집된다</p></div>
<div class="card"><h3>Android 웹뷰는 Chrome 기준</h3><p>Chromium 기반이라 기기에 설치된 WebView 버전의 Chrome과 같다</p></div>
<div class="card"><h3>CLS 건수가 적은 것은 정상</h3><p>web-vitals는 지원하지 않는 브라우저에서 오류 없이 아무 값도 보고하지 않는다</p></div>
</div>

<div class="src">MDN browser-compat-data 8.1.6</div>

<!-- 예외가 하나 있다. Apple은 EU에서만 브라우저 앱과 인앱 브라우저가 WebKit이 아닌 엔진(Chromium의 Blink, Firefox의 Gecko 등)을 쓰도록 허용한다. 별도 권한(Web Browser Engine Entitlement, Embedded Browser Engine Entitlement)을 받아야 하고 iOS 17.4, iPadOS 18 이상에서만 된다(developer.apple.com/support/alternative-browser-engines). EU 디지털 시장법(DMA) 대응으로 생긴 제도다. 실제로 이 방식으로 배포 중인 브라우저가 있는지는 확인하지 않았다. 한국 사용자에게는 해당하지 않으니 iOS는 Safari 기준으로 보면 된다. -->

<!-- 웹뷰: BCD 8.1.6에서 webview_android는 Chrome Android를, webview_ios는 Safari iOS를 그대로 따른다(mirror). 측정 API는 같지만, 앱이 웹뷰를 닫을 때 숨겨짐 신호가 오지 않는 경우가 있어 4장의 "떠나는 순간" 전송이 더 어렵다. 블로그 시리즈 3편의 주제다. -->

---

<!-- _class: chapter -->
<!-- _header: '' -->
<!-- _paginate: false -->

<div class="num">4장</div>

# 수집할 때 생기는 질문 다섯 가지

값이 서버에 도착하지 않는 경우

---

<!-- _class: statement -->
<!-- header: '4장 수집할 때 생기는 질문' -->

<div class="k">이 장의 원칙</div>

## CWV 값은 사용자가<br>**페이지를 떠날 때** 확정된다

<div class="cards">
<div class="card lcp"><p><b class="c-lcp">LCP</b> 후보가 바뀐다</p></div>
<div class="card cls"><p><b class="c-cls">CLS</b> 이동 점수가 누적된다</p></div>
<div class="card inp"><p><b class="c-inp">INP</b> 더 느린 클릭이 나오면 갱신된다</p></div>
</div>

<!-- 가게 마감 정산과 같다. 영업 중에는 매출이 계속 바뀌고, 셔터를 내릴 때 확정된다. 그때 보내지 못하면 그날 장부는 없다. 이 장의 다섯 질문은 모두 이 원칙으로 답한다. -->

---

## 이 장에서 답할 질문 다섯 가지

<div class="asks">
<div><em>1</em><span>"<code>console.log</code>를 걸었는데 왜 안 찍혀요?"</span></div>
<div><em>2</em><span>"CLS가 같은 페이지에서 두 번 왔어요"</span></div>
<div><em>3</em><span>"LCP가 FCP보다 적게 모여요. 데이터 빠진 거죠?"</span></div>
<div><em>4</em><span>"떠날 때 fetch로 보내면 되죠?"</span></div>
<div><em>5</em><span>"Next.js의 <code>useReportWebVitals</code>면 되죠?"</span></div>
</div>

---

<!-- _class: ask -->

<div class="q"><em>1</em>"왜 안 찍혀요?"</div>

## 지표마다 보고 시점이 다르다

<div class="tl">
<div><b>문서 열림</b><i></i></div>
<div><b>첫 그림</b><i></i><span class="tag ghost">FCP</span></div>
<div><b><code>load</code></b><i></i><span class="tag ghost">TTFB</span></div>
<div class="hl"><b>첫 클릭, 키 입력</b><i></i><span class="tag lcp">LCP</span><small>조금 뒤</small></div>
<div><b>계속 사용</b><i></i></div>
<div class="end"><b>숨겨질 때</b><i></i><span class="tag cls">CLS</span><span class="tag inp">INP</span><small>입력이 없었다면 LCP도</small></div>
</div>

<div class="cards">
<div class="card"><h3>TTFB</h3><p><code>load</code> 전에 떠나면 보고되지 않는다</p></div>
<div class="card"><h3>FCP, LCP</h3><p>그리기 전에 숨겨지면 보고되지 않는다</p></div>
<div class="card hl"><h3>CLS, INP</h3><p>원래 <b>숨겨질 때</b> 보고된다</p></div>
</div>

<!-- reportAllChanges 없이 쓴 경우다. reportAllChanges: true를 주면 값이 바뀔 때마다 콜백이 불린다. 근거는 블로그 시리즈 1편의 지표별 report(true) 위치. -->

---

<!-- _class: ask -->

<div class="q"><em>1</em>"왜 안 찍혀요?"</div>

## 값이 없으면 콜백도 불리지 않는다

<div class="flow">
<div><b>① 측정한 값이 있나</b><code>value &gt;= 0</code></div>
<div><b>② 보고 시점인가</b>숨겨질 때, 확정될 때</div>
<div><b>③ 첫 보고이거나 값이 바뀌었나</b>같은 값이면 다시 부르지 않는다</div>
<div class="ok"><b>콜백 호출</b><code>onINP(cb)</code>의 <code>cb</code></div>
</div>

<div class="cards two">
<div class="card bad"><div class="k">상호작용이 없던 페이지 로드</div><div class="big">INP 없음</div><p>0이 아니라 값 자체가 없다. 지표 값은 <b>-1</b>에서 시작한다</p></div>
<div class="card good"><div class="k">레이아웃 이동이 없던 페이지 로드</div><div class="big">CLS 0</div><p>CLS만 0에서 시작하므로 0도 보고된다</p></div>
</div>

<div class="src">서버에 INP가 없다는 사실만으로는 클릭이 없었던 것인지, 클릭은 있었지만 받지 못한 것인지 구별할 수 없다. web-vitals 6.2.3 src/lib/bindReporter.ts</div>

<!--
bindReporter를 줄이면 이렇다.
if (metric.value >= 0) { if (forceReport || reportAllChanges) { delta = value - (prevValue ?? 0); if (delta || prevValue === undefined) callback(metric) } }
-->

---

<!-- _class: ask -->

<div class="q"><em>2</em>"두 번 왔어요"</div>

## 숨겨질 때마다 다시 보고된다

<div class="tl">
<div><b>보임</b><i></i></div>
<div class="end"><b>숨김</b><i></i><span class="tag cls">CLS 0</span></div>
<div><b>다시 보임</b><i></i></div>
<div class="hl"><b>위쪽에 요소 삽입</b><i></i><small>레이아웃 이동 발생</small></div>
<div class="end"><b>숨김</b><i></i><span class="tag cls">CLS 0.1124</span><small>같은 id</small></div>
</div>

<div class="cards two">
<div class="card"><h3>서버에서는 <code>id</code>로 묶는다</h3><p>따로 세면 탭을 오간 페이지 로드가 두 번 집계된다. 첫 값과 마지막 값 중 무엇을 쓸지 정해 둔다</p></div>
<div class="card"><h3>콜백이 받는 <code>metric</code>은 매번 같은 객체</h3><p>저장해 둔 객체를 나중에 읽으면 그사이 바뀐 값이 보인다</p></div>
</div>

<!-- 다시 보일지는 알 수 없으니 숨겨질 때마다 그때까지의 값을 보낸다. 블로그 시리즈 3편의 재현 환경(e12-browser-lifecycle) 탭 복귀 시나리오에서 잰 값이다. LCP, FCP, TTFB는 페이지 로드마다 한 번만 보고된다. -->

---

<!-- _class: ask -->

<div class="q"><em>3</em>"적게 모여요"</div>

## 백그라운드 탭으로 열린 페이지는 그리기 지표가 빠진다

<div class="cards five">
<div class="card good"><div class="k">TTFB</div><div class="big">5/5</div></div>
<div class="card bad"><div class="k">FCP</div><div class="big">0/5</div></div>
<div class="card bad"><div class="k">LCP</div><div class="big">0/5</div></div>
<div class="card bad"><div class="k">CLS</div><div class="big">0/5</div></div>
<div class="card good"><div class="k">INP</div><div class="big">5/5</div></div>
</div>

- 10ms에서 18ms 만에 화면에 보이게 됐지만 결과는 같았다
- 한 번이라도 숨겨졌다면 **그 뒤에 그려진 것은 측정하지 않는다.** 사용자가 보지 못한 화면이기 때문이다
- CLS는 FCP가 보고된 뒤부터 측정하므로 함께 빠졌다

<div class="src">Chromium 156, web-vitals 6.2.3. 보고된 페이지 로드 수 / 전체. 포그라운드 탭(대조)은 다섯 지표 모두 5/5</div>

<!-- CDP의 Target.createTarget에 background: true를 주고 2초 뒤 앞으로 가져와 100ms짜리 버튼을 누르고 닫았다. Chromium이 왜 10ms 만에 보이게 바꿨는지는 확인하지 못했다. -->

---

<!-- _class: ask -->

<div class="q"><em>3</em>"적게 모여요"</div>

## `load` 전에 떠나면 TTFB가 보고되지 않는다

<div class="flow">
<div class="ok"><b>첫 바이트 도착</b>TTFB 값은 이때 이미 정해진다</div>
<div class="warn"><b><code>load</code>까지 대기</b>navigation 기록의 나머지 값이 채워질 때까지</div>
<div><b>보고</b><code>load</code> 다음 task</div>
</div>

<div class="cards two">
<div class="card bad"><div class="k">1초 뒤 이탈 (load는 3초)</div><div class="big">TTFB 0/5</div><p>FCP, LCP, CLS는 5/5</p></div>
<div class="card good"><div class="k">5초 뒤 이탈</div><div class="big">TTFB 5/5</div><p>모두 보고됨</p></div>
</div>

수집된 TTFB는 **`load`까지 머문 페이지 로드에 치우친다.** 서버가 느려 `load`도 늦어지는 페이지일수록 느린 TTFB가 더 많이 빠질 수 있다.

<!-- Chromium 156, web-vitals 6.2.3. 이미지 하나로 load를 3초 늦췄다. 치우침이 실제로 얼마인지는 재지 않았다. -->

---

<!-- _class: ask -->

<div class="q"><em>3</em>"적게 모여요"</div>

## 뒤로 가기로 복원된 페이지는 측정 기준이 다르다

<div class="cols">
<div>

| 클릭 핸들러 | 처음 로드 INP | 복원 뒤 INP |
| ----------- | ------------- | ----------- |
| 0ms         | 16ms          | **8ms**     |
| 20ms        | 24ms          | **8ms**     |
| 60ms        | 64ms          | 64ms        |

</div>
<div>

- **bfcache**: 뒤로, 앞으로 가기 때 페이지를 메모리에 보관했다가 그대로 복원하는 캐시
- 복원 뒤 FCP와 LCP는 프레임 두 개를 그리는 시간(약 16ms에서 39ms)
- 복원 뒤 TTFB는 0
- 40ms 미만의 클릭은 8ms로 보고된다

</div>
</div>

<div class="card hl"><code>navigationType</code>이 <code>back-forward-cache</code>인 값은 <b>따로 분리해서 본다.</b> 복원 비율이 늘면 LCP p75가 좋아진 것처럼 보일 수 있다.</div>

<!-- Chromium 156, web-vitals 6.2.3, Playwright가 기본으로 넣는 --disable-back-forward-cache를 뺐다. 15번 모두 복원됐다. 40ms는 web-vitals가 이벤트를 관찰하는 하한(durationThreshold)이다. -->

---

<!-- _class: ask -->

<div class="q"><em>4</em>"fetch로 보내면 되죠?"</div>

## 페이지가 닫혀도 전송되는 방법 두 가지

<div class="cards two">
<div class="card"><div class="k">방법 1</div><h3><code>navigator.sendBeacon(url, data)</code></h3><p>응답을 확인할 수 없다. 반환값 <code>true</code>는 <b>전송 대기열에 넣었다</b>는 뜻이다</p></div>
<div class="card"><div class="k">방법 2</div><h3><code>fetch(url, {keepalive: true, ...})</code></h3><p>문서가 아직 남아 있다면 응답을 확인할 수 있다</p></div>
</div>

- 일반 요청은 문서가 사라질 때 **함께 취소될 수 있다**
- 진행 중인 keepalive 요청의 본문 크기는 **모두 합쳐 64KiB로 제한된다**(Fetch 명세). 원인 정보를 많이 담으면 한도에 가까워진다

---

<!-- _class: ask -->

<div class="q"><em>4</em>"fetch로 보내면 되죠?"</div>

## sendBeacon이 성공해도 서버에 도착했다는 뜻은 아니다

| 보내는 방법     | 서버가 200 | 서버가 400 | CORS 거부                |
| --------------- | ---------- | ---------- | ------------------------ |
| sendBeacon      | 성공       | **성공**   | **성공** (서버 수신 0건) |
| keepalive fetch | 성공       | 실패       | 실패                     |

<div class="card hl">sendBeacon의 반환값으로 알 수 있는 것은 <b>브라우저 전송 대기열에 들어갔는지</b>까지다. 400 응답이나 CORS 거부도 대기열에는 들어간다.</div>

<div class="src">OpenTelemetry JS OTLP exporter 0.57.2(sendBeacon)와 0.223.0(keepalive fetch)이 보고한 결과, 각 조합 2번. 블로그 시리즈 2편, cwv-otel-grafana/experiments/e2-exporter</div>

---

<!-- _class: ask -->

<div class="q"><em>4</em>"fetch로 보내면 되죠?"</div>

## 페이지를 떠날 때 이벤트 발생 순서

<div class="steps">
<div><em>1</em><code>beforeunload</code><span class="tag ghost">visible</span><span></span></div>
<div class="hl"><em>2</em><code>pagehide</code><span class="tag ghost">visible</span><span>window에서만 발생한다. <b>document에 등록한 리스너는 호출되지 않았다</b></span></div>
<div class="hl"><em>3</em><code>visibilitychange</code><span class="tag hidden">hidden</span><span>web-vitals가 여기서 CLS와 INP를 보고한다</span></div>
<div><em>4</em><code>unload</code><span class="tag hidden">hidden</span><span></span></div>
</div>

<div class="card bad"><b>pagehide에서만 전송하면</b> CLS와 INP는 아직 보고되기 전이다. <span class="tag ghost">web-vitals 다음 배포부터는 pagehide에서도 보고</span></div>

<div class="src">같은 탭에서 링크로 이동, Chromium 156에서 2번 모두 같은 결과. visibilitychange는 window capture, document, window 순으로 호출됐다. cwv-otel-grafana/experiments/e3-lifecycle</div>

---

<!-- _class: ask -->

<div class="q"><em>4</em>"fetch로 보내면 되죠?"</div>

## 권장하는 방법: 숨겨질 때 모아서 한 번에

<div class="cols wide-left">
<div class="small-code">

```js
import {onCLS, onINP, onLCP} from 'web-vitals'

const queue = new Set()
const add = (metric) => queue.add(metric)
onCLS(add)
onINP(add)
onLCP(add)

addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'hidden') return
  if (queue.size === 0) return
  const body = JSON.stringify([...queue])
  navigator.sendBeacon('/analytics', body)
  queue.clear()
})
```

</div>
<div>

<div class="card hl"><h3>왜 visibilitychange인가</h3><p>web-vitals는 같은 이벤트를 window capture 단계에서 <b>먼저</b> 받는다. 그래서 이 리스너가 실행될 때는 CLS와 INP가 이미 queue에 들어 있다</p></div>

<div class="card"><h3>왜 <code>Set</code>인가</h3><p>같은 지표가 다시 보고돼도 같은 객체이므로 하나만 남는다</p></div>

</div>
</div>

<div class="src">web-vitals README의 "Batch multiple reports together"를 줄인 것</div>

---

<!-- _class: ask -->

<div class="q"><em>4</em>"fetch로 보내면 되죠?"</div>

## 도착 여부는 서버에서 센다

| 보낸 곳                        | TTFB | FCP | LCP | INP   | CLS   |
| ------------------------------ | ---- | --- | --- | ----- | ----- |
| `pagehide`, `visibilitychange` | 1    | 1   | 1   | **0** | **0** |
| `visibilitychange`만           | 1    | 1   | 1   | 1     | 1     |

<div class="cards two">
<div class="card"><h3>오류는 어디에도 없었다</h3><p>콘솔과 서버 로그 모두 오류가 없었다. TTFB, FCP, LCP만 확인했다면 정상이라고 판단했을 것이다</p></div>
<div class="card hl"><h3>그래서</h3><p>수집 코드를 바꾸면 실제로 페이지를 떠나 보고, <b>지표별로 보낸 건수와 받은 건수</b>를 비교한다</p></div>
</div>

<div class="src">원인은 OpenTelemetry JS(sdk-metrics 2.9.0 이상)가 두 번째 전송 전에 첫 번째 전송의 응답을 기다린 것이다(블로그 시리즈 2편). 도구가 달라도 확인 방법은 같다</div>

<!-- README의 sendBeacon 모양은 응답을 기다리지 않으므로 이 문제는 없을 것으로 보지만, 그 조합은 재지 않았다. 측정 코드는 cwv-otel-grafana/experiments/e7-unload-flush. -->

---

<!-- _class: ask -->

<div class="q"><em>5</em>"useReportWebVitals면 되죠?"</div>

## INP는 클릭 직후 처리되지 않는다

<div class="flow inp">
<div><b>클릭</b>핸들러 실행</div>
<div><b>다음 화면</b>렌더링</div>
<div><b>기록 도착</b>web-vitals가 수신</div>
<div class="warn"><b>idle까지 대기</b>6.2.3: 최대 1초<br>4.2.1: 상한 없음</div>
<div class="ok"><b>INP 반영</b>이후 보고 가능</div>
</div>

<div class="cards two">
<div class="card"><h3>왜 기다리나</h3><p>한 상호작용에 속한 기록이 모두 도착할 때까지 기다리기 위해서다(소스 주석). idle은 브라우저가 할 일이 없는 때다</p></div>
<div class="card hl"><h3>측정해 보니</h3><p>Chromium에서는 클릭 뒤 <b>다른 입력이 없으면</b> idle 콜백이 호출되지 않았다. 마우스를 움직이면 그때 처리된다</p></div>
</div>

<!-- requestIdleCallback이다. Chromium 153, 156, Chrome Beta 157에서 같았고, 원인이 스케줄러 어디에 있는지는 확인하지 못했다. Safari에는 requestIdleCallback이 없어 setTimeout으로 다음 task에 처리한다(소스로만 확인). -->

---

<!-- _class: ask -->

<div class="q"><em>5</em>"useReportWebVitals면 되죠?"</div>

## 대기 중에 페이지를 떠나면 INP가 보고되지 않는다

<div class="cols">
<div>
<div class="k">WEB-VITALS 6.2.3 (지금 npm 최신)</div>
<div class="bars">
<div class="bar"><span>0.1초 뒤</span><i><i class="w10"></i></i><em>1/10</em></div>
<div class="bar"><span>0.3초 뒤</span><i><i class="w0"></i></i><em>0/10</em></div>
<div class="bar"><span>0.6초 뒤</span><i><i class="w10"></i></i><em>1/10</em></div>
<div class="bar"><span>1.5초 뒤</span><i><i class="w100"></i></i><em>10/10</em></div>
</div>
</div>
<div>
<div class="k">MAIN (다음 배포 예정, #809)</div>
<div class="bars">
<div class="bar"><span>0.1초 뒤</span><i><i class="w100"></i></i><em>10/10</em></div>
<div class="bar"><span>0.3초 뒤</span><i><i class="w100"></i></i><em>10/10</em></div>
<div class="bar"><span>0.6초 뒤</span><i><i class="w100"></i></i><em>10/10</em></div>
<div class="bar"><span>1.5초 뒤</span><i><i class="w100"></i></i><em>10/10</em></div>
</div>
</div>
</div>

<div class="cards">
<div class="card"><h3>원인</h3><p>숨겨질 때 보고 처리가 먼저 실행되고, 대기 중이던 클릭 처리는 그 뒤에 실행된다</p></div>
<div class="card"><h3>수정</h3><p>2026-10-15 main에 병합. npm 최신은 아직 6.2.3</p></div>
<div class="card hl"><h3>배포 전까지는</h3><p>INP만 <code>reportAllChanges</code>를 켜고 <code>id</code>별 마지막 값을 보낸다(30번 중 30번 보고)</p></div>
</div>

<!-- 클릭 핸들러 200ms, 끝난 뒤 location.href로 이동, 메인 스레드가 한가한 페이지, Chromium 156. 바쁜 페이지는 1.5초 전 조건이 전부 0/10이었다. 원인은 window capture 단계에 visibilitychange 리스너 두 개가 등록된 순서다(가시성 감시가 먼저, 대기 중인 처리가 나중). 이슈는 web-vitals #805. 실제 사용자는 클릭 뒤 마우스를 움직이는 경우가 많아 손실은 이 표보다 작을 수 있다. 터치 기기에서는 재지 않았다. -->

---

<!-- _class: ask -->

<div class="q"><em>5</em>"useReportWebVitals면 되죠?"</div>

## Next.js 16.3.5에는 web-vitals 4.2.1이 내장되어 있다

<div class="cards">
<div class="card bad"><div class="k">4.2.1, 클릭 1.5초 뒤 이탈</div><div class="big">0/5</div><p>클릭 뒤 다른 입력 없음</p></div>
<div class="card bad"><div class="k">4.2.1, 클릭 4.5초 뒤 이탈</div><div class="big">0/5</div><p>클릭 뒤 다른 입력 없음</p></div>
<div class="card good"><div class="k">6.2.3, 클릭 1.5초 뒤 이탈</div><div class="big">10/10</div><p>1초 상한이 있어 보고됨</p></div>
</div>

- `useReportWebVitals`는 Next.js에 내장된 4.2.1로 측정한다. 4.2.1에는 **대기 시간 1초 상한이 없다**
- 직접 설치한 web-vitals와 함께 쓰면 같은 페이지 로드라도 값과 `id`가 서로 다르다
- 프레임워크가 제공하는 도구도 **내장된 web-vitals 버전**이 페이지를 떠나기 직전의 값을 어떻게 처리하는지 확인해야 한다

<!-- Next.js v16.3.5 packages/next/package.json의 devDependencies web-vitals: 4.2.1. 4.2.1의 src/lib/whenIdle.ts는 requestIdleCallback을 timeout 없이 부른다. 4.2.1은 한가한 페이지, 바쁜 페이지 모두 0/5. -->

---

<!-- _class: ask -->

<div class="q"><em>5</em>"useReportWebVitals면 되죠?"</div>

## 측정 이야기: 1.28초의 정체

<div class="cards">
<div class="card"><div class="k">클릭을 1.3초 간격으로 반복</div><div class="big">1.28초</div><p>"클릭 후 1.28초면 idle이 온다"?</p></div>
<div class="card"><div class="k">클릭을 3.5초 간격으로 반복</div><div class="big">약 3.5초</div><p>idle 시각도 함께 늦어졌다</p></div>
<div class="card hl"><div class="k">정체</div><div class="big">다음 클릭</div><p>idle이 아니라 다음 입력이 들어간 시각이었다</p></div>
</div>

- 측정 방법 자체가 결과에 영향을 준다. **조건을 하나 바꿔 다시 측정하면** 드러난다
- 자동화 도구 때문인지 확인하려고 DevTools를 연결하지 않은 Chrome에 OS 수준 클릭으로 다시 측정했다. 결과는 같았다
- 확인한 것은 "다른 입력이 없으면 idle 콜백이 호출되지 않는다"까지이고, 원인은 아직 확인하지 못했다

<!-- 자동화 도구는 Chrome DevTools Protocol(CDP)이다. OS 수준 클릭은 macOS의 HID 이벤트 탭(CGEvent)으로 넣었다. -->

---

<!-- header: '정리' -->

## 원칙 하나로 다섯 질문에 답하기

> CWV 값은 사용자가 **페이지를 떠날 때** 확정된다

<div class="qa">
<div><span>1. 왜 안 찍혀요?</span><span>값이 확정될 때 보고되고, 값이 없으면 보고되지 않는다</span></div>
<div><span>2. 왜 두 번 와요?</span><span>숨겨질 때마다 그때까지의 값을 보고한다. 돌아와서 값이 바뀌면 다시 온다</span></div>
<div><span>3. 왜 적게 모여요?</span><span>보지 않은 화면, 확정 전에 떠난 페이지, 지원하지 않는 브라우저에서는 값이 원래 없다</span></div>
<div><span>4. 언제 어떻게 보내요?</span><span>숨겨질 때 한 번, 페이지가 닫혀도 전송되는 방법으로 보낸다. 도착 여부는 서버에서 센다</span></div>
<div><span>5. 기본 도구면 돼요?</span><span>내장된 web-vitals 버전이 떠나기 직전의 값을 어떻게 처리하는지 확인한다</span></div>
</div>

---

<!-- _class: quiz -->
<!-- _header: '종합 퀴즈' -->

## 종합 퀴즈

<div class="cards two">
<div class="card"><div class="k">Q1</div><p>페이지를 열고 5초 동안 <b>스크롤만</b> 하다가 탭을 닫았다. 탭을 닫을 때 처음 보고되는 지표는?</p></div>
<div class="card"><div class="k">Q2</div><p>Lighthouse 점수는 100인데, 실제 사용자의 INP p75는 450ms다. 어떻게 가능할까?</p></div>
<div class="card"><div class="k">Q3</div><p>같은 <code>id</code>의 CLS가 0.02, 0.15로 두 번 도착했다. 둘 다 분포에 넣으면 어떤 문제가 생길까?</p></div>
<div class="card"><div class="k">Q4</div><p>Next.js 16.3.5 앱에서 "구매하기"를 누르면 곧바로 다른 도메인의 결제 페이지로 간다. 이 클릭의 INP는 서버에 보고될까?</p></div>
</div>

<!--
Q1. LCP, CLS. 스크롤은 LCP를 확정하는 신호가 아니라서(프로그램으로도 만들 수 있어 web-vitals가 뺐다) LCP는 숨겨질 때 확정된다. 이 경우는 소스로만 확인했다. CLS는 원래 숨겨질 때 보고한다. 스크롤은 상호작용이 아니라서 INP는 없다. TTFB는 load 뒤, FCP는 그린 직후에 이미 보고됐다.
Q2. Lighthouse는 클릭하지 않으므로 INP를 재지 않는다. 개발 노트북과 사용자의 휴대폰은 다르다. 무거운 핸들러는 로드가 끝난 뒤 사용자가 누를 때 드러난다.
Q3. 같은 페이지 로드를 두 번 센다. 탭을 오간 로드만 두 번 들어가 분포가 틀어진다. id로 묶어 마지막 값(또는 처음 값) 하나만 쓴다.
Q4. 거의 오지 않는다. useReportWebVitals는 web-vitals 4.2.1이라 클릭 뒤 다른 입력이 없으면 클릭을 처리하기 전에 페이지가 사라진다. web-vitals 6.2.3을 직접 써도 1초 안에 떠나면 대부분 빠진다. 다음 배포(#809 포함)부터는 온다.
-->

---

<!-- _class: quiz -->
<!-- _header: '종합 퀴즈' -->

## 종합 퀴즈

<div class="card"><div class="k">Q5</div><p>이 수집 코드의 문제 두 가지는?</p></div>

```js
document.addEventListener('pagehide', () => {
  navigator.sendBeacon('/vitals', JSON.stringify([...queue]))
})
```

<!--
1. pagehide는 window에서만 발생하므로 document에 등록한 리스너는 호출되지 않는다.
2. window에 걸어도 pagehide는 visibilitychange보다 먼저 와서, 6.2.3 기준 CLS와 INP가 아직 큐에 없다. visibilitychange의 hidden에서 보내야 한다.
-->

---

<!-- header: '참고' -->

## 참고

- 블로그 시리즈 1편: [web-vitals는 언제 핵심 웹 지표를 보고하는가](https://yceffort.kr/2026/10/web-vitals-report-timing)
- 블로그 시리즈 2편: [페이지를 떠나는 순간 핵심 웹 지표를 보내기](https://yceffort.kr/2026/10/cwv-send-on-page-leave)
- [web-vitals v6.2.3](https://github.com/GoogleChrome/web-vitals/tree/v6.2.3), INP 누락 [#805](https://github.com/GoogleChrome/web-vitals/issues/805)와 수정 [#809](https://github.com/GoogleChrome/web-vitals/pull/809)
- web.dev: [LCP](https://web.dev/articles/lcp), [INP](https://web.dev/articles/inp), [CLS](https://web.dev/articles/cls), [기준값을 정한 방법](https://web.dev/articles/defining-core-web-vitals-thresholds), [랩과 필드의 차이](https://web.dev/articles/lab-and-field-data-differences)
- 사례: [Vodafone](https://web.dev/case-studies/vodafone), [redBus](https://web.dev/case-studies/redbus-inp), [Yahoo! JAPAN 뉴스](https://web.dev/case-studies/yahoo-japan-news)
- [Google Search Central: 페이지 경험](https://developers.google.com/search/docs/appearance/page-experience), [CrUX 수집 방법](https://developer.chrome.com/docs/crux/methodology)
- 측정 코드: [blog-experiments/cwv-otel-grafana](https://github.com/yceffort/blog-experiments/tree/main/cwv-otel-grafana)
