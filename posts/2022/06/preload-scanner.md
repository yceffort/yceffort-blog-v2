---
title: '브라우저의 프리로드 스캐너 (pre-load scanner)와 파싱 동작의 이해'
tags:
  - web
  - browser
published: true
date: 2022-06-12 18:10:40
description: '브라우저랑 싸우지마'
---

## Table of Contents

## Introduction

웹 개발자가 웹 페이지 속도를 개선하기 위해서 가장 먼저 알아야할 것은, 웹 페이지 로딩 속도 개선을 위한 테크닉이나 팁이 아닌 바로 브라우저의 동작 원리다. 우리가 최적화 하려고 노력하는 것 만큼, 브라우저도 웹페이지를 로딩할 때 최적화 하려고 더 노력한다. 하지만 우리의 이런 최적화 노력이 의도치 않게 브라우저의 최적화 노력을 방해할 수도 있다.

페이지 속도 개선에 있어 가장 먼저 이해해야할 것 중 하나는 바로 브라우저의 프리로드 스캐너다. 프리로드 스캐너는 무엇인지, 그리고 우리가 이 작업을 방해하지 ㅇ낳기 위해서는 무엇을 해야 하는지 알아보자.

## Pre-load Scanner란 무엇인가

모든 브라우저는 raw markup 상태의 파일을 [토큰화](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization) 하고, [객체 모델](https://developer.mozilla.org/ko/docs/Web/API/Document_Object_Model)로 처리하는 HTML 파서를 기본적으로 가지고 있다. 이 모든 작업은 이 파서가 `<link />` 또는 `async` `defer`가 없는 `<script />`와 같은 [블로킹 리소스](https://web.dev/render-blocking-resources/)를 만나기 전 까지 계속 된다.

![html-parser](https://web-dev.imgix.net/image/jL3OLOhcWUQDnR4XjewLBx4e3PC3/mXRoJneD6CbMAqaTqNZW.svg)

CSS 파일의 경우, [스타일링이 적용되지 않는 콘텐츠가 잠깐 뜨는 현상(Flash of unstyled content, AKA FOUC)](https://en.wikipedia.org/wiki/Flash_of_unstyled_content)을 방지하기 위해 파싱과 렌더링이 차단된다.

그리고 자바스크립트의 경우에는, 앞서 언급했듯 `async` `defer`가 없는 `<script/>`를 만나게 되면 파싱과 렌더링 작업이 중단된다.

> `<script />`에 `type=module`이 있다면 기본적으로 `defer`로 동작한다.
