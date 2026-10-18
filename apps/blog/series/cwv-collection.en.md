---
name: 'Collecting Core Web Vitals Yourself'
title: 'Collecting <em>Core Web Vitals</em> Yourself'
description: 'A series that splits the CWV measured by web-vitals into OpenTelemetry traces and metrics, and re-measures in a reproduction environment every point where a value disappeared without an error on its way from the browser to a Grafana panel'
---

The hard part of collecting CWV yourself was not the measuring but the road from a measured value to a dashboard. The values web-vitals computed vanished inside the browser, at the moment the page was left, in the collector and when the stores were queried, and none of those places raised an error. This series is a record of finding those points one by one and measuring them again in a public reproduction environment.

The same question was worked through in two settings. One sends the values of screens opened in an app webview to an internal observability platform shared by many teams, where the constraints were a webview that gives no signal on leaving and a collector whose settings could not be changed. The other puts the same collection on this blog and runs the collection server as well, where every setting can be changed but visits are rare.

The measurement code and experiments are in [blog-experiments/cwv-otel-grafana](https://github.com/yceffort/blog-experiments/tree/main/cwv-otel-grafana), and each post and each experiment's README states the versions its numbers were measured against.
