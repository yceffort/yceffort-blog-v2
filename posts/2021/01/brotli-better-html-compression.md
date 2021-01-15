---
title: '더 나은 압축 알고리즘, Brotli'
tags:
  - browser
published: true
date: 2021-01-07 23:57:07
description: '왜 이걸 이제 알았나 자괴감 들고 괴로워'
---

웹 사이트의 크기는 해가 갈 수록 커지고 있다. 3년 전과 비교 했을 때, 데스크톱 사이트는 20.5%, 모바일 웹 사이트의 경우에는 24.1%나 커졌다.

![total-kb](./images/website-total-kilobytes.png)

https://httparchive.org/reports/page-weight?start=2016_11_15&end=latest&view=list

다행히도(?) 단순히 사이즈만 커져가지는 않았다. 웹사이트 방문자들에게 가능한 최소한의 사이즈로 제공하기 위한 많은 노력들이 있다. 그중에 하나가 HTTP 압축이다.

## HTTP Compression

간단하게 얘기해서, 압축을 해서 웹서버에서 더 작은 파일을 서빙할 수 있도록 하는 기술이다. 사용자가 URL을 통해서 웹사이트에 접속을 하면, 브라우저는 웹서버가 압축을 해서 보낸 다는 것을 인지한다. 이러한 압축은 네트워크 요청을 빠르게 하여 가능한 로딩을 빠르게 도와준다. (압축을 푸는 것이 네트워크 요청보다 더 빠르므로)

### Gzip

이러한 압축 알고리즘으로 가장 널리 알려져 있는 것이 [gzip](https://ko.wikipedia.org/wiki/Gzip)이다. 최대 70%까지 사이즈를 줄여주는 가장 보편적인 압축 알고리즘이다. Gzip은 웹사이트 파일의 중복코드, 띄어쓰기의 양을 줄여서 동작하고, 9단계에 걸친 옵션을 제공하여 압축량과 압축에 걸리는 시간을 세세하게 조정할 수도 있다.

웹 사이트에서 Gzip을 제공하는 방법은 호스팅 공급자와 웹서버에 따라 달라진다.

아파치 서버의 경우, `.htacess`에 아래 코드를 추가하면 된다.

```bash
AddOutputFilterByType DEFLATE text/plain
AddOutputFilterByType DEFLATE text/html
AddOutputFilterByType DEFLATE text/xml
AddOutputFilterByType DEFLATE text/css
AddOutputFilterByType DEFLATE application/xml
AddOutputFilterByType DEFLATE application/xhtml+xml
AddOutputFilterByType DEFLATE application/rss+xml
AddOutputFilterByType DEFLATE application/javascript
AddOutputFilterByType DEFLATE application/x-javascript
```

nginx의 경우에는 다음과 같다.

```bash
gzip on;
gzip_comp_level 2;
gzip_http_version 1.0;
gzip_proxied any;
gzip_min_length 1100;
gzip_buffers 16 8k;
gzip_types text/plain text/html text/css application/x-javascript text/xml application/xml application/xml+rss text/javascript;
gzip_disable "MSIE [1-6].(?!.*SV1)";
gzip_vary on;
```

> 물론 이거 같다 붙힌다고 만사 해결이 되는 것은 아니다. 반드시 세세한 설정을 거쳐야 한다.

이렇듯 gzip은 설치가 용이하고, 압축도 잘되며, 압축에 걸리는 시간도 빠르기 때문에 널리 사용되고 잇다.

## Brotli

- https://github.com/google/brotli
- https://en.wikipedia.org/wiki/Brotli
- https://aws.amazon.com/ko/about-aws/whats-new/2020/09/cloudfront-brotli-compression/

2013년, 구글은 웹 폰트 압축을 위해 Brotli라는 새로운 압축 알고리즘을 만들었으며, 2년 뒤인 2015년에는 HTTP 압축에 사용할 버전을 만들었다. 이는 앞서 언급한 GZip에 비해 많은 우위를 가지고 있었다.

![css file size](https://miro.medium.com/max/2000/1*j-3dAHj0pu5E1GmtD6eMuw.png)

![javascript file size](https://miro.medium.com/max/2000/1*i3xJiqPfF84H1h2Rt7na9w.png)

![overall](https://miro.medium.com/max/1400/1*_GVXtCwykvrPzclpIcpAsw.png)

[이 글](https://certsimple.com/blog/nginx-brotli)에 따르면 Brotli를 활용했을 경우 자바스크립트 파일의 경우 gzip에 비해 14%, HTML은 21%, css는 17% 더 작게 만들어 준다고 나와있다.

https://tools.keycdn.com/brotli-test 에서 brotli를 지원하는지 확인할 수 있다.

![yceffort blog](./images/yceffort-brotli.png)

> content-encoding이 br로 되어 있으면 brotli를 사용한다는 뜻이다.

### 적용하는 방법

**apache**

```bash
<VirtualHost *:443>
…
…
RewriteEngine On
RewriteCond %{HTTP:Accept-Encoding} br
RewriteCond %{DOCUMENT_ROOT}/%{REQUEST_FILENAME}.br -f
RewriteRule ^(.*)$ $1.br [L]
RewriteRule ".br$" "-" [NS,E=no-gzip:1,E=dont-vary:1]
<Files *.js.br>
  AddType "text/javascript" .br
  AddEncoding br .br
</Files>
<Files *.css.br>
  AddType "text/css" .br
  AddEncoding br .br
</Files>
<Files *.svg.br>
    AddType "image/svg+xml" .br
    AddEncoding br .br
</Files>
<Files *.html.br>
    AddType "text/html" .br
    AddEncoding br .br
</Files>
</VirtualHost>
```

[nginx-brotli](https://github.com/google/ngx_brotli)를 설치하여 제공할 수 있다.

**nginx**

```bash
http{
    brotli_static on;
    brotli_types text/plain text/css application/javascript application/x-javascript text/xml application/xml application/xml+rss text/javascript image/x-icon image/vnd.microsoft.icon image/bmp image/svg+xml;
}
```

자세한 내용은 https://www.tezify.com/how-to/use-brotli-compression/ 를 참조.

주의 할 것은, jpg, jpeg와 같이 이미 압축되어 있는 컨텐츠를 다시 압축할 필요는 없다는 것이다. 또한 html, js만 압축할 것이 아니라 xml, json등의 파일도 압축 대상에 포함시켜야 한다.

## 그러나

그러나 brotli는 ie에서 지원하지 않는다.

https://caniuse.com/brotli

![can-i-use-brotli](./images/can-i-use-brotli.png)

따라서, ie 환경을 고려하고 있다면 gzip 방식으로도 컨텐츠를 제공해야 한다.
