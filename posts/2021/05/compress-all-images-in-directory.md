---
title: '디렉토리에 있는 모든 이미지 최적화 하기'
tags:
  - blog
  - images
published: true
date: 2021-05-02 20:18:24
description: 'PNG는 좀 느립니다'
---

블로그에서 추가되는 이미지를 자동으로 최적화하기 위해 https://imgbot.net/ 을 사용하고 있다. imgbot을 사용하면, 새롭게 추가되는 이미지들에 대해서 최적화를 해주는 PR을 만들어 준다. https://github.com/yceffort/yceffort-blog-v2/pull/298 (빌드가 실패한 것에 대해선 정말 긴 히스토리가 있다, imgbot 때문이 아니다)

그러나 기존에 추가되었던 이미지들에 대해서는 최적화가 안되기 때문에, 이를 위해서 방법을 찾아보다가 아래와 같은 라이브러리를 사용했다.

## PNG

```bash
brew install optipng
apt-get install optipng
```

```bash
find . -iname "*.png" -exec optipng -o7 {} \;
```

http://optipng.sourceforge.net/

PNG의 경우에는 굉장히 오래걸렸다.

## JPG, JPEG

```bash
brew install jpegoptim
sudo apt-get install jpegoptim
```

```bash
find . -iname "*.jpg" -exec jpegoptim -m80 -o -p {} \;
```

## GIF

```bash
brew install gifsicle
sudo apt-get install gifsicle
```

```bash
find . -iname "*.gif" -exec gifsicle --batch -V -O2 {} \;
```

## GUI application

일괄적으로 모든 파일을 수정하기 위해서는 위 명령어를 사용했지만, GUI application도 있어서 필요할 때 사용하면 좋을 것 같다.

https://imageoptim.com/mac
