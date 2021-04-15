---
title: Docker 공부 (3) - 도커 이미지
tags:
  - docker
published: true
date: 2020-08-09 03:48:47
description:
  '## 도커 이미지 npm에서 다양한 도커 관련 패키지를 관리하듯, 도커는 기본적으로 [Docker
  Hub](https://hub.docker.com/)라는 중앙 이미지 저장소에서 다양한 이미지를 내려받을 수 있다. Docker Hub는
  도커가 제공하고 있는 이미지 저장소로, 누구나 도커 계정을 가지고 있다면 쉽게 이미지를 공유할 수 있다.  `docker...'
category: docker
slug: /2020/08/docker-study-3/
template: post
---

## 도커 이미지

npm에서 다양한 도커 관련 패키지를 관리하듯, 도커는 기본적으로 [Docker Hub](https://hub.docker.com/)라는 중앙 이미지 저장소에서 다양한 이미지를 내려받을 수 있다. Docker Hub는 도커가 제공하고 있는 이미지 저장소로, 누구나 도커 계정을 가지고 있다면 쉽게 이미지를 공유할 수 있다.

`docker create`, `docker run`, `docker pull` 등의 명령어로 이미지를 내려받을 때는 이 Docker hub에서 검색한 뒤에 내려받는다. 다만 주의 할 것은 누구나 올릴 수 있으므로, `official` 딱지가 붙어있는 이미지를 사용하는 것이 좋다.

```shell
ubuntu@study:~$ docker search ubuntu
NAME                                                      DESCRIPTION                                     STARS               OFFICIAL            AUTOMATED
ubuntu                                                    Ubuntu is a Debian-based Linux operating sys…   11187               [OK]
dorowu/ubuntu-desktop-lxde-vnc                            Docker image to provide HTML5 VNC interface …   452                                     [OK]
rastasheep/ubuntu-sshd                                    Dockerized SSH service, built on top of offi…   246                                     [OK]
consol/ubuntu-xfce-vnc                                    Ubuntu container with "headless" VNC session…   222                                     [OK]
ubuntu-upstart                                            Upstart is an event-based replacement for th…   110                 [OK]
neurodebian                                               NeuroDebian provides neuroscience research s…   68                  [OK]
1and1internet/ubuntu-16-nginx-php-phpmyadmin-mysql-5      ubuntu-16-nginx-php-phpmyadmin-mysql-5          50                                      [OK]
ubuntu-debootstrap                                        debootstrap --variant=minbase --components=m…   44                  [OK]
nuagebec/ubuntu                                           Simple always updated Ubuntu docker images w…   24                                      [OK]
i386/ubuntu                                               Ubuntu is a Debian-based Linux operating sys…   22
1and1internet/ubuntu-16-apache-php-5.6                    ubuntu-16-apache-php-5.6                        14                                      [OK]
1and1internet/ubuntu-16-apache-php-7.0                    ubuntu-16-apache-php-7.0                        13                                      [OK]
1and1internet/ubuntu-16-nginx-php-phpmyadmin-mariadb-10   ubuntu-16-nginx-php-phpmyadmin-mariadb-10       11                                      [OK]
1and1internet/ubuntu-16-nginx-php-5.6                     ubuntu-16-nginx-php-5.6                         8                                       [OK]
1and1internet/ubuntu-16-nginx-php-5.6-wordpress-4         ubuntu-16-nginx-php-5.6-wordpress-4             7                                       [OK]
1and1internet/ubuntu-16-nginx-php-5.6-wordpress-4         ubuntu-16-nginx-php-5.6-wordpress-4             7                                       [OK]
1and1internet/ubuntu-16-apache-php-7.1                    ubuntu-16-apache-php-7.1                        6                                       [OK]
darksheer/ubuntu                                          Base Ubuntu Image -- Updated hourly             5                                       [OK]
pivotaldata/ubuntu                                        A quick freshening-up of the base Ubuntu doc…   4
1and1internet/ubuntu-16-nginx-php-7.0                     ubuntu-16-nginx-php-7.0                         4                                       [OK]
pivotaldata/ubuntu16.04-build                             Ubuntu 16.04 image for GPDB compilation         2
pivotaldata/ubuntu-gpdb-dev                               Ubuntu images for GPDB development              1
1and1internet/ubuntu-16-sshd                              ubuntu-16-sshd                                  1                                       [OK]
smartentry/ubuntu                                         ubuntu with smartentry                          1                                       [OK]
1and1internet/ubuntu-16-php-7.1                           ubuntu-16-php-7.1                               1                                       [OK]
pivotaldata/ubuntu16.04-test                              Ubuntu 16.04 image for GPDB testing             0
```

ubuntu를 검색하면 다양한 이미지가 있는 것을 볼 수 있다.

## 나만의 이미지 만들기

```shell
ubuntu@study:~$ docker run -i -t --name commit_test ubuntu:14.04
root@db92d7141b48:/# echo first_test! >> first
root@db92d7141b48:/# exit
exit
ubuntu@study:~$ docker commit -a 'yceffort-test' -m 'my first commit' commit_test commit_test:first
sha256:0ae047cd0bdeacff0145fce31f7abdeef169cfe077db5f95053399e2be8f9497
ubuntu@study:~$
```

이미지 이름을 `commit_test`로, 태그는 `first`로 했다. `-a`는 제작자(author)를 의미한다. 이제 이미지가 생성되었는지 확인해보자.

```shell
ubuntu@study:~$ docker images
REPOSITORY              TAG                 IMAGE ID            CREATED              SIZE
commit_test             first               0ae047cd0bde        About a minute ago   197MB
ubuntu@study:~$
```

이제 같은 방법으로 `commit_test:first`를 활용하여 두번째 이미지를 만들어보자.

```shell
ubuntu@study:~$ docker run -i -t --name commit_test2 commit_test:first
root@42a13487a0bf:/# echo second_test! >> second
root@42a13487a0bf:/# exit
exit
ubuntu@study:~$ docker commit -a 'yceffort' -m 'my second commit' commit_test2 commit_test:second
sha256:c5d7289a7e1eaec8e34050d78e6006c181b0080743126c357308df8664916b3a
```

```shell
ubuntu@study:~$ docker images
REPOSITORY              TAG                 IMAGE ID            CREATED             SIZE
commit_test             second              c5d7289a7e1e        12 seconds ago      197MB
commit_test             first               0ae047cd0bde        2 minutes ago       197MB
```

정상적으로 생성되어 있는 것을 볼 수 있다.

## 도커 이미지 구조

`docker inspect 이미지명` 명령어로 이미지의 구조를 확인해볼 수 있다. 다만 너무 길어져서 layer 부분만 따로 떼어 내본다.

```shell
ubuntu@study:~$ docker inspect ubuntu:14.04
```

```json
[
  "sha256:f2fa9f4cf8fd0a521d40e34492b522cee3f35004047e617c75fadeb8bfd1e6b7",
  "sha256:48dc77435ad5c63ea60d91e6ad4828c70e7e61755f99982b0505abb8aaa00872",
  "sha256:3da511183950aa462f667f43fcda0bb5484c5c73eaa94fcd0a94bbd4db396e1c"
]
```

```shell
ubuntu@study:~$ docker inspect commit_test:first
```

```json
[
  "sha256:f2fa9f4cf8fd0a521d40e34492b522cee3f35004047e617c75fadeb8bfd1e6b7",
  "sha256:48dc77435ad5c63ea60d91e6ad4828c70e7e61755f99982b0505abb8aaa00872",
  "sha256:3da511183950aa462f667f43fcda0bb5484c5c73eaa94fcd0a94bbd4db396e1c",
  "sha256:40a2c0b1240bac592d4874d3b6ba3c29d65dfcb53131b0954d2a4f5f31eba285"
]
```

```shell
ubuntu@study:~$ docker inspect commit_test:second
```

```json
[
  "sha256:f2fa9f4cf8fd0a521d40e34492b522cee3f35004047e617c75fadeb8bfd1e6b7",
  "sha256:48dc77435ad5c63ea60d91e6ad4828c70e7e61755f99982b0505abb8aaa00872",
  "sha256:3da511183950aa462f667f43fcda0bb5484c5c73eaa94fcd0a94bbd4db396e1c",
  "sha256:40a2c0b1240bac592d4874d3b6ba3c29d65dfcb53131b0954d2a4f5f31eba285",
  "sha256:02d3b6e97ca5091aa41a9b8b5b160254831eb0b41c35e14dafbccd5cfee20b0f"
]
```

뭔가 앞에서 부터 레이어가 하나씩 쌓여있는 것을 볼 수 있다. 이로 미루어보았을때, 이미지 커밋을 할때 변경된 사항만 새로운 레이어로 저장하고, 기존에 것은 별도의 레이어로 둔다는 것을 알 수 있다.

삭제를 하기 위해서는 `docker rmi`를 사용하면 된다.

```shell
ubuntu@study:~$ docker rmi commit_test:first
Error response from daemon: conflict: unable to remove repository reference "commit_test:first" (must force) - container 42a13487a0bf is using its referenced image 0ae047cd0bde
```

그러나 해당 이미지를 사용하는 컨테이너가 존재해서 삭제가 안된다. 따라서 컨테이너를 삭제한 후에 이미지를 삭제해야한다.

사실 `commit_test:first`를 삭제했다고 해서, 실제로 해당 이미지의 레이어 파일이 삭제되는 것은 아니다. 왜냐하면 이 이미지를 기반으로 한 `commit_test:second`가 존재하기 때문이다. 따라서 실제 이미지 파일을 삭제하지 않고, 그냥 레이어에 부여한 이름만 삭제한다.

```shell
ubuntu@study:~$ docker rmi commit_test:second
Untagged: commit_test:second
Deleted: sha256:c5d7289a7e1eaec8e34050d78e6006c181b0080743126c357308df8664916b3a
Deleted: sha256:994c3bba470f6e08dbd29fbb0523d05994796aa687c085ddda6a0eebe8e66284
```

`commit_test:second`를 기반으로한 이미지는 없으므로 바로 삭제되는 것을 볼 수 있다.

## 이미지 추출하고 로드하기

```shell
ubuntu@study:~$ docker save -o ubuntu_14_04.tar ubuntu:14.04
```

```shell
ubuntu@study:~$ docker load -i ubuntu_14_04.tar
Loaded image: ubuntu:14.04
```

`save`, `load` 와 비슷한 `import` `export`가 있다. 차이는, `export`는 컨테이너의 파일 시스템을 tar로 추출하지만, 컨테이너 및 이미지에 대한 정보는 저장하지 않는다.
