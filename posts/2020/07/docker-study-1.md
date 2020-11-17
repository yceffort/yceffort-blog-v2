---

title: Docker 공부 (1) - 도커 기초부터 볼륨 공유까지
tags:
  - docker
published: true
date: 2020-07-28 08:25:27
description: "`toc tight: true, from-heading: 2 to-heading: 3 ` ## Docker 는
무엇인가? 리눅스 컨테이너에 여러가지 기능을 추가하여 애플리케이션을 컨테이너로서 좀더 쉽게 사용할 수 있도록 만든 오픈소스. 이에 대해
정리 해 놓은 [좋은 글](https://subicura.com/2017/01/19/docker-g..."
category: docker
slug: /2020/07/docker-study-1/
template: post
---

## Table of Contents

## Docker 는 무엇인가?

리눅스 컨테이너에 여러가지 기능을 추가하여 애플리케이션을 컨테이너로서 좀더 쉽게 사용할 수 있도록 만든 오픈소스. 이에 대해 정리 해 놓은 [좋은 글](https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html)이 있으니 여기를 참고.

> Developing apps today requires so much more than writing code. Multiple languages, frameworks, architectures, and discontinuous interfaces between tools for each lifecycle stage creates enormous complexity. Docker simplifies and accelerates your workflow, while giving developers the freedom to innovate with their choice of tools, application stacks, and deployment environments for each project.

## 설치하는 법

[공식 문서](https://docs.docker.com/engine/install/)를 참고

## 실습 위치

이번에 나온 [Toast Compute Instance](https://console.toast.com/) 를 활용. AWS free tier는 진작에 소진 했고, 돈내고 쓸 수 있는 곳 중에서 가장 싼 인스턴스를 제공하는 서비스는 Toast 였다.

## 컨테이너 생성

```shell
$ docker run -i -t ubuntu:14.04

Unable to find image 'ubuntu:14.04' locally
14.04: Pulling from library/ubuntu
2e6e20c8e2e6: Pull complete
30bb187ac3fc: Pull complete
b7a5bcc4a58a: Pull complete
Digest: sha256:ffc76f71dd8be8c9e222d420dc96901a07b61616689a44c7b3ef6a10b7213de4
Status: Downloaded newer image for ubuntu:14.04

root@ec01158d61da:/# ls

bin  boot  dev  etc  home  lib  lib64  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var
```

명령어 실행과 동시에 컨테이너 생성, 실행, 컨테이너 내부로 진입이 한꺼번에 이뤄짐. 기본 사용자 root에 호스트이름은 무작위 16진수 해쉬값을 가지게 된다.

> docker 명령어에서 shell을 사용하기 위해서는 -i (상호입출력) -t (tty)를 활성화 시켜야 한다.

종료는 `exit`를 하면 된다.

저장소를 단순히 내려 받고 싶을 땐, pull을 사용하면 된다.

```shell
ubuntu@study:~$ docker pull centos:7

7: Pulling from library/centos
524b0c1e57f8: Pull complete
Digest: sha256:e9ce0b76f29f942502facd849f3e468232492b259b9d9f076f71b392293f1582
Status: Downloaded newer image for centos:7
docker.io/library/centos:7
```

`images`로 현재 있는 이미지를 확인할 수 있다.

```shell
ubuntu@study:~$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
centos              7                   b5b4d78bc90c        2 months ago        203MB
ubuntu              14.04               6e4f1fe62ff1        7 months ago        197MB
```

`create`으로 컨테이너를 생성할 수도 있다.

```shell
ubuntu@study:~$ docker create -i -t --name mycentos centos:7
42d20904cc1afff9dc499363f158789fd1860f7ede99dcadfaa8bbc201bbc119
```

`start` 와 `attach`로 컨테이너 시작 및 진입을 할 수 있음.

```shell
ubuntu@study:~$ docker start mycentos
mycentos
ubuntu@study:~$ docker attach mycentos
[root@42d20904cc1a /]# ls
anaconda-post.log  bin  dev  etc  home  lib  lib64  media  mnt  opt  proc  root  run  sbin  srv  sys  tmp  usr  var
```

`ps`로 지금까지 생성한 컨테이너 목록 확인

```shell
ubuntu@study:~$ docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
42d20904cc1a        centos:7            "/bin/bash"         2 minutes ago       Up 9 seconds                            mycentos
```

`ps` 명령어는 정지 되지 않은 컨테이너 목록만 출력. 정지 하지 않고, 단순히 빠져 나오기만 하고 싶다면, `control+p+q`로 나올 수 있다. 모든 컨테이너를 보고 싶다면 `-a`를 붙이면 된다.

```shell
ubuntu@study:~$ docker ps -a
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                      PORTS               NAMES
42d20904cc1a        centos:7            "/bin/bash"         4 minutes ago       Up About a minute                               mycentos
ec01158d61da        ubuntu:14.04        "/bin/bash"         18 minutes ago      Exited (0) 16 minutes ago                       goofy_aryabhata
```

`rm` 명령어를 이용해서 삭제할 수 있다.

```shell
ubuntu@study:~$ docker rm mycentos
Error response from daemon: You cannot remove a running container 42d20904cc1afff9dc499363f158789fd1860f7ede99dcadfaa8bbc201bbc119. Stop the container before attempting removal or force remove
ubuntu@study:~$ docker stop mycentos
mycentos
ubuntu@study:~$ docker ps -a
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                       PORTS               NAMES
42d20904cc1a        centos:7            "/bin/bash"         10 minutes ago      Exited (137) 2 minutes ago                       mycentos
ec01158d61da        ubuntu:14.04        "/bin/bash"         24 minutes ago      Exited (0) 22 minutes ago                        goofy_aryabhata
ubuntu@study:~$ docker rm mycentos
mycentos
ubuntu@study:~$ docker ps -a
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                      PORTS               NAMES
ec01158d61da        ubuntu:14.04        "/bin/bash"         24 minutes ago      Exited (0) 22 minutes ago                       goofy_aryabhata
ubuntu@study:~$
```

## 컨테이너를 외부에 노출 시키기

컨테이너도 가상 머신과 마찬가지로, 가상 IP 주소를 할당 받을 수 있다. 기본적으로 `178.17.0.X`를 순차적으로 할당한다.

```shell
ubuntu@study:~$ docker run -i -t --name network_test ubuntu:14.04
root@b505ea44b935:/# ifconfig
eth0      Link encap:Ethernet  HWaddr 02:42:ac:11:00:02
          inet addr:172.17.0.2  Bcast:172.17.255.255  Mask:255.255.0.0
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:9 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0
          RX bytes:766 (766.0 B)  TX bytes:0 (0.0 B)

lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)

root@b505ea44b935:/#
```

아무런 설정도 하지 않았다면, 외부에서 접근할 수 없으며 도커가 설치된 호스트에서만 접근할 수 있다.

```shell
ubuntu@study:~$ docker run -i -t --name network_test -p 80:80 ubuntu:14.04
```

연결해보면, 아파치 서버가 정상적으로 실행되서 연결된 것을 알 수 있다.

```
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
  <!--
    Modified from the Debian original for Ubuntu
    Last updated: 2014-03-19
    See: https://launchpad.net/bugs/1288690
  -->
  <head>
  ...
```

실제 아파치 서버가 설치 된 것은 컨테이너 내부이므로, 호스트에는 아무런 영향이 없다.

다시 정리하자면, 호스트의 80번 포트를 컨테이너의 80번 포트와 연결했고, 아파치 웹서비스의 80번 포트가 컨테이너의 포트와 연결되어 있는 것이다.

```
80번 호스트 포트 > 80번 컨테이너 포트 > 아파치 웹서비스 80번 포트
```

## 컨테이너 애플리케이션 구축

이번엔 데이터베이서 컨테이너와 웹서버 컨테이너를 각각 별도로 설치하여 연결해보자.

```shell
ubuntu@study:~$ docker run -d --name wordpressdb -e MYSQL_ROOT_PASSWORD=test -e MYSQL_DATABASE=wordpress mysql:5.7
ubuntu@study:~$ docker run -d -e WORDPRESS_DB_PASSWORD=test --name wordpress --link wordpressdb:mysql -p 80 wordpress
ubuntu@study:~$ docker port wordpress
80/tcp -> 0.0.0.0:32768
```

해당 포트로 접근해보면, 워드프레스가 실행되는 것을 볼 수 있다.

여기서 실행할 때 사용한 몇가지 파라미터에 대해 알아보자

- `-d`: detach 모드로 실행한다. docker 내에서 정의한 프로그램을 background에서 실행한다. `-i -t`와는 다르게 입출력이 없는 상태에서 시작하게 된다.
- `-e`: 환경변수를 설정한다.
- `--link`: 내부 IP를 알 필요 없이, 항상 컨테이너에 alias로 접근할 수 있도록 하는 것. 즉 두 번째로 싱행된 웹서버 컨테이너는, wordpressdb의 ip를 몰라도, `mysql`이라는 호스트 이름으로 요청을 전송하면, `wordpressdb` 컨테이너의 내부 IP로 접근할 수 있다. 그러나 해당 명령어는 deprecated 되어 있으며, 도커 브릿지를 사용해서 연결해야 한다.

```shell
ubuntu@study:~$ docker exec wordpress curl mysql:3306 --silent
J
5.7.31
???ziqa7)NDmmysql_native_password!??#08S01Got packets out of order
```

## 도커 볼륨

도커 이미지로 컺테이너를 생성하면, 이미지는 읽기 전용이 되며 컨테이너의 변경사항만 별도로 저장해서 각 컨테이ㅓ의 정보를 보존하게 된다. 예를 들어, mysql 컨테이너는 `mysql:5.7`이라는 이미지로 생성되었지만, 워드프레스 블로그를 위한 데이터베이스 정보는 컨에티너가 가지고 있다. 이미지는 어떠한 경우에서도 변경되지 않으며, 컨테이너 계층에 변경된 정보가 저장된다. 그러나 이러한 구조 때문에 컨테이너를 삭제하게 되면 데이터베이스 정보까지 삭제된다는 단점이 있다.

컨테이너의 persistent 데이터를 활용할 수 있는 방법으로는 볼륨이 있다.

### 1. 호스트와 볼륨을 공유하는 방법

```shell
ubuntu@study:~$ docker run -d --name wordpressdb_hostvolume -e MYSQL_ROOT_PASSWORD=test -e MYSQL_DATABASE=wordpress -v /home/wordpress_db:/var/lib/mysql mysql:5.7
f5dcb2c9ae66b4e20486a63961f607ce7a341a71a8a975d45388aa8de70bd468

ubuntu@study:~$ docker run -d -e WORDPRESS_DB_PASSWORD=test --name wordpress_hostvolume --link wordpressdb_hostvolume:mysql -p 80 wordpress
0fa6a24a817e9267b867c7a7bfe3b9ff4c688d7e66d05c357aba5ba53c4cdd93

ubuntu@study:~$ ls /home/wordpress_db/
auto.cnf    ca.pem           client-key.pem  ibdata1      ib_logfile1  mysql               private_key.pem  server-cert.pem  sys
ca-key.pem  client-cert.pem  ib_buffer_pool  ib_logfile0  ibtmp1       performance_schema  public_key.pem   server-key.pem   wordpress
```

미리 지정해 놓은 `/home/wordpress_db` 디렉토리에 mysql 관련 데이터베이스 파일이 들어 있는 것을 볼 수 있다. 컨테이너의 `/var/lib/mysql`과 호스트의 `/home/wordpress_db`는 완전히 같은 디렉토리다.

### 2. 볼륨 컨테이너

`-v` 옵션으로 볼륨을 사용하는 컨테이너를, 다른 컨테이너와 공유하는 것이다. 컨테이너 생성시 `--volumes-from `을 사용하면 `-v`를 사용한 컨테이너의 볼륨 디렉토리를 공유할 수 있다.

```shell
ubuntu@study:~$ docker run -i -t --name volume_overide -v /home/wordpress_db:/home/testdir_2 alicek106/volume_test
Unable to find image 'alicek106/volume_test:latest' locally
latest: Pulling from alicek106/volume_test
56eb14001ceb: Pull complete
7ff49c327d83: Pull complete
6e532f87f96d: Pull complete
3ce63537e70c: Pull complete
587f7dba3172: Pull complete
Digest: sha256:e0287b5cfd550b270e4243344093994b7b1df07112b4661c1bf324d9ac9c04aa
Status: Downloaded newer image for alicek106/volume_test:latest
root@e3364f2e3f69:/# exit
exit
ubuntu@study:~$ docker run -i -t --name volumes_from_container --volumes-from volume_overide ubuntu:14.04
root@03ae3b0ae3c0:/# ls /home/testdir_2/
auto.cnf    ca.pem           client-key.pem  ib_logfile0  ibdata1  performance_schema  public_key.pem   server-key.pem  wordpress
ca-key.pem  client-cert.pem  ib_buffer_pool  ib_logfile1  mysql    private_key.pem     server-cert.pem  sys
root@03ae3b0ae3c0:/#
```

컨테이너 생서이 `--volumes-from`을 사용하면, `-v`를 적용한 컨테이너의 볼륨디렉토리를 사용할 수 있다. 첫번째 예제에서, `-v`를 사용해서 `volume_test`를 만들었고, 두번째 예제에서 `-volumes-from`을 사용해서 `volume_test`의 볼륨을 사용하고 있는 모습이다. 이러한 옵션을 활용하면, 단순히 볼륨을 공유해주는 역할만 하는 컨테이너를 만들 수도 있다.

### 3. 도커 볼륨

마지막 방법은 docker에서 제공해주는 방법을 사용하는 것이다.

```shell
ubuntu@study:~$ docker volume create --name myvolume
myvolume
ubuntu@study:~$ docker volume ls
DRIVER              VOLUME NAME
local               2832c28b81f0d6dd856b124167df067ae1372fb432fa216d97af07f1544b5fa6
local               b59792aa2847d7a2c8e4b35b665d1591a6cfbf081eefc3024f6cf47e23318ff8
local               ec05ad99c7d764924a96e2abba77d9ac8f44de47ac62dd5027686f64c35d35ab
local               myvolume
```

그리구 위의 볼륨을 활용해 컨테이너를 만들 수 있다.

```shell
ubuntu@study:~$ docker run -i -t --name myvolume_1 -v myvolume:/root/ ubuntu:14.04
```

```shell
ubuntu@study:~$ docker run -i -t --name myvolume_1 -v myvolume:/root/ ubuntu:14.04
root@f351516e97a1:/# echo hell, volume! >> /root/volume
root@f351516e97a1:/# exit
exit
ubuntu@study:~$ docker run -i -t --name myvolume_2 -v myvolume:/root/ ubuntu:14.04
root@e2131dc32f0a:/# cat /root/volume
hell, volume!
```

같은 볼륨을 공유하는 두개의 컨테이너에서, 하나는 파일을 생성하고, 다른 하나에서는 생성한 파일을 읽었는데 모두 정상적으로 작동하는 것을 보아 볼륨을 공유하고 있다는 것을 알 수 있다.

```shell
ubuntu@study:~$ docker inspect --type volume myvolume
[
    {
        "CreatedAt": "2020-07-28T12:58:47+09:00",
        "Driver": "local",
        "Labels": {},
        "Mountpoint": "/var/lib/docker/volumes/myvolume/_data",
        "Name": "myvolume",
        "Options": {},
        "Scope": "local"
    }
]
```

`docker inspect` 를 활용해 볼륨의 정보를 알 수 있다.

`docker volume create`를 사용하지 않아도, `-v`옵션으로 볼륨을 그냥 만들어 버릴 수 있다. `-v \root` 형태로 실행하면, 무작위 형태의 이름을 가진 볼륨을 생성해 자동으로 그것을 사용한다.

```shell
ubuntu@study:~$ docker volume prune
WARNING! This will remove all local volumes not used by at least one container.
Are you sure you want to continue? [y/N] y
Deleted Volumes:
ec05ad99c7d764924a96e2abba77d9ac8f44de47ac62dd5027686f64c35d35ab
b59792aa2847d7a2c8e4b35b665d1591a6cfbf081eefc3024f6cf47e23318ff8
2832c28b81f0d6dd856b124167df067ae1372fb432fa216d97af07f1544b5fa6

Total reclaimed space: 300.5MB
```

마찬가지로 `docker volume prune`을 사용해 볼륨 정보를 모두 날릴 수 있다.

### 결론

컨테이너가 아닌 외부에서 데이터를 저장하고, 컨테이너는 컨테이너 그 데이터 만으로 동작하도록 설계하는 것을 stateless 컨테이너라고 한다. 컨테이너 자체에는 어떠한 정보나 상태가 존재하지 않고, 다만 그 컨테이너에 있는 정보를 기반으로 실행할 뿐이다. 필요한 유동적인 정보는 외부 volume에서 받는다. 반대로 컨테이너 내부에 상태나 정보가 있으면 stateful 컨테이너라고 한다. 그러나 이와 같은 컨테이너는 컨테이너 자체에 정보를 보관하므로, 이러한 설계는 지양하는 것이 좋다.
