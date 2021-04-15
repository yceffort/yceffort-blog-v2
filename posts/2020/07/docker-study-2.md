---
title: Docker 공부 (2) - 도커 네트워크
tags:
  - docker
published: true
date: 2020-07-29 05:17:31
description:
  '## 도커 네트워크 도커는 컨테이너에 내부 IP를 순차적으로 할당하며, 이 IP는 컨테이너가 재시작 될 때 마다
  변경된다. 이 내부 IP는 내부망에서만 쓸 수 있으므로 외부와 연결될 필요가 있는데, 이 과정은 컨테이너가 시작할 때마다 호스트에
  `veth` 라는 네트워크 인터페이스를 생성하면서 이루어진다. 이 `veth`인터페이스는 직접 생성하는게 아니라,...'
category: docker
slug: /2020/07/docker-study-2/
template: post
---

## 도커 네트워크

도커는 컨테이너에 내부 IP를 순차적으로 할당하며, 이 IP는 컨테이너가 재시작 될 때 마다 변경된다. 이 내부 IP는 내부망에서만 쓸 수 있으므로 외부와 연결될 필요가 있는데, 이 과정은 컨테이너가 시작할 때마다 호스트에 `veth` 라는 네트워크 인터페이스를 생성하면서 이루어진다. 이 `veth`인터페이스는 직접 생성하는게 아니라, 컨테이너가 생성될 때 도커 엔진이 자동으로 생성한다.

도커가 설치된 호스트에서 `ifconfig` 나 `ip addr`과 같은 명령어로 네트워크를 확인해보자.

```
ubuntu@study:~$ ifconfig
docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255
        inet6 fe80::42:a7ff:fec2:fa90  prefixlen 64  scopeid 0x20<link>
        ether 02:42:a7:c2:fa:90  txqueuelen 0  (Ethernet)
        RX packets 0  bytes 0 (0.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 9  bytes 806 (806.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1454
        inet 192.168.0.4  netmask 255.255.255.0  broadcast 192.168.0.255
        inet6 fe80::f816:3eff:fe64:daf3  prefixlen 64  scopeid 0x20<link>
        ether fa:16:3e:64:da:f3  txqueuelen 1000  (Ethernet)
        RX packets 4889  bytes 29037050 (29.0 MB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 4182  bytes 383098 (383.0 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu 65536
        inet 127.0.0.1  netmask 255.0.0.0
        inet6 ::1  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen 1000  (Local Loopback)
        RX packets 44  bytes 4228 (4.2 KB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 44  bytes 4228 (4.2 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

veth4e8339b: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet6 fe80::30e5:e9ff:fedf:15f3  prefixlen 64  scopeid 0x20<link>
        ether 32:e5:e9:df:15:f3  txqueuelen 0  (Ethernet)
        RX packets 0  bytes 0 (0.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 9  bytes 766 (766.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0

veth7dda7a9: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet6 fe80::4841:73ff:fe01:d09d  prefixlen 64  scopeid 0x20<link>
        ether 4a:41:73:01:d0:9d  txqueuelen 0  (Ethernet)
        RX packets 0  bytes 0 (0.0 B)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 5  bytes 426 (426.0 B)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```

`eth0`은 공인IP 또는 내부IP가 할당되어, 실제로 외부와 통신할 수 있는 호스트의 네트워크 인터페이스다. `veth...`는 컨테이너를 시작할때 생성되었으며, 이는 각 컨테이너의 `eth0`과 연결되어 있다.

그리고 `docker0` 이라고 하는 브릿지도 존재하는데, 이는 각 `veth`와 바인딩 되어 연결하는 역할을 해준다.

![](https://blog.daocloud.io/wp-content/uploads/2015/01/17.jpg)

기본적으로 `docker0` 브릿지를 통해 외부와 연결 할 수 있지만, 다른 네트워크 드라이버를 사용할 수 있다. `bridge` `host` `none` `container` `overlay` 등등이 있다.

```shell
ubuntu@study:~$ docker network ls
NETWORK ID          NAME                DRIVER              SCOPE
eeff6a1308cc        bridge              bridge              local
da0bbc153406        host                host                local
33a0f0fd85e2        none                null                local
```

### bridge

컨테이너를 생성할 때 자동으로 여결되는 `docker0` 브릿지를 활용하도록 설정되어 있다. 이 네트워크는 `172.17.0.x`를 순차적으로 할당한다.

```shell
ubuntu@study:~$ docker network inspect bridge
[
    {
        "Name": "bridge",
        "Id": "eeff6a1308cca652363d7e236f95c2ec97b852fe1a9ed92a1658ce248d21243d",
        "Created": "2020-07-28T17:21:59.821050359+09:00",
        "Scope": "local",
        "Driver": "bridge",
        "EnableIPv6": false,
        "IPAM": {
            "Driver": "default",
            "Options": null,
            "Config": [
                {
                    "Subnet": "172.17.0.0/16",
                    "Gateway": "172.17.0.1"
                }
            ]
        },
        "Internal": false,
        "Attachable": false,
        "Ingress": false,
        "ConfigFrom": {
            "Network": ""
        },
        "ConfigOnly": false,
        "Containers": {
            "1369cc681f26f07679ceb59b975a8f5e9fe02a65f26fc7a54701ca50ad8b8861": {
                "Name": "wordpress",
                "EndpointID": "4ce24808e36381f409853a39457db1940c80dd16c7905d515106b1ad98325603",
                "MacAddress": "02:42:ac:11:00:03",
                "IPv4Address": "172.17.0.3/16",
                "IPv6Address": ""
            },
            "886832f53107fb732f9bb4fc15818fe9cd80fa536230987078fe58f69135fedb": {
                "Name": "wordpressdb",
                "EndpointID": "9ecad5d42dd536cdeb7dedeeaac0e3151a667973804d2518e69c107bc554f224",
                "MacAddress": "02:42:ac:11:00:02",
                "IPv4Address": "172.17.0.2/16",
                "IPv6Address": ""
            }
        },
        "Options": {
            "com.docker.network.bridge.default_bridge": "true",
            "com.docker.network.bridge.enable_icc": "true",
            "com.docker.network.bridge.enable_ip_masquerade": "true",
            "com.docker.network.bridge.host_binding_ipv4": "0.0.0.0",
            "com.docker.network.bridge.name": "docker0",
            "com.docker.network.driver.mtu": "1500"
        },
        "Labels": {}
    }
]
```

### 브릿지 네트워크

`docker0`과 비슷하게, 브릿지 네트워크는 사용자 정의 브릿지를 새로 생성해 각 네트워크에 연결하는 네트워크 구조다.

```shell
ubuntu@study:~$ docker network create --driver bridge mybridge
c3235dd3f5cf8269822e66c435f422850e287c52ed14138a8da33eabb6e93aab
ubuntu@study:~$ docker run -i -t --name mynetwork_container --net mybridge ubuntu:14.04
root@873f7231c461:/# ifconfig
eth0      Link encap:Ethernet  HWaddr 02:42:ac:12:00:02
          inet addr:172.18.0.2  Bcast:172.18.255.255  Mask:255.255.0.0
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:12 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0
          RX bytes:1032 (1.0 KB)  TX bytes:0 (0.0 B)

lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)

root@873f7231c461:/#
```

`mybridge`라는 새로운 네트워크를 생성하고, 그 네트워크를 활용해서 연결했다. 그리고 내부 IP가 `172.18.x.x`로 시작하는 것을 볼 수 있다. 그리고 이러한 네트워크는 수동으로 연결하고 끊을 수 있다.

```shell
ubuntu@study:~$ docker network disconnect mybridge mynetwork_container
ubuntu@study:~$ docker network connect mybridge mynetwork_container
```

### 호스트 네트워크

네트워크를 호스트로 설정하면 , 호스트의 네트워크 환경을 그대로 사용하게 된다. 별도 설정필요 없이 `host`를 사용하면 된다.

```shell
ubuntu@study:~$ docker run -i -t --name network_host --net host ubuntu:14.04
root@study:/# ifconfig
br-c3235dd3f5cf Link encap:Ethernet  HWaddr 02:42:1b:92:84:80
          inet addr:172.18.0.1  Bcast:172.18.255.255  Mask:255.255.0.0
          inet6 addr: fe80::42:1bff:fe92:8480/64 Scope:Link
          UP BROADCAST MULTICAST  MTU:1500  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:5 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0
          RX bytes:0 (0.0 B)  TX bytes:446 (446.0 B)

docker0   Link encap:Ethernet  HWaddr 02:42:a7:c2:fa:90
          inet addr:172.17.0.1  Bcast:172.17.255.255  Mask:255.255.0.0
          inet6 addr: fe80::42:a7ff:fec2:fa90/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:1 errors:0 dropped:0 overruns:0 frame:0
          TX packets:9 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0
          RX bytes:28 (28.0 B)  TX bytes:806 (806.0 B)

eth0      Link encap:Ethernet  HWaddr fa:16:3e:64:da:f3
          inet addr:192.168.0.4  Bcast:192.168.0.255  Mask:255.255.255.0
          inet6 addr: fe80::f816:3eff:fe64:daf3/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1454  Metric:1
          RX packets:6303 errors:0 dropped:0 overruns:0 frame:0
          TX packets:5054 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:29145891 (29.1 MB)  TX bytes:675899 (675.8 KB)

lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:44 errors:0 dropped:0 overruns:0 frame:0
          TX packets:44 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:4228 (4.2 KB)  TX bytes:4228 (4.2 KB)

veth4e8339b Link encap:Ethernet  HWaddr 32:e5:e9:df:15:f3
          inet6 addr: fe80::30e5:e9ff:fedf:15f3/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:8 errors:0 dropped:0 overruns:0 frame:0
          TX packets:26 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0
          RX bytes:600 (600.0 B)  TX bytes:2084 (2.0 KB)

veth7dda7a9 Link encap:Ethernet  HWaddr 4a:41:73:01:d0:9d
          inet6 addr: fe80::4841:73ff:fe01:d09d/64 Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:10 errors:0 dropped:0 overruns:0 frame:0
          TX packets:22 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0
          RX bytes:828 (828.0 B)  TX bytes:1676 (1.6 KB)
```

ifconfig 의 결과가 실제 호스트에서 때린 결과와 비슷한 것을 볼 수 있다.

컨테이너의 네트워크를 호스트모드로 설정하면, 컨테이너 내부의 애플리케이션을 별도로 포트포워딩 하지 않아도 바로 서비스 할 수 있다.

### none

말 그대로 네트워크를 사용하지 않는 것이다.

```shell
ubuntu@study:~$ docker run -i -t --name network_none --net none ubuntu:14.04
root@ab2595314554:/# ifconfig
lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)
```

로컬호스트외에 어떠한 네트워크도 없음을 알 수 있다.

### Container network

--net 옵션으로 컨테이너를 입력하면, 다른 컨테이너의 네트워크 네임스페이스 환경을 공유할 수 있다. 여기서 공유되는 것은 다음과 같다.

- 내부IP
- 네트워크의 맥 주소

```shell
ubuntu@study:~$ docker run -i -t -d --name network_container_1 ubuntu:14.04
e5b3da0af26e970b446d5bdf3e24fa3121a5e2f02615f77e74667783c2709b37
ubuntu@study:~$ docker run -i -t -d --name network_container_2 --net container:network_container_1 ubuntu:14.04
5a54709d03f7e9374e842896b1ea0a57bac4127fd5f836171310dd3f79ad934d
ubuntu@study:~$ docker exec network_container_1 ifconfig
eth0      Link encap:Ethernet  HWaddr 02:42:ac:11:00:02
          inet addr:172.17.0.2  Bcast:172.17.255.255  Mask:255.255.0.0
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:10 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0
          RX bytes:836 (836.0 B)  TX bytes:0 (0.0 B)

lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)

ubuntu@study:~$ docker exec network_container_2 ifconfig
eth0      Link encap:Ethernet  HWaddr 02:42:ac:11:00:02
          inet addr:172.17.0.2  Bcast:172.17.255.255  Mask:255.255.0.0
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:10 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:0
          RX bytes:836 (836.0 B)  TX bytes:0 (0.0 B)

lo        Link encap:Local Loopback
          inet addr:127.0.0.1  Mask:255.0.0.0
          UP LOOPBACK RUNNING  MTU:65536  Metric:1
          RX packets:0 errors:0 dropped:0 overruns:0 frame:0
          TX packets:0 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000
          RX bytes:0 (0.0 B)  TX bytes:0 (0.0 B)
```

`inet addr:172.17.0.2` 과 `HWaddr 02:42:ac:11:00:02`가 두개다 동일한 것을 알 수 있다.

즉, 두 컨테이너가 같은 `eth0`으로 네트워킹 하는 것이다.

```shell
ubuntu@study:~$ docker run -i -t -d --name network_alias_container1 --net mybridge --net-alias yceffort ubuntu:14.04
da00dbc42f111940c3d3aa1909f6a0cdc61bf5eb15133c75195945925af7de00
ubuntu@study:~$ docker run -i -t -d --name network_alias_container2 --net mybridge --net-alias yceffort ubuntu:14.04
d5478e84304af611f5d74ff98432cf843f0c451c42e6d8740e8b0b319c8a2262
ubuntu@study:~$ docker run -i -t -d --name network_alias_container3 --net mybridge --net-alias yceffort ubuntu:14.04
e7f3cc92dfb2f63cc5cd95baa200b5b6690daa400cf4c4a891b351bc5e63f15a
ubuntu@study:~$ docker inspect network_alias_container1 | grep IPAddress
            "SecondaryIPAddresses": null,
            "IPAddress": "",
                    "IPAddress": "172.18.0.2",
```

그리고 핑을 한번 날려보자.

```shell
ubuntu@study:~$ docker run -i -t --name network_alias_ping --net mybridge ubuntu:14.04
root@a02263c69cd1:/# ping -c 1 yceffort

--- yceffort ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 0.108/0.108/0.108/0.000 ms
root@a02263c69cd1:/# ping -c 1 yceffort
PING yceffort (172.18.0.4) 56(84) bytes of data.
64 bytes from network_alias_container3.mybridge (172.18.0.4): icmp_seq=1 ttl=64 time=0.062 ms

--- yceffort ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 0.064/0.064/0.064/0.000 ms
root@a02263c69cd1:/# ping -c 1 yceffort
PING yceffort (172.18.0.2) 56(84) bytes of data.
64 bytes from network_alias_container1.mybridge (172.18.0.2): icmp_seq=1 ttl=64 time=0.048 ms

--- yceffort ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 0.135/0.135/0.135/0.000 ms
root@a02263c69cd1:/# ping -c 1 yceffort
PING yceffort (172.18.0.3) 56(84) bytes of data.
64 bytes from network_alias_container2.mybridge (172.18.0.3): icmp_seq=1 ttl=64 time=0.053 ms
```

각 세개의 컨테이너로 ping이 전송되는 것을 알 수 있다. 라운드 로빈 방식으로 핑이 전송된다. 이는 도커 엔진에 내장된 DNS가 yceffort라는 호스트 이름을 --net-alias 옵션으로 yceffort를 설정한 컨테이너로 변환하기 때문이다.

![출처: https://jungwoon.github.io/docker/2019/01/13/Docker-4/](https://cdn-images-1.medium.com/max/2400/1*5Ts6bzLOp07PO08BYO4VVQ.png)

```shell
root@a02263c69cd1:/# dig yceffort

; <<>> DiG 9.9.5-3ubuntu0.19-Ubuntu <<>> yceffort
;; global options: +cmd
;; Got answer:
;; ->>HEADER<<- opcode: QUERY, status: NOERROR, id: 17204
;; flags: qr rd ra; QUERY: 1, ANSWER: 3, AUTHORITY: 0, ADDITIONAL: 0

;; QUESTION SECTION:
;yceffort.                      IN      A

;; ANSWER SECTION:
yceffort.               600     IN      A       172.18.0.2
yceffort.               600     IN      A       172.18.0.4
yceffort.               600     IN      A       172.18.0.3

;; Query time: 4 msec
;; SERVER: 127.0.0.11#53(127.0.0.11)
;; WHEN: Tue Jul 28 09:15:36 UTC 2020
;; MSG SIZE  rcvd: 98

root@a02263c69cd1:/#
```
