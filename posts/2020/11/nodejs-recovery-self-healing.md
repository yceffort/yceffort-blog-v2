---
title: 'Nodejs 서비스 Recovery 전략'
tags:
  - javascript
  - nodejs
published: true
date: 2020-11-20 23:59:25
description: '아 내 서비스는 완벽해서 그런거 필요 없다니까요?'
---

100%의 테스트 리커버리가 도달한 이상적인 세계가 왔다. 오류 처리 또한 완벽했고, 모든 실패는 우아하게 처리되었다. 모든 시스템이 정말로 완벽에 도달했기 때문에, 이런 오류에서의 회복 같은 논의 따위는 필요가 없다.

그러나 나부터 시작해서, 2020년의 지구에는 아직 그런 이상적인 시스템이 있는 곳은 단언코 아무 곳도 없을 것이다. 누군가의 서버는 여전히 프로덕션에서 박살나고 있다.

이 글에서, 서버를 더욱 탄력적으로 만들고 프로세스 관리 능력을 향상시키는 몇가지 개념과 도구를 살펴보자.

## `node index.js`

Nodjs, 특히 서버 관련 작업을 처음 접하는 경우 원격 프로덕션이나, 개발 환경이든간에 동일한 방식으로 앱을 실행한다.

nodejs를 설치하고, repo를 clone하고, `npm install` 이후에 `node index.s`또는 `npm start` 로 시작한다.

이는 모든 프로젝트를 시작하는 완벽한 방법 처럼 보인다. 만약 이게 정말 제대로 작동만 한다면, 우리는 더 이상 수정할 필요가 없다.

그러나 사전에 예상하지 못했던 발생할 수가 있다. vm또는 호스트가 재시작 된다면? 서버 크래쉬가 발생한다면?

복구(Recovery)는 다양한 방법으로 처리할 수 있다. 크래시 이후 서버를 다시 시작하는 편리한 솔루션도 있고, 프로덕션을 크래쉬로 부터 안전하게 만드는 것보다 더 우아한 방법들이 많다.

우리가 실행하려는 환경에 따라서, 복구는 다른 특성을 가지고 있다. 개발환경에서의 목표는 편의성이고 (코드를 수정하면 알아서 재시작 되는 것과같은), 프로덕션의 경우에는 오류로부터의 탄력성이라 볼 수 있다.

## 문제가 발생하자 마자 해결하기

개발 환경에서 nodejs 서버를 작성하고 있다고 상상해보자. 몇줄의 코드를 고칠 때마다 탭을 전환하여 `node index`나 `npm start`로 프로세스를 실행한다. 이는 몇번 반복하게 되면 끔찍할 정도로 지루해진다.

코드를 변경한 뒤에 자체적으로 그냥 재시작하면 좋지 않을까?

여기에서 도움이 되는 것이 바로 `nodemon`이다.

```bash
nodemon index.js
```

혹은 `Supervisor`를 동일한 방식으로 사용할 수도 있다.

```bash
supervisor index.js
```

두 서비스 보두 인기만큼이나 유용하다. 두 개의 차이점은, `nodemon`은 코드(파일)이 변경되면 재시작 되는 반면, `supervisor`는 에러가 발생할 때 다시 시작한다는 것이다.

개발 환경에서의 문제는 해결이 쉽다. 그러나 프로덕션 환경의 문제는 다르다. 프로덕션 서버에 내보내기 시작하면 대학을 보낸 부모마냥 안절부절해진다. 여기에서 사용하는 것이 프로세스 관리자다.

## 프로세스 관리

앱을 실행하게 되면, 프로세스가 생성된다.

개발환경에서는, 터미널 윈도우를 열고 코맨드를 실행한다. foreground proccess가 만들어 진것이며, 앱이 실행된다.

그러나 터미널 윈도우를 닫는다면, 앱 또한 종료된다. 또한 터미널 윈도우는 더 이상 추가적인 작업을 할 수 없는 상태가 된다. `Ctrl+c`로 끄지 않는 이상, 터미널은 사용할 수 없게 된다.앱 실행이 터미널 윈도우와 강하게 연결되어 있기 때문에, 로그와 에러 또한 볼 수 있다. 그러나 프로덕션 서버에서는 앱을 백그라운드에서 실행해야 하기 때문에 이러한 장점을 잃게 된다.

이 때 등장하는 것이 프로세스 관리자다.

### PM2

nodejs에서 가장 널리 쓰이는 프로세스 관리자는 [PM2](https://github.com/Unitech/pm2)다. `PM2`를 설치하고 아래 명령어로 실행하면 된다.

```bash
pm2 start index.js
```

```bash
===============================================================================
--- PM2 development mode ------------------------------------------------------
Apps started         : src
Processes started    : 1
Watch and Restart    : Enabled
Ignored folder       : node_modules
===============================================================================
```

실행하게 되면, 무언가 다른점을 눈치챌 수 있을 것이다. 마치 아무것도 일어나지 않는 것처럼 보이지만, 앱의 엔드포인트로 가면 앱이 실행중인 것을 알 수 있다. `PM2`는 앞서 언급된 것처럼, 앱을 백그라운드에서 실행하게 해준다. 또한 `--watch` 명령어로 pm2가 파일을 감시하고 재시작하는 것을 볼 수도 있다.

```bash
pm2 start index.js --watch
```

그럼에도 여전히 가시성이 조금은 부족하다. 서버로그를 보기 위해서는, 아래와 같은 명령어를 쓰면 된다.

https://blog.appsignal.com/2020/09/09/nodejs-resiliency-concepts-recovery-and-self-healing.html

| command           | Description                                                                                                           |
| ----------------- | --------------------------------------------------------------------------------------------------------------------- |
| `pm2 list`        | 앱의 목록을 보여준다. pm2에서 관리하고 있는 애플리케이션의 ID를 볼 수 있다. 이 아이디로 아래 명령어를 실행할 수 있다. |
| `pm2 logs <id>`   | 앱의 로그를 확인한다.                                                                                                 |
| `pm2 stop <id>`   | 프로세스를 중단한다. 단순히 중단만 되는 것이므로, 프로세스까지 제거하기 위해서는 `delete` 를 사용해야 한다.           |
| `pm2 delete <id>` | 프로세스를 지운다. 지우게 되면 `stop` 도 이루어진다.                                                                  |

`pm2` 는 설정하기 용이하며, 로드밸런싱도 수행할 수 있고, hot reload도 가능하다.

`pm2`는 놀랍도록 편리하지만, 몇가지더 살펴볼 것이 존재한다.

## Systemd

만약 Linux VM에서 앱을 실행하려고 계획중이라면, 컨테이너와 오케스트레이터 개념을 깊게 들어가기이전에 `systemd`를 언급할 필요가 있다. (그러나 Azure, AWS Lambda, GCP App Engine 등에서 실행하려고 준비중이라면 그다지 필요가 없다.)

`systemd`는 프로세스를 시작, 중지, 재시작을 할수가 있다. VM이 재시작 된다면, `systemd`는 앱이 다시 시작하게끔 도와준다.

`systemd`는 다음 시스템에서 사용가능하다.

- Ubuntu Xenial 또는 그 이상
- Centos 7 / RHGEL 7
- Debian Jessie 또는 그 이상
- Fedora 15 또는 그 이상

추가로 확인해야 할 것은 해당 유저가 `sudo`권한이 있어야 한다는 것이다.

이제부터 예제는 `Ubuntu`를 사용하고 있고, 홈 디렉토리는 `/home/user/`라고 가정하며,`index.js`는 이 홈 디렉토리에 있다고 가정한다.

### systemd 서비스 파일

`systemd` 파일은 서비스에 대한 설정을 보관할 수 있는 시스템영역을 만드는데 도움을 주는 파일이다. 먼저 한번 설정해보자.

`systemd` 파일은 아래 디렉토리에 존재한다.

```bash
/lib/systemd/system
```

```bash
cd /lib/systemd/system
```

```bash
sudo nano myapp.service
```

그리고 이 파일을 만들어보자.

```bash
# /lib/systemd/system/myapp.service

[Unit]
Description=My awesome server
Documentation=https://awesomeserver.com
After=network.target

[Service]
Environment=NODE_PORT=3000
Environment=NODE_ENV=production
Type=simple
User=user
ExecStart=/usr/bin/node /home/user/index.js
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

몇가지 설정들은 꽤 분명하게 되어 있지만, `After` `Type`에 대해서만 알아보자.

`After=network.target`은 포트가 필요하기 때문에, 서버의 네트워킹이 가동되고 실행될 때 까지 기다려야 한다는 것을 의미한다. `Type`은 단순히 미친짓 하지말고 실행하라는 것을 의미한다.

### systemctl로 앱 실행하기

파일을 생성했으니, `systemd`에게 새롭게 변경된 파일을 이용하라고 알려줄 때다. 이 파일에 변화가 있으면, 아래 명령어를 계속 실행해야 한다.

```bash
sudo systemctl daemon-reload
```

이제 `systemctl` 명령을 사용하여 서비스를 시작하고 중지할 수 있어야 한다.

```bash
sudo systemctl start myapp
```

멈추고 싶다면 `stop`을, 재시작하고 싶다면 `restart`를 쓰면 된다.

이제 제일 중요한 부분이다. VM이 실행될 때 애플리케이션이 자동으로 시작되게 하고 싶다면, 아래 명령어를 입력하면 된다.

```bash
sudo systemctl enable myapp
```

작동중지는 `disable`을 쓰면 된다.

이게 전부다. 이제 Node.js가 아닌 프로세스를 관리를 하는 다른 시스템이 생겼다.

하지만 아마도 🤔 요즘 이렇게 리눅스 VM에 직접 서비스를 올려서 굴리는 시스템은 별로 없을 것이다......... 컨테이너로 넘어갈 차례다.

## 컨테이너란 무엇인가

Mesos, CoreOS, LXC, OpenVZ 등 다양한 컨테이너 런타임 환경이 존재하지만, 아마도 가장 유명한 것, 그리고 컨테이너의 진정한 동의어로 취급받는 것은 Docker다. 현재 컨테이너로 굴러가는 시스템의 80%가 Docker로 이루어져 있고, 사람들이 컨테이너를 이야기 하면 아마두 열에 아홉은 도커를 이야기 하고 있다고 봐도 무방하다.

컨테이너가 하는 일은 정확히 무엇인가? 컨테이넌 말그대로 컨테이너다. 그리고 이 컨테이너에는 무엇을 보관하고 있을까?

컨테이너에는 응용프로그램과 그에 필요한 모든 종속성이 포함되어 있다. 단순히 그것 뿐이다. 그렇다면 Nodejs 서버가 가지고 있어야 할 것이 있는지 생각해보자. Nodejs, index.js파일, 그리도 npm 패키지 종속성들이 필요할 것이다. 따라서 컨테이너를 만든다면, 이것이 존재하며 제대로 담겨있는지 확인하고 싶을 것이다. 그리고 컨테이너가 준비되어 있다면, 컨테이너를 컨테이너 엔진(도커)를 이용해 실행 시킬 수 있다.

### Container vs VM

Docker의 마법은 동일한 기반의 물리적 시스템과 운영체제를 사용하여 서로 충돌하지 않고, 다양한 형태의, 다양한 애플리케이션을 원활하게 실행할 수 있다는 것이다.

### 도커 컨테이너 만들기

도커 컨테이너를 만드는 것은 정말로 쉽다. 로컬 머신에 Docker를 먼저 설치하면 된다. Docker는 어떤 운영체제에서든 설치가 가능하지만, 프로덕션 머신에는 Linux 환경에서 설치하기를 권한다.

도커는 `Dockerfile`이라고 불리우는 파일을 참조하며, 도커 이미지라고 불리우는 컨테이너의 레시피를 만드는데 사용할 것이다. 따라서 컨테이너를 만들기 전에 이 파일을 생성해야 한다. `index.js`와 동일한 위치에 생성하면 된다.

```Dockerfile
# Dockerfile

# Base image (we need Node)
FROM node:12

# Work directory
WORKDIR /usr/myapp

# Install dependencies
COPY ./package*.json ./

RUN npm install

# Copy app source code
COPY ./ ./

# Set environment variables you need (if you need any)
ENV NODE_ENV='production'
ENV PORT=3000

# Expose the port 3000 on the container so we can access it
EXPOSE 3000

# Specify your start command, divided by commas
CMD [ "node", "index.js" ]
```

`.dockerignore`를 사용하면 `node_modules`와 같이 복사를 원치 않는 파일목록을 명시할 수 있다. `.gitignore`와 정확히 똑같이 동작한다.

```Dockerfile
# .dockerignore

node_modules
npm-debug.log
```

이제 설정이 완료되었고, Docker Image를 만들 차례다.

이미지는 컨테이너를 만드는 일종의 레시피다. 또는 소프트웨어를 인스톨을 하기 위한 플로피 디스크/씨디롬과 같은 존재로도 볼 수 있다. 이는 실제 작동하는 소프트웨어가 아니지만, 패키징된 소프트웨어 데이터를 포함하고 있다고 보면 된다.

```bash
docker build -t myapp .
```

이미지가 준비되어있다면, 이미지 목록에서 확인할 수 있다.

```bash
docker image ls
```

그리고 실행은 아래와 같이 하면 된다.

```bash
docker run -p 3000:3000 myapp
```

컨테이너에서 시작하는 서버를 볼 수 있고, 그 과정에서 로그도 확인할 수 있다. 백그라운드에서 실행시키기 위해서는 `-d` 플래그를 사용하면 된다. 또한 컨테이너를 백그라운드에서 실행중인 경우, 아래 명령어를 활용하여 컨테이너 목록을 확인할 수 있다.

```bash
docker container ls
```

이제 컨테이너에 대한 이해는 마쳤으므로, 원래 글의 주제인 복구와 매우 밀접한 오케스트레이션에 대해 알아보자.

## Orchestration

이상적으로 생각해봤을때, 레고블록과 같은 인프라는 개별적으로 관리하는 것은 굉장히 손이 많이 가기 때문에 어렵다. 앞서 이야기한 프로세스 관리자 처럼, 다른 존재가 이러한 관리를 대신 해주는 것이 좋을 것이다. 여기에서 오케스트레이터가 활약한다.

오케스트레이터는 컨에테이너를 관리하고 스케줄링 할 수 있도록 도와주며, 여러 위치에 분산되어 흩어져있는 VM (컨테이너 호스트)에 걸쳐서 이러한 작업을 할 수 있도록 해준다.

여기에서 우리가 특히 관심을 가져야 하는 것은 `Replication`이다.

### Replication과 높은 가용성

크래시가 발생했을 때 서버가 재기동하는 것은 좋은일이다. 그러나 재기동 중에는 어떤 일이 생기는가? 사용자들이 서비스가 다시 시작 되길 기다려야 하는가? 우리의 목표는 서비스를 고 가용성으로 만드는 것인데, 이는 사용자들이 서버에서 크래시가 있어도 앱을 사용할 수 있어야 함을 의미한다. 어떻게 하면 가능할까?

답은 간단하다. 서버의 복사본을 만들어두고, 동시에 실행하는 것이다.

처음에 이를 설정하는 것은 골치 아프지만, 다행히도 이 매커니즘을 가능하게 하는 모든 것을 가지고 있다. 일단 앱이 컨테이너화가 되어 있으면, 원하는 만큼의 복사본을 실행할 수 있는데 이를 Replica라고 한다.

이제 컨테이너 오케스트레이션 엔진을 활용하여, 어떻게 이를 설정할지를 알아보자. 여러가지 방법이 있지만, 도커 오케스트레이션 엔진과 가장 쉽게 할 수 있는 방법은 Docker Swarm이다.

### swarm 내의 replication

머신에 Docker가 설치 되어있다면, docker swarm은 비교적 간단하게 시작할 수 있다.

```bash
docker swarm init
```

이 명령어로 Docker swarm을 사용할 수 있으며, 다른 VM을 스웜에 연결하여 분산 클러스터를 구성할 수 있다. 이번 예제에서는, 단순히 하나의 머신만 사용할 것이다.

스웜이 활성화 되어 있으면, 서비스라 불리우는 컴포넌트에 접근할 수 있게 된다. 이는 일종의 마이크로 서비스 아키텍쳐의 빵과 버터인데, replica를 만들기 쉽게 해준다.

이제 서비스를 만들어 보자.

```bash
docker service create --name myawesomeservice --replicas 3 myapp
```

위 명령어는 `myawesomeservice`라 불리우는 서비스를 만들어주며, `myapp`이라고 하는 이미지를 활용하여 3개의 동일한 컨테이너를 만들게 된다.

```bash
docker service ls
```

이제 서버가 복제되어 실행되며, 컨테이너가 크래쉬가 되도 재시작 되며, 프로세스 전체에 결쳐 온전한 컨테이너 엑세스를 제공할 수 있다. 서비스 replica의 수를 조정하기 위해서는 아래 명령어를 활용하면 된다.

```bash
docker service scale <name_of_service>=<number_of_replicas>
```

```bash
docker service scale myapp=5
```

이제 원하는 만큼 복제본을 만들어 실행시킬 수 있다.

### 쿠버네틱스에서의 replication

오케스트레이션을 논하는데 쿠버네틱스를 빼먹으면 섭섭하다. 컨테이너하면 도커 듯이, 오케스트레이션하면 쿠버네틱스다.

개인적인 의견으로는, 도커의 스웜보다 쿠버네틱스가 더 학습하기 어렵다. 따라서 이제 막 컨테이너를 시작했다면, 도커 스웜을 먼저 해보는 것이 좋다. 그렇긴하더라도, 쿠버네틱스 세계가 어떻게 작동하는지 이해하는 것도 나쁘지 않다.

- https://minikube.sigs.k8s.io/docs/start/
- https://labs.play-with-k8s.com/

이 예제에서는 두개의 yaml파일을 만들어 설정을 진행한다. 하나는 A 클러스터 IP 인데, 이는 앱과 통신할 수 있는 포트를 연다. 또다른 하나는 도커 스웜과 같은 서비스이다.

`cluster-ip.yml`

```yaml
# cluster-ip.yml

apiVersion: v1
kind: Service
metadata:
  name: cluster-ip-service
spec:
  type: ClusterIP
  selector:
    component: server
  ports:
    - port: 3000
      targetPort: 3000
```

`development.yml`

```yaml
# deployment.yml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: server-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      component: server
  template:
    metadata:
      labels:
        component: server
    spec:
      containers:
        - name: server
          image: your_docker_user/your_image
          ports:
            - containerPort: 3000
```

`your_docker_user/your_image`를 실제 도커 유저와 이미지로 교체하고, 도커 레포에 해당 이미지가 호스팅 되고 있는지 확인해야 한다.

이제 아래 명령어를 실행해보자.

```bash
kubectl apply -f .
```

이제 서비스가 실행되고 있는지, 아래 명령어로 확인해볼 수 있다.

```bash
kubectl get deployments
kubectl get services
```

모든 것이 정상적으로 되고 있다면, `cluster-ip-service`에서 보여주는 IP와 포트를 복사하여 브라우저에 붙여넣기 한다면, 애플리케이션에 접속할 수 있을 것이다.

생성된 복제본을 보고 싶다면, 아래 명령어를 활용하면 된다.

```bash
kubectl get pods
```

나열된 pod는 `deployment.yaml`에 지정한 replica의 수와 일치해야 한다. 모든 컴포넌트를 정리하기 위해서는, 아래 명령어를 사용하면 된다.

```bash
kubectl delete -f .
```

## 결론

이제 우리는 고 가용성의 복구 가능한 애플리케이션을 가지고 있다. 물론, 이게 다가 아니다.

실제로 애플리케이션이 고장이 나지 않는데, 어떤 문제가 있는지 어떻게 알까?

로그를 확인해야할까? 만약 엔드포인트를 확인할 때마다 앱이 작동한다면, 아마도 일년에 한두번 정도만 로그를 볼 것이다. 따라서 앱이 개선되고 있는지 확인하기 위해서는 모니터링, 오류처리 및 오류 전파에 대해서 생각해볼 필요가 있다. 문제가 발생할 때마다 인지하고, 서버를 다운시키지 않더라도 문제를 해결할 수 있도록 해봐야 한다.

출처: https://blog.appsignal.com/2020/09/09/nodejs-resiliency-concepts-recovery-and-self-healing.html
