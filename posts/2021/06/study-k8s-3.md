---
title: 'K8s 공부 (3)'
tags:
  - K8s
published: true
date: 2021-06-12 22:32:44
description: '무지성에서 시작하는 K8s 공부해보기 시리즈(3)'
---

[K8s 공부 (2)](/2021/06/study-k8s-2)에서 이어집니다.

## K8s yaml 설정파일 알아보기

`nginx-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
        - name: nginx
          image: nginx:1.16
          ports:
            - containerPort: 8080
```

`nginx-service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
```

K8s의 설정파일은 모두 3가지 파트로 이루어져 있음.

1. metadata
2. specification
3. status: 이는 자동으로 K8s에서 자동으로 생성되어서 붙게됨. K8s는 항상 spec에 적혀있는 내용과 현재 상태를 비교함. 이 두 상태가 일치하지 않다면, 고쳐야 할 것이 있다는 것으로 인식. (=self-healing) 이런한 상태를 가져오는 것이 `etcd`이다. `etcd`는 언제나 모든 컴포넌트의 상태를 계속해서 가지고 있음.

`yaml`형태로 이루어져있기 때문에, indent에 주의해야함.

이 설정파일은 실제 애플리케이션 코드와 함께 있거나 혹은 자체 레파지토리에서 보관하는 것이 좋음.

`spec`의 하위에 `template`이 있는데, 여기에도 동일하게 `metadata`와 `spec`이 있음. (configuration 내부의 configuration) 이 `template`이 pod에 적용되는 설정임. `pod`의 blueprint라고 볼 수 있음.

`labels` & `selectors`: `metadata`는 `labels`를 가지고 있고, `spec`은 `selector`를 가지고 있음.

- 위 예제에서는, `app`이 `nginx`를 가지고 있는데, 이것이 컴포넌트와 연결되어 있는 것임.
- 그렇게 되면 `deployment`가 이 `pod`가 어디와 연결되어 있는지 알 수 있음
- `service`에는 `selector`가 있는데, 여기에 있는 내용으로 `deployment`가 무엇과 연관되어 있는지 알 수 있음.
- service에는 port가 존재. `containerPort`와 `targetPort`를 연결 시키면됨.

## Demo

```bash
» kubectl apply -f nginx-deployment.yaml
deployment.apps/nginx-deployment created

» kubectl apply -f nginx-service.yaml
service/nginx-service created

» kubectl get pod
NAME                                READY   STATUS    RESTARTS   AGE
nginx-deployment-644599b9c9-8wfww   1/1     Running   0          62s
nginx-deployment-644599b9c9-fwp5g   1/1     Running   0          62s

» kubectl get service
NAME            TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)   AGE
kubernetes      ClusterIP   10.96.0.1        <none>        443/TCP   2d
nginx-service   ClusterIP   10.101.115.234   <none>        80/TCP    43s
```

`kubernetes`는 default로 항상 켜져 있다고 보면 된다.

```bash
» kubectl describe service nginx-service
Name:              nginx-service
Namespace:         default
Labels:            <none>
Annotations:       <none>
Selector:          app=nginx
Type:              ClusterIP
IP Families:       <none>
IP:                10.101.115.234
IPs:               10.101.115.234
Port:              <unset>  80/TCP
TargetPort:        8080/TCP
Endpoints:         172.17.0.3:8080,172.17.0.4:8080
Session Affinity:  None
Events:            <none>
```

```bash
» kubectl get pod -o wide
NAME                                READY   STATUS    RESTARTS   AGE     IP           NODE       NOMINATED NODE   READINESS GATES
nginx-deployment-644599b9c9-8wfww   1/1     Running   0          3m27s   172.17.0.4   minikube   <none>           <none>
nginx-deployment-644599b9c9-fwp5g   1/1     Running   0          3m27s   172.17.0.3   minikube   <none>           <none>
```

이번엔 자동으로 생성된다던 status를 살펴보자.

```bash
» kubectl get deployment nginx-deployment -o yaml
```

yaml 파일을 보면, 우리가 생성한 것 외에 추가적인 정보가 더 생겼다는 것을 알 수가 있다. (`status`를 제외 하더라도) 따라서 이것들을 복사해서 바로 사용하지 말고 주의해서 사용해야 한다.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  annotations:
    deployment.kubernetes.io/revision: '1'
    kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{},"labels":{"app":"nginx"},"name":"nginx-deployment","namespace":"default"},"spec":{"replicas":2,"selector":{"matchLabels":{"app":"nginx"}},"template":{"metadata":{"labels":{"app":"nginx"}},"spec":{"containers":[{"image":"nginx:1.16","name":"nginx","ports":[{"containerPort":80}]}]}}}}
  creationTimestamp: '2021-06-12T14:04:22Z' # 생성시간
  generation: 1
  labels:
    app: nginx
  name: nginx-deployment
  namespace: default
  resourceVersion: '49389'
  uid: 23fb2be1-a62f-4875-a6ae-7298ebd2b49c
spec:
  progressDeadlineSeconds: 600
  replicas: 2
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      app: nginx
  strategy:
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
    type: RollingUpdate
  template:
    metadata:
      creationTimestamp: null
      labels:
        app: nginx
    spec:
      containers:
        - image: nginx:1.16
          imagePullPolicy: IfNotPresent
          name: nginx
          ports:
            - containerPort: 80
              protocol: TCP
          resources: {}
          terminationMessagePath: /dev/termination-log
          terminationMessagePolicy: File
      dnsPolicy: ClusterFirst
      restartPolicy: Always
      schedulerName: default-scheduler
      securityContext: {}
      terminationGracePeriodSeconds: 30
status:
  availableReplicas: 2
  conditions:
    - lastTransitionTime: '2021-06-12T14:04:26Z'
      lastUpdateTime: '2021-06-12T14:04:26Z'
      message: Deployment has minimum availability.
      reason: MinimumReplicasAvailable
      status: 'True'
      type: Available
    - lastTransitionTime: '2021-06-12T14:04:22Z'
      lastUpdateTime: '2021-06-12T14:04:26Z'
      message: ReplicaSet "nginx-deployment-644599b9c9" has successfully progressed.
      reason: NewReplicaSetAvailable
      status: 'True'
      type: Progressing
  observedGeneration: 1
  readyReplicas: 2
  replicas: 2
  updatedReplicas: 2
```

```bash
» kubectl delete -f nginx-deployment.yaml
deployment.apps "nginx-deployment" deleted

» kubectl delete -f nginx-service.yaml
service "nginx-service" deleted
```

## 예제

### 구성

- MongoDB: pod로 internal 서비스로 만들어서, 외부에서 요청을 받지 못하도록 한다. (같은 클러스터에서만 받도록)
- Mongo Express: DB와 연결, 인증, deployment.yaml로 생성. 외부에서 연결되도록 external service로 만든다.

브라우저 → Mongo Express External service → Mongo Express Pod → MongoDB internal Service → Mongo DB Pod

`mongodb-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mongodb-deployment
  labels:
    app: mongodb
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mongodb
    template: # pod에 관한 정보
      metadata:
        labels:
          app: mongodb
      spec:
        containers:
          - name: mongodb
            image: mongo
```

mongo image가 어떻게 되어있는지 살펴보자.

https://hub.docker.com/_/mongo

- 기본 포트가 `27017`이다.
- Environment Variable: `MONGO_INITDB_ROOT_USERNAME` `MONGO_INITDB_ROOT_PASSWORD`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mongodb-deployment
  labels:
    app: mongodb
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mongodb
    template: # pod에 관한 정보
      metadata:
        labels:
          app: mongodb
      spec:
        containers:
          - name: mongodb
            image: mongo
            ports:
              - containerPort: 27017
            env:
              - name: MONGO_INITDB_ROOT_USERNAME
                value:
              - name: MONGO_INITDB_ROOT_PASSWORD
                value:
```

여기서 아이디와 암호를 직접 넣을 수는 없으므로, `Secret`을 활용할 것이다.

```yaml
apiVersion: v1
kind: Secret # secret
metadata:
  name: mongodb-secret # 이름
type: Opaque # 기본. key-value 타입, TLS... 등이 있음.
data: # 실제 키 값. 여기서 값은 base 64여야한다!! 터미널에서 만들기를 추천
  mongo-root-username: dXNlcm5hbWU=
  mongo-root-password: c2V4eWd1eTEwMjQ=
```

```bash
» echo -n 'username' | base64
dXNlcm5hbWU=

» echo -n 'sexyguy1024' | base64
c2V4eWd1eTEwMjQ=
```

이제 이 값을 추가하자.

```bash
» kubectl apply -f mongo-secret.yml
secret/mongodb-secret created

» kubectl get secret
NAME                  TYPE                                  DATA   AGE
default-token-tffzh   kubernetes.io/service-account-token   3      2d
mongodb-secret        Opaque                                2      8s
```

이를 이제 deployment에서 참조하자.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mongodb-deployment
  labels:
    app: mongodb
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mongodb
    template: # pod에 관한 정보
      metadata:
        labels:
          app: mongodb
      spec:
        containers:
          - name: mongodb
            image: mongo
            ports:
              - containerPort: 27017
            env:
              - name: MONGO_INITDB_ROOT_USERNAME
                valueFrom:
                  secretKeyRef:
                    name: mongodb-secret # secret 메타 데이터 이름
                    key: mongo-root-username # secret 키
              - name: MONGO_INITDB_ROOT_PASSWORD
                valueFrom:
                  secretKeyRef:
                    name: mongodb-secret # secret 메타 데이터 이름
                    key: mongo-root-password # secret 키
```

```bash
» kubectl apply -f mongodb-deployment.yaml
deployment.apps/mongodb-deployment created

» kubectl get all
NAME                                     READY   STATUS              RESTARTS   AGE
pod/mongodb-deployment-8f6675bc5-pg2rh   0/1     ContainerCreating   0          13s

NAME                 TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
service/kubernetes   ClusterIP   10.96.0.1    <none>        443/TCP   2d

NAME                                 READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/mongodb-deployment   0/1     1            0           13s

NAME                                           DESIRED   CURRENT   READY   AGE
replicaset.apps/mongodb-deployment-8f6675bc5   1         1         0       13s

» kubectl get pod
NAME                                 READY   STATUS              RESTARTS   AGE
mongodb-deployment-8f6675bc5-pg2rh   0/1     ContainerCreating   0          29s

» kubectl describe pod mongodb-deployment-8f6675bc5-pg2rh
Name:         mongodb-deployment-8f6675bc5-pg2rh
Namespace:    default
Priority:     0
Node:         minikube/192.168.64.3
Start Time:   Sat, 12 Jun 2021 23:34:01 +0900
Labels:       app=mongodb
              pod-template-hash=8f6675bc5
Annotations:  <none>
Status:       Running
IP:           172.17.0.3
IPs:
  IP:           172.17.0.3
Controlled By:  ReplicaSet/mongodb-deployment-8f6675bc5
Containers:
  mongodb:
    Container ID:   docker://9fb6bbe32ecdd525fccfb92d89af73cce896401be1b4b6d7bb5f23b360fb0080
    Image:          mongo
    Image ID:       docker-pullable://mongo@sha256:482a562bf25f42f02ce589458f72866bbe9eded5b6f8fa5b1213313f0e00bba2
    Port:           27017/TCP
    Host Port:      0/TCP
    State:          Running
      Started:      Sat, 12 Jun 2021 23:34:35 +0900
    Ready:          True
    Restart Count:  0
    Environment:
      MONGO_INITDB_ROOT_USERNAME:  <set to the key 'mongo-root-username' in secret 'mongodb-secret'>  Optional: false
      MONGO_INITDB_ROOT_PASSWORD:  <set to the key 'mongo-root-password' in secret 'mongodb-secret'>  Optional: false
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from default-token-tffzh (ro)
Conditions:
  Type              Status
  Initialized       True
  Ready             True
  ContainersReady   True
  PodScheduled      True
Volumes:
  default-token-tffzh:
    Type:        Secret (a volume populated by a Secret)
    SecretName:  default-token-tffzh
    Optional:    false
QoS Class:       BestEffort
Node-Selectors:  <none>
Tolerations:     node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                 node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
Events:
  Type    Reason     Age   From               Message
  ----    ------     ----  ----               -------
  Normal  Scheduled  55s   default-scheduler  Successfully assigned default/mongodb-deployment-8f6675bc5-pg2rh to minikube
  Normal  Pulling    54s   kubelet            Pulling image "mongo"
  Normal  Pulled     21s   kubelet            Successfully pulled image "mongo" in 33.366837514s
  Normal  Created    21s   kubelet            Created container mongodb
  Normal  Started    20s   kubelet            Started container mongodb
```

정상적으로 시작되었음을 알 수 있다.

이제 internal service를 만들어보자. 근데 일반적으로 deployment와 service는 하나의 파일에 둔다. `yaml`파일 하단에 `---`로 선언해두면, 그 다음 파일 설정을 만들어 둘 수 있다.

```yaml
---
apiVersion: v1
kind: Service
metadata:
  name: mongodb-service
spec:
  selector:
    app: mongodb # 앞서 `labels`로 설정해두었던 값들
  ports:
    - protocol: TCP
      port: 27017
      targetPort: 27017
```

```bash
» kubectl apply -f mongodb-deployment.yaml
deployment.apps/mongodb-deployment unchanged
service/mongodb-service created

» kubectl get service
NAME              TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)     AGE
kubernetes        ClusterIP   10.96.0.1        <none>        443/TCP     2d
mongodb-service   ClusterIP   10.111.181.214   <none>        27017/TCP   34s

» kubectl describe service mongodb-service
Name:              mongodb-service
Namespace:         default
Labels:            <none>
Annotations:       <none>
Selector:          app=mongodb
Type:              ClusterIP
IP Families:       <none>
IP:                10.111.181.214
IPs:               10.111.181.214
Port:              <unset>  27017/TCP
TargetPort:        27017/TCP
Endpoints:         172.17.0.3:27017
Session Affinity:  None
Events:            <none>

NAME                                 READY   STATUS    RESTARTS   AGE     IP           NODE       NOMINATED NODE   READINESS GATES
mongodb-deployment-8f6675bc5-pg2rh   1/1     Running   0          7m35s   172.17.0.3   minikube   <none>           <none>
```

한번에 보고 싶다면,

```bash
» kubectl get all
NAME                                     READY   STATUS    RESTARTS   AGE
pod/mongodb-deployment-8f6675bc5-pg2rh   1/1     Running   0          8m24s

NAME                      TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)     AGE
service/kubernetes        ClusterIP   10.96.0.1        <none>        443/TCP     2d
service/mongodb-service   ClusterIP   10.111.181.214   <none>        27017/TCP   2m14s

NAME                                 READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/mongodb-deployment   1/1     1            1           8m24s

NAME                                           DESIRED   CURRENT   READY   AGE
replicaset.apps/mongodb-deployment-8f6675bc5   1         1         1       8m24s
```

이제 mongo express와 external service를 만들자.

`mongo-express.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mongo-express
  labels:
    app: mongo-express
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mongo-express
  template:
    metadata:
      labels:
        app: mongo-express
    spec:
      containers:
        - name: mongo-express
          image: mongo-express
```

https://hub.docker.com/_/mongo-express

- `port`: 8081
- `ME_CONFIG_MONGODB_ADMINUSERNAME`
- `ME_CONFIG_MONGODB_ADMINPASSWORD`
- `ME_CONFIG_MONGODB_PORT`: 는 기본 27017을 써서 상관없을듯
- `ME_CONFIG_MONGODB_SERVER`

위 정보를 추가하자. `ME_CONFIG_MONGODB_SERVER`는 secret이 아닌 `ConfigMap`을 사용하면 좋을듯.

`mongo-configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mongodb-configmap
data:
  database_url: mongodb-service # 서비스 메타데이터 네임을 그대로 가져온다.
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mongo-express
  labels:
    app: mongo-express
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mongo-express
  template:
    metadata:
      labels:
        app: mongo-express
    spec:
      containers:
        - name: mongo-express
          image: mongo-express
          ports:
            - containerPort: 8081
          env:
            - name: ME_CONFIG_MONGODB_ADMINUSERNAME
              valueFrom:
                secretKeyRef:
                  name: mongodb-secret # secret 메타 데이터 이름
                  key: mongo-root-username # secret 키
            - name: ME_CONFIG_MONGODB_ADMINPASSWORD
              valueFrom:
                secretKeyRef:
                  name: mongodb-secret # secret 메타 데이터 이름
                  key: mongo-root-password # secret 키
            - name: ME_CONFIG_MONGODB_SERVER
              valueFrom:
                configMapKeyRef:
                  name: mongodb-configmap # configmap 메타 데이터 이름
                  key: database_url # configmap 키
```

configmap부터 적용해보자.

```bash
» kubectl apply -f mongo-configmap.yaml
configmap/mongodb-configmap created

» kubectl apply -f mongo-express.yaml
deployment.apps/mongo-express created

» kubectl get pod
NAME                                 READY   STATUS    RESTARTS   AGE
mongo-express-78fcf796b8-vjmgd       1/1     Running   0          20s
mongodb-deployment-8f6675bc5-pg2rh   1/1     Running   0          19m

» kubectl logs mongo-express-78fcf796b8-vjmgd
Waiting for mongodb-service:27017...
Welcome to mongo-express
------------------------


Mongo Express server listening at http://0.0.0.0:8081
Server is open to allow connections from anyone (0.0.0.0)
basicAuth credentials are "admin:pass", it is recommended you change this in your config.js!
Database connected
Admin Database connected
```

정상적으로 서비스가 연결되었다.

이제 브라우저에서 연결되게 해보자. 서비스도 마찬가지로 `mongo-express.yaml`의 하단에 기재한다.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mongo-express-service
spec:
  selector:
    app: mongo-express
  type: LoadBalancer # 인터널 서비스도 로드밸런서로 동작한다. 그냥 여기에서는 external IP 주소를 할당하는 목적이라고 보면 될 것 같다.
  ports:
    - protocol: TCP
      port: 8081
      targetPort: 8081
      nodePort: 30000 # 외부 ip에 열어둘 port 3000~32767 사이만 가능
```

```bash
» kubectl apply -f mongo-express.yaml
deployment.apps/mongo-express unchanged
service/mongo-express-service created

» kubectl get service
NAME                    TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)          AGE
kubernetes              ClusterIP      10.96.0.1        <none>        443/TCP          2d
mongo-express-service   LoadBalancer   10.99.198.37     <pending>     8081:30000/TCP   10s
mongodb-service         ClusterIP      10.111.181.214   <none>        27017/TCP        18m
```

`CLUSTER-IP`는 모두 내부 IP다. `EXTERNAL-IP`가 pending으로 나와 있는 것은, minikube가 실제 K8s와 다른점이다. K8s에서는 실제 주소를 볼 수 있다.

minikube에서는 아이피를 아래 명령어로 줘야 한다.

```bash
» minikube service mongo-express-service
|-----------|-----------------------|-------------|---------------------------|
| NAMESPACE |         NAME          | TARGET PORT |            URL            |
|-----------|-----------------------|-------------|---------------------------|
| default   | mongo-express-service |        8081 | http://192.168.64.3:30000 |
|-----------|-----------------------|-------------|---------------------------|
🎉  Opening service default/mongo-express-service in default browser...
```

![mongo-express](./images/mongo-express.png)

실화냐? 가슴이 웅장해진다. K8s는 전설이다.