---
title: 'v8에서의 메모리 관리'
tags:
  - nodejs
  - v8
published: true
date: 2020-11-18 23:01:36
description: 'V8의 깊고 더 어두운 곳으로...'
---

## V8 메모리 구조

![V8 memory](https://i.imgur.com/kSgatSL.png)

### Heap Memory

이 영역이 V8이 객체나 다이나믹 데이터를 담아두는 영역이다. 여기는 메모리 영역 중에서 가장 큰 부분을 차지하며, 가비지 컬렉션이 발생하는 곳이다. 전체 힙메모리가 가비지 컬렉팅이 되는 것은 아니며, 오직 New Space와 Old Space만 가비지 컬렉팅의 대상이 된다.

- New Space: New Space 또는 `Young generation`이라 불리는 곳이며, 새로운 객체 또는 단기간 유효한 객체들이 존재하는 곳이다. 이 영역은 상대적으로 작고, 두개의 별도 공간인 `Semi Space`가 존재한다. 이는 JVM의 `S0` `S1`과 비슷하다고 볼 수 있다. 이 공간은 `Scavenger`이른바 `Minor GC`에 의해서 관리된다. 이 사이즈의 영역은 `--mini_semi_space_size`와 `--max-_semi_space_size`로 조절할 수 있다.
- Old Space: Old Space 또는 `Old generation`이라고 불리는 곳이며, `new space`에서 minor GC 사이클로부터 살아남은 객체들이 이동하는 곳이다. 이 영역은 `Major GC(Mark-Sweep & Mark-Compact)`에 의해서 관리된다. 이 공간의 사이즈는 `--initial_old_space_size`와 `--max_old_space_size`로 설정할 수 있다. 이 영역은 두개로 나눠진다.
  - Old Pointer Space: 다른 객체를 가르키는 객체가 보관 되는 곳
  - Old Data Space: 단순히 데이터만 가지고 있는 객체 (특정 객체를 가르키지 않음). Strings, boxed numbers, unboxed doubles의 배열이 `New Space`의 두번의 minor GC Cycle로 부터 살아남는다면 이쪽으로 이동하게 된다.
- Large object Space: 다른 Space에 있기에 너무 큰 객체들이 여기에 존재하게 된다. 각 객체들은 [mmap](https://en.wikipedia.org/wiki/Mmap)을 갖게 된다. 큰 객체들은 절대 가비지 콜렉터에 의해 이동하지 않는다.
- Code space: `Just In Time(JIT)` 컴파일러가 컴파일된 코드 블록을 보관하는 곳이다. 실행가능한 메모리가 존재할 수 있는 유일한 곳이다. (코드의 양이 커져서 `Large Object Space`로 가더라도, 여전히 실행 가능하다.)
- Cell space, property cell space, map space: 이는 각각 `Cells` `PropertyCells` `Maps`를 가지고 있는다. 각각의 공간에는 모두 동일한 크기의 객체가 포함되어 있으며, 어떤 종류의 객체를 가리킬 수 있는지에 대한 제한이 있기 때문에 수집을 단순화 한다.

각각의 공간은 pages의 세트로 구성되어 있다. 여기서 페이지란, 운영체제 mmap에서 할당된 연속적인 메모리 청크를 의미한다. `Large Object Space`를 제외하고는, 각각 1MB이다.

### Stack

스택 메모리 영역으로, V8 프로세스 하나당 한개의 스택을 가지고 있다. 메서드/함수 프레임, 원시 값, 객체를 가르키는 포인터등 정적인 데이터를 보유하고 있는 곳이다. 이 스택 메모리의 크기는 `-stack_size`로 결정할 수 있다.

## V8의 메모리 사용 (Stack vs Heap)

메모리가 어떤 구조로 되어 있는지 알아봤으니, 이제는 중요한 부분 인 프로그램이 실행될 때 각 부분이 어떻게 사용되는지를 알아보자. 아래 예제 코드를 살펴보자.

```javascript
class Employee {
    constructor(name, salary, sales) {
        this.name = name;
        this.salary = salary;
        this.sales = sales;
    }
}

const BONUS_PERCENTAGE = 10;

function getBonusPercentage(salary) {
    const percentage = (salary * BONUS_PERCENTAGE) / 100;
    return percentage;
}

function findEmployeeBonus(salary, noOfSales) {
    const bonusPercentage = getBonusPercentage(salary);
    const bonus = bonusPercentage * noOfSales;
    return bonus;
}

let john = new Employee("John", 5000, 5);
john.bonus = findEmployeeBonus(john.salary, john.sales);
console.log(john.bonus);
```

<script async class="speakerdeck-embed" data-id="e89e2e48a797417eb8692897dcada584" data-ratio="1.77777777777778" src="//speakerdeck.com/assets/embed.js"></script>

- `Global Scope`는 스택의 `Global Frame`내에 존재한다.
- 모든 함수 호출은 스택 메모리에 `frame-block` 형태로 추가된다.
- 모든 지역변수, arguments, 그리고 리턴 값은 위에서 언급한 함수 `frame-block` 내에 저장된다.
- 모든 원시값은 스택에 바로 저장된다. 이는 전역변수에 있어도 마찬가지다.
- 모든 객체는 힙에 생성되며, 스택에서 스택 포인터를 활용하여 참조된다. 함수는 자바스크립트에서 단순히 객체다. 이는 전역변수에서도 마찬가지다.
- 현재 함수에서 실행된 새로운 함수는 스택의 맨 위에 쌓인다.
- 함수가 프레임을 리턴하면 이는 스택에서 제거 된다.
- 메인 프로세스가 완료되면, 힙에 있는 객체는 스택에서 더이상 가리키는 포인터가 없으므로 고립되어 버린다.
- 따로 복제를 명시적으로 만들어두지 않는 이상, 다른 객체안에 있는 모든 객체 참조는 참조 포인터를 사용해서 완료된다. 

보시다시피, 스택은 자동으로 관리되며, 이는 V8이 아닌 운영체재가 수행한다. 따라서 우리는 스택에 대해서 많은 신경을 쓸필요가 없다. 반면 힙은 OS에 의해 자동으로 관리되지 않으며, 메모리 공간도 가장 크고, 동적데이터를 보유하고 있기 때문에 시간이 지남에 따라 프로그램의 메모리가 바닥날 수도 있다. 또한 시간이 지남에 따라 파편화가 되면서 애플리케이션의 속도도 느려질 수 있다. 여기가 바로 가비지 컬렉터가 들어오는 곳이다.

힙의 포인터와 데이터를 구별하는 것은 가비지 컬렉션에서 중요한 부분이며, 이를 위해 V8은 `태그된 포인터`라는 접근 방식을 사용한다. 이 방식은 각 단어의 끝에 비트를 표시해두어 포인터인지 데이터인지를 구별한다. 이 접근 방식은 컴파일러지원이 필요하지만서도, 간단하면서도 효율적인 방식이다.

## 가비지 컬렉팅

프로그램이 자유롭게 사용할 수 있는 것보다 더 많은 메모리를 힙에 할당하려고 한다면, V8은 메모리 부족 오류를 발생시킨다. 또는 잘못 관리된 힙도 메모리 누수를 이르킬 수 있다.

V8은 가비지 컬렉팅을 활용하여 힙 메모리를 관리한다. 간단히 얘기하자면, 고립된 객체, 즉 더이상 스택에서 직/간접적으로 참조되지 않은 객체들은 메모리에서 해제하며 다른 객체 생성을 위한 메모리 공간을 확보하게 해준다.

V8의 가비지 컬렉터는 V8 프로세스에서 재사용하기 위하여, 사용 중이지 않은 메모리를 화수하는 역할을 한다. V8 가비지 컬렉터는 힙에 있는 객체를 수명별로 분리하여 각각 다른 단계에서 처리한다. 여기에는 두가지 다른 단계가 있고, 3가지 다른 알고리즘을 사용하여 V8에서 가비지 컬렉팅을 한다.

### Minor GC (Scavenger)

이 GC는 young/new space를 간결하고 깨끗하게 유지하는 역할을 한다. 상대적으로 작은 객체(1~8bm)는 `New Space`에 위치하게 된다. `New Space`에 있는 비용은 매우 저렴하다. 여기에는 새로운 객체를 위한 공간을 할당하고 싶을 때마다 등가시키는 할당 포인터가 있다. 할당 포인터가 `New Space`의 끝에 도달하면, 마이너 GC가 트리거 된다. 이 과정은 Scavenger 라고도 불리우며, [체니의 알고리즘](https://en.wikipedia.org/wiki/Cheney's_algorithm)으로 구현되어 있다. 이 과정은 굉장히 빈번하게 발생되며, 병렬로 스레드를 활용해 이루어지기 때문에 굉장히 빠르다.

마이너 GC의 처리과정을 살짝 보자.

앞서 말했듯, `New Space`는 두개의 같은 사이즈인 `semi-space`로 이루어져 있다. 하나는 `to-space`고 다른 하나는 `from-space`다. 대부분의 할당은 `from-space`에서 이루어진다. (`old space`에 할당되는 실행가능한 코드들은 여기에 저장되지 않는다) `from-space`가 가득차게 되면 마이너 GC 가 가동된다.ㅣ

<script async class="speakerdeck-embed" data-id="5fff2548e55c4bb0a9c837c7eb598bee" data-ratio="1.77777777777778" src="//speakerdeck.com/assets/embed.js"></script>

1. 코드 시작단계에서 `from-space`에 이미 객체가 있다고 가정해보자. (01~06)
2. 프로세스가 새로운 객체인 07을 만들어 낸다.
3. V8은 `from-space`로 부터 메모리를 요청하지만, 더이상 객체를 할당할 메모리가 존재하지 않는다. 따라서 V8은 마이너 GC를 트리거 한다.
4. 마이너 GC는 스택 포인터 (GC 루트)에서 시작하여 `from-space`에서 객체 그래프를 재귀적으로 탐색하여, 사용되거나 살아있는 객체를 찾는다. 이러한 객체는 `to-space`로 이동한다. 이러한 객체가 참조하는 모든 객체도 마찬가지로 이동하게 되며, 이들의 포인터 또한 업데이트 된다. 이는 `from-space`내의 모든 객체를 모두 스캔할 때 까지 실행된다. 이 작업이 완료되면 `to-space`는 자동으로 파편화를 줄이기 위하여 압축된다.
5. 마이너 GC는 `from-space`에 남아 있는 객체들은 모두 가비지로 판단하여 비우게 된다.
6. 마이너 GC는 `to-space`와 `from-space`를 스왑한다. 따라서 모든 객체들은 `from-space`에 존재하며, `to-space`는 비어 있게 된다.
7. 새로운 객체는 `from-space`의 메모리에 할당된다.
8. 이제 `from-space`에 시간이 흘러서 객체가 더 들어 왔다고 가정해보자.
9. 애플리케이션이 새로운 객체를 만든다.
10. V8은 `from-space`로 부터 메모리를 요청하지만, 더이상 객체를 할당할 메모리가 존재하지 않는다. 따라서 V8은 마이너 GC를 트리거한다.
11. 위 작업이 반복되며, 두번째 마이너 GC로부터 살아남은 객체들은 `old-space`로 이동하게 된다. 첫번째 생존자들은 `to-space`로 이동하게 되고, `from-space`에는 가비지만 남아있고, 이를 비우게 된다.
12. 마이너 GC는 `to-space`와 `from-space`를 스왑하며, 모든 객체들은 `from-space`로 이동하고 `to-space` 는 비어지게 된다.
13. 새로운 객체가 `from-space`에 할당된다.

마이너 GC가 어떻게 `young generation`에 공간을 요청하고 이를 간결하게 유지하는지 살펴보았다. 이러한 일련의 과정은 프로세스를 중단시키지만, 너무 빠르고 효율적이기 떄문에 대부분의 경우 무시할 수 있는 수준이다. 이 프로세스는 `nes space`의 참조를 위해 `old space`의 객체를 스캔하지 않기 떄문에, 이전 space에서 새로운 메모리에 이르는 모든 포인터의 레지스터를 사용한다. 이는 [write barrier](https://www.memorymanagement.org/glossary/w.html#term-write-barrier)라고 불리는 과정에 의해 버퍼에 기록된다.

### Major GC

이 GC는 `old generation` 공간을 간결하고 깨끗하게 유지해준다. V8이 `old space`에 더이상 충분한 공간이 없다고 판단했을 때 시작된다.

스캐빈저 알고리즘은 작은 데이터 사이즈에는 매우 완벽하지만, `old space`와 같이 힙사이즈가 큰 경우에는 메모리 과부하를 일으킬 수 있어서 메이저 GC에는 `Mark-Sweep-Compact` 알고리즘을 사용한다. 이 알고리즘은 3색 표시 시스템(흰색, 회색, 검은색) 을 사용한다. 따라서 메이저 GC는 3단계 과정을 거치게 된다.

![Mark-Sweep-Compact](https://i.imgur.com/rcjSZ0T.gif)

- Marking: 첫번째 단계로, 두 알고리즘에 공통으로 사용된다. 가비지 컬렉터가 사용중인 객체와 사용하지 않는 객체를 식별하는 단계다. 사용 중이거나 GC 루트 (스택 포인터)에서 도달할 수 있는 객체는 활성 상태로 표시된다. 
- Sweeping: 가비지 컬렉터는 힙을 탐색하고, 활성으로 표시되지 않은 객체의 메모리 주소를 기록한다. 이제 이 공간은 사용 가능한 공간으로 표시되며, 다른 객체를 저장하는데 사용할 수 있다.
- Compact: 청 소 필요한 경우 모든 살아남은 객체가 이동하게 된다. 이렇게 하게 되면 파편화가 줄어들고, 새로운 객체에 대한 메모리 할당 성능이 증가하게 된다.

이러한 유형의 GC는 GC를 수행하는 동안 프로세스의 일시중지를 야기 하기 때문에 `stop-the-world` GC라고도 한다. 이를 피하기 위해 V8은 아래와 같은 방법을 사용한다.

![Major GC](https://v8.dev/_img/trash-talk/09.svg)

- 증분 GC: GC는 하나가 아닌 여러 증분 단계로 수행된다.
- 동시 marking: 마킹은 메인 자바스크립트 스레드에 영향을 주지 않기 위해 여러 헬퍼 스레드를 사용하여 동시에 수행된다. `Writes Barrier`는 헬퍼가 마킹을 하는 동안 자바스크립트하 생성하는 객체 간에 새로운 참조를 추적하기 위하여 사용된다.
- 동시 Sweeping, compacting: 메인 자바스크립트 스레드에 영향을 주지 않기 위해 Sweeping과 Compacting은 헬퍼 쓰레드에서 동시에 이루어진다.
- 게으른 Sweeping: 게으른 Sweeping 은 메모리가 필요로 할 떄까지 페이지에서 가비지 삭제를 지연 시키는 것을 포함한다.

Major GC의 프로세스를 살펴보자.

1. 많은 마이너 GC 사이클이 지나고 `old space`는 거의 가득차서 V8이 메이서 GC를 트리거 했다고 가정해보자.
2. 메이저 GC는 스택포인터에서 시작하여 객체 그래프를 재귀적으로 순회하며, `old space`에 있는 사용중인 객체와 가비지를 별개로 표시해둔다. 이 작업은 여러개의 동시 헬퍼 스레드를 사용하여 수행되며, 각 헬퍼는 포인터를 따른다. 이는 주 메인 쓰레드에 영향을 미치지 않는다.
3. 동시 marking이 끝나거나, 메모리가 제한에 도달하면 GC는 메인스레드를 사용하여 Marking 단계를 마무리한다. 이는 작은 일시정지 시간을 만든다.
4. 이제 메이저 GC는 동시 스윕 스레드를 사용하여, 모든 가비지 객체의 메모리를 사용가능한 것으로 표시해둔다. 또한 병렬 압축 작업이 트리거 되어, 파편화를 방지하기 위하여 관련 메모리 블록을 모두 동일한 페이지로 이동시킨다. 이 단계에서 포인터가 업데이트 된다.


출처: https://deepu.tech/memory-management-in-v8/