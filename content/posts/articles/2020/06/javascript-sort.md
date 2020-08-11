---
title: 자바스크립트로 구현해보는 다양한 정렬
tags:
  - javascript
  - algorithm
published: true
date: 2020-07-01 07:42:01
description: "## 거품(버블)정렬 - 가까운 두 원소를 비교해서 정렬하는 방식이다. - `O(N^2)` - 코드가 단순하고 구현하기
  쉽다 -
  느리다.  ![bubble-sort](https://upload.wikimedia.org/wikipedia/commons/3/37/Bubb\
  le_sort_animation.gif)  ```javascript function bub..."
category: javascript
slug: /2020/06/javascript-sort/
template: post
---
## 거품(버블)정렬

- 가까운 두 원소를 비교해서 정렬하는 방식이다.
- `O(N^2)`
- 코드가 단순하고 구현하기 쉽다
- 느리다.

![bubble-sort](https://upload.wikimedia.org/wikipedia/commons/3/37/Bubble_sort_animation.gif)

```javascript
function bubbleSort(arr){
   for (var i=arr.length-1; i>=0; i--){
     for(var j=1; j<=i; j++){
       if(arr[j-1]>arr[j]){
           var temp = arr[j-1];
           arr[j-1] = arr[j];
           arr[j] = temp;
        }
     }
   }
   return arr;
}
```

## 선택정렬

- 배열에서 가장 작은 값을 찾아, 그 값을 배치 한다.
- `O(N^2)`
- 코드가 단순하고 구현하기 쉽다.
- 느리다.

![selection-sort](https://upload.wikimedia.org/wikipedia/commons/b/b0/Selection_sort_animation.gif)

```javascript
function selectionSort(arr){
  var minIndex, temp, len=arr.length;
  for(var i=0; i < len; i++){
    minIndex = i;
    for(var j = i+1; j<len; j++){
       if(arr[j]<arr[minIndex]){
          minIndex = j;
       }
    }
    temp = arr[i];
    arr[i] = arr[minIndex];
    arr[minIndex] = temp;
  }
  return arr;
}
```

## 삽입정렬

- 배열의 요소를 차례대로 순회하면서, 이미 정렬된 배열과 비교하여 해당 요소를 올바른 위치에 삽입하는 것
- `O(N^2)`
- 구현하기 쉽다
- 배열이 길어질 수록 정렬할 경우의 수가 많아져서 느려진다.

![insertion-sort](https://upload.wikimedia.org/wikipedia/commons/4/42/Insertion_sort.gif)

```javascript
function insertionSort(arr) {
  const result = [...arr];

  for (let i = 1; i < result.length; i++) {
    let temp = result[i];
    let aux = i - 1;

    // 배열 요소가 0보다 같거나 크고, 왼쪽 값이 더 클 때마다 계속해서 바꿔 나간다.
    while (aux >= 0 && result[aux] > temp) {
      result[aux + 1] = result[aux];
      aux--;
    }

    result[aux + 1] = temp;
  }

  return result;
}
```

## 퀵정렬

- 리스트 가운데에서 하나의 원소를 고른다. 이 원소를 피벗이라고 한다.
- 피벗을 기준으로 피벗 앞에는 피벗 보다 작은 값을, 뒤에는 큰 값들이 오도록하고 그렇게 리스트를 둘로 나눈다.
- 분할된 리스트에 대해 이 작업을 리스트의 크기가 0 또는 1이 될 때까지 반복한다.
- 제법 빠르지만, 별도의 메모리 공간이 필요해서 공간 낭비가 있다.
  

```javascript
function quickSort (array) {
  if (array.length < 2) {
    return array;
  }

  const pivot = [array[0]];
  const left = [];
  const right = [];

  for (let i = 1; i < array.length; i++) {
    if (array[i] < pivot) {
      left.push(array[i]);
    } else if (array[i] > pivot) {
      right.push(array[i]);
    } else {
      pivot.push(array[i]);
    }
  }
  return [quickSort(left).concat(pivot, quickSort(right))].flat(Infinity);
}
```

## 병합정렬

- 정렬되지 않은 리스트를 반으로 잘라 비슷한 크기의 두 배열로 나눈다. (길이가 1이면 정렬되었다고 본다)
- 분할된 각 원소에 대해 비교하여 정렬하고 합친다.
- 위 과정을 반복한다.
- `O(N * logN)`

```javascript
function mergeSort(arr) {
  // 이미 배열 한개짜리는 정렬되었다.
  if (arr.length === 1) return arr

  const middleIndex = Math.floor(arr.length / 2)
  const left = arr.slice(0, middle)
  const right = arr.slice(middle)

  return merge(mergeSort(left), mergeSort(right))
}

function merge(left, right) {
  const result = []
  let leftIndex = 0;
  let rightIndex = 0;

  while (leftIndex < left.length && rightIndex < right.index) {
    if (left[leftIndex] < right[rightIndex]) {
      result.push(left[leftIndex])
      leftIndex++
    } else {
      result.push(right[rightIndex])
      rightIndex++
    }
  }

  return [...result, ...left.slice(leftIndex), ...right.slice(rightIndex)]
}
```


