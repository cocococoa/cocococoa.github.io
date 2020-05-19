---
title: CUDA C++ Programming Guide メモ(1)
tags: GPU
---

* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

## Chapter 1.

three key abstractions
:    a hierarchy of thread groups
:    shared memories
:    barrier synchronization

## Chapter 2.

* Thread Hierarchy: gridはいくかのblockから構成され、blcokはいくつかのthreadから構成される
* Memory Hierarchy: threadごとにlocal memory、blockごとにshared memory、gridにはglobal memoryが割り当てられる
    * 他にも2つ、read onlyなメモリ領域がある
        * constant memory space
        * texture memory space
* unified memoryはmanaged memoryを提供する
    * managed memoryとは、__coherent__ な、どのCPU、GPUからもアクセスできる、 アドレス空間を共有する単一のメモリイメージのことである
* compute casapability (!= cuda version)
    * 7 -> Volta
        * 7.5 -> Turing
    * 6 -> Pascal
    * 5 -> Maxwell
    * 3 -> Kepler
    * 2 -> Fermi
    * 1 -> Tesla


## Chapter 3.

* Binary Compatibility: cubinオブジェクトはアーキテクチャ特異的に生成できる
    * -code=sm_35: compute capability 3.5のデバイス用のバイナリを生成する
* PTX Compatibility: PTX命令によってはあるcompute capability以上でないとサポートされていない、ということがある
    * -arch=compute_30: compute capability 3.0で対応している命令のみ生成する
* CUDA Runtime: cudart.lib or libcudart.a or cudart.dll or libcudart.so でruntimeは実装されている
* GPUの初期化は明示的に呼ばない
    * デバイス関数が呼ばれた時点で初期化され、primary contextを作成する
    * hostで `cudaDeviceReset()` を呼ばれるとprimary contextを破壊する
* device memoryはlinear array or CUDA arrayとして確保される
    * CUDA arrayはtexture fetchingに最適化されたopaque memory layout
    * linear memory (linear array)は __単一の__ ユニファイドアドレス空間の連続した領域にアロケートされる
        * `cudaMalloc()`, `cudaFree()`, `cudaMemcpy()`
* shared memory: global memoryより速いことが期待される
    * 行列積の例(略)
* Page-Locked Host Memory: pinned memory(OSにページングされないメモリ領域)
    * `cudaHostAlloc()`, `cudaFreeHost()`
    * `cudaHostRegister()`
    * 利点
        * pinned memory <--> device memoryのコピーをカーネル実行に並行して行える
        * deviceによってはpinned memoryはdeviceのアドレス空間にmapされる(詳細はMapped Memoryの章で)
        * front-side busのあるシステムでは転送のバンド幅が大きくなる(Write-Combining Memoryならもっと速くなりうる)
* Portable Memory: __?__
* Write-Combining Memory: hostのL1, L2 cacheにキャッシュされないメモリで、snoopもされない( -> 40% 程度速度改善)
    ```
    バススヌーピングとは分散共有メモリとマルチプロセッサを備えたシステムで、キャッシュコヒーレンスを実現するために用いられる技術
    各キャッシュコントローラはバスを監視し、バス上に流れるブロードキャストの通知を受け、必要に応じて特定のキャッシュラインを無効にする
    ```
    * hostにおいて、Write-Combining Memoryからの読み取りは遅い
* Mapped Memory: `cudaMapHostMemory` property
    * __?__
* 3.2.5: Asynchronous Concurrent Execution: __PASS__
    * 3.2.5.6: Graph: 操作がnode、操作間の依存関係がエッジ
* Multi Device System
    * Peer to Peer Memory Access
* Unified Virtual Address Space


