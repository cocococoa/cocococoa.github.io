---
title: CUDA C++ Programming Guide メモ(1)
tags: GPU
---

* https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

## Chapter 1.

three key abstractions:
    a hierarchy of thread groups
    shared memories
    barrier synchronization

## Chapter 2.

* Thread Hierarchy: grid はいくかの block で構成され、blcok はいくつかの thread で構成される   
* Memory Hierarchy: thread ごとに local memory 、block ごとに shared memory、grid には global memory が割り当てられる   
    * 他にも2つ、read onlyなメモリ領域がある
        * constant memory space
        * texture memory space
    * この2つと global memory はカーネルの起動中存在し続ける
* unified Memory は managed memory を提供する   
    * managed memory とは、 __coherent__ な、どの CPU, GPU からもアクセスできる、 アドレス空間を共有する単一のメモリイメージのことである   
* compute casapability (!= cuda version)
    * 7 -> Volta
        * 7.5 -> Turing
    * 6 -> Pascal
    * 5 -> Maxwell
    * 3 -> Kepler
    * 2 -> Fermi
    * 1 -> Tesla


## Chapter 3.

* Binary Compatibility: cubin オブジェクトはアーキテクチャ特異的に生成できる
    * -code=sm_35: compute capability 3.5 のデバイス用のバイナリを生成する
* PTX Compatibility: PTX命令によってはある compute capability 以上でないとサポートされていない、ということがある
    * -arch=compute_30: compute capability 3.0 で対応している命令のみ生成する
* CUDA Runtime: cudart.lib or libcudart.a or cudart.dll or libcudart.so でruntimeは実装されている
* GPUの初期化は明示的に呼ばない
    * 関数を呼ばれた時点で初期化され、primary context を作成する
    * hostで cudaDeviceReset() を呼ばれるとprimary context を破壊する
* device memory は linear array or CUDA array として確保される
    * CUDA array はtexture fetching に最適化された opaque memory layout
    * linear memory (linear array) は __単一の__ ユニファイドアドレス空間の連続した領域にアロケートされる
        * cudaMalloc, cudaFree, cudaMemcpy
* shared memory: global memory より速いことが期待される
    * 行列積の例(略)
* Page-Locked Host Memory: pinned memory(OSにページングされないメモリ領域)
    * cudaHostAlloc, cudaFreeHost
    * cudaHostRegister
    * 利点
        * pinned memory <--> device memory のコピーをカーネル実行に並行して行える
        * deviceによってはpinned memoryはdeviceのアドレス空間にmapされる(詳細は Mapped Memory の章で)
        * front-side busのあるシステムでは転送のバンド幅が大きくなる(Write-Combining Memoryならもっと速くなりうる)
