---
title: CUDA C++ Programming Guide メモ(1)
tags: GPU
---

* <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>

## Chapter 1. Introduction

three key abstractions
:    a hierarchy of thread groups
:    shared memories
:    barrier synchronization

## Chapter 2. Programming Model

* Thread Hierarchy: gridはいくかのblockから構成され、blcokはいくつかのthreadから構成される
* Memory Hierarchy: threadごとにlocal memory、blockごとにshared memory、gridにはglobal memoryが割り当てられる
    * 他にも2つ、read onlyなメモリ領域がある
        * constant memory space
        * texture memory space
* unified memoryはmanaged memoryを提供する
    * managed memoryとは、**coherent** な、どのCPU、GPUからもアクセスできる、 アドレス空間を共有する単一のメモリイメージのことである
* compute casapability (!= cuda version)
    * 8 -> Ampere
    * 7 -> Volta
        * 7.5 -> Turing
    * 6 -> Pascal
    * 5 -> Maxwell
    * 3 -> Kepler
    * 2 -> Fermi
    * 1 -> Tesla


## Chapter 3. Programming Interface

### 3.1. Compilation with NVCC

* runtime: `CUDA RUNTIME`は「デバイスメモリを確保、ホストメモリとデバイスメモリの通信、複数のデバイスの管理」等を行うC/C++関数を提供する
* Binary Compatibility: binary codeはarchitecture specificである。`-code`でarchitectureを指定すれば目的の`cubin` objectを生成できる
    * `-code=sm_35`: compute capability 3.5のデバイス用のバイナリを生成する
* PTX Compatibility: あるcompute capability以上のデバイスでしかサポートしていないPTX命令がある。`-arch`でarchitectureを指定すれば、そのような命令を使うことができる
    * 例
        * Warp SHuffle Functions: compute capability 3.0 以上のみサポート
    * `-arch=compute_30`: compute capability 3.0で対応している命令のみ生成する
* 注意
    * `-arch`で指定されたcompute capabilityをXとすると、このPTXコードはcompute capabilityがX以上のcubin objectに必ずコンパイルできる
    * ただし、このような場合、新しいcompute capabitliyがサポートしている機能は使っていない可能性がある
    * 例: `-arch=compute_60 -code=sm_70`
        * Pascal(6.0)ではTensor Core命令はないので、そのようなPTX命令は出力されない

### 3.2. CUDA Runtime

* CUDA Runtime: `cudart.lib` or `libcudart.a` or `cudart.dll` or `libcudart.so` でruntimeは実装されている

| memory type                | brief description                                        |
|----------------------------|----------------------------------------------------------|
| Device Memory              | device memory                                            |
| Shared Memory              | shared memory                                            |
| Page-Locked Host Memory    | kernelの実行とhost-device通信を同時に行うために必要      |
| Texture and Surface Memory | another way to access device memory (主にレンダリング用) |

* GPUの初期化は明示的に呼ばない
    * デバイス関数が呼ばれた時点で初期化され、`primary context`を作成する
    * hostで `cudaDeviceReset()` を呼ぶとprimary contextを破壊する
* device memoryは`linear array` or `CUDA array`として確保される
    * CUDA arrayはtexture fetchingに最適化されたopaque memory layout
    * linear memory (linear array)は **単一の** ユニファイドアドレス空間の連続した領域にアロケートされる
        * `cudaMalloc()`, `cudaFree()`, `cudaMemcpy()`
* 3.2.3. Device Memory L2 Access Management
    * カーネル中で何度もアクセスするメモリ: `persisting`
    * 逆に一度だけアクセスするメモリ: `streaming`
    * CUDA11.0, CC>=8.0から、L2キャッシュを操作しpersistence of data in the L2 cacheが可能となった
    1. 操作可能なL2キャッシュ領域を確保する
        ```cpp
        cudaGetDeciveProperties(&prop, device_id);
        size_t size = min(int(prop.l2CacheSize * 0.75), prop.persisteingL2CacheMaxSize);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // L2キャッシュの3/4をpersisting access用に確保
        ```
    2. L2 Policy for persisting acces
        ```cpp
        cudaStreamAttrValue stream_attribute;
        stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(d_ptr);
        stream_attribute.accessPolicyWindow.num_bytes = num_bytes;
        stream_attribute.accessPolicyWindow.hitRatio = 0.6;                         // キャッシュヒット率のヒント
        stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting; // キャッシュヒットした時の access property
        stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming; // キャッシュミスし時の access property
        ```
    3. Reset L2 Access to Normal
        * `cudaAccessPropertyormal()`, `cudaCtxResetPersistingL2Cahce()`
    * 注意: 複数のストリームがそれぞれ異なるaccess policy windowを使っている場合に、L2 set-aside cache portionはすべての実行中のカーネルで共有されている
* shared memory: global memoryより速いことが期待される
    * 行列積の例(略)
* Page-Locked Host Memory: page-locked host memory(`pinned-memory`ともいう。OSにページングされないメモリ領域)
    * `cudaHostAlloc()`, `cudaFreeHost()`
    * `cudaHostRegister()`
    * 利点
        * pinned memory <---> device memoryのコピーをカーネル実行に並行して行える
        * (deviceによっては、) page-locked host memoryはdeviceのアドレス空間にmapされる(詳細はMapped Memoryの章で)
        * front-side busのあるシステムでは転送のバンド幅が大きくなる(Write-Combining Memoryならもっと速くなりうる)
* Portable Memory: デフォルトだとpage-locked memoryがアロケートされているデバイスでしか利点を受けられない。複数のデバイスでも利点を享受するために、`cudaHostAllocPortable`を`cudaHostAlloc()`に渡す。
* Write-Combining Memory: hostのL1, L2 cacheにキャッシュされないメモリで、snoopもされない( -> 40% 程度速度改善)
    ```
    バススヌーピングとは分散共有メモリとマルチプロセッサを備えたシステムで、キャッシュコヒーレンスを実現するために用いられる技術
    各キャッシュコントローラはバスを監視し、バス上に流れるブロードキャストの通知を受け、必要に応じて特定のキャッシュラインを無効にする
    ```
    * hostにおいて、Write-Combining Memoryからの読み取りは遅い
        * hostは書き込むだけのメモリを`write-combinig`にすることが普通
* Mapped Memory: page-locked host memoryはdeviceのアドレス空間にマップすることができる
    * flag: `cudaHostAllocMapped` -> `cudaHostAlloc()`, `cudaHostRegisterMapped` -> `cudaHostRegister()` 
    * つまりmapped memoryは二つのアドレスを持つ: host(`cudaHostAlloc`の返り値)と、device(`cudaHostGetDevicePointer()`の返り値)
        * unified address spaceの場合は例外
* 3.2.6. Asynchronous Concurrent Execution: **PASS** (**TODO**: いずれ読む)
    * 3.2.6.6. CUDA Graphs: 操作がnode、操作間の依存関係がエッジ
* 3.2.7. Multi-Device System
    * 3.2.7.4. Peer-to-Peer Memory Access: `cudaDeviceEnablePeerAccess()`を呼ばなければならない
    * `cudaMemcpyPeer()`
* 3.2.8. Unified Virtual Address Space
    * A **single address space** is used for the host and all devices
    * `cudaPointerGetAttributes()`
    * `cudaMemcpyDefault`
    * `cudaHostAlloc()`で確保したメモリは自動的にportableとなる。そして`cudaHostAlloc()`の返り値をカーネルに渡せる
        * `cudaHostGetDevicePointer()`でデバイスポインタを獲得する必要はない
* 3.2.9. Interprocess Communication: **PASS**
* 3.2.10. Error Checking: **PASS**
* 3.2.11. Call Stack: **PASS**
* 3.2.12. Texture and Surface Memory: **PASS**
* 3.2.13. Graphics Interoperability: **PASS**
* 3.2.14. External Resource Interoperability: **PASS**

### 3.3. Versioning and Compatibility

### 3.4. Compute Modes

### 3.5. Mode Switches

### 3.6. Tesla Compute Cluster Mode for Windows


