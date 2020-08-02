---
title: CUDA C++ Programming Guide メモ(2)
tags: GPU
---

* <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>

## Chapter 4. Hardware Implementation

NVIDIA GPUは**Streaming Multiprocessors**(**SMs**)から構成される。
mutliprocessorは数百ものスレッドを同時に実行するように設計されている。
**Single-Instruction, Mutliple-Thread**(**SIMT**) architectureを採用することでこれを実現している。

* 採用している
    * pipeline
    * instruction-level parallelism within a single thread
* 採用していない
    * out of order実行
    * branch prediction
    * 投機的実行

### 4.1. SIMT Architecture

multiprocessorは**warp**(32スレッド)単位でスレッドを生成、管理、スケジュール、実行する。
warp内の各スレッドは同一のプログラムアドレスから命令を実行し始めるが、
スレッドごとにaddress counterとregister stateを持っている。
このため分岐が可能であったり、
命令の実行を独立に実行できたりする。

multiprocessorでblockを実行する時、
blockをwarpに分割し、
各warpは**warp scheduler**がスケジュールする。
blockのwarpへの分割方法は常に同じで、
スレッドID: 0, ..., 31のwarp、
スレッドID: 32, ..., 63のwarp
といった具合である。

warp内のスレッドは共通の命令を実行する。
よってすべてのスレッドのexecution pathが同一の時に最大効率となる。
そうならず**diverge**した場合、
異なるexecution pathのスレッドはストール(?)させて、
一部のスレッドだけで実行するようにする。
(Branch divergenceはwarp内でのみ起こることであることに注意。異なるwarpは関係ない)

Voltaの前はwarpで一つのプログラムカウンタを共有し、
active threadを指定するような**active mask**を使っていた。
この場合、divergent regionにおいて`signal each other` or `exchange data`が出来なかった。

Voltaから**Independent Thread Scheduling**が導入され、
full concurrency between threads, regardless of warpとなった。
threadごとにexecution state(program counter and call stack)を保持しthrread単位で命令を発行することで、
実行リソースを上手く利用したり、他のスレッドが作成するデータを待ったりできる。
schedule optimizerはwarpのactive threadsをSIMT unitsにグループ化する方法を決定する。

> threads can now diverge and reconverge at sub-warp granularity.

* **active** threads   : participating in the current instruction
* **inactive** threads : not on the current instruction

### 4.2. Hardware Multithreading

execution contextはwarpが生きている間はon-chipに置かれている。
execution contextのスイッチはno costである。
命令発行時はwarp schedulerがactive threadsを選択して実行する。

## Chapter 5. Performance Guidelines

### 5.1. Overall Performance Optimization Strategies

* three basic strategirs
    * Maximize parallel execution to achieve maximum utilization
    * Optimize memory usage to achieve maximum memory throughput
    * Optimize instruction usage to achieve maximum instruction throughput

### 5.2. Maximize Utilization

> the application should maximize parallel execution between the various functional units within a multiprocessor.   
> Utilization is therefore directly linked to the number of resident warps.

* **latency**: warpが次の命令を実行できるようになるまでに要するクロック数

理想は、あるwarpのlatencyの間は他のwarpがずっと起動している状態である(latency is completely hidden)。
latencyのクロック数をLとして、latencyを隠すために必要なwarp数は次の通り

* 4L for cc 5.x, 6.1, 6.2, 7.x, 8.0
* 2L for cc 6.0
* 8L for cc 3.x

warpが実行可能状態でない主な理由はオペランドがまだ利用可能でないことである。

* register dependencies: cc 7.xでは算術命令のclockはだいたい4なので、4*4=16のactive warps per multiprocessorがあればよい。
* operands resides in off-chip memory: 数百クロックはかかる。
    * **arithmetic intensity of the program**
* waiting at some memory fence or synchronization point

レジスタやshared memoryの使用率はコンパイラにオプション`-ptxas-options=-v`を渡せば分かる。

使用するshared memoryの量はstatically allocatedとdynamically allocatedの和である。

カーネルが使用するレジスタの数はresident warpsの数に超絶寄与する。
このためコンパイラはレジスタスピルを極力抑えつつも使用するレジスタの数を最小にとどめようと最適化をかける。
使用するレジスタの数は`maxrregcount`オプションで無理やり指定することもできる。

GPUのレジスタファイルは32bitレジスタで構成されている。`uint8_t`は1つの32bitレジスタを使用するし、`double`は2つのレジスタを使用する。

execution configuratinのパフォーマンスへの寄与はカーネルのコードによってまちまちなので実験することが望ましい。

ただ、configurationを決定するのを補助するAPIはいくつも用意してある。

* **Occupancy**
    * `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
    * `cudaOccupancyMaxPotentialBlockSize`, `cudaOccupancyMaxPotentialBlockSizeVariableSMem`

### 5.3. Maximize Memory Throughput

> The first step in maximizing overall memory throughput for the application is to minimize data transfers with low bandwidth.

つまり、hostとdeviceの通信は最小限にとどめる。
次に、global memoryとdeviceの通信も最小限にとどめる。つまりon-chipメモリ(shared memory, caches)を有効に使おう。

> The next step in maximizing memory throughput is therefore to organize memory accesses at optimally as possible based on the optimal memory access patterns.

* Global Memory: 32-, 64-, or 128-byte transactions
    * global memory throughputを最大化するためには、
        * (ccごとの)最適なアクセスパターンに従う
        * sizeやalignmentに気を使う: runtime APIでmallocしたデバイスメモリは少なくとも256bytesにアラインされている。
        ```
        struct __align__(16) {
            float x;
            float y;
            float z;
        };
        ```
        * 適度にpaddingする
* Local Memory: resides in device memory
    * 次のようなデータがlocal memoryに配置される
        * Arrays for which it cannot determine that they are indexed with constant quantites
        * Large structure or arrays that would consume too much register space
        * Any variable if the kernel uses more registers than availabe
    * PTXを見れば`first compilation phases`(?)にてlocal memoryに配置されているデータが何か分かる
        * `.local` mnemonicを使って宣言され、`ld.local`や`st.local`mnemonicを使ってアクセスされる
    * ただしその後の最適化でどうなったかは分からない
    * このような場合はcubinを`cuobjdump`すれば分かる
    * また`--ptxas-options=-v`オプションを使えば、使用しているlocal memoryのサイズをコンパイラが報告する
    * local memoryはアクセスがfully coalescedとなるように配置される 
* Shared Memory: on-chip
    * **bank**(equally sized memory modules)に分割されている
    * n個のshared memoryへのリクエスト(r/w)がすべて異なるbankへのリクエストだった場合、非常にhigh performance
    * しかし、同一bankへのアクセスが複数存在する場合、アクセスがserializeされる (**bank confilct**)
* Constant Memory: **PASS**
* Texture and Surface Memory: **PASS**

### 5.4. Maximize Instruction Throughput

* 命令のスループットをあげるには
    * スループットの小さい算術命令を使わない
        * トレードオフがある: regular functions v.s. insrinsic functions, single-precision v.s. double-precision
    * divergent warpをなくす
    * 命令数を減らす
        * syncを減らす
        * `__restrict__` を使う

このセクションでは、スループットを、　(mulitprocessor単位で)1クロック当たりの操作数(number of operations)と定義する。
warpの1命令は32操作に対応するので、スループットがNのとき、命令のスループットはN/32となる。


