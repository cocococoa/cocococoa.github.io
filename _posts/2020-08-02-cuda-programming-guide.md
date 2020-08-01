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


### 5.3. Maximize Memory Throughput

### 5.4. Maximize Instruction Throughput


