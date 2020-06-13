---
title: Processingの環境構築(WSL2 + VSCode)
tags: processing
---

## 目次

1.  WSL2のGUI環境を構築
2.  WSL2にProcessingをインストール
3.  VSCodeでProcessing用の環境を構築

### 1. WSL2のGUI環境を構築

この記事通りインストールすればほぼ完了。
僕の環境ではこれに加えて、ファイヤーウォールの設定もしなければならなかった。

-   <https://qiita.com/ryoi084/items/0dff11134592d0bb895c>

要するにVcXsrvでWindows側にXのサーバを立ち上げておいて、WSL2側からこのサーバに接続する感じだと思う。   
余談だけど、WSL1ではサーバのアドレスは `DISPLAY=:0.0` という感じで良かったけれど、
WSL2でVMになったせいか、ネットワーク系の設定がWindows側とWSL2側で大きく異なることを意識して設定しないといけなくなった気がする。

### 2. WSL2にProcessingをインストール

公式ホームページからインストーラをダウンロードする。(もちろん Linux 64-bitを)

-   <https://processing.org/download/>

解凍すればすでにバイナリがあるので、ここにパスを通せば完了。

### 3. VSCodeでProcessing用の環境を構築

この拡張をインストールするだけ。シンタックスハイライト、インテリセンス、タスク等がサポートされる。

-   <https://marketplace.visualstudio.com/items?itemName=Tobiah.language-pde>

## 補足

たぶん追加でJavaをインストールしないといけない。
僕はすでにインストールしていたから良く分からない。
