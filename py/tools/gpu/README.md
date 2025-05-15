# QuantRS2 GPU Tools

このディレクトリには、QuantRS2のGPUサポートに関するビルドスクリプトとテストツールが含まれています。

## ビルドスクリプト

### 1. GPU機能付きのビルド（スタブ実装）

```bash
./tools/gpu/build_with_gpu_stub.sh
```

GPUスタブ実装を使ってパッケージをビルドします。このバージョンでは実際のGPUアクセラレーションは行いませんが、GPUコードパスが機能します。

### 2. GPUサポートの構築（将来的な実装用）

```bash
./tools/gpu/build_with_gpu.sh
```

本物のGPU実装でビルドを試みます（現在は不完全な実装のため完了しない可能性があります）。

### 3. 簡易ビルド

```bash
./tools/gpu/try_gpu_build.sh
```

デバッグ目的で単純なGPU機能のビルドを行います。

## テストツール

### 1. GPUサポートチェック

```bash
python tools/gpu/check_gpu_support.py
```

システムがGPUサポートを持っているかどうかをチェックします。

### 2. シンプルGPUテスト

```bash
python tools/gpu/simple_gpu_test.py
```

基本的なGPUコードパスのテストを行います。

### 3. ミニマルGPUテスト

```bash
python tools/gpu/minimal_gpu.py
```

最小限の機能でGPUコードパスをテストします。