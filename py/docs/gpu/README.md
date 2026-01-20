# QuantRS2 GPU Documentation

このディレクトリには、QuantRS2のGPUアクセラレーション機能に関する文書が含まれています。

## 文書一覧

### 1. [GPU_SUPPORT.md](./GPU_SUPPORT.md)

GPUサポートの概要と使い方についての詳細なガイドです。ビルド方法、使用方法、パフォーマンスに関する考慮事項が含まれています。

### 2. [GPU_SUPPORT_STATUS.md](./GPU_SUPPORT_STATUS.md)

GPUサポートの現在の状態、既知の問題、将来的な計画についての情報が含まれています。

### 3. [GPU_FIXES.md](./GPU_FIXES.md)

GPUサポートの実装中に行われた修正や改善点の詳細です。開発者向けの技術的な情報が含まれています。

## GPUサポートの使用方法

GPUサポートを使うには、次のステップに従ってください：

1. GPUスタブ実装でビルドする
   ```bash
   cd $quantrs/py
   ./tools/gpu/build_with_gpu_stub.sh
   source .venv/bin/activate
   ```

2. GPUサポートのテスト
   ```bash
   python tools/gpu/simple_gpu_test.py
   ```

3. サンプルの実行
   ```bash
   python examples/gpu/gpu_demo.py
   ```

4. 自分のコードでGPU機能を使用する
   ```python
   import quantrs2 as qr
   
   # 回路を作成
   circuit = qr.PyCircuit(10)
   circuit.h(0)
   circuit.cnot(0, 1)
   # さらにゲートを追加...
   
   # GPUでシミュレーション
   result = circuit.run(use_gpu=True)
   ```