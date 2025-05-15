# QuantRS2 GPU Support

このファイルでは、QuantRS2のGPUサポート機能の概要と使い方を説明します。

## ディレクトリ構造

GPUサポート関連のファイルは次のように整理されています：

```
py/
├── docs/gpu/             # GPUサポートに関するドキュメント
├── examples/gpu/         # GPUデモとベンチマークスクリプト
└── tools/gpu/            # GPUビルドおよびテストツール
```

## 使用開始

1. GPUスタブ実装をビルドする
   ```bash
   cd /Users/kitasan/work/quantrs/py
   ./tools/gpu/build_with_gpu_stub.sh
   source .venv/bin/activate
   ```

2. GPUサポートがビルドされたことを確認する
   ```bash
   python tools/gpu/simple_gpu_test.py
   ```

3. GPUデモを実行する
   ```bash
   python examples/gpu/gpu_demo.py
   ```

## 詳細情報

- [ドキュメント](./docs/gpu/README.md) - GPUサポートの詳細情報
- [サンプル](./examples/gpu/README.md) - GPUデモとベンチマーク
- [ツール](./tools/gpu/README.md) - ビルドおよびテストツール

## 現在の状態

現在のGPU実装はスタブであり、実際のGPUアクセラレーションは行いませんが、`use_gpu=True`フラグを使って将来的なGPUサポートを想定したコードを書くことができます。

完全なGPU実装は現在開発中です。