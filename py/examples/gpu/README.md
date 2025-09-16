# QuantRS2 GPU Examples

このディレクトリには、QuantRS2のGPUアクセラレーション機能のデモとベンチマーク用のスクリプトが含まれています。

## 使用方法

QuantRS2をGPUスタブ実装でビルドする必要があります：

```bash
cd $quantrs/py
./tools/gpu/build_with_gpu_stub.sh
source .venv/bin/activate
```

## 含まれるサンプル

### 1. 基本的なGPUデモ

```bash
python examples/gpu/gpu_demo.py
```

GHZ状態と量子フーリエ変換の2種類の量子回路で、様々な量子ビット数でのCPUとGPUのパフォーマンスを比較します。

### 2. 極端なGPUデモ（より明確な効果）

```bash
python examples/gpu/extreme_gpu_demo.py
```

量子ビット数に応じたGPUの優位性をより極端にシミュレートするデモです。特に大規模回路での劇的なスピードアップを示します。

### 3. 複雑な回路ベンチマーク

```bash
python examples/gpu/complex_benchmark.py
```

深さと量子ビット数の異なる複雑な回路でのCPUとGPUのパフォーマンスを測定します。

### 4. 量子ベンチマーク

```bash
python examples/gpu/quantum_benchmark.py
```

様々な量子回路サイズでのCPUとGPUのパフォーマンスを比較します。

### 5. インタラクティブ量子回路デザイナー

```bash
python examples/gpu/interactive_quantum.py
```

対話型のインターフェースで量子回路を作成し、CPUとGPUの両方で実行できます。

## GPUアダプター

`gpu_adapter.py`は、本物のGPU実装がなくても将来的なGPUの効果をシミュレートするツールです。以下のように使用できます：

```python
import examples.gpu.gpu_adapter as ga
ga.install_gpu_adapter()

# この後のGPU実行は現実的な効果をシミュレートします
```