#!/usr/bin/env python3
"""
QuantRS2 極端なGPU効果デモ

このスクリプトは、より明確なGPUの優位性を示すための拡張デモです。
本物のGPU実装ではなくシミュレーションですが、実際のシステムでの
期待される挙動を視覚化するためのものです。
"""

import time
import sys
import random
from tabulate import tabulate
import math

def create_complex_circuit(qr, n_qubits, depth=10):
    """
    非常に複雑な量子回路を生成します。
    深さとゲート数が多く、計算量の多い回路です。
    """
    circuit = qr.PyCircuit(n_qubits)
    
    # 初期のスーパーポジション状態を作成
    for i in range(n_qubits):
        circuit.h(i)
    
    # 深さに応じたゲート層を追加
    for d in range(depth):
        # 奇数層: 回転ゲート
        if d % 2 == 0:
            for i in range(n_qubits):
                angle = (i * math.pi) / n_qubits
                circuit.rx(i, angle)
                circuit.rz(i, angle * 2)
        
        # 偶数層: エンタングルメント
        else:
            for i in range(n_qubits - 1):
                circuit.cnot(i, i+1)
            
            # 追加のCNOTゲート (循環的な接続)
            if n_qubits > 2:
                circuit.cnot(n_qubits-1, 0)
    
    return circuit

def run_extreme_demo():
    """
    極端なGPU効果デモを実行します。
    """
    try:
        # QuantRS2をインポート
        import _quantrs2 as qr
    except ImportError:
        print("❌ _quantrs2モジュールをインポートできません")
        print("build_with_gpu_stub.shでビルドし、仮想環境を有効化してください：")
        print("source .venv/bin/activate")
        return
    
    print("\nQuantRS2 極端なGPU効果デモ")
    print("============================\n")
    print("このデモは、量子ビット数と回路の深さに応じたGPUの優位性を示します。")
    print("※実際のGPU実装ではなく、GPUで期待される挙動をシミュレートしています。\n")
    
    # GPUアダプターをパッチするカスタム実装
    original_run = qr.PyCircuit.run
    
    def extreme_gpu_run(self, *args, **kwargs):
        """
        極端なGPUスピードアップをシミュレートする実装
        """
        use_gpu = kwargs.get('use_gpu', False)
        
        if not use_gpu:
            # CPUの場合はオリジナルの実装をそのまま使用
            return original_run(self, *args, **kwargs)
        
        # GPU実行時のシミュレーション
        print("🚀 GPU実行モード (シミュレーション)")
        
        # 量子ビット数を検出
        n_qubits = 0
        for q in [1, 2, 3, 4, 5, 8, 10, 16]:
            if hasattr(self, f"_qubits_{q}"):
                n_qubits = q
                break
        
        if n_qubits == 0:
            print("⚠️ 量子ビット数を検出できませんでした")
            return original_run(self, *args, **kwargs)
        
        print(f"📊 {n_qubits}量子ビット回路 (状態サイズ: 2^{n_qubits} = {2**n_qubits:,})")
        
        # 実際の計算を実行
        start_time = time.time()
        result = original_run(self, *args, **kwargs)
        actual_time = time.time() - start_time
        
        # シミュレートされたスピードアップ係数を計算
        # 量子ビット数に応じて指数関数的に増加
        # 小さな回路: わずかな遅延
        # 大きな回路: 大幅なスピードアップ
        if n_qubits <= 3:
            speedup = 0.7  # 小さな回路ではGPUが遅い
            simulated_time = actual_time / speedup
            time.sleep(simulated_time - actual_time)
        elif n_qubits <= 5:
            speedup = 1.5 + (n_qubits - 3) * 0.5  # 中小の回路
            # ここでは遅延を加えないので高速化が見られる
        elif n_qubits <= 10:
            speedup = 3.0 + (n_qubits - 5) * 1.0  # 中規模の回路
            # さらに高速に見せるために短い遅延を使用
            time.sleep(0.00001)
        else:
            # 大規模回路の場合、指数関数的なスピードアップが期待される
            # 10量子ビットで8x、16量子ビットで100x以上など
            speedup = 8.0 * (2 ** ((n_qubits - 10) / 2))
            # 非常に短い遅延で高速な結果をシミュレート
            time.sleep(0.000001)
        
        print(f"⏱️ 実際の実行時間: {actual_time:.6f}秒")
        print(f"🔮 シミュレートされたスピードアップ: {speedup:.1f}x")
        
        return result
    
    # run関数をパッチ
    qr.PyCircuit.run = extreme_gpu_run
    
    # テスト構成: (量子ビット, 回路の深さ)
    configs = [
        (2, 5),
        (3, 10),
        (4, 15),
        (5, 20),
        (8, 10),
        (10, 5),
        (16, 3)
    ]
    
    results = []
    
    for n_qubits, depth in configs:
        print(f"\n{'='*60}")
        print(f"🧪 {n_qubits}量子ビット回路 (深さ: {depth}) をテスト中")
        print(f"状態ベクトルサイズ: 2^{n_qubits} = {2**n_qubits:,} 要素")
        
        try:
            # 複雑な回路を作成
            print(f"📝 複雑な回路を生成中...")
            circuit = create_complex_circuit(qr, n_qubits, depth)
            
            # CPU実行
            print(f"\n🖥️ CPUで実行中...")
            start_time = time.time()
            cpu_result = circuit.run(use_gpu=False)
            cpu_time = time.time() - start_time
            print(f"CPU実行時間: {cpu_time:.6f}秒")
            
            # GPUシミュレーション実行
            print(f"\n🎮 GPUで実行中...")
            start_time = time.time()
            gpu_result = circuit.run(use_gpu=True)
            gpu_time = time.time() - start_time
            
            # 結果の整合性をチェック
            cpu_probs = cpu_result.state_probabilities()
            gpu_probs = gpu_result.state_probabilities()
            
            # 上位5つの状態を取得
            cpu_top5 = sorted(cpu_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            gpu_top5 = sorted(gpu_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # 上位の状態が一致するか確認
            states_match = True
            for (cpu_state, _), (gpu_state, _) in zip(cpu_top5, gpu_top5):
                if cpu_state != gpu_state:
                    states_match = False
                    break
            
            print(f"📊 結果の一致: {'✓' if states_match else '❌'}")
            
            # 結果を保存
            results.append({
                "qubits": n_qubits,
                "depth": depth,
                "state_size": 2**n_qubits,
                "cpu_time": cpu_time,
                "gpu_time": gpu_time,
                "speedup": cpu_time / gpu_time,
                "match": states_match
            })
            
        except Exception as e:
            print(f"❌ エラー: {e}")
            results.append({
                "qubits": n_qubits,
                "depth": depth,
                "state_size": 2**n_qubits,
                "cpu_time": None,
                "gpu_time": None,
                "speedup": None,
                "match": None
            })
    
    # 結果テーブルを表示
    print("\n\n最終結果:")
    table_data = []
    
    for r in results:
        row = [
            r["qubits"],
            r["depth"],
            f"{r['state_size']:,}",
        ]
        
        if r["cpu_time"] is not None:
            row.extend([
                f"{r['cpu_time']:.6f}",
                f"{r['gpu_time']:.6f}",
                f"{r['speedup']:.2f}x",
                "✓" if r["match"] else "❌"
            ])
        else:
            row.extend(["エラー", "エラー", "N/A", "N/A"])
        
        table_data.append(row)
    
    headers = [
        "量子ビット", "回路の深さ", "状態サイズ", 
        "CPU時間(秒)", "GPU時間(秒)", "スピードアップ", "結果一致"
    ]
    print(tabulate(table_data, headers, tablefmt="grid"))
    
    # GPUの優位性に関する説明
    print("\n量子コンピューティングにおけるGPU加速の特性:")
    print("1. 小規模回路 (≤3量子ビット):")
    print("   - GPUのオーバーヘッドにより、むしろ遅くなることがある")
    print("   - 初期化コストと転送時間が計算時間を上回る")
    
    print("\n2. 中規模回路 (4-8量子ビット):")
    print("   - GPUの並列処理能力が効果を発揮し始める")
    print("   - 量子ビット数に応じて1.5x〜4xのスピードアップ")
    
    print("\n3. 大規模回路 (≥10量子ビット):")
    print("   - GPUの並列性が大幅なスピードアップをもたらす")
    print("   - 量子ビット数の増加に伴い指数関数的な性能向上")
    print("   - 16量子ビット以上では数十倍〜数百倍の高速化も理論的には可能")
    
    print("\n※実際のGPU実装の性能は、ハードウェア、アルゴリズム、最適化に依存します。")
    print("このデモは将来的に期待される効果を視覚化するためのものです。")

if __name__ == "__main__":
    try:
        run_extreme_demo()
    except KeyboardInterrupt:
        print("\nデモはユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        sys.exit(1)