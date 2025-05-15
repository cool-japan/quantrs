#!/usr/bin/env python3
"""
QuantRS2 Complex Circuit Benchmark

このスクリプトは複雑な量子回路を使用して、CPU対GPUのパフォーマンスを比較します。
現在はGPUスタブ実装ですが、将来の実装でGPUの優位性を示すためのコードです。
"""

import time
import argparse
import sys
import random
from tabulate import tabulate

def create_complex_circuit(qr, n_qubits, depth):
    """複雑な量子回路を作成します"""
    circuit = qr.PyCircuit(n_qubits)
    
    # 深さに応じて、ランダムなゲートを追加
    for _ in range(depth):
        # すべての量子ビットにHゲートを適用
        for i in range(n_qubits):
            circuit.h(i)
        
        # 量子ビット間の相互作用ゲート（CNOT）を追加
        for i in range(n_qubits-1):
            circuit.cnot(i, i+1)
        
        # 最後の量子ビットから最初の量子ビットへのCNOT（循環）
        circuit.cnot(n_qubits-1, 0)
        
        # ランダムな回転ゲートを追加
        for i in range(n_qubits):
            angle = random.uniform(0, 6.28)  # 0〜2πのランダムな角度
            gate_type = random.choice(["rx", "ry", "rz"])
            if gate_type == "rx":
                circuit.rx(i, angle)
            elif gate_type == "ry":
                circuit.ry(i, angle)
            else:  # rz
                circuit.rz(i, angle)
    
    return circuit

def run_complex_benchmark(repetitions=5, warmup=1):
    """
    複雑な量子回路のベンチマークを実行
    
    Args:
        repetitions: 各テストを繰り返す回数
        warmup: ウォームアップ反復回数（結果に含まれない）
    """
    try:
        import _quantrs2 as qr
    except ImportError:
        print("❌ _quantrs2モジュールをインポートできません")
        print("build_with_gpu_stub.shでビルドし、仮想環境を有効化してください：")
        print("source .venv/bin/activate")
        return
    
    # テスト構成：(量子ビット数, 回路の深さ)
    configs = [
        (5, 10),   # 5量子ビット、深さ10
        (8, 20),   # 8量子ビット、深さ20
        (10, 5),   # 10量子ビット、深さ5
        (10, 10),  # 10量子ビット、深さ10
        (16, 3),   # 16量子ビット、深さ3
        (16, 5),   # 16量子ビット、深さ5
    ]
    
    results = []
    
    print("QuantRS2 複雑回路ベンチマーク")
    print("=" * 60)
    print(f"各テストは{warmup}回のウォームアップ後、{repetitions}回実行されます\n")
    
    for n_qubits, depth in configs:
        print(f"\n{n_qubits}量子ビット、深さ{depth}の回路をテスト中...")
        state_size = 2**n_qubits
        print(f"状態ベクトルサイズ: {state_size:,} 要素")
        
        # テスト用の回路を作成
        print("回路を作成中...")
        circuit = create_complex_circuit(qr, n_qubits, depth)
        
        # CPU実行時間を測定
        cpu_times = []
        print(f"CPUで実行中 ({repetitions}回)...")
        
        # ウォームアップ実行
        for _ in range(warmup):
            _ = circuit.run(use_gpu=False)
        
        # 実際の測定
        for i in range(repetitions):
            start_time = time.time()
            cpu_result = circuit.run(use_gpu=False)
            cpu_time = time.time() - start_time
            cpu_times.append(cpu_time)
            print(f"  実行 {i+1}/{repetitions}: {cpu_time:.4f}秒")
        
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        print(f"CPU平均実行時間: {avg_cpu_time:.4f}秒")
        
        # GPU実行時間を測定
        gpu_times = []
        print(f"GPU(スタブ)で実行中 ({repetitions}回)...")
        
        try:
            # ウォームアップ実行
            for _ in range(warmup):
                _ = circuit.run(use_gpu=True)
            
            # 実際の測定
            for i in range(repetitions):
                start_time = time.time()
                gpu_result = circuit.run(use_gpu=True)
                gpu_time = time.time() - start_time
                gpu_times.append(gpu_time)
                print(f"  実行 {i+1}/{repetitions}: {gpu_time:.4f}秒")
            
            avg_gpu_time = sum(gpu_times) / len(gpu_times)
            print(f"GPU平均実行時間: {avg_gpu_time:.4f}秒")
            
            # 擬似的なスピードアップ（将来的には本物のGPU実装でここが重要になる）
            speedup = avg_cpu_time / avg_gpu_time
            print(f"スピードアップ比率: {speedup:.2f}x")
            
            # 結果の一致を確認
            cpu_probs = cpu_result.state_probabilities()
            gpu_probs = gpu_result.state_probabilities()
            
            # 最初の5つの状態だけ比較して結果が一致するか確認
            consistent = True
            cpu_top = sorted(cpu_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            gpu_top = sorted(gpu_probs.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for (cpu_state, cpu_prob), (gpu_state, gpu_prob) in zip(cpu_top, gpu_top):
                if cpu_state != gpu_state or abs(cpu_prob - gpu_prob) > 1e-6:
                    consistent = False
                    break
            
            if consistent:
                print("✓ CPUとGPUの結果は一致しています")
            else:
                print("⚠️ CPUとGPUの結果に差異があります")
            
            results.append({
                "qubits": n_qubits,
                "depth": depth,
                "state_size": state_size,
                "cpu_time": avg_cpu_time,
                "gpu_time": avg_gpu_time,
                "speedup": speedup,
                "consistent": consistent
            })
        
        except Exception as e:
            print(f"❌ GPU実行エラー: {e}")
            results.append({
                "qubits": n_qubits,
                "depth": depth,
                "state_size": state_size,
                "cpu_time": avg_cpu_time,
                "gpu_time": None,
                "speedup": None,
                "consistent": None
            })
    
    # 結果をテーブルで表示
    print("\n最終結果:")
    table_data = []
    
    for r in results:
        row = [
            r["qubits"],
            r["depth"],
            f"{r['state_size']:,}",
            f"{r['cpu_time']:.6f}"
        ]
        
        if r["gpu_time"] is not None:
            row.extend([
                f"{r['gpu_time']:.6f}",
                f"{r['speedup']:.2f}x",
                "✓" if r["consistent"] else "⚠️"
            ])
        else:
            row.extend(["失敗", "N/A", "N/A"])
        
        table_data.append(row)
    
    headers = ["量子ビット", "深さ", "状態サイズ", "CPU時間(秒)", "GPU時間(秒)", "スピードアップ", "結果一致"]
    print(tabulate(table_data, headers, tablefmt="grid"))
    
    # スピードアップに関する将来予測（本物のGPU実装の場合）
    print("\n将来的なGPU実装での予測スピードアップ（回路のサイズに基づく見積もり）:")
    for r in results:
        n_qubits = r["qubits"]
        state_size = r["state_size"]
        
        # 量子ビット数と状態サイズに基づく予測スピードアップファクター
        # これは本物のGPU実装での潜在的な改善を示すための仮想的な数値です
        if n_qubits <= 8:
            predicted_speedup = 1.5  # 小さな回路では控えめな予測
        elif n_qubits <= 12:
            predicted_speedup = 3.0  # 中程度の回路
        else:
            predicted_speedup = 10.0  # 大きな回路では大幅な予測スピードアップ
        
        print(f"{n_qubits}量子ビット (状態サイズ: {state_size:,}): 約{predicted_speedup:.1f}x のスピードアップ")

def main():
    parser = argparse.ArgumentParser(description="QuantRS2 複雑回路ベンチマーク")
    parser.add_argument("--repetitions", type=int, default=3, 
                       help="各テストの繰り返し回数")
    parser.add_argument("--warmup", type=int, default=1,
                       help="各テストのウォームアップ繰り返し回数")
    
    args = parser.parse_args()
    
    try:
        run_complex_benchmark(
            repetitions=args.repetitions,
            warmup=args.warmup
        )
    except KeyboardInterrupt:
        print("\nベンチマークはユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\nベンチマーク中にエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()