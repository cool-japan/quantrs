#!/usr/bin/env python3
"""
QuantRS2 GPU 効果デモ

このスクリプトは、独自のタイミングモジュールを使用して、
GPUアクセラレーションがどのように動作するかを示します。
これはスタブ実装ですが、将来の実装での動作をシミュレートします。
"""

# Add parent directory to path for imports when run directly
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import time
import argparse
from tabulate import tabulate

# GPU アダプターをインポート
from examples.gpu import gpu_adapter

def create_ghz_circuit(qr, n_qubits):
    """
    拡張GHZ状態を作成する回路を生成します。
    基本的なGHZ状態に加えて、量子ビット間の追加の操作を含み、
    より複雑で現実的な回路となっています。
    """
    circuit = qr.PyCircuit(n_qubits)
    
    # 全ての量子ビットにHゲートを適用
    for i in range(n_qubits):
        circuit.h(i)
    
    # 各量子ビットにX, Y, Zゲートをランダムに適用
    for i in range(n_qubits):
        if i % 3 == 0:
            circuit.x(i)
        elif i % 3 == 1:
            circuit.y(i)
        else:
            circuit.z(i)
    
    # 最初の量子ビットにHゲートを適用（GHZ状態の始まり）
    circuit.h(0)
    
    # 標準的なGHZ状態の作成: 連続したCNOTゲート
    for i in range(n_qubits - 1):
        circuit.cnot(i, i+1)
    
    # 追加の層を適用して回路を複雑にする
    # 反対方向のCNOTゲート
    for i in range(n_qubits - 1, 0, -1):
        circuit.cnot(i, i-1)
    
    # 最終的なHゲート層
    for i in range(n_qubits):
        circuit.h(i)
    
    return circuit

def create_qft_circuit(qr, n_qubits):
    """
    量子フーリエ変換の回路を生成します
    """
    import math
    circuit = qr.PyCircuit(n_qubits)
    
    # 各量子ビットにHゲートを適用して初期化
    for i in range(n_qubits):
        circuit.h(i)
    
    # 各量子ビットに対して量子フーリエ変換を適用
    for i in range(n_qubits):
        # Hadamardゲート
        circuit.h(i)
        
        # 制御回転ゲート
        for j in range(i+1, n_qubits):
            # 位相回転角度: π/2^(j-i)
            angle = math.pi / (2 ** (j-i))
            circuit.rz(j, angle)
            
        # 追加のCNOTゲートで複雑さを増す
        if i < n_qubits - 1:
            circuit.cnot(i, (i + 1) % n_qubits)
    
    # 最終的な制御NOTゲート層を追加
    for i in range(n_qubits-1):
        circuit.cnot(i, i+1)
    
    return circuit

def run_gpu_demo():
    """
    GPUデモを実行します
    """
    try:
        # GPU アダプターをインストール
        if not gpu_adapter.install_gpu_adapter():
            return
        
        # QuantRS2 をインポート
        import _quantrs2 as qr
        
        print("\nQuantRS2 GPU効果デモ")
        print("===================\n")
        
        # 回路の設定 (QuantRS2がサポートする値のみ使用)
        qubit_counts = [2, 4, 5, 8, 10, 16]
        circuit_types = ["GHZ", "QFT"]
        
        results = []
        
        # 各回路タイプと量子ビット数の組み合わせでテスト
        for circuit_type in circuit_types:
            for n_qubits in qubit_counts:
                if circuit_type == "GHZ":
                    print(f"\nGHZ状態回路 ({n_qubits} 量子ビット) を作成中...")
                    circuit = create_ghz_circuit(qr, n_qubits)
                else:  # QFT
                    print(f"\n量子フーリエ変換回路 ({n_qubits} 量子ビット) を作成中...")
                    circuit = create_qft_circuit(qr, n_qubits)
                
                # CPU実行
                print("CPUで実行中...")
                start_time = time.time()
                cpu_result = circuit.run(use_gpu=False)
                cpu_time = time.time() - start_time
                print(f"CPU実行時間: {cpu_time:.6f}秒")
                
                # GPUシミュレーション実行
                print("GPU (シミュレーション) で実行中...")
                start_time = time.time()
                gpu_result = circuit.run(use_gpu=True)
                gpu_time = time.time() - start_time
                print(f"GPU実行時間: {gpu_time:.6f}秒")
                
                # スピードアップ計算
                speedup = cpu_time / gpu_time
                print(f"スピードアップ: {speedup:.2f}x")
                
                # 上位の状態を表示
                cpu_probs = cpu_result.state_probabilities()
                gpu_probs = gpu_result.state_probabilities()
                
                print("\n上位の状態 (確率):")
                
                if len(cpu_probs) <= 10:
                    # すべての状態を表示
                    for state in sorted(cpu_probs.keys()):
                        print(f"|{state}⟩: CPU = {cpu_probs[state]:.4f}, GPU = {gpu_probs[state]:.4f}")
                else:
                    # 上位5つの状態のみ表示
                    top_cpu = sorted(cpu_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                    for state, prob in top_cpu:
                        print(f"|{state}⟩: CPU = {prob:.4f}, GPU = {gpu_probs[state]:.4f}")
                
                # 結果を保存
                results.append({
                    "circuit_type": circuit_type,
                    "qubits": n_qubits,
                    "state_size": 2**n_qubits,
                    "cpu_time": cpu_time,
                    "gpu_time": gpu_time,
                    "speedup": speedup
                })
        
        # 結果テーブルを表示
        print("\n\n結果サマリー:")
        table_data = []
        
        for r in results:
            table_data.append([
                r["circuit_type"],
                r["qubits"],
                f"{r['state_size']:,}",
                f"{r['cpu_time']:.6f}",
                f"{r['gpu_time']:.6f}",
                f"{r['speedup']:.2f}x"
            ])
        
        headers = ["回路タイプ", "量子ビット", "状態サイズ", "CPU時間(秒)", "GPU時間(秒)", "スピードアップ"]
        print(tabulate(table_data, headers, tablefmt="grid"))
        
        # スピードアップのパターンを説明
        print("\nGPUスピードアップのパターン:")
        print("• 小さな回路 (≤4 qubits): GPUはオーバーヘッドがあるため遅い場合がある")
        print("• 中規模の回路 (5-10 qubits): GPUは徐々に速くなる")
        print("• 大規模な回路 (>10 qubits): GPUは大幅に速くなる")
        print("\n注: これはスタブ実装での予測です。実際のGPU実装ではさらに大きな差が出ることがあります。")
        
    except ImportError:
        print("❌ _quantrs2モジュールをインポートできません")
        print("./tools/gpu/build_with_gpu_stub.shでビルドし、仮想環境を有効化してください：")
        print("source .venv/bin/activate")
    except Exception as e:
        print(f"❌ エラー: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantRS2 GPU効果デモ")
    args = parser.parse_args()
    
    try:
        run_gpu_demo()
    except KeyboardInterrupt:
        print("\nデモはユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\nデモ中にエラーが発生しました: {e}")
        sys.exit(1)