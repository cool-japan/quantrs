#!/usr/bin/env python3
"""
QuantRS2のGPUサポートを確認するスクリプト
"""

import sys
import traceback

def check_gpu_support():
    """QuantRS2のGPUサポートを確認します"""
    try:
        # まずネイティブモジュールをインポート
        import _quantrs2 as qr
        
        print("_quantrs2モジュールを正常にインポートしました")
        
        # GPUサポートを確認
        try:
            # 最小の回路でGPUを使用
            circuit = qr.PyCircuit(2)
            circuit.h(0)
            circuit.cnot(0, 1)
            
            # GPU機能を使用してみる
            print("GPUシミュレーションを試行中...")
            result = circuit.run(use_gpu=True)
            
            print(f"GPU実行成功！結果: {result}")
            if hasattr(result, 'state_probabilities'):
                probs = result.state_probabilities()
                print("状態確率:")
                for state, prob in probs.items():
                    print(f"|{state}⟩: {prob:.6f}")
            
            return True, "GPUサポートは有効で正常に動作しています"
        except Exception as gpu_err:
            print(f"GPUシミュレーション中にエラーが発生: {gpu_err}")
            traceback.print_exc()
            
            # エラーメッセージからGPUサポートの状態を判断
            error_msg = str(gpu_err).lower()
            if "not compiled" in error_msg or "feature" in error_msg:
                return False, "GPUサポートがコンパイルされていません。--features gpu フラグを使用してコンパイルし直してください。"
            elif "not available" in error_msg:
                return False, "GPUは検出されましたが、利用できません。ドライバーが正しくインストールされていることを確認してください。"
            else:
                return False, f"GPUシミュレーション中に予期しないエラーが発生しました: {gpu_err}"
    
    except ImportError as e:
        return False, f"_quantrs2モジュールのインポートに失敗しました: {e}"
    except Exception as e:
        return False, f"予期しないエラーが発生しました: {e}"

def main():
    """メイン関数"""
    print("QuantRS2 GPUサポート確認ツール")
    print("===========================")
    print(f"Python バージョン: {sys.version}")
    
    # GPUサポートの確認
    success, message = check_gpu_support()
    
    print("\n結果:")
    print(f"GPUサポート: {'有効' if success else '無効'}")
    print(f"詳細: {message}")
    
    # CPUでの実行も試してみる
    try:
        import _quantrs2 as qr
        print("\nCPUシミュレーションを試行中...")
        circuit = qr.PyCircuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        result = circuit.run(use_gpu=False)
        print("CPUシミュレーション成功")
        probs = result.state_probabilities()
        print("状態確率:")
        for state, prob in probs.items():
            print(f"|{state}⟩: {prob:.6f}")
    except Exception as e:
        print(f"CPUシミュレーション中にエラーが発生: {e}")
    
    # 終了
    if success:
        print("\nおめでとうございます！QuantRS2のGPUサポートが有効になっています。")
        return 0
    else:
        print("\nQuantRS2のGPUサポートが有効になっていないか、問題があります。")
        print("GPUサポートを有効にするには、`maturin develop --features gpu`を実行してください。")
        return 1

if __name__ == "__main__":
    sys.exit(main())