#!/usr/bin/env python3
"""
Comprehensive tests for NumRS2 integration in QuantRS2-Core Python bindings.

This test suite validates the complete integration between QuantRS2-Core and NumRS2,
ensuring seamless data exchange and quantum operations with high-performance
numerical arrays.
"""

import sys
import traceback
import numpy as np
from typing import List, Tuple, Optional

def test_numrs2_integration():
    """Test NumRS2 integration with QuantRS2-Core Python bindings."""
    print("="*80)
    print("🧮 NumRS2 Integration Test Suite")
    print("Testing seamless data exchange between QuantRS2-Core and NumRS2")
    print("="*80)
    
    test_results = []
    
    try:
        # Import the core module with NumRS2 support
        import quantrs2.core as core
        print("✅ Successfully imported quantrs2.core with NumRS2 support")
        test_results.append(("Core import", True))
    except ImportError as e:
        print(f"❌ Failed to import quantrs2.core: {e}")
        test_results.append(("Core import", False))
        return test_results
    
    # Test 1: Basic NumRS2Array creation
    print("\n📋 Test 1: Basic NumRS2Array Creation")
    try:
        # Create a 3x3 NumRS2 array
        shape = [3, 3]
        numrs2_array = core.create_numrs2_array(shape)
        print(f"✅ Created NumRS2Array with shape {shape}")
        print(f"   Array details: {numrs2_array}")
        print(f"   Shape: {numrs2_array.shape}")
        print(f"   Size: {numrs2_array.size}")
        print(f"   Dimensions: {numrs2_array.ndim}")
        test_results.append(("NumRS2Array creation", True))
    except Exception as e:
        print(f"❌ NumRS2Array creation failed: {e}")
        test_results.append(("NumRS2Array creation", False))
    
    # Test 2: NumRS2 array creation with specific values
    print("\n📋 Test 2: NumRS2Array Creation with Specific Values")
    try:
        # Create zeros array
        zeros_array = core.numrs2_zeros([2, 2])
        print(f"✅ Created zeros array: {zeros_array}")
        
        # Create ones array
        ones_array = core.numrs2_ones([2, 2])
        print(f"✅ Created ones array: {ones_array}")
        
        # Create array from vector
        data = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]  # Complex numbers as (real, imag) tuples
        vec_array = core.numrs2_from_vec(data, [2, 2])
        print(f"✅ Created array from vector: {vec_array}")
        test_results.append(("NumRS2Array specialized creation", True))
    except Exception as e:
        print(f"❌ NumRS2Array specialized creation failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        test_results.append(("NumRS2Array specialized creation", False))
    
    # Test 3: NumPy ↔ NumRS2 conversion
    print("\n📋 Test 3: NumPy ↔ NumRS2 Conversion")
    try:
        # Create a NumPy array
        np_array = np.array([[1+2j, 3+4j], [5+6j, 7+8j]], dtype=np.complex128)
        print(f"✅ Created NumPy array:\n{np_array}")
        
        # Convert NumPy to NumRS2
        numrs2_from_numpy = core.numpy_to_numrs2(np_array)
        print(f"✅ Converted NumPy to NumRS2: {numrs2_from_numpy}")
        
        # Convert NumRS2 back to NumPy
        numpy_from_numrs2 = numrs2_from_numpy.to_numpy()
        print(f"✅ Converted NumRS2 back to NumPy:\n{numpy_from_numrs2}")
        
        # Verify conversion accuracy
        conversion_error = np.max(np.abs(np_array - numpy_from_numrs2))
        print(f"✅ Conversion round-trip error: {conversion_error:.2e}")
        
        if conversion_error < 1e-15:
            print("✅ NumPy ↔ NumRS2 conversion is lossless")
        else:
            print(f"⚠️  NumPy ↔ NumRS2 conversion has error: {conversion_error}")
        
        test_results.append(("NumPy ↔ NumRS2 conversion", conversion_error < 1e-12))
    except Exception as e:
        print(f"❌ NumPy ↔ NumRS2 conversion failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        test_results.append(("NumPy ↔ NumRS2 conversion", False))
    
    # Test 4: NumRS2 arithmetic operations
    print("\n📋 Test 4: NumRS2 Arithmetic Operations")
    try:
        # Create test arrays
        a = core.numrs2_from_vec([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)], [2, 2])
        b = core.numrs2_from_vec([(1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)], [2, 2])
        
        # Test addition
        c_add = a.add(b)
        print(f"✅ Addition successful: {c_add}")
        
        # Test multiplication
        c_mul = a.multiply(b)
        print(f"✅ Element-wise multiplication successful: {c_mul}")
        
        # Test matrix multiplication
        c_matmul = a.matmul(b)
        print(f"✅ Matrix multiplication successful: {c_matmul}")
        
        test_results.append(("NumRS2 arithmetic operations", True))
    except Exception as e:
        print(f"❌ NumRS2 arithmetic operations failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        test_results.append(("NumRS2 arithmetic operations", False))
    
    # Test 5: NumRS2 array manipulation
    print("\n📋 Test 5: NumRS2 Array Manipulation")
    try:
        # Create test array
        original = core.numrs2_from_vec([(i, i*0.5) for i in range(6)], [2, 3])
        print(f"✅ Original array (2x3): {original}")
        
        # Test reshape
        reshaped = original.reshape([3, 2])
        print(f"✅ Reshaped to (3x2): {reshaped}")
        print(f"   New shape: {reshaped.shape}")
        
        # Test transpose
        transposed = original.transpose()
        print(f"✅ Transposed: {transposed}")
        print(f"   Transposed shape: {transposed.shape}")
        
        test_results.append(("NumRS2 array manipulation", True))
    except Exception as e:
        print(f"❌ NumRS2 array manipulation failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        test_results.append(("NumRS2 array manipulation", False))
    
    # Test 6: NumRS2 indexing operations
    print("\n📋 Test 6: NumRS2 Indexing Operations")
    try:
        # Create test array
        test_array = core.numrs2_from_vec([(i+1, i*2) for i in range(9)], [3, 3])
        print(f"✅ Created test array (3x3): {test_array}")
        
        # Test getting elements
        element = test_array.get_item([1, 1])
        print(f"✅ Element at [1,1]: {element}")
        
        # Test setting elements
        test_array.set_item([0, 0], (10.0, 20.0))
        updated_element = test_array.get_item([0, 0])
        print(f"✅ Updated element at [0,0]: {updated_element}")
        
        test_results.append(("NumRS2 indexing operations", True))
    except Exception as e:
        print(f"❌ NumRS2 indexing operations failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        test_results.append(("NumRS2 indexing operations", False))
    
    # Test 7: Quantum gate operations with NumRS2
    print("\n📋 Test 7: Quantum Gate Operations with NumRS2")
    try:
        # Create a quantum state as NumRS2 array (2^2 = 4 elements for 2 qubits)
        quantum_state = core.numrs2_from_vec([
            (1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)  # |00⟩ state
        ], [4, 1])
        print(f"✅ Created quantum state: {quantum_state}")
        
        # Create a quantum gate
        hadamard_gate = core.create_hadamard_gate(0)
        print(f"✅ Created Hadamard gate: {hadamard_gate}")
        
        # Test applying gate to NumRS2 array (this validates quantum state structure)
        quantum_state.apply_gate(hadamard_gate)
        print("✅ Successfully applied quantum gate to NumRS2 array")
        
        test_results.append(("Quantum gate operations with NumRS2", True))
    except Exception as e:
        print(f"❌ Quantum gate operations with NumRS2 failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        test_results.append(("Quantum gate operations with NumRS2", False))
    
    # Test 8: Performance and memory efficiency
    print("\n📋 Test 8: Performance and Memory Efficiency")
    try:
        import time
        
        # Test large array creation performance
        start_time = time.time()
        large_shape = [1000, 1000]
        large_array = core.numrs2_zeros(large_shape)
        creation_time = time.time() - start_time
        print(f"✅ Created large NumRS2 array {large_shape} in {creation_time:.4f}s")
        
        # Test memory usage
        memory_size_mb = (large_array.size * 16) / (1024 * 1024)  # Complex64 = 16 bytes
        print(f"✅ Large array memory usage: {memory_size_mb:.2f} MB")
        
        # Test arithmetic performance
        start_time = time.time()
        large_ones = core.numrs2_ones(large_shape)
        result = large_array.add(large_ones)
        arithmetic_time = time.time() - start_time
        print(f"✅ Large array addition in {arithmetic_time:.4f}s")
        
        test_results.append(("Performance and memory efficiency", True))
    except Exception as e:
        print(f"❌ Performance and memory efficiency test failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        test_results.append(("Performance and memory efficiency", False))
    
    # Test 9: NumRS2 with quantum circuits
    print("\n📋 Test 9: NumRS2 with Quantum Circuits")
    try:
        # Create a variational circuit that can work with NumRS2 arrays
        circuit = core.VariationalCircuit(2)  # 2-qubit circuit
        circuit.add_rotation_layer("X")
        circuit.add_entangling_layer()
        print(f"✅ Created variational circuit: {circuit}")
        
        # Create parameters as NumRS2 array
        params_data = [(np.pi/4, 0.0), (np.pi/3, 0.0)]  # 2 rotation parameters
        params_array = core.numrs2_from_vec(params_data, [2, 1])
        print(f"✅ Created parameters as NumRS2 array: {params_array}")
        
        test_results.append(("NumRS2 with quantum circuits", True))
    except Exception as e:
        print(f"❌ NumRS2 with quantum circuits failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        test_results.append(("NumRS2 with quantum circuits", False))
    
    # Test 10: Advanced NumRS2 features
    print("\n📋 Test 10: Advanced NumRS2 Features")
    try:
        # Test creating NumRS2Array from static methods
        numrs2_instance = core.NumRS2Array([4, 4])
        print(f"✅ Created NumRS2Array instance: {numrs2_instance}")
        
        # Test from_numpy static method
        np_test = np.array([[1+0j, 2+0j], [3+0j, 4+0j]], dtype=np.complex128)
        numrs2_from_static = core.NumRS2Array.from_numpy(np_test)
        print(f"✅ Created NumRS2Array from NumPy using static method: {numrs2_from_static}")
        
        test_results.append(("Advanced NumRS2 features", True))
    except Exception as e:
        print(f"❌ Advanced NumRS2 features failed: {e}")
        print(f"   Error details: {traceback.format_exc()}")
        test_results.append(("Advanced NumRS2 features", False))
    
    # Summary
    print("\n" + "="*80)
    print("📊 NumRS2 Integration Test Results Summary")
    print("="*80)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for _, passed in test_results if passed)
    failed_tests = total_tests - passed_tests
    
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {failed_tests}")
    print(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 ALL TESTS PASSED! NumRS2 integration is working perfectly!")
        print("🚀 QuantRS2-Core now has seamless high-performance numerical computing support!")
    else:
        print(f"\n⚠️  {failed_tests} test(s) failed. NumRS2 integration needs attention.")
    
    return test_results

if __name__ == "__main__":
    try:
        results = test_numrs2_integration()
        # Exit with appropriate code
        failed_count = sum(1 for _, passed in results if not passed)
        sys.exit(failed_count)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        print(f"Error details: {traceback.format_exc()}")
        sys.exit(1)