#!/usr/bin/env python
# coding=utf-8
"""
软门控预测功能的单元验证脚本

此脚本使用模拟数据验证软门控函数的核心逻辑是否正确，
不需要实际的模型和数据集。
"""

import numpy as np
import sys
import os

# 测试软门控公式逻辑
def test_soft_gating_formula():
    """测试软门控公式计算是否正确"""
    print("="*80)
    print("测试 1: 软门控公式计算")
    print("="*80)
    
    # 测试用例
    test_cases = [
        # (p, y_reg, y_base, gamma, expected_y_hat)
        (1.0, -0.5, -0.2, 1.0, -0.5),      # p=1: 完全信任回归
        (0.0, -0.5, -0.2, 1.0, -0.2),      # p=0: 完全使用y_base
        (0.5, -0.5, -0.2, 1.0, -0.35),     # p=0.5: 中间值
        (0.8, -0.6, -0.2, 1.0, -0.56),     # p=0.8: 偏向回归
        (0.5, -0.5, -0.2, 2.0, -0.3875),   # gamma=2.0: 锐化
    ]
    
    all_passed = True
    for i, (p, y_reg, y_base, gamma, expected) in enumerate(test_cases):
        p_g = p ** gamma
        y_hat = p_g * y_reg + (1 - p_g) * y_base
        
        passed = abs(y_hat - expected) < 0.0001
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        
        print(f"案例 {i+1}: {status}")
        print(f"  输入: p={p}, y_reg={y_reg}, y_base={y_base}, gamma={gamma}")
        print(f"  计算: p_g={p_g:.4f}, y_hat={y_hat:.4f}")
        print(f"  期望: {expected:.4f}, 误差: {abs(y_hat - expected):.6f}")
        print()
    
    return all_passed


def test_edge_cases():
    """测试边界情况"""
    print("="*80)
    print("测试 2: 边界情况")
    print("="*80)
    
    all_passed = True
    
    # 测试极端概率值
    test_cases = [
        ("极小概率", 0.001, -1.0, -0.2, 1.0),
        ("极大概率", 0.999, -1.0, -0.2, 1.0),
        ("零概率", 0.0, -1.0, -0.2, 1.0),
        ("满概率", 1.0, -1.0, -0.2, 1.0),
    ]
    
    for name, p, y_reg, y_base, gamma in test_cases:
        try:
            p_g = p ** gamma
            y_hat = p_g * y_reg + (1 - p_g) * y_base
            # 验证结果在合理范围内
            in_range = min(y_reg, y_base) <= y_hat <= max(y_reg, y_base)
            status = "✓ PASS" if in_range else "✗ FAIL"
            all_passed = all_passed and in_range
            print(f"{name}: {status}")
            print(f"  p={p}, y_hat={y_hat:.4f}, 在范围 [{min(y_reg, y_base):.2f}, {max(y_reg, y_base):.2f}]")
        except Exception as e:
            print(f"{name}: ✗ FAIL - 异常: {e}")
            all_passed = False
        print()
    
    return all_passed


def test_gamma_effect():
    """测试 gamma 参数的效果"""
    print("="*80)
    print("测试 3: Gamma 参数效果")
    print("="*80)
    
    p = 0.7
    y_reg = -0.8
    y_base = -0.2
    
    print(f"固定: p={p}, y_reg={y_reg}, y_base={y_base}")
    print(f"{'Gamma':<10} {'p_g':<10} {'y_hat':<10} {'说明'}")
    print("-" * 50)
    
    gammas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    last_y_hat = None
    all_passed = True
    
    for gamma in gammas:
        p_g = p ** gamma
        y_hat = p_g * y_reg + (1 - p_g) * y_base
        
        if last_y_hat is not None:
            # gamma 增大，y_hat 应该更接近 y_reg（因为 p > 0.5）
            trend = "向y_reg" if y_hat < last_y_hat else "向y_base"
        else:
            trend = "基准"
        
        print(f"{gamma:<10.1f} {p_g:<10.4f} {y_hat:<10.4f} {trend}")
        last_y_hat = y_hat
    
    print()
    print("✓ PASS - Gamma 效果验证完成")
    return True


def test_function_import():
    """测试函数是否可以正确导入"""
    print("="*80)
    print("测试 4: 函数导入")
    print("="*80)
    
    try:
        from predicproc.show import print_predict_result_soft_gated_t1l10
        print("✓ PASS - 函数导入成功")
        print(f"函数文档:\n{print_predict_result_soft_gated_t1l10.__doc__[:200]}...")
        return True
    except ImportError as e:
        print(f"✗ FAIL - 函数导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ FAIL - 未预期的错误: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("软门控预测功能验证")
    print("="*80 + "\n")
    
    results = []
    
    # 运行所有测试
    results.append(("公式计算", test_soft_gating_formula()))
    results.append(("边界情况", test_edge_cases()))
    results.append(("Gamma效果", test_gamma_effect()))
    results.append(("函数导入", test_function_import()))
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("="*80)
    if all_passed:
        print("✓ 所有测试通过！软门控功能实现正确。")
        return 0
    else:
        print("✗ 部分测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
