import numpy as np
import matplotlib.pyplot as plt
from ElectricDiskJaxOptimizer import ElectricDiskJaxOptimizer
import pickle

def main():
    """
    演示如何使用JAX优化器进行电场包络优化
    """
    print("=== JAX自动微分与优化演示 ===")
    
    # 自定义目标区域: 圆形 r=0.5, theta=pi/4, 半径=0.1
    target_region = {"type": "circle", "center": (0.5, np.pi/4), "radius": 0.1}
    
    # 创建优化器实例
    print("\n创建JAX优化器实例...")
    optimizer = ElectricDiskJaxOptimizer(R=1.0, target_region=target_region)
    
    # 确认目标区域
    print("\n确认目标区域...")
    optimizer.confirm_target_region()
    
    # 设置初始电流参数
    initial_currents = np.array([1.0, np.pi/3, 4*np.pi/3, 
                                1.0, 5*np.pi/3, np.pi/2])
    
    print("\n初始电流参数:")
    for i in range(0, len(initial_currents), 3):
        print(f"电流对 {i//3 + 1}: I={initial_currents[i]:.2f}A, θ₁={initial_currents[i+1]:.2f}, θ₂={initial_currents[i+2]:.2f}")
    
    # 计算初始目标函数值
    initial_objective = optimizer.objective_function(initial_currents)
    print(f"\n初始目标函数值: {initial_objective:.6f}")
    
    # 可视化初始结果
    print("\n可视化初始电场分布...")
    optimizer.visualize_results(initial_currents)
    
    # 执行JAX优化
    print("\n开始JAX优化过程...")
    result = optimizer.jax_optimize(
        initial_currents=initial_currents,
        learning_rate=0.01, 
        num_iterations=300
    )
    
    # 输出优化结果
    print("\n优化后的电流参数:")
    for i in range(0, len(result), 3):
        print(f"电流对 {i//3 + 1}: I={result[i]:.2f}A, θ₁={result[i+1]:.2f}, θ₂={result[i+2]:.2f}")
    
    # 计算优化后的目标函数值
    final_objective = optimizer.objective_function(result)
    print(f"\n优化后的目标函数值: {final_objective:.6f}")
    print(f"改进比例: {(initial_objective - final_objective) / abs(initial_objective) * 100:.2f}%")
    
    # 可视化优化结果
    print("\n可视化优化后的电场分布...")
    optimizer.visualize_results(result)
    
    # 保存优化结果到pickle文件
    with open("jax_optimization_result.pkl", "wb") as f:
        pickle.dump(result, f)
    print("\n优化结果已保存到 'jax_optimization_result.pkl'")
    
    # 比较与原始优化器的结果
    try:
        with open("optimization_result.pkl", "rb") as f:
            original_result = pickle.load(f)
        
        print("\n比较JAX优化器与原始优化器的结果:")
        original_objective = optimizer.objective_function(original_result)
        print(f"原始优化器目标函数值: {original_objective:.6f}")
        print(f"JAX优化器目标函数值: {final_objective:.6f}")
        print(f"改进比例: {(original_objective - final_objective) / abs(original_objective) * 100:.2f}%")
    except:
        print("\n未找到原始优化器结果文件，跳过比较")

if __name__ == "__main__":
    main()
