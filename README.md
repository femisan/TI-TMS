# 电场包络优化 (Electric Field Envelope Optimization)

本项目实现了导电圆盘上电场包络的计算与优化，使用JAX进行自动微分和优化。

## 项目概述

该项目模拟了导电圆盘上的电场分布，并通过优化电流对的参数（电流大小和电极位置），使特定目标区域的电场强度最大化，同时使非目标区域的电场强度最小化。

主要功能：
- 计算导电圆盘上的电势和电场分布
- 计算两个电场的包络线幅度
- 使用JAX进行自动微分和优化
- 可视化电场分布和优化结果

## 安装依赖

1. 克隆仓库：
```bash
git clone <repository-url>
cd <repository-directory>
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 验证依赖安装：
```bash
python check_dependencies.py
```

## 文件说明

- `ElectricDisk.py`: 导电圆盘电场计算的基础类
- `ElectricDiskOptimizer.py`: 原始优化器实现（使用数值梯度）
- `ElectricDiskJaxOptimizer.py`: 基于JAX的优化器实现（使用自动微分）
- `jax_optimization_demo.py`: JAX优化器的演示脚本
- `check_dependencies.py`: 依赖检查脚本
- `requirements.txt`: 项目依赖列表

## 使用方法

### 运行JAX优化演示

```bash
python jax_optimization_demo.py
```

这将执行以下步骤：
1. 创建优化器实例
2. 确认目标区域
3. 计算初始电场分布
4. 执行JAX优化
5. 可视化优化结果
6. 保存优化结果
7. 与原始优化器结果比较（如果有）

### 自定义优化参数

您可以通过修改 `jax_optimization_demo.py` 中的参数来自定义优化过程：

```python
# 自定义目标区域
target_region = {"type": "circle", "center": (0.5, np.pi/4), "radius": 0.1}

# 自定义初始电流参数
initial_currents = np.array([1.0, np.pi/3, 4*np.pi/3, 
                            1.0, 5*np.pi/3, np.pi/2])

# 自定义优化参数
result = optimizer.jax_optimize(
    initial_currents=initial_currents,
    learning_rate=0.01, 
    num_iterations=300
)
```

## JAX优化器的优势

与原始优化器相比，JAX优化器具有以下优势：

1. **自动微分**：不需要手动计算梯度，减少了错误可能性
2. **JIT编译**：提高了计算性能
3. **函数式编程**：使代码更简洁、更易于理解
4. **更先进的优化器**：使用Adam优化器，收敛更快更稳定

## 示例结果

优化过程将生成以下可视化结果：

1. 目标区域和掩模可视化
2. 优化过程中目标函数值的变化
3. 优化后的电场包络分布
4. 优化后的电流对配置

优化结果将保存在 `jax_optimization_result.pkl` 文件中。
