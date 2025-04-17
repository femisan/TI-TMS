import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import optax
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from ElectricDisk import ElectricDisk
import pickle

class ElectricDiskJaxOptimizer:
    def __init__(self, R=1.0, sigma=1.0, n_points=200, target_region=None):
        """初始化JAX优化器"""
        self.R = R
        self.sigma = sigma
        self.n_points = n_points
        
        # 创建网格
        self.r = np.linspace(0, R, n_points)
        self.theta = np.linspace(0, 2 * np.pi, n_points)
        self.R_grid, self.Theta_grid = np.meshgrid(self.r, self.theta)
        self.X = self.R_grid * np.cos(self.Theta_grid)
        self.Y = self.R_grid * np.sin(self.Theta_grid)
        
        # 设置目标区域 (默认: 圆形 r=0.5, theta=pi/4, 半径=0.1)
        if target_region is None:
            target_region = {"type": "circle", "center": (0.5, np.pi/4), "radius": 0.1}
        self.target_region = target_region
        
        # 创建目标区域掩模
        self.hr_mask = self.create_region_mask(self.target_region)
        
        # 创建非目标区域掩模
        self.nh_mask = ~self.hr_mask
        print(f"Number of True values in hr_mask: {np.sum(self.hr_mask)}")
        print(f"Number of True values in nh_mask: {np.sum(self.nh_mask)}")

    def create_region_mask(self, region):
        """创建目标区域掩模"""
        if region["type"] == "circle":
            r_center, theta_center = region["center"]
            x_center = r_center * np.cos(theta_center)
            y_center = r_center * np.sin(theta_center)
            distance = np.sqrt((self.X - x_center) ** 2 + (self.Y - y_center) ** 2)
            mask = distance <= region["radius"]
            
            return mask
        
        elif region["type"] == "sector":
            start_angle, end_angle = region["angles"]
            angle_mask = (self.Theta_grid >= start_angle) & (self.Theta_grid <= end_angle)
            radius_mask = self.R_grid > 0
            mask = angle_mask & radius_mask

            return mask
        
        else:
            raise ValueError("Unsupported region type. Use 'circle' or 'sector'.")

    def calculate_envelope(self, currents):
        """
        计算电场包络，利用 ElectricDisk 类中的 calculate_envelope 方法
        """
        # 重新组织电流参数
        current_pairs = []
        for i in range(0, len(currents), 3):
            current_pairs.append((currents[i], currents[i+1], currents[i+2]))
        
        # 创建 ElectricDisk 实例
        disk = ElectricDisk(R=self.R, sigma=self.sigma, 
                            currents=current_pairs,
                            n_points=self.n_points)
        
        # 调用 ElectricDisk 的 calculate_envelope 方法
        envelope = disk.calculate_envelope()
        
        return envelope
    
    def objective_function(self, currents):
        """
        目标函数 Ratio = E_area^Hr / E_area^Nh
        
        这个函数将被JAX用于自动微分
        """
        envelope = self.calculate_envelope(currents)
        
        # 计算区域平均场强
        E_hr = jnp.mean(envelope[self.hr_mask])
        E_nh = jnp.mean(envelope[self.nh_mask])
        
        # 防止除以零
        E_nh_safe = jnp.where(E_nh < 1e-6, 1e-6, E_nh)
        ratio = -E_hr / E_nh_safe
        
        return ratio
    
    @jit
    def jit_objective(self, currents):
        """JIT编译的目标函数，用于加速计算"""
        return self.objective_function(currents)
    
    def jax_optimize(self, initial_currents=None, bounds=None, learning_rate=0.01, num_iterations=1000):
        """
        使用JAX的自动微分和optax优化器进行优化
        
        参数:
            initial_currents: 初始电流参数
            bounds: 参数边界，格式为[(min1, max1), (min2, max2), ...]
            learning_rate: 学习率
            num_iterations: 迭代次数
            
        返回:
            优化后的电流参数
        """
        if initial_currents is None:
            initial_currents = np.array([1.0, np.pi/3, 4*np.pi/3, 
                                        1.0, 5*np.pi/3, np.pi/2])
        
        if bounds is None:
            # 默认边界：电流[0.1, 2]，角度[0, 2π]
            bounds = [(0.1, 2)] + [(0, 2*np.pi)]*2
            bounds *= len(initial_currents) // 3
        
        # 将初始参数转换为JAX数组
        params = jnp.array(initial_currents)
        
        # 创建优化器
        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(params)
        
        # 定义梯度函数
        value_and_grad_fn = jit(value_and_grad(self.objective_function))
        
        # 定义更新步骤
        @jit
        def step(params, opt_state):
            value, grads = value_and_grad_fn(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # 应用边界约束
            for i, (lower, upper) in enumerate(bounds):
                params = params.at[i].set(jnp.clip(params[i], lower, upper))
                
            return params, opt_state, value
        
        # 执行优化
        values = []
        for i in range(num_iterations):
            params, opt_state, value = step(params, opt_state)
            values.append(float(value))
            
            if i % 50 == 0:
                print(f"Iteration {i}, Objective: {value:.6f}")
        
        # 绘制优化过程
        plt.figure(figsize=(10, 5))
        plt.plot(values)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Optimization Progress')
        plt.grid(True)
        plt.show()
        
        return np.array(params)
    
    def visualize_results(self, currents):
        """可视化优化结果"""
        current_pairs = []
        for i in range(0, len(currents), 3):
            current_pairs.append((currents[i], currents[i+1], currents[i+2]))
        
        envelope = self.calculate_envelope(currents)
        
        plt.figure(figsize=(12, 5))
        
        # 绘制包络分布
        plt.subplot(121)
        contour = plt.contourf(self.X, self.Y, envelope, 
                             levels=100, cmap='RdGy_r', alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(contour, label='Envelope Field Strength')
        
        # 标记目标区域
        ax = plt.gca()
        self.draw_target_region(ax)
        
        # 标记电极位置
        for i, (_, theta1, theta2) in enumerate(current_pairs):
            x1, y1 = self.R * np.cos(theta1), self.R * np.sin(theta1)
            x2, y2 = self.R * np.cos(theta2), self.R * np.sin(theta2)
            plt.scatter([x1, x2], [y1, y2], 
                       color=f'C{i}',
                       marker='o' if i == 0 else 's',
                       s=100,
                       label=f'Pair {i+1}')
        
        plt.title(f'Optimized Field Distribution\nRatio = {-self.objective_function(currents):.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.axis('equal')
        
        # 绘制电流对配置
        plt.subplot(122)
        for i, (I, theta1, theta2) in enumerate(current_pairs):
            plt.plot([theta1, theta2], [I, I], 
                    marker='o', 
                    linestyle='-',
                    label=f'Pair {i+1}: I={I:.2f}A')
        
        plt.xticks(np.linspace(0, 2*np.pi, 5),
                   ['0', 'π/2', 'π', '3π/2', '2π'])
        plt.xlabel('Electrode Position (rad)')
        plt.ylabel('Current (A)')
        plt.title('Current Pair Configuration')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def draw_target_region(self, ax):
        """绘制目标区域"""
        if self.target_region["type"] == "circle":
            r_center, theta_center = self.target_region["center"]
            x_center = r_center * np.cos(theta_center)
            y_center = r_center * np.sin(theta_center)
            circle = plt.Circle((x_center, y_center), self.target_region["radius"], 
                                color='r', fill=False, linestyle='--', linewidth=2, label='Target Region')
            ax.add_patch(circle)
        elif self.target_region["type"] == "sector":
            start_angle, end_angle = np.degrees(self.target_region["angles"])
            wedge = Wedge((0, 0), self.R, start_angle, end_angle, 
                          color='r', fill=False, linestyle='--', linewidth=2, label='Target Region')
            ax.add_patch(wedge)
        else:
            raise ValueError("Unsupported region type. Use 'circle' or 'sector'.")
    
    def confirm_target_region(self):
        """确认目标区域的位置，同时显示 hr_mask 和 nh_mask 区域"""
        plt.figure(figsize=(8, 8))

        # 绘制 hr_mask 区域
        plt.contourf(self.X, self.Y, self.hr_mask.astype(int), levels=[0, 1], colors=['red'], alpha=0.5, label='HR Mask')

        # 绘制 nh_mask 区域
        plt.contourf(self.X, self.Y, self.nh_mask.astype(int), levels=[0, 1], colors=['blue'], alpha=0.3, label='NH Mask')

        # 绘制目标区域
        ax = plt.gca()
        self.draw_target_region(ax)

        # 设置图形属性
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Target Region and Masks (HR and NH)")
        plt.axis("equal")
        plt.legend(['HR Mask', 'NH Mask', 'Target Region'])
        plt.grid(True)
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 自定义目标区域: 圆形 r=0.5, theta=pi/4, 半径=0.1
    target_region = {"type": "circle", "center": (0.5, np.pi/4), "radius": 0.1}
    
    # 创建优化器实例
    optimizer = ElectricDiskJaxOptimizer(R=1.0, target_region=target_region)
    
    # 确认目标区域
    optimizer.confirm_target_region()
    
    # 执行JAX优化
    result = optimizer.jax_optimize(learning_rate=0.01, num_iterations=500)
    print("Optimized currents:", result)
    
    # 可视化优化结果
    optimizer.visualize_results(result)
    
    # 保存优化结果到pickle文件
    with open("jax_optimization_result.pkl", "wb") as f:
        pickle.dump(result, f)
    print("Optimization result saved to 'jax_optimization_result.pkl'")
