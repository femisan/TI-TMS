import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from scipy.ndimage import zoom
from ElectricDisk import ElectricDisk
import pickle  # 导入pickle模块

class ElectricDiskOptimizer:
    def __init__(self, R=1.0, sigma=1.0, n_points=200, target_region=None):
        """初始化优化器"""
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

    def resize_mask(self, mask, target_shape):
        """Resize a mask to match the target shape."""
        zoom_factors = (
            target_shape[0] / mask.shape[0],
            target_shape[1] / mask.shape[1]
        )
        return zoom(mask, zoom_factors, order=0)  # Use nearest-neighbor interpolation
    
    def safe_normalize(self, x, y):
        """安全归一化向量，避免除以零"""
        norm = np.sqrt(x**2 + y**2)
        mask = norm > 0
        nx, ny = np.zeros_like(x), np.zeros_like(y)
        nx[mask] = x[mask]/norm[mask]
        ny[mask] = y[mask]/norm[mask]
        return nx, ny
    
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
        """
        envelope = self.calculate_envelope(currents)
        
        # Debugging: Print shapes
        # print("Envelope shape:", envelope.shape)
        # print("HR mask shape:", self.hr_mask.shape)
        # print("NH mask shape:", self.nh_mask.shape)
        
        # 计算区域平均场强
        E_hr = np.mean(envelope[self.hr_mask])
        E_nh = np.mean(envelope[self.nh_mask])
        
        # 防止除以零
        if E_nh < 1e-6:
            ratio = -E_hr
        else:
            ratio = -E_hr / E_nh
        
        # 实时打印目标函数值
        print(f"Objective function value: {ratio:.6f}")
        
        return ratio
    
    def optimize(self, initial_currents=None, bounds=None):
        """执行优化"""
        if initial_currents is None:
            initial_currents = [1.0, np.pi/3, 4*np.pi/3, 
                              1.0, 5*np.pi/3, np.pi/2]
        
        if bounds is None:
            # 默认边界：电流[0.1, 2]，角度[0, 2π]
            bounds = [(0.1, 2)] + [(0, 2*np.pi)]*2
            bounds *= len(initial_currents) // 3
        
        # 定义回调函数
        def callback(xk):
            current_value = self.objective_function(xk)
            print(f"Current objective value during optimization: {current_value:.6f}")
        
        result = minimize(self.objective_function,
                         initial_currents,
                         bounds=bounds,
                         method='L-BFGS-B',
                         callback=callback,  # 添加回调函数
                         options={'maxiter': 1000, 'disp': True})
        
        return result
    
    def sgd_optimize(self, initial_currents=None, bounds=None, learning_rate=0.01, max_iter=1000):
        """使用随机梯度下降 (SGD) 优化"""
        if initial_currents is None:
            initial_currents = [1.0, np.pi/3, 4*np.pi/3, 
                                1.0, 5*np.pi/3, np.pi/2]
        
        if bounds is None:
            # 默认边界：电流[0.1, 2]，角度[0, 2π]
            bounds = [(0.1, 2)] + [(0, 2*np.pi)] * 2
            bounds *= len(initial_currents) // 3

        # 初始化参数
        currents = np.array(initial_currents)
        
        for iteration in range(max_iter):
            # 计算目标函数值
            loss = self.objective_function(currents)
            
            # 打印当前迭代信息
            print(f"Iteration {iteration + 1}/{max_iter}, Loss: {loss:.6f}")
            
            # 计算梯度 (数值梯度)
            gradients = np.zeros_like(currents)
            epsilon = 1e-6
            for i in range(len(currents)):
                currents[i] += epsilon
                loss_plus = self.objective_function(currents)
                currents[i] -= 2 * epsilon
                loss_minus = self.objective_function(currents)
                currents[i] += epsilon
                gradients[i] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # 更新参数 (梯度下降)
            currents -= learning_rate * gradients
            
            # 应用边界约束
            for i, (lower, upper) in enumerate(bounds):
                currents[i] = np.clip(currents[i], lower, upper)
        
        return currents
    
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

    def confirm_target_region_3d(self):
        """在3D表面图中显示 hr_mask 和 nh_mask 区域"""
        from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 创建一个高度矩阵，用于区分不同的掩模
        z = np.zeros_like(self.R_grid)
        z[self.hr_mask] = 1  # hr_mask 区域高度为 1
        z[self.nh_mask] = 0  # nh_mask 区域高度为 0.5

        # 绘制3D表面
        surf = ax.plot_surface(self.X, self.Y, z, cmap='coolwarm', edgecolor='k', alpha=0.8)

        # 绘制目标区域边界
        self.draw_target_region_3d(ax)

        # 设置图形属性
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Mask Value")
        ax.set_title("3D Visualization of HR and NH Masks")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Mask Value")
        plt.show()

    def draw_target_region_3d(self, ax):
        """在3D图中绘制目标区域"""
        if self.target_region["type"] == "circle":
            r_center, theta_center = self.target_region["center"]
            x_center = r_center * np.cos(theta_center)
            y_center = r_center * np.sin(theta_center)

            # 绘制目标区域的圆形边界
            theta = np.linspace(0, 2 * np.pi, 100)
            x_circle = x_center + self.target_region["radius"] * np.cos(theta)
            y_circle = y_center + self.target_region["radius"] * np.sin(theta)
            z_circle = np.ones_like(x_circle)  # 设置圆的高度
            ax.plot(x_circle, y_circle, z_circle, color='r', linestyle='--', linewidth=2, label='Target Region')
        elif self.target_region["type"] == "sector":
            start_angle, end_angle = self.target_region["angles"]
            theta = np.linspace(start_angle, end_angle, 100)
            r = self.R
            x_sector = r * np.cos(theta)
            y_sector = r * np.sin(theta)
            z_sector = np.ones_like(x_sector)
            ax.plot(x_sector, y_sector, z_sector, color='r', linestyle='--', linewidth=2, label='Target Region')
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

# filepath: /Users/nero/Development/Personal/TI-TMS/ElectricDiskOptimizer.py
if __name__ == "__main__":
    # 自定义目标区域: 圆形 r=0.5, theta=pi/4, 半径=0.1
    target_region = {"type": "circle", "center": (0.5, np.pi/4), "radius": 0.1}
    
    optimizer = ElectricDiskOptimizer(R=1.0, target_region=target_region)

    optimizer.confirm_target_region_3d()
    # # 执行优化
    # result = optimizer.optimize()
    # print("Optimization result:", result.x)
    # print("Final ratio:", -result.fun)
    # optimizer.visualize_results(result.x)

    # 执行随机梯度下降优化
    result = optimizer.sgd_optimize(learning_rate=0.01, max_iter=500)
    print("Optimized currents:", result)
    optimizer.visualize_results(result)
    
    # 保存优化结果到pickle文件
    with open("optimization_result.pkl", "wb") as f:
        pickle.dump(result, f)
    print("Optimization result saved to 'optimization_result.pkl'")
