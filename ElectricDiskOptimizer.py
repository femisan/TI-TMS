import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from ElectricDisk import ElectricDisk

class ElectricDiskOptimizer:
    def __init__(self, R=1.0, sigma=1.0, n_points=100):
        """初始化优化器"""
        self.R = R
        self.sigma = sigma
        self.n_points = n_points
        
        # 定义目标区域（示例：60度扇形区域）
        self.target_angle = (np.pi/3, 2*np.pi/3)  # Hr区域
        self.non_target_angle = (0, np.pi/3)      # Nh区域
        
        # 创建网格
        self.r = np.linspace(0, R, n_points)
        self.theta = np.linspace(0, 2*np.pi, n_points)
        self.R_grid, self.Theta_grid = np.meshgrid(self.r, self.theta)
        self.X = self.R_grid * np.cos(self.Theta_grid)
        self.Y = self.R_grid * np.sin(self.Theta_grid)
        
        # 创建区域掩模
        self.hr_mask = self.create_region_mask(*self.target_angle)
        self.nh_mask = self.create_region_mask(*self.non_target_angle)
    
    def create_region_mask(self, start_angle, end_angle):
        """创建区域掩模"""
        angle_mask = (self.Theta_grid >= start_angle) & (self.Theta_grid <= end_angle)
        radius_mask = self.R_grid > 0  # 排除中心点
        return angle_mask & radius_mask
    
    def safe_normalize(self, x, y):
        """安全归一化向量，避免除以零"""
        norm = np.sqrt(x**2 + y**2)
        mask = norm > 0
        nx, ny = np.zeros_like(x), np.zeros_like(y)
        nx[mask] = x[mask]/norm[mask]
        ny[mask] = y[mask]/norm[mask]
        return nx, ny
    
    def calculate_envelope(self, currents):
        """计算电场包络"""
        # Ensure envelope has the same shape as the grid
        envelope = np.zeros_like(self.R_grid)
        
        # 重新组织电流参数
        current_pairs = []
        for i in range(0, len(currents), 3):
            current_pairs.append((currents[i], currents[i+1], currents[i+2]))
        
        # 创建ElectricDisk实例
        disk = ElectricDisk(R=self.R, sigma=self.sigma, 
                          currents=current_pairs,
                          n_points=self.n_points)
        
        # 获取前两个电流对的电场
        if len(disk.individual_fields) < 2:
            return np.zeros_like(disk.X)
        
        E1x, E1y = disk.individual_fields[0]
        E2x, E2y = disk.individual_fields[1]
        
        # 计算单位法向量场（径向方向）
        nx, ny = self.safe_normalize(disk.X, disk.Y)
        
        # 计算E1+E2和E1-E2
        E_plus_x = E1x + E2x
        E_plus_y = E1y + E2y
        E_minus_x = E1x - E2x
        E_minus_y = E1y - E2y
        
        # 计算点积
        dot_plus = E_plus_x * nx + E_plus_y * ny
        dot_minus = E_minus_x * nx + E_minus_y * ny
        
        # 计算包络
        envelope = np.abs(np.abs(dot_plus) - np.abs(dot_minus))
        return envelope
    
    def objective_function(self, currents):
        """
        目标函数 Ratio = E_area^Hr / E_area^Nh
        """
        envelope = self.calculate_envelope(currents)
        
        # Debugging: Print shapes
        print("Envelope shape:", envelope.shape)
        print("HR mask shape:", self.hr_mask.shape)
        print("NH mask shape:", self.nh_mask.shape)
        
        # 计算区域平均场强
        E_hr = np.mean(envelope[self.hr_mask])
        E_nh = np.mean(envelope[self.nh_mask])
        
        # 防止除以零
        if E_nh < 1e-6:
            return -E_hr
        return -E_hr / E_nh
    
    def optimize(self, initial_currents=None, bounds=None):
        """执行优化"""
        if initial_currents is None:
            initial_currents = [1.0, np.pi/3, 4*np.pi/3, 
                              1.0, 5*np.pi/3, np.pi/2]
        
        if bounds is None:
            # 默认边界：电流[0.1, 2]，角度[0, 2π]
            bounds = [(0.1, 2)] + [(0, 2*np.pi)]*2
            bounds *= len(initial_currents) // 3
        
        result = minimize(self.objective_function,
                         initial_currents,
                         bounds=bounds,
                         method='L-BFGS-B',
                         options={'maxiter': 50})
        
        return result
    
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
                             levels=100, cmap='viridis')
        plt.colorbar(contour, label='Envelope Field Strength')
        
        # 标记目标区域
        wedge_hr = Wedge((0,0), self.R, 
                        *np.degrees(self.target_angle),
                        fc='none', ec='r', lw=2, linestyle='--')
        wedge_nh = Wedge((0,0), self.R,
                        *np.degrees(self.non_target_angle),
                        fc='none', ec='b', lw=2, linestyle='--')
        plt.gca().add_patch(wedge_hr)
        plt.gca().add_patch(wedge_nh)
        
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

# 使用示例
if __name__ == "__main__":
    optimizer = ElectricDiskOptimizer(R=1.0)
    
    # 执行优化
    result = optimizer.optimize()
    print("Optimization result:", result.x)
    print("Final ratio:", -result.fun)
    
    # 可视化结果
    optimizer.visualize_results(result.x)