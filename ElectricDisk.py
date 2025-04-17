import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

class ElectricDisk:
    def __init__(self, R=1.0, sigma=1.0, currents=None, n_points=100):
        """
        初始化导电圆盘的电势和电场计算类
        
        参数:
            R: 圆盘半径 (m)
            sigma: 电导率 (S/m)
            currents: 电流对列表，每个元素为(I, theta1, theta2)
        """
        self.R = R
        self.sigma = sigma
        self.currents = currents if currents is not None else [
            (1.0, 2*np.pi/3, np.pi),
            (1.0, np.pi/3, 0)
        ]
        self.n_points = n_points
        
        # 设置颜色表 (使用tab10颜色映射，最多支持10个电流对)
        self.color_table = plt.cm.tab10(np.linspace(0, 1, len(self.currents)))
        
        # 创建网格
        self.r = np.linspace(0, R, 200)
        self.theta = np.linspace(0, 2*np.pi, 200)
        self.R_grid, self.Theta_grid = np.meshgrid(self.r, self.theta)

        self.X = self.R_grid * np.cos(self.Theta_grid)
        self.Y = self.R_grid * np.sin(self.Theta_grid)

        self.U = self.calculate_potential()
        self.Ex, self.Ey = self.calculate_electric_field()
        self.individual_fields = self.calculate_individual_fields()
        self.envelope = self.calculate_envelope()
        
    def calculate_potential(self, N_terms=100):
        """
        计算圆盘上的总电势分布
        
        参数:
            N_terms: 傅里叶级数展开的项数
            
        返回:
            电势分布数组
        """
        r = self.R_grid
        theta = self.Theta_grid
        U_total = np.zeros_like(r)
        
        for I, theta1, theta2 in self.currents:
            U = np.zeros_like(r)
            for n in range(1, N_terms + 1):
                An = I / (self.sigma * n * np.pi * self.R**n) * (np.cos(n * theta1) - np.cos(n * theta2))
                Bn = I / (self.sigma * n * np.pi * self.R**n) * (np.sin(n * theta1) - np.sin(n * theta2))
                U += r**n * (An * np.cos(n * theta) + Bn * np.sin(n * theta))
            U_total += U
        
        return U_total
    
    def calculate_individual_fields(self, N_terms=100):
        """
        计算每个电流对单独产生的电场分布
        
        返回:
            列表，包含每个电流对的(Ex, Ey)电场分量
        """
        individual_fields = []
        r = self.R_grid
        theta = self.Theta_grid
        
        for I, theta1, theta2 in self.currents:
            # 计算单个电流对的电势
            U = np.zeros_like(r)
            for n in range(1, N_terms + 1):
                An = I / (self.sigma * n * np.pi * self.R**n) * (np.cos(n * theta1) - np.cos(n * theta2))
                Bn = I / (self.sigma * n * np.pi * self.R**n) * (np.sin(n * theta1) - np.sin(n * theta2))
                U += r**n * (An * np.cos(n * theta) + Bn * np.sin(n * theta))
            
            # 计算单个电流对的电场
            dr = self.r[1] - self.r[0]
            dtheta = self.theta[1] - self.theta[0]
            
            dU_dr = np.gradient(U, dr, axis=1)
            dU_dtheta = np.gradient(U, dtheta, axis=0)
            
            Er = -dU_dr
            Etheta = np.zeros_like(self.R_grid)
            mask = self.R_grid > 0
            Etheta[mask] = -dU_dtheta[mask] / self.R_grid[mask]
            
            Ex = Er * np.cos(self.Theta_grid) - Etheta * np.sin(self.Theta_grid)
            Ey = Er * np.sin(self.Theta_grid) + Etheta * np.cos(self.Theta_grid)
            
            individual_fields.append((Ex, Ey))
        
        return individual_fields
    
    def calculate_electric_field(self):
        """计算总电场强度分布 (E = -∇U)"""
        dr = self.r[1] - self.r[0]
        dtheta = self.theta[1] - self.theta[0]
        
        dU_dr = np.gradient(self.U, dr, axis=1)
        dU_dtheta = np.gradient(self.U, dtheta, axis=0)
        
        Er = -dU_dr
        Etheta = np.zeros_like(self.R_grid)
        mask = self.R_grid > 0
        Etheta[mask] = -dU_dtheta[mask] / self.R_grid[mask]
        
        Ex = Er * np.cos(self.Theta_grid) - Etheta * np.sin(self.Theta_grid)
        Ey = Er * np.sin(self.Theta_grid) + Etheta * np.cos(self.Theta_grid)
        
        return Ex, Ey
    
    def plot_individual_fields(self, stride=10):
        """绘制每个电流对的独立电场分布"""
        X = self.R_grid * np.cos(self.Theta_grid)
        Y = self.R_grid * np.sin(self.Theta_grid)
        
        fig, axes = plt.subplots(1, len(self.currents), figsize=(15, 6))
        
        for i, ((Ex, Ey), (I, theta1, theta2)) in enumerate(zip(self.individual_fields, self.currents)):
            ax = axes[i] if len(self.currents) > 1 else axes
            X_sub = X[::stride, ::stride]
            Y_sub = Y[::stride, ::stride]
            Ex_sub = Ex[::stride, ::stride]
            Ey_sub = Ey[::stride, ::stride]
            
            ax.quiver(X_sub, Y_sub, Ex_sub, Ey_sub)
            ax.set_title(f'Current Pair {i+1}\nI={I}A, θ₁={theta1:.2f}, θ₂={theta2:.2f}')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.axis('equal')
        
        plt.tight_layout()
        plt.show()
    
    def plot_combined_field(self, stride=10):
        """绘制总电场分布，不同电流对使用不同颜色"""
        X = self.R_grid * np.cos(self.Theta_grid)
        Y = self.R_grid * np.sin(self.Theta_grid)
        
        plt.figure(figsize=(8, 6))
        
        # 绘制总电势分布
        plt.contourf(X, Y, self.U, levels=100, cmap='jet', alpha=0.7)
        plt.colorbar(label='Potential U (V)')
        
        # 绘制每个电流对的电场（不同颜色）
        for i, (Ex, Ey) in enumerate(self.individual_fields):
            X_sub = X[::stride, ::stride]
            Y_sub = Y[::stride, ::stride]
            Ex_sub = Ex[::stride, ::stride]
            Ey_sub = Ey[::stride, ::stride]
            
            # 使用颜色表中对应的颜色
            plt.quiver(X_sub, Y_sub, Ex_sub, Ey_sub, 
                      color=self.color_table[i],
                      label=f'Current Pair {i+1}',
                      alpha=0.8)
        
        plt.title('Combined Electric Field and Potential Distribution')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()
        plt.axis('equal')
        plt.show()
    
    def plot3D(self, step=5, potential_cmap='viridis', field_color='r'):
        """绘制3D总电势曲面和总电场矢量"""
        X = self.R_grid * np.cos(self.Theta_grid)
        Y = self.R_grid * np.sin(self.Theta_grid)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, self.U, 
                             cmap=potential_cmap, 
                             alpha=0.8,
                             rstride=1,
                             cstride=1,
                             linewidth=0,
                             antialiased=False)
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Potential U (V)')
        
        ax.quiver(X[::step, ::step], 
                 Y[::step, ::step], 
                 self.U[::step, ::step],
                 self.Ex[::step, ::step], 
                 self.Ey[::step, ::step], 
                 np.zeros_like(self.Ex[::step, ::step]),
                 length=0.1, 
                 color=field_color, 
                 normalize=True,
                 arrow_length_ratio=0.3)
        
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('Potential U (V)')
        ax.set_title('3D Combined Potential & Electric Field')
        plt.tight_layout()
        plt.show()

    def calculate_envelope(self):
        """
        按公式计算两个电场的包络线幅度
        """
        if len(self.individual_fields) < 2:
            return np.zeros_like(self.R_grid)

        E1x, E1y = self.individual_fields[0]
        E2x, E2y = self.individual_fields[1]

        # 计算模长
        E1_abs = np.sqrt(E1x**2 + E1y**2)
        E2_abs = np.sqrt(E2x**2 + E2y**2)

        # 计算 cosα
        dot = E1x * E2x + E1y * E2y
        E1_abs_safe = np.where(E1_abs == 0, 1e-12, E1_abs)
        E2_abs_safe = np.where(E2_abs == 0, 1e-12, E2_abs)
        cos_alpha = dot / (E1_abs_safe * E2_abs_safe)

        # 计算 |E1-E2| 及其模长
        dEx = E1x - E2x
        dEy = E1y - E2y
        dE_abs = np.sqrt(dEx**2 + dEy**2) + 1e-12  # 防止除零

        # 计算二维叉积
        cross1 = E2x * dEy - E2y * dEx  # E2 × (E1 - E2)
        cross2 = E1x * (-dEy) - E1y * (-dEx)  # E1 × (E2 - E1)
        cross1_abs = np.abs(cross1)
        cross2_abs = np.abs(cross2)

        # 条件判断
        cond1 = (E1_abs > E2_abs) & (E2_abs < E1_abs * cos_alpha)
        cond2 = (E1_abs > E2_abs) & (E2_abs >= E1_abs * cos_alpha)
        cond3 = (E1_abs <= E2_abs) & (E1_abs < E2_abs * cos_alpha)
        cond4 = (E1_abs <= E2_abs) & (E1_abs >= E2_abs * cos_alpha)

        # 根据条件计算包络幅度
        envelope = np.zeros_like(E1_abs)
        envelope = np.where(cond1, 2 * E2_abs, envelope)
        envelope = np.where(cond2, 2 * cross1_abs / dE_abs, envelope)
        envelope = np.where(cond3, 2 * E1_abs, envelope)
        envelope = np.where(cond4, 2 * cross2_abs / dE_abs, envelope)

        return envelope

    def plot_envelope(self):
        """绘制电场包络分布"""
        plt.figure(figsize=(8, 6))
        
        # 绘制包络场强
        contour = plt.contourf(self.X, self.Y, self.envelope, 
                        levels=100, cmap='RdGy_r', alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(contour, label='Envelope Field Strength |E_AM|')

        
        # 标记电极位置
        for i, (_, theta1, theta2) in enumerate(self.currents):
            x1, y1 = self.R * np.cos(theta1), self.R * np.sin(theta1)
            x2, y2 = self.R * np.cos(theta2), self.R * np.sin(theta2)
            plt.scatter([x1, x2], [y1, y2], 
                       color=self.color_table[i],
                       marker='o' if i == 0 else 's',
                       s=100, 
                       label=f'Pair {i+1} Electrodes')
        
        plt.title('Electric Field Envelope Distribution')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()
        plt.axis('equal')
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 定义两对电流
    currents = [
        (1.0, 2*np.pi/3, np.pi),  # 第一对电流
        (1.0, np.pi/3, 0)         # 第二对电流
    ]
    
    disk = ElectricDisk(R=1.0, sigma=1.0, currents=currents)
    
    # 绘制各个电流对的独立电场
    disk.plot_individual_fields(stride=8)
    
    # 绘制总电场和电势
    disk.plot_combined_field(stride=8)
    
    # 绘制3D总电势和电场
    disk.plot3D(step=8, field_color='r')

    # 绘制电场包络
    disk.plot_envelope()
