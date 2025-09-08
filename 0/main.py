import numpy as np
import matplotlib.pyplot as plt

# --- 1. 全局绘图参数设置 (rcParams) ---
# 设置 Matplotlib 的默认参数，以获得更专业的图表外观
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 设置字体为 Times New Roman
    'mathtext.fontset': 'stix',       # 数学公式字体
    'axes.facecolor': '#ffffff',       # 坐标系背景色为白色
    'figure.facecolor': '#ffffff',     # 图形背景色为白色
    'axes.edgecolor': '#333333',       # 坐标轴边框颜色
    'grid.color': '#cccccc',          # 网格线颜色
    'grid.linestyle': '-',            # 网格线样式 (统一为实线，更简洁)
    'grid.linewidth': 0.7,            # 网格线宽度
    'grid.alpha': 0.6,                # 网格线透明度
    'xtick.direction': 'in',          # x轴刻度线方向向内
    'ytick.direction': 'in',          # y轴刻度线方向向内
    'axes.labelcolor': '#333333',     # 轴标签颜色
    'xtick.color': '#333333',         # x轴刻度标签颜色
    'ytick.color': '#333333',         # y轴刻度标签颜色
    'legend.frameon': True,           # 图例显示边框
    'legend.edgecolor': '#cccccc',    # 图例边框颜色
    'legend.facecolor': 'white',      # 图例背景色
    'legend.fancybox': True,          # 图例圆角
    'legend.shadow': False,           # 图例阴影
})

# --- 2. 定义常用的绘图常量，提高代码可读性和易维护性 ---
LINE_WIDTH = 2
MARKER_SIZE = 100  # 标记尺寸保持不变，已足够大
ALPHA = 0.8       # 标记透明度
CMAP_NAME = 'viridis' # 颜色映射名称，可以尝试 'plasma', 'cividis', 'magma'
DATA_THINNING_INTERVAL = 1000 # 数据稀疏化间隔保持不变

# --- 字体大小调整 (保持上次的放大值) ---
# TITLE_FONTSIZE = 22  # 标题已移除，此常量不再直接使用，但保留以防未来需要
LABEL_FONTSIZE = 20  # 轴标签字体增大
TICK_LABELSIZE = 18  # 刻度标签字体增大
LEGEND_FONTSIZE = 16 # 图例字体增大
CBAR_LABEL_FONTSIZE = 18 # 颜色条标签字体增大

GRID_LINESTYLE = '-'
GRID_LINEWIDTH = 0.7
GRID_ALPHA = 0.6

REFERENCE_LINE_COLOR = 'gray'
REFERENCE_LINE_STYLE = '--'
REFERENCE_LINE_WIDTH = 1.5
REFERENCE_LINE_LABEL = 'Reference Line' # 为参考线添加标签

# --- 3. 仿真模型类 (保持不变，但为了完整性包含在内) ---
dt = 0.01 # 时间步长

class Missile:
    def __init__(self, position, velocity, r, N=3):
        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)
        self.r = r
        # 确保速度不为零，避免除以零
        vel_norm = np.linalg.norm(self.vel)
        self.max_acc = vel_norm ** 2 / r if vel_norm > 1e-6 else 1e6 # 避免除以零
        self.N = N

    def calculate_bearing(self, target_pos):
        dx = target_pos[0] - self.pos[0]
        dy = target_pos[1] - self.pos[1]
        return np.arctan2(dy, dx)

    def run(self, target_pos, target_vel):
        bearing_to_target = self.calculate_bearing(target_pos)
        R = np.linalg.norm(target_pos - self.pos)
        # 避免除以零
        if R < 1e-8:
            bearing_rate = 0.0
        else:
            bearing_rate = ((target_vel[0] - self.vel[0]) * np.sin(bearing_to_target) -
                            (target_vel[1] - self.vel[1]) * np.cos(bearing_to_target)) / R

        acc_hat = self.N * bearing_rate * np.linalg.norm(self.vel)
        acc = np.clip(acc_hat, -self.max_acc, self.max_acc)

        # 确保速度不为零，避免除以零
        vel_norm = np.linalg.norm(self.vel)
        if vel_norm < 1e-8:
            acc_x = 0.0
            acc_y = 0.0
        else:
            acc_x = acc * self.vel[1] / vel_norm
            acc_y = -acc * self.vel[0] / vel_norm

        self.vel += np.array([acc_x, acc_y]) * dt
        self.pos += self.vel * dt

        return np.copy(self.pos), np.copy(self.vel)


class FakeTarget:
    def __init__(self, attack_angle, k1, k2, k3, r):
        self.attack_angle = attack_angle
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.r = r
        self.pos = np.array([0.0, 0.0], dtype=float) # 确保是浮点数

    def run(self, missle_pos, missle_vel):
        R = np.linalg.norm(missle_pos)
        theta = np.arctan2(-missle_vel[1], -missle_vel[0])
        d_theta = (self.attack_angle - theta + np.pi) % (2 * np.pi) - np.pi
        beta = self.attack_angle + self.k1 * d_theta
        L = R * self.k2 * (1 + self.k3 * d_theta**2)
        pos = np.array([L * np.cos(beta), L * np.sin(beta)])
        self.pos = pos

        return np.copy(self.pos), np.array([0.0, 0.0]) # 假目标速度为零

# --- 4. 仿真运行函数 ---
def run_simulation(initial_missile_pos, initial_missile_vel, r, N,
                   attack_angle, k1, k2, k3, total_time, dt, data_thinning_interval):
    """
    运行导弹和假目标的仿真，并收集轨迹数据。

    Args:
        initial_missile_pos (list): 导弹初始位置 [x, y]。
        initial_missile_vel (list): 导弹初始速度 [vx, vy]。
        r (float): 导弹最大加速度相关参数。
        N (int): 比例导航常数。
        attack_angle (float): 假目标攻击角度。
        k1, k2, k3 (float): 假目标运动参数。
        total_time (float): 仿真总时长。
        dt (float): 仿真时间步长。
        data_thinning_interval (int): 每隔多少个时间步保存一次数据。

    Returns:
        tuple: (missile_positions, fake_target_positions)
               missile_positions (np.array): 导弹轨迹点数组。
               fake_target_positions (np.array): 假目标轨迹点数组。
    """
    missile = Missile(position=initial_missile_pos, velocity=initial_missile_vel, r=r, N=N)
    fake_target = FakeTarget(attack_angle=attack_angle, k1=k1, k2=k2, k3=k3, r=r)

    missile_positions = [np.copy(missile.pos)]
    fake_target_positions = [np.copy(fake_target.pos)]

    time_steps = int(total_time / dt)
    min_dis = np.linalg.norm(missile.pos) # 记录最小距离，用于判断是否接近目标

    for i in range(time_steps):
        fake_target_pos, fake_target_vel = fake_target.run(missile.pos, missile.vel)
        missile_pos, missile_vel = missile.run(fake_target_pos, fake_target_vel)

        # 每隔指定时间步保存一次位置，实现数据稀疏化
        if i % data_thinning_interval == 0:
            missile_positions.append(np.copy(missile_pos))
            fake_target_positions.append(np.copy(fake_target_pos))

        current_dis = np.linalg.norm(missile.pos)
        # 如果距离开始增大且已经很接近目标，则认为仿真结束
        # 这里的判断条件可以根据实际需求调整
        if current_dis > min_dis and min_dis < 100: # 假设100m内算接近
            print(f"Simulation stopped at time step {i*dt:.2f}s due to close approach and divergence.")
            break
        else:
            min_dis = min(min_dis, current_dis) # 更新最小距离

    return np.array(missile_positions), np.array(fake_target_positions)

# --- 5. 绘图函数 ---
def plot_trajectories(missile_positions, fake_target_positions, dt, data_thinning_interval):
    """
    绘制导弹和假目标的轨迹图。

    Args:
        missile_positions (np.array): 导弹轨迹点数组。
        fake_target_positions (np.array): 假目标轨迹点数组。
        dt (float): 仿真时间步长。
        data_thinning_interval (int): 数据稀疏化间隔。
    """
    fig, ax = plt.subplots(figsize=(10, 8)) # 调整图表尺寸，使其更宽更高

    # 创建颜色映射和归一化
    # 颜色条表示的是实际时间，所以需要乘以 dt 和 data_thinning_interval
    # 确保颜色条的范围覆盖所有数据点的时间
    max_time_m = (len(missile_positions) - 1) * dt * data_thinning_interval
    max_time_f = (len(fake_target_positions) - 1) * dt * data_thinning_interval
    max_plot_time = max(max_time_m, max_time_f)
    norm = plt.Normalize(0, max_plot_time)
    cmap = plt.get_cmap(CMAP_NAME)

    # 导弹轨迹 (使用散点图表示时间梯度)
    # c 参数使用实际时间，而不是简单的索引
    sc_m = ax.scatter(missile_positions[:, 0], missile_positions[:, 1],
                      c=np.arange(len(missile_positions)) * dt * data_thinning_interval,
                      cmap=cmap, norm=norm, s=MARKER_SIZE, marker='o', label='Missile', alpha=ALPHA,
                      edgecolors='black', linewidths=0.5) # 添加黑色边框，使标记更显眼

    # 假目标轨迹 (使用散点图表示时间梯度)
    sc_f = ax.scatter(fake_target_positions[:, 0], fake_target_positions[:, 1],
                      c=np.arange(len(fake_target_positions)) * dt * data_thinning_interval,
                      cmap=cmap, norm=norm, s=MARKER_SIZE, marker='^', label='Virtual Target', alpha=ALPHA,
                      edgecolors='black', linewidths=0.5) # 添加黑色边框

    # 添加参考线 (例如，目标原点或边界)
    ax.plot([0, 0], [0, 60000], color=REFERENCE_LINE_COLOR, linestyle=REFERENCE_LINE_STYLE,
            linewidth=REFERENCE_LINE_WIDTH, label=REFERENCE_LINE_LABEL)

    # 添加颜色条，并缩短其长度
    cbar = fig.colorbar(sc_m, ax=ax, label='Time (s)', orientation='vertical', pad=0.02, shrink=0.8) # <--- 缩短颜色条
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)
    cbar.set_label('Time (s)', fontsize=CBAR_LABEL_FONTSIZE)

    # 设置轴标签
    ax.set_xlabel('$x$ (m)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('$y$ (m)', fontsize=LABEL_FONTSIZE)
    # ax.set_title('Missile and Virtual Target Trajectories with Time Gradient', fontsize=TITLE_FONTSIZE, fontweight='bold') # <--- 移除标题

    # 设置 X 和 Y 轴刻度标签的字体大小
    ax.tick_params(axis='x', labelsize=TICK_LABELSIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABELSIZE)

    # 保证 X Y 轴的长度比例 1:1
    ax.set_aspect('equal', adjustable='box')

    # 设置网格
    ax.grid(True, which='both', linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)

    # 显示图例
    ax.legend(loc='upper right', fontsize=LEGEND_FONTSIZE) # 图例位置保持右上角

    # 自动调整子图参数以填充整个图形区域
    plt.tight_layout()

    # 显示图表
    plt.show()

# --- 6. 主程序执行块 ---
if __name__ == '__main__':
    # 仿真参数
    attack_angle = 0.5 * np.pi
    r = 1.0
    N = 4
    initial_missile_pos = [50000.0, 0.0]
    initial_missile_vel = [300.0, 300.0]
    k1, k2, k3 = 0.1, 0.4, 0.2
    total_time = 1000 # 仿真总时长

    # 运行仿真并获取数据
    missile_positions, fake_target_positions = run_simulation(
        initial_missile_pos, initial_missile_vel, r, N,
        attack_angle, k1, k2, k3, total_time, dt, DATA_THINNING_INTERVAL
    )

    # 绘制轨迹图
    plot_trajectories(missile_positions, fake_target_positions, dt, DATA_THINNING_INTERVAL)
