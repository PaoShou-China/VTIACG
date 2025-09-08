import matplotlib.pyplot as plt
import numpy as np
from simulation import guidance_data



# --- 全局绘图参数设置 ---
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

# 定义常用的绘图常量，提高代码可读性和易维护性
LINE_WIDTH = 2
MARKER_SIZE = 6
MARKER_EVERY = 3000 # <--- 修改为每隔 3000 个数据点绘制一个标记
ALPHA = 0.8
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_LABELSIZE = 12
LEGEND_FONTSIZE = 10
LEGEND_TITLE_FONTSIZE = 12
GRID_LINEWIDTH = 0.7
GRID_ALPHA = 0.6
SUBPLOT_HSPACE = 0.5 # 子图之间的垂直间距

def generate_and_plot(parameter):
    """
    生成并绘制制导律仿真结果。

    Args:
        parameter (list): 传递给 guidance_data 函数的参数列表。
                          预期格式: [x0, y0, theta0, theta_f]
    """
    fig = plt.figure(figsize=(9, 9))
    gs = fig.add_gridspec(3, 1, hspace=SUBPLOT_HSPACE)

    ax1 = fig.add_subplot(gs[0, 0]) # 轨迹图
    ax2 = fig.add_subplot(gs[1, 0]) # 加速度图
    ax3 = fig.add_subplot(gs[2, 0]) # 角度误差图

    # 定义要绘制的制导律及其属性（索引、标签、颜色、线型、标记）
    guidance_laws_to_plot = [
        {'index': 1, 'label': 'VTACG', 'color': '#e6194B', 'linestyle': '-', 'marker': 'o'},  # 红色实线，圆圈标记
        {'index': 2, 'label': 'LIACG', 'color': '#4363d8', 'linestyle': '--', 'marker': 's'}, # 蓝色虚线，方块标记
        {'index': 4, 'label': 'NIACG', 'color': '#f58231', 'linestyle': ':', 'marker': '^'}   # 橙色点线，三角形标记
    ]

    # 遍历并绘制每种制导律的数据
    for law_info in guidance_laws_to_plot:
        law_idx = law_info['index']
        label = law_info['label']
        color = law_info['color']
        linestyle = law_info['linestyle']
        marker = law_info['marker'] # 获取标记样式

        data_VT = guidance_data(parameter, law_idx)

        # 检查数据是否足够，避免索引错误
        if len(data_VT['t']) < 2 or len(data_VT['x']) < 2 or \
           len(data_VT['y']) < 2 or len(data_VT['a']) < 2 or \
           len(data_VT['e_theta']) < 2:
            print(f"Warning: Not enough data for {label}. Skipping plot.")
            continue

        # 绘制轨迹 (x-y 平面)
        ax1.plot(data_VT['x'], data_VT['y'],
                 linewidth=LINE_WIDTH, label=label,
                 color=color, linestyle=linestyle, marker=marker,
                 markersize=MARKER_SIZE, alpha=ALPHA, markevery=MARKER_EVERY) # <--- 应用 markevery

        # 绘制加速度随时间变化
        ax2.plot(data_VT['t'][1:], data_VT['a'][1:],
                 linewidth=LINE_WIDTH, label=label,
                 color=color, linestyle=linestyle, marker=marker,
                 markersize=MARKER_SIZE, alpha=ALPHA, markevery=MARKER_EVERY) # <--- 应用 markevery

        # 绘制角度误差随时间变化
        ax3.plot(data_VT['t'][1:], np.rad2deg(data_VT['e_theta'][1:]) - 360,
                 linewidth=LINE_WIDTH, label=label,
                 color=color, linestyle=linestyle, marker=marker,
                 markersize=MARKER_SIZE, alpha=ALPHA, markevery=MARKER_EVERY) # <--- 应用 markevery

    # 在轨迹图 (ax1) 上添加一条固定的参考线
    ax1.plot([0, 0], [0, 30000], color='gray', linestyle='--', linewidth=1.5, label='Reference Line')


    # 配置每个子图的标题、轴标签、网格和刻度
    plot_configs = [
        {'ax': ax1, 'title': 'Trajectory', 'ylabel': '$y (m)$', 'xlabel': '$x (m)$'},
        {'ax': ax2, 'title': 'Acceleration Over Time', 'ylabel': '$a (m/s^2)$', 'xlabel': '$t (s)$'},
        {'ax': ax3, 'title': 'Angle Error Over Time', 'ylabel': r'$\varepsilon (^{\circ})$', 'xlabel': '$t (s)$'}
    ]

    for config in plot_configs:
        ax = config['ax']
        ax.set_title(config['title'], fontsize=TITLE_FONTSIZE, fontweight='bold')
        ax.set_xlabel(config['xlabel'], fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(config['ylabel'], fontsize=LABEL_FONTSIZE)
        ax.grid(True, which='both', linestyle='-', linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
        ax.tick_params(axis='both', which='major', labelsize=TICK_LABELSIZE)
        ax.legend(title="Guidance Laws", fontsize=LEGEND_FONTSIZE,
                  title_fontsize=LEGEND_TITLE_FONTSIZE, loc='upper right')

    plt.tight_layout(rect=[0.0, 0.05, 1, 1])
    plt.show()


if __name__ == '__main__':
    sigma0 = -3 * np.pi / 4
    rs = 1.0
    lambda0 = np.pi
    theta0 = lambda0 + sigma0
    if theta0 > np.pi:
        theta0 -= 2 * np.pi
    elif theta0 < -np.pi:
        theta0 += 2 * np.pi
    theta_f = -np.pi / 2
    parameter = [rs * np.cos(lambda0 + np.pi), rs * np.sin(lambda0 + np.pi), theta0, theta_f]

    generate_and_plot(parameter)
