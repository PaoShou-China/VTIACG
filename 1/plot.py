import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
from simulation import our_guidance_data  # 假设这个函数已经定义

# 设置全局参数
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix'
})

# 参数定义等其他部分保持不变...
sigma0 = -3 * np.pi / 4
rs = 1.0
lambda0 = np.pi
theta0 = lambda0 + sigma0
if theta0 > np.pi:
    theta0 = theta0 - 2 * np.pi
elif theta0 < -np.pi:
    theta0 = theta0 + 2 * np.pi
theta_f = -np.pi / 2
parameter = [rs * np.cos(lambda0 + np.pi), rs * np.sin(lambda0 + np.pi), theta0, theta_f]

def generate_and_plot(variable_param, fixed_params, range_var, num_points):
    """
    根据给定的可变参数和固定参数生成数据，并绘制结果。
    """
    k_values = np.linspace(range_var[0], range_var[1], num_points)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(k_values.min(), k_values.max())
    scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array([])

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    for idx, k in enumerate(k_values):
        params = fixed_params.copy()
        params[variable_param] = k
        data_VT = our_guidance_data(parameter, **params)
        color = cmap(norm(k))

        ax1.plot(data_VT['x'], data_VT['y'], linewidth=1.5, color=color)
        ax1.plot([0, 0], [0, 30000], color='gray', linestyle='--', linewidth=1.5)
        ax2.plot(data_VT['t'][1:], data_VT['a'][1:], linewidth=1.5, color=color)
        ax3.plot(data_VT['t'][1:], np.rad2deg(data_VT['e_theta'][1:])-360, linewidth=1.5, color=color)

    if variable_param == 'k1':
        variable_param_label = '$k_{\\theta}$'
    elif variable_param == 'k2':
        variable_param_label = '$k_{R}$'
    else:
        variable_param_label = '$k_{\epsilon}$'

    cbar = fig.colorbar(scalar_map, ax=ax1, orientation='horizontal', shrink=0.8, pad=0.1)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(f'{variable_param_label}', fontsize=18)

    pos = ax1.get_position()
    cbar.ax.set_position([pos.x0, pos.y1, pos.width, 0])

    # 添加局部放大图
    axins = inset_axes(ax1, width="40%", height="40%", loc=2, bbox_to_anchor=(0.4, -0.2, 0.6, 0.6),
                       bbox_transform=ax1.transAxes)

    for idx, k in enumerate(k_values):
        params = fixed_params.copy()
        params[variable_param] = k
        data_VT = our_guidance_data(parameter, **params)
        color = cmap(norm(k))
        axins.plot(data_VT['x'], data_VT['y'], linewidth=1.5, color=color)
        axins.plot([0, 0], [0, 30000], color='gray', linestyle='--', linewidth=1.5)


    # 调整局部放大图的坐标范围以聚焦于原点附近
    axins.set_xlim(-100, 100)  # 示例范围，请根据实际情况调整
    axins.set_ylim(0, 2000)  # 示例范围，请根据实际情况调整

    # 标记局部放大区域与主图的关系
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # 继续设置各子图标题、标签等...
    ax1.set_xlabel('$x (m)$', fontsize=18)
    ax1.set_ylabel('$y (m)$', fontsize=18)
    ax2.set_xlabel('$t (s)$', fontsize=18);
    ax2.set_ylabel('$a (m/s^2)$', fontsize=18)
    ax3.set_xlabel('$t (s)$', fontsize=18);
    ax3.set_ylabel(r'$\varepsilon (^{\circ})$', fontsize=18)

    for ax in [ax1, ax2, ax3]:
        ax.grid('on', linewidth=0.2)
        ax.tick_params(axis='both', which='major', labelsize=16)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

if __name__ == '__main__':

    variable_param = 'k1'  # 可选择 'k1', 'k2', 或 'k3'
    fixed_params = {'k2': 0.4, 'k3': 0.2}
    range_var = (0.0, 0.25)

    # variable_param = 'k2'  # 可选择 'k1', 'k2', 或 'k3'
    # fixed_params = {'k1': 0.1, 'k3': 0.2}
    # range_var = (0.3, 0.5)

    # variable_param = 'k3'  # 可选择 'k1', 'k2', 或 'k3'
    # fixed_params = {'k1': 0.1, 'k2': 0.4}
    # range_var = (0.0, 0.3)

    generate_and_plot(variable_param, fixed_params, range_var, 50)