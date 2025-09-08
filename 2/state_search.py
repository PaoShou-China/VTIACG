from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np
from simulation import guidance_data
from contextlib import contextmanager
import os
import sys


# 禁用print输出的上下文管理器
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# 修正后的参数搜索空间
search_space = [
    Real(0.7, 1.3, name='rs'),  # 径向距离系数
    Real(-np.pi, np.pi, name='sigma0'),  # 初始角度（弧度）
    Real(-np.pi, np.pi, name='lambda0'),  # 初始经度角（弧度）
    Real(-np.pi, np.pi, name='theta_f'),  # 终端角度（弧度）
]


def run_simulation(rs, sigma0, lambda0, theta_f, law):
    # 生成制导参数向量
    theta0 = lambda0 + sigma0
    if theta0 > np.pi:
        theta0 = theta0 - 2 * np.pi
    elif theta0 < -np.pi:
        theta0 = theta0 + 2 * np.pi
    parameter = [
        rs * np.cos(lambda0 + np.pi),
        rs * np.sin(lambda0 + np.pi),
        theta0,
        theta_f
    ]

    # 执行仿真计算（抑制控制台输出）
    with suppress_stdout():
        try:
            data = guidance_data(parameter, law)
        except Exception as e:
            print(f"仿真异常: {str(e)}")
            return 1e8

    # 计算终端指标
    miss_distance = np.hypot(data['x'][-1], data['y'][-1])  # 脱靶量
    e_theta = (data['e_theta'][-1] + np.pi) % (2 * np.pi) - np.pi  # 角度偏差归一化

    # 约束条件判断
    if miss_distance < 120 and abs(e_theta) < 0.017:  # 0.017 rad≈1°
        return data['total_a']
    return 1e5  # 不满足约束时返回大惩罚值


# 贝叶斯优化目标函数
@use_named_args(search_space)
def objective(rs, sigma0, lambda0,  theta_f):
    """优化目标函数接口"""
    metrics = run_simulation(rs, lambda0, sigma0, theta_f, 1) - run_simulation(rs, lambda0, sigma0, theta_f, 2)
    print(metrics)
    return metrics


if __name__ == '__main__':
    # 执行参数优化
    optimization_result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=100,  # 总评估次数
        n_jobs=-1,  # 使用全部CPU核心
        random_state=42  # 随机种子保证可复现性
    )

    # 打印优化结果
    print("\n优化完成！最佳参数组合：")
    param_names = ['rs', 'sigma0', 'lambda0',  'theta_f']
    for name, value in zip(param_names, optimization_result.x):
        print(f"{name.ljust(8)} = {value:.4f}")

    print(f"\n最优性能指标: {optimization_result.fun:.2f}")