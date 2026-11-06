from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import numpy as np
from simulation import our_guidance_data, parameter_list
import os
import sys
from multiprocessing import Pool
from tqdm import tqdm

flag = 0
_simulation_pool = None


def _silence_worker_stdout():
    """Redirect worker output once, rather than reopening /dev/null per simulation."""
    sys.stdout = open(os.devnull, "w")


def _run_simulation_job(job):
    parameter, k1, k2, k3, N = job
    return run_simulation(parameter, k1, k2, k3, N)


# 定义搜索空间
search_space = [
    Real(0, 0.2, name='k1'),
    Real(0.1, 0.6, name='k2'),
    Real(0, 0.2, name='k3'),
    Real(1, 6, name='N')
]


def run_simulation(parameter, k1, k2, k3, N):
    """
    运行模拟并返回结果
    :param parameter: 参数列表
    :param k1, k2, k3, N: 控制器参数
    :return: 当前仿真目标值
    """
    data = our_guidance_data(parameter, k1, k2, k3, N)
    miss_distance = (data['x'][-1] ** 2 + data['y'][-1] ** 2) ** 0.5
    e_theta_f = (data['e_theta'][-1] + np.pi) % (2 * np.pi) - np.pi
    e_lambda_f = (data['e_lambda'][-1] + np.pi) % (2 * np.pi) - np.pi
    final_a = abs(data['a'][-1])
    total_a = data['total_a']

    if miss_distance < 12 and e_theta_f < 0.017:
        return total_a
    else:
        return 1e8


# 目标函数，用于贝叶斯优化
@use_named_args(search_space)
def objective(**params):
    global flag
    k1, k2, k3, N = params['k1'], params['k2'], params['k3'], params['N']
    jobs = [(parameter, k1, k2, k3, N) for parameter in parameter_list]

    if _simulation_pool is None:
        # Keep direct objective() calls usable outside main().
        with Pool(processes=min(len(jobs), os.cpu_count() or 1),
                  initializer=_silence_worker_stdout) as pool:
            metrics = sum(tqdm(pool.imap(_run_simulation_job, jobs), total=len(jobs)))
    else:
        metrics = sum(tqdm(_simulation_pool.imap(_run_simulation_job, jobs), total=len(jobs)))

    print(flag)
    flag += 1
    return metrics


if __name__ == '__main__':
    # Reuse a bounded worker pool across all Bayesian-optimization evaluations.
    _simulation_pool = Pool(
        processes=min(len(parameter_list), os.cpu_count() or 1),
        initializer=_silence_worker_stdout,
    )
    try:
        result = gp_minimize(objective, search_space, n_calls=100, random_state=42)
    finally:
        _simulation_pool.close()
        _simulation_pool.join()
        _simulation_pool = None

    # 输出最佳参数和对应的性能指标
    print("Best parameters:", result.x)
    print("Best performance metric:", result.fun)
