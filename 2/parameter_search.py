from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import numpy as np
from simulation import our_guidance_data, parameter_list
from contextlib import contextmanager
import os
import sys
from multiprocessing import Process, Queue, Value
from tqdm import tqdm
import time
flag = 0
# 用于临时禁用print输出的上下文管理器
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


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
    :param law: 制导法则编号
    :return: 包含参数和结果的字典
    """
    with suppress_stdout():  # 禁用print输出
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




def run_simulation_wrapper(parameter, k1, k2, k3, N, queue, completed):
    """
    包装run_simulation函数，使其结果可以放入队列中。
    :param parameter: 当前参数
    :param k1, k2, k3, N: 控制器参数
    :param queue: 用于存储结果的队列
    :param completed: 共享的完成计数器
    """
    result = run_simulation(parameter, k1, k2, k3, N)
    queue.put(result)
    with completed.get_lock():  # 确保线程安全地增加计数器
        completed.value += 1

# 目标函数，用于贝叶斯优化
@use_named_args(search_space)
def objective(**params):
    global flag
    k1, k2, k3, N = params['k1'], params['k2'], params['k3'], params['N']
    processes = []
    result_queue = Queue()
    completed = Value('i', 0)  # 初始化一个共享的整数变量作为计数器

    for parameter in parameter_list:
        p = Process(target=run_simulation_wrapper, args=(parameter, k1, k2, k3, N, result_queue, completed))
        processes.append(p)
        p.start()

    metrics = 0
    with tqdm(total=len(parameter_list)) as pbar:
        while completed.value < len(parameter_list):
            if not result_queue.empty():
                metrics += result_queue.get()
                pbar.update(1)  # 更新进度条
            else:
                # 防止过快轮询导致CPU占用过高
                time.sleep(0.01)

    for p in processes:
        p.join()

    print(flag)
    flag += 1

    return metrics


if __name__ == '__main__':
    # 使用贝叶斯优化进行参数搜索
    result = gp_minimize(objective, search_space, n_calls=100, random_state=42)

    # 输出最佳参数和对应的性能指标
    print("Best parameters:", result.x)
    print("Best performance metric:", result.fun)
