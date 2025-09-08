from simulation import guidance_data, parameter_list
import numpy as np
import pandas as pd
from multiprocessing import Process
from tqdm import tqdm
import sys
import os
from contextlib import contextmanager


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


def run_simulation(parameter, law):
    """
    运行模拟并返回结果
    :param parameter: 参数列表
    :param law: 制导法则编号
    :return: 包含参数和结果的字典
    """
    with suppress_stdout():  # 禁用print输出
        data = guidance_data(parameter, law)

    miss_distance = (data['x'][-1] ** 2 + data['y'][-1] ** 2) ** 0.5
    e_theta_f = (data['e_theta'][-1] + np.pi) % (2 * np.pi) - np.pi
    e_lambda_f = (data['e_lambda'][-1] + np.pi) % (2 * np.pi) - np.pi
    final_a = abs(data['a'][-1])
    total_a = data['total_a']


    # 将parameter列表展开为单独的列
    result = {
        'law': law,
        'miss_distance': miss_distance.item(),
        'e_theta_f': e_theta_f.item(),
        'e_lambda_f': e_lambda_f.item(),
        'final_a': final_a.item(),
        'total_a': total_a
    }

    # 假设parameter列表长度固定为4
    for i, param in enumerate(parameter):
        result[f'param_{i + 1}'] = param

    return result


def process_law(parameters, law, output_file):
    """
    处理特定制导法则的任务并将结果写入文件
    :param parameters: 参数列表
    :param law: 制导法则编号
    :param output_file: 输出文件名
    """
    results = []
    for parameter in tqdm(parameters, desc=f"Law {law} Progress", file=sys.stderr):  # 使用stderr显示进度条
        result = run_simulation(parameter, law)
        results.append(result)

    # 将结果保存到指定文件
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)


def main(parameter_list):
    # 按照制导法则分组任务
    tasks_by_law = {1: [], 2: [], 4: []}
    for parameter in parameter_list:
        for law in [1, 2, 4]:
            tasks_by_law[law].append(parameter)

    # 创建进程列表
    processes = []
    for law in [1, 2, 4]:
        output_file = f'{law}.csv'
        p = Process(target=process_law, args=(tasks_by_law[law], law, output_file))
        processes.append(p)

    # 启动所有进程
    for p in processes:
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print("All simulations completed and results saved.")

if __name__ == '__main__':
    main(parameter_list)