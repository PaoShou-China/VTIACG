import os
import numpy as np
import random
import batch
import compare
from simulation import random_parameters

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    log_file = 'output.txt'  # 定义日志文件名
    for i in range(100):
        set_seed(i)
        parameter_list = random_parameters(100)
        batch.main(parameter_list)
        a, b, c = compare.main()

        # 准备要写入文件的内容
        output_str = f"{a} {b} {c}\n"

        # 打开文件并追加内容
        with open(log_file, 'a') as file:
            file.write(output_str)

        # 如果你仍然希望在控制台打印这些信息，可以保留这行
        print(output_str.strip())
