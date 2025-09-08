import pandas as pd
import numpy as np


def evaluate(filename):
    df = pd.read_csv(filename, encoding="utf-8")
    total_a_list = []
    final_a_list = []
    miss_distance_list = []
    e_theta_f_list = []
    success_count = 0

    for index, row in df.iterrows():
        miss_distance = row['miss_distance']
        e_theta_f = abs(row['e_theta_f'])
        final_a = abs(row['final_a'])
        total_a = abs(row['total_a'])

        if miss_distance < 6.5 and e_theta_f < 0.018:
            success_count += 1
            total_a_list.append(total_a)
            final_a_list.append(final_a)
            miss_distance_list.append(miss_distance)
            e_theta_f_list.append(e_theta_f)

    success_rate = success_count / len(df)
    mean_total_a = np.mean(np.array(total_a_list))
    std_total_a = np.std(np.array(total_a_list))
    mean_final_a = np.mean(np.array(final_a_list))
    std_final_a = np.std(np.array(final_a_list))
    mean_miss_distance = np.mean(np.array(miss_distance_list))
    std_miss_distance = np.std(np.array(miss_distance_list))
    mean_e_theta_f = np.mean(np.array(e_theta_f_list))
    std_e_theta_f = np.std(np.array(e_theta_f_list))

    # 使用字典来组织返回的数据
    result = {
        'success_rate': success_rate,
        'mean_total_a': mean_total_a,
        'std_total_a': std_total_a,
        'mean_final_a': mean_final_a,
        'std_final_a': std_final_a,
        'mean_normalized_miss_distance': mean_miss_distance/3.0,
        'std_normalized_miss_distance': std_miss_distance/3.0,
        'mean_e_theta_f': mean_e_theta_f,
        'std_e_theta_f': std_e_theta_f
    }

    return result


def main():
    results = {}
    results['file_1'] = evaluate('1.csv')
    results['file_2'] = evaluate('2.csv')
    results['file_4'] = evaluate('4.csv')

    return results


# 打印结果
print(main())