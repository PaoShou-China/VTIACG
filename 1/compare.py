import pandas as pd
import ast


def evaluate(filename):
    # 读取CSV文件，并为特定列指定数据类型
    df = pd.read_csv(filename, encoding="utf-8")

    ta, fa = 0, 0

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        miss_distance = row['miss_distance']
        e_theta_f = row['e_theta_f']
        final_a = row['final_a']
        total_a = row['total_a']

        if miss_distance < 6.5 and e_theta_f < 0.018:
            ta += total_a
            fa += final_a
        else:
            # print(row)
            ta += 1e8
            fa += 1e8


    return ta / 100

def main():
    a = evaluate('1.csv')
    b = evaluate('2.csv')
    c = evaluate('4.csv')
    return a, b, c

print(main())