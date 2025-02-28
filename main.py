import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from Model import Strategy
from Model.Network import Network


def show_result(data_dict, save_path):
    """
    绘制并保存结果图表
    :param data_dict: 包含数据的字典
    :param save_path: 图表保存路径
    """
    for key, values in data_dict.items():
        plt.figure()
        plt.plot(values)
        plt.title(f'Plot for {key}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.savefig(f'{save_path}{key}.png')
        plt.close()


def load_json_file(file_path):
    """
    加载JSON文件
    :param file_path: JSON文件路径
    :return: 加载后的字典
    """
    with open(file_path, 'r', encoding="utf-8") as f:
        return json.load(f)


def run_base_calculation(network, load_dict, save_path):
    """
    执行基础计算模块
    :param network: Network对象
    :param load_dict: 加载的JSON数据
    :param save_path: 结果保存路径
    """
    result = network.run_base(load_dict)
    df_measure = pd.DataFrame(result)
    df_measure.to_csv(f'{save_path}result.csv', index=False, header=True)
    show_result(df_measure, save_path)


def run_sensitivity_analysis(network, load_dict, save_path):
    """
    执行灵敏度分析模块
    :param network: Network对象
    :param load_dict: 加载的JSON数据
    :param save_path: 结果保存路径
    """
    mode = load_dict["Sensitivity_analysis"]["mode"]
    # 单个参数变化
    # if mode ==0:
    #     result_before = network.run_MC(load_dict)
    #     print(network.broken)
    #     result_after = network.sensitive_analysis(load_dict)
    #     pd.DataFrame(result_after).to_csv(f'{save_path}DE_modified.csv')
    #     print(network.broken)


    # 概率分布，多参数变化结合
    if mode == 0:
        network.Distance = 100
        # result_before = network.run_MC(load_dict)
        FO_matrix, SAF_matrix = network.sensitive_MC(load_dict)
        print(FO_matrix)
    # 单次计算，单个参数改变
    elif mode == 1:
        result_before, results_after = network.sensitive_analysis(load_dict)
        if results_after:
            for param_type, result in results_after.items():
                df_after = pd.DataFrame(
                    result if network.global_set(load_dict).get("Hybrid_method", 0) == 1 else result[0])
                df_after.to_csv(f'{save_path}{param_type.lower()}_modified.csv')

    # 单次计算，多个参数改变
    elif mode == 2:
        result_before, result_after = network.sensitive_analysis(load_dict)
        if result_after:
            pd.DataFrame(
                result_after if network.global_set(load_dict).get("Hybrid_method", 0) == 1 else result_after[0]).to_csv(
                f'{save_path}combined_modified.csv')
            print(f"  Combined: {result_after}")
            show_result(result_after, path)
    # 多次计算，多个参数改变
    elif mode == 3:
        result_before, results_after = network.sensitive_analysis(load_dict)
        if results_after:
            for param_type, result in results_after.items():
                df_after = pd.DataFrame(
                    result if network.global_set(load_dict).get("Hybrid_method", 0) == 1 else result[0])
                # 保存到 save_path 而不是 output_path，与其他模式保持一致
                df_after.to_csv(f'{save_path}{param_type.lower()}_sequential.csv')
                print(f"  {param_type}: {result}")
                show_result(result, path)


def run_monte_carlo_simulation(network, load_dict, save_path):
    """
    执行蒙特卡洛模拟模块
    :param network: Network对象
    :param load_dict: 加载的JSON数据
    :param save_path: 结果保存路径
    """
    distance = network.Pre_run_MC(load_dict)
    network = Network()
    network.Distance = distance
    result = network.run_MC(load_dict)
    name = "Heidler_perfect_10000_IP100"
    df = pd.DataFrame(result, index=[name])
    df.to_csv(f'{save_path}summary_values_ins.csv', mode='a')


if __name__ == '__main__':
    # 配置路径和文件名
    path = "Data/input/case3_nonlinear/"
    file_name = "nonlinear_ye"
    json_file_path = f'{path}{file_name}.json'

    # 加载JSON文件
    load_dict = load_json_file(json_file_path)

    # 初始化Network对象
    network = Network()

    # 根据计算模型执行不同的逻辑
    calculation = load_dict["Global"]["Calculation_Model"]
    if calculation == 0:
        run_base_calculation(network, load_dict, path)
    elif calculation == 1:
        run_sensitivity_analysis(network, load_dict, path)
    elif calculation == 2:
        run_monte_carlo_simulation(network, load_dict, path)