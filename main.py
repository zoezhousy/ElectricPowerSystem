import pandas as pd
import pickle
import multiprocessing
from multiprocessing import Process, Manager
from Model import Strategy
from Model.Network import Network
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd


def show_result(dict):
    for key, values in dict.items():
        plt.figure()  # 创建一个新的图形
        plt.plot(values)
        plt.title(f'Plot for {key}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        # 保存每个图表为图片文件
        plt.savefig(f'{path}{key}.png')
        plt.close()  # 关闭图形，避免内存泄漏
    for key, values in dict.items():
        plt.figure()
        plt.plot(values)
        plt.title(f'Plot for {key}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.show()


if __name__ == '__main__':
    # 1. 接收到创建新电网指令
    #file_name = "01_8_ye"
    # file_name = "case2_linear/Assessment_simple"
    #file_name = "case3_nonlinear/nonlinear_simple3"

    #file_name = "case3_nonlinear/nonlinear_ye"
    #file_name = "case2_linear/Assessment_18_1213_Yeung"
    #file_name = "case2_linear/Assessment_18_1213_Yeung(withRL)"
    #path = "Data/input/case4_linear/"
    path = "Data/input/case3_nonlinear/"
    file_name = "nonlinear_ye"
    #file_name = "inducedvoltagetest_threephase_withSW"
    json_file_path = path + file_name + ".json"
    # 0. read json file
    with open(json_file_path, 'r', encoding="utf-8") as j:
        load_dict = json.load(j)
    varied_frequency = np.arange(0, 37, 9)
    network = Network()
    #change = Strategy.Change_DE_max()
    #strategy = Strategy.variant_frequency()
    #network.run(load_dict,change)
    #network.run_individual(load_dict)
    #pd.DataFrame(network.run_measure()).to_csv("Data/Output/"+file_name+"_output.csv")

    calculation = load_dict["Global"]["Calculation_Model"]
    # 基础模块
    if calculation == 0:
        result = network.run_base(load_dict)
        df_measure = pd.DataFrame(result)
        df_measure.to_csv(path+'result.csv', index=False, header=True)
        show_result(df_measure)
    # 灵敏度分析模块
    elif calculation == 1:
        network.Distance = 600
        result_before = network.run_MC(load_dict)
        print(network.broken)
        result_after = network.sensitive_analysis(load_dict)
        pd.DataFrame(result_after).to_csv(path+"DE_modified.csv", index=False, header=True)
        print(network.broken)
    elif calculation == 2:
        distance = network.Pre_run_MC(load_dict)
        network = Network()
        #distance = 600
        # file_name = "case3_nonlinear/nonlinear_ye"
        # json_file_path = "Data/input/" + file_name + ".json"
        # with open(json_file_path, 'r', encoding="utf-8") as j:
        #     load_dict = json.load(j)
        result = network.run_MC(load_dict,distance)

        name = "Heidler_perfect_10000_IP100"
        df = pd.DataFrame(result, index=[name])
        df.to_csv(f'{path}summary_values_ins.csv', mode='a')




    # 二、灵敏度分析模块
    #network.sensitive_analysis(load_dict)

    # 三、蒙特卡洛，大量模拟运算

    #network.run_MC(load_dict)

#    pickle.dump(network, open("network.pkl", 'wb'))  # 序列化

    #print(network.solution)
    #print(network.measurement)





