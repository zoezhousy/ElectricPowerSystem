import pickle

from PIL.ImageChops import offset
from Risk_Evaluate.pcnumber import *
from Risk_Evaluate.edge_image import *
from Risk_Evaluate.lighting_parameters_distribution import *
from Risk_Evaluate.Lightning_stroke_location import *
from Risk_Evaluate.current_waveform_generator import *
from Utils.Math import distance
import numpy as np
import json
import pandas as pd
from Model.Lightning import Stroke,Lightning,Channel
import math
import copy
import sys
import random

# network = pickle.load(open('../Data/output/network.pkl', 'rb'))
# print(network.dt)

def run_MC(network,load_dict,Dmax):

    Line = {}
    Line["Node_all"] = np.empty((0, 5))
    Line["Edge_all"] = np.empty((0, 3))
    Line["SWnumber"] = []
    Line["Pcnumber"] = []
    Line["segments"] = []
    Line["Suppose_OHLP"] = []
    Line["Span"] = []
    Line["Tower"] = []
    max_length = 50
    def find_height(id):
        for tower in network.towers:
            if tower.info.ID == id:
                return tower.info.Pole_Height
    for tower in network.towers:
        new_row = np.array([[tower.info.ID,tower.info.position[0],tower.info.position[1],tower.info.position[2]
                          ,tower.info.Pole_Height]]).astype(float)
        # 将新行添加到ndarray中
        Line["Node_all"] = np.vstack([Line["Node_all"], new_row])
        Line["Tower"].append(int(tower.info.ID))
    for ohl in network.OHLs:
        Line["Span"].append(int(ohl.info.ID))
        new_row = np.array([[ohl.info.ID, ohl.info.HeadTower_id,ohl.info.TailTower_id]]).astype(float)
        Line["Edge_all"] = np.vstack([Line["Edge_all"] , new_row])

        s = np.empty((0, 9))
        span = np.array([[ohl.info.HeadTower_pos[0],ohl.info.HeadTower_pos[1],ohl.info.HeadTower_pos[2],
                         ohl.info.TailTower_pos[0],ohl.info.TailTower_pos[1],ohl.info.TailTower_pos[2]
                          ,0,0,0]]).astype(float)
        s = np.vstack([s, span])

        sw = 0
        pc = 0
        for wire in ohl.wires.air_wires:
            if wire.name[5] == "S":
                sw += 1
                new_wire = np.array([[wire.start_node.x, wire.start_node.y, wire.start_node.z,
                                  wire.end_node.x, wire.end_node.y, wire.end_node.z, wire.offset,
                                  wire.name[1:5],0]]).astype(float)
                s = np.vstack([s, new_wire])
            pc +=1
            new_wire = np.array([[wire.start_node.x, wire.start_node.y, wire.start_node.z,
                                  wire.end_node.x, wire.end_node.y, wire.end_node.z, wire.offset,
                                  wire.name[1:5], pc]]).astype(float)
            s = np.vstack([s, new_wire])

        Line["Suppose_OHLP"].append(s)
        Line["SWnumber"].append(sw)
        Line["Pcnumber"].append(ohl.phase_num)
        length = distance(ohl.info.HeadTower_pos, ohl.info.TailTower_pos)
        segment_num = int(np.ceil(length / network.max_length))
        Line["segments"].append(segment_num)


    Line["SWnumber"] = np.array(Line["SWnumber"]).reshape(1, len(network.OHLs))
    Line["Pcnumber"] = np.array(Line["Pcnumber"]).reshape(1, len(network.OHLs))
    Line["segments"] = np.array(Line["segments"]).reshape(1, len(network.OHLs))

    # 每条span的斜坡角度[上（左），下（右）各一个角度（度数）]，斜坡起点是最初的落雷点位置到最近的pole-pole线的垂足点
    Line['Slope_angle'] = np.tile(np.array([10, 20]), (len(Line["Edge_all"]),1))
    # 多个building
    Line['buildings'] = [
        {
            'building_XY': np.array([[50, 0], [70, 0], [50, 20], [70, 20]]),  # building四个点的XY坐标（顺序依次是左下角，右下角，左上角，右上角）
            'building_height': 8.5  # building的高度
        },
        {
            'building_XY': np.array([[30, 10], [40, 10], [30, 25], [40, 25]]),  # 第二个建筑物
            'building_height': 10.0
        }
    ]

    Line['OHLPf'] = pcnumber(Line)  # 判断phase conductor有几相-OHLPf

    # 函数
    Line['Node'] = Line['Node_all'][:, 1: 3]
    Line['Edges'] = Line['Edge_all'][:, 1: 3]

    # json_file_path = "Data/input/MC.json"
    # # 0. read json file
    # with open(json_file_path, 'r', encoding="utf-8") as j:
    #     load_dict = json.load(j)

    #每个杆子的端点pole状态
    Line['polestate'] = np.array([random.choice([0, 1]) for _ in range(Line['Node'].shape[0])])
    # 判断横着的线长度是否大于等于1000m 找到x轴最中间的pointmdm的两个点
    Coordinates = Line['Node']
    y_zero_points = Coordinates[Coordinates[:, 1] == 0]  # 找到 y 坐标为 0 的点
    x_max_point = y_zero_points[np.argmax(y_zero_points[:, 0])]  # 找出 x 最大的点
    x_min_point = y_zero_points[np.argmin(y_zero_points[:, 0])]  # 找出 x 最最小的点
    x_max = x_max_point[0]  # 计算 x 坐标的差值
    x_min = x_min_point[0]
    x_diff = x_max - x_min
    MC_lgtn = load_dict["MC"]["MC_lgtn"]
    casemodel = MC_lgtn['casemodel']  # 选择模式 1-一般情况 非1-特定情况
    pointmd = MC_lgtn['pointmd']  # x轴最中间pointmd距离的两个点
    # 如果长度小于1000m，显示警告并只能操作一般模式
    if casemodel !=1 and x_diff < 1000:
        print(
            "Error: Length is less than 1000m, cannot partition and extract indirect lightning data. Exiting the program.")
        sys.exit()
    else :
        num_points = int(x_diff // pointmd)  # 按 pointmd 米间隔分段
        if num_points ==0:
            MC_lgtn['mid_points'] = np.array([[x_min, 0], [x_max, 0]])
        else:
            x_values = np.linspace(x_min, x_max, num_points + 1)  # 计算分段的 x 坐标
            mid_index = num_points // 2  # 获取中间部分的两个点
            MC_lgtn['mid_points'] = np.array([[x_values[mid_index], 0], [x_values[mid_index + 1], 0]])

    DSave = load_dict["MC"]["DSave"]
    AR = load_dict["MC"]["AR"]
    surrounding_distance = Dmax
    if Dmax is None:
        surrounding_distance =  load_dict["MC"]["surrounding_distance"]
    foldname = "Data/output"
    Wave_Model = 1
    # 1. 画范围框
    [resultedge, XY_need3] = edge_image(Line, DSave, surrounding_distance, foldname)

    # 2. number of flashes and number of strokes
    [resultcur, parameterst, flash_stroke, points_need] = lighting_parameters_distribution(MC_lgtn, DSave, Line,
                                                                                           resultedge, foldname)
    # 3. 判断落雷点位置
    [resultstro, dataSTS, stroke_result] = Lightning_stroke_location(Line, resultcur, DSave, foldname,
                                                                     resultedge, AR)
    PoleXY = Line['Node_all']
    # 4. 描述电流的波形
    light_final = current_waveform_generator(Wave_Model, DSave, resultstro, resultcur, foldname)
    light_final = np.array(light_final)

    DIND = stroke_result[0]
    DINDs = np.array([1 if x == 'Direct' else 0 for x in DIND])
    DIND2 = DINDs[resultcur['siteone']]
    flash_stroke = np.hstack((flash_stroke, DIND2.reshape(-1, 1)))

   # 一般情况，直接输出间接雷数据和直接雷数据
    if casemodel == 1:
        # 将double 数组转换为table
        df27 = pd.DataFrame(flash_stroke, columns=['flash', 'stroke', 'Direct1_Indirect2'])
        return df27, light_final, stroke_result, PoleXY
    else: # 特定情况
        # 间接雷数据按照Y轴正半轴（离x轴的距离）分区
        filtered_indices_ind = []
        for idx in range(len(stroke_result[0])):
            if stroke_result[0][idx] == 'Indirect':
                filtered_indices_ind.append(idx)

        df27 = []
        parameterst_list = []
        stroke_result_list = []
        Dymax = MC_lgtn['Dymax']
        Dyp = MC_lgtn["Dyp"]
        num_zones = Dymax // Dyp  # 计算区间数量
        filtered_indices_indall = [[] for _ in range(num_zones)]
        for idx in filtered_indices_ind:
            y_coord = stroke_result[7][idx][1]  # 获取 y 坐标
            zone_index = int(y_coord // Dyp)  # 计算 y 坐标所在的区间（区间索引）
            if zone_index < num_zones:  # 防止越界
                filtered_indices_indall[zone_index].append(idx)

        # 遍历每个区间
        for i in range(len(filtered_indices_indall)):
            indices = filtered_indices_indall[i]  # 第 i 区间的索引
            stroke_resultindq = []
            parameterstindq = None
            flash_strokeindq = None
            # 对于每个区间，从 stroke_result 中提取相应的索引数据
            for j in range(len(stroke_result)):
                # 获取 stroke_result 中的 jth list 和 filtered_indices_indall 中第 i 个区间的索引

                stroke_resultindq.append([stroke_result[j][idx] for idx in indices])

            # globals()[f"stroke_resultindq_{i}"] = stroke_resultindq  # 使用 globals() 动态创建一个新的变量名
            stroke_result_list.append(stroke_resultindq)
            parameterstindq = light_final[indices]
            parameterst_list.append(parameterstindq)
            # globals()[f"parameterstindq_{i}"] = parameterstindq
            flash_strokeindq = flash_stroke[indices]
            df27ind = pd.DataFrame(flash_strokeindq, columns=['flash', 'stroke', 'Direct1_Indirect2'])
            df27.append(df27ind)
            # globals()[f"df27indq_{i}"] = df27ind

        return df27, parameterst_list, stroke_result_list, PoleXY