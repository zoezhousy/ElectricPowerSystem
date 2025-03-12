# from abc import ABC, abstractmethod
# import numpy as np
#
# class Strategy(ABC):
#
#     def __init__(self):
#         self.capacitance_matrix = None
#
#     @abstractmethod
#     def apply(self,netwotk,dt,Nt):
#         C = np.array(netwotk.capacitance_matrix)  # 点点
#         G = np.array(netwotk.conductance_matrix)
#         L = np.array(netwotk.inductance_matrix)  # 线线
#         R = np.array(netwotk.resistance_matrix)
#         ima = np.array(netwotk.incidence_matrix_A)  # 线点
#         imb = np.array(netwotk.incidence_matrix_B.T)  # 点线
#
#
#
#
# class Change_light_pos(Strategy):
#     def apply(self,wire,new_pos):
#         print("change light position calculation is used")
#
#         #lightning.channel.hit_pos = new_pos
#         cir_id = wire['cir_id']
#         if wire['type'] == 'SW':
#             bran = 'Y' + str(cir_id) + wire['phase']
#             #network.sources = network.source_calculate(lightning, area, bran, new_pos)
#         elif wire['type'] == 'CIRO':
#             bran = 'Y' + str(cir_id) + wire['phase']
#             #network.sources = network.source_calculate(lightning,area,bran, new_pos)
#         else:
#             bran = wire["name"]
#         return bran
#
# class Change_light_waveform(Strategy):
#     def apply(self, network, lightning, new_waveform, pos_dict):
#         print("change light waveform calculation is used")
#
#         lightning.channel.hit_pos = new_waveform
#         network.sources = network.source_calculate(lightning, pos_dict)
#         network.calculate(network.dt, network.H, network.sources)
#
# class Change_light_parameters(Strategy):
#     def apply(self, network, lightning, new_parameters, pos_dict):
#         print("change light parameters calculation is used")
#         for stroke in lightning.strokes:
#             stroke.parameters = new_parameters #
#             stroke.current_waveform = []
#             stroke.calculate()
#         network.sources = network.source_calculate(lightning, pos_dict)
#         network.calculate(network.dt, network.H, network.sources)
#
# class Change_ROD(Strategy):
#     def apply(self, network, load_dict,lumpname,r,l):
#         print("change ROD calculation is used")
#         for tower in load_dict["Tower"]:
#             for lump in tower["Lump"]:
#                 if lump["name"] == lumpname:
#                     lump["value1"] = r
#                     lump["value2"] = l
#         network.towers = []
#         network.initial_tower(load_dict)
#         network.combine_parameter_matrix()
#         network.calculate(network.dt, network.H, network.sources)
#         pd.DataFrame(network.run_measure()).to_csv("ROD_modified.csv")
#
# class Change_ground(Strategy):
#     def apply(self, load_dict, new_parameter):
#         print("change ground calculation")
#         load_dict['Global']["ground"] = new_parameter  # 修改值
#
#         with open("modified", 'w') as file:
#             json.dump(load_dict, file, indent=4)
#
#         network = Network()
#         network.run(load_dict)
#
# class Change_Arrestor_pos(Strategy):
#     def apply(self, network, load_dict, remove_tower_list):
#         print("changing arrestor position calculation")
#         # for tower in network.towers:
#         #     for device in tower.devices:
#         #         for arrestor in device.arrestors:
#         #             if arrestor.name == name:
#         #                 arrestor.capacitance_matrix = arrestor.capacitance_matrix.rename(columns=node_mapping,index=node_mapping)
#         #                 arrestor.inductance_matrix = arrestor.inductance_matrix.rename(columns=wire_mapping,index=wire_mapping)
#         #                 arrestor.incidence_matrix_A = arrestor.incidence_matrix_A.rename(columns=node_mapping,index=wire_mapping)
#         #                 arrestor.inductance_matrix_B = arrestor.inductance_matrix_B.rename(columns=node_mapping,index=wire_mapping)
#         #                 arrestor.resistance_matrix = arrestor.resistance_matrix.rename(columns=wire_mapping,index=wire_mapping)
#         #                 arrestor.conductance_matrix = arrestor.conductance_matrix.rename(columns=node_mapping,index=node_mapping)
#         #     tower.reset_matrix()
#         #     gnd = network.ground if network.global_ground == 1 else tower.ground
#         #     tower_building(tower, network.f0, network.max_length, gnd, network.varied_frequency)
#         #
#         #     network.combine_parameter_matrix()
#         #     network.calculate(network.dt,network.H,network.sources)
#         for tower in load_dict["Tower"]:
#             if tower['name'] in remove_tower_list:
#                 tower["Device"] = [device for device in tower["Device"] if device['name'].split("_") != "Arrester"]
#         network.run(load_dict)
#         with open("modified_Arrester", 'w') as file:
#             json.dump(load_dict, file, indent=4)
#
# class Change_SW(Strategy):
#     def apply(self, network, load_dict, name):
#
#         for ohl in load_dict['OHL']:
#             #if ohl.name == name:
#                 # if position:
#                 #     ohl["position"] = position # 修改值
#                 # if bran:
#                 #     ohl["bran"] = bran # 修改值
#                 # load_dict['OHL'][index] = ohl
#             if ohl.name == name:
#                 ohl["Wire"] = [wire for wire in ohl["Wire"] if wire['type'] != "SW"]
#         with open("modified_SW", 'w') as file:
#             json.dump(load_dict, file, indent=4)
#
#         network.run(load_dict)
#
# class Change_DE_max(Strategy):
#     def apply(self, network, DE_max):
#         print("Changing DE max calculation is used")
#         for index, lump in enumerate(network.switch_disruptive_effect_models):
#             lump.parameters['DE_max'] = DE_max
#             switch_disruptive_copy = copy.deepcopy(network.switch_disruptive_effect_models)
#             switch_disruptive_copy[index] = lump
#         network.H["switch_disruptive_effect_models"] = switch_disruptive_copy


# sensitivity_calculator.py
# sensitivity_calculator.py
# sensitivity_calculator.py
# sensitivity_calculator.py
import pandas as pd

from Utils.Math import  distance
# sensitivity_calculator.py
import pandas as pd
import os
from Utils.Math import  distance
from Model.Contant import Constant
import itertools
import numpy as np
from Driver.modeling.tower_modeling import tower_building, tower_building_variant_frequency, tower_building_with_tube, \
    tower_building_variant_frequency_with_tube
from Driver.modeling.OHL_modeling import OHL_building
from Model.Lightning import Stroke,Channel,Lightning
from Function.Calculators.InducedVoltage_calculate import InducedVoltage_calculate_direct,LightningCurrent_calculate_direct
from collections import defaultdict, deque

def run_sensitivity_analysis(network, load_dict, sa_dict, use_hybrid, mode,
                             output_path):

    def rerun_solve(network, sources):
        if use_hybrid:
            result, other = network.calculate_of_hybrid_mode(
                line_matrix, tower_matrix, sources, network.Nt, network.dt, network.GPU_calculation
            )
            broken_arrestor_list = [network.arrestor_bran2Tower[bran][1] for bran in other["NLR"] if
                                    bran in network.arrestor_bran2Tower.keys()]
            network.broken = list(set(broken_arrestor_list))
            return network.run_measure(result)
        else:
            return network.calculate(network.Nt, network.dt, network.H, sources)

    def rerun_full(network, load_dict):
        if use_hybrid:
            return network.run_hybrid(load_dict)
        else:
            return network.run_base(load_dict)

    def modify_stroke_position(load_dict, params):
        lightning_data = load_dict["Source"]["Lightning"]
        lightning_data["position"] = params["position"]
        lightning_data["type"] = params["type"]
        if params["type"] == "Direct":
            lightning_data["area"] = params["area"]
            lightning_data["wire"] = params["wire"]
            if params.get("cir_id"):
                lightning_data["cir_id"] = params["cir_id"]
            if params.get("phase"):
                lightning_data["phase"] = params["phase"]
        else:
            lightning_data["area"] = "Ground"

        return load_dict

    def modify_soil_sig(network, value):
        if value:
            print(f"Modifying soil parameters: {value}")
            network.sig = value
            for tower in network.towers:
                swhs_node = [[swh.name, swh.node1[0], swh.node2[0]] for ins in
                             tower.devices.insulators for swh in ins.switch_disruptive_effect_models]
                tower.reset_matrix()
                if network.global_ground == 1:
                    network.ground.sig = value
                    gnd = network.ground
                else:
                    tower.ground.sig = value
                    gnd = network.ground

                tower_building(tower, gnd)

            for ohl in network.OHLs:
                ohl.reset_matrix()
                if network.global_ground == 1:
                    network.ground.sig = value
                    gnd = network.ground
                else:
                    ohl.ground.sig = value
                    gnd = network.ground

                OHL_building(ohl, network.max_length, gnd, network.fixed_frequency)

            network.sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)
            return swhs_node
    def modify_soil_epr(network, value):
        print(f"Modifying soil epr parameters: {value}")
        network.epr = value
        for tower in network.towers:
            swhs_node = [[swh.name, swh.node1[0], swh.node2[0]] for ins in
                         tower.devices.insulators for swh in ins.switch_disruptive_effect_models]
            tower.reset_matrix()
            if network.global_ground == 1:
                network.ground.epr = value
                gnd = network.ground
            else:
                tower.ground.epr = value
                gnd = network.ground

            tower_building(tower, gnd)

        for ohl in network.OHLs:
            ohl.reset_matrix()
            if network.global_ground == 1:
                network.ground.epr = value
                gnd = network.ground
            else:
                ohl.ground.epr = value
                gnd = network.ground

            OHL_building(ohl, network.max_length, gnd, network.fixed_frequency)

        network.sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)
        return swhs_node
    def modify_arrester_name(network, params):
        if params.get("name") is not None:
            arrester_name = params["name"]
            print(f"Removing arrester: {arrester_name}")
            for tower in network.towers:
                tower.devices.arrestors = [device for device in tower.devices.arrestors if device.name != arrester_name]
        if params.get("distance"):
            # 根据距离间隔保留 arrester
            distance_interval = params["distance"]
            print(f"Keeping arresters every {distance_interval} meters")
            # 按塔的 position 排序
            towers_sorted = sorted(network.towers, key=lambda t: t.info.position[0])  # 假设沿 x 轴排序
            last_kept_position = None

            for tower in towers_sorted:
                current_pos = tower.info.position
                if last_kept_position is None:
                    last_kept_position = current_pos
                else:
                    dist = distance(last_kept_position, current_pos)
                    if dist < distance_interval:
                        # 删除此塔的所有 arrester
                        tower.devices.arrestors = []
                    else:
                        # 保留此塔的 arrester，更新 last_kept_position
                        last_kept_position = current_pos
        for tower in network.towers:
            if tower.info.name == value:
                tower.reset_matrix()
                gnd = network.ground if network.global_ground == 1 else tower.ground
                network.tower_building()
                tower_building(tower, gnd)


    def modify_arrester_distance(network, value):

        # 根据距离间隔保留 arrester
        distance_interval = value
        print(f"Keeping arresters every {distance_interval} meters")
        # 按塔的 position 排序
        towers_sorted = sorted(network.towers, key=lambda t: t.info.position[0])  # 假设沿 x 轴排序
        last_kept_position = None

        for tower in towers_sorted:
            print(tower.name)
            if not hasattr(tower, 'devices'):
                print(f"Warning: Invalid tower object in network.towers: {tower}")
                continue
            current_pos = tower.info.position
            if last_kept_position is None:
                last_kept_position = current_pos
            else:
                dist = distance(last_kept_position, current_pos)
                if dist < distance_interval:
                    # 删除此塔的所有 arrester
                    tower.devices.arrestors = []
                    tower.devices.arrestors_bran = []
                else:
                    # 保留此塔的 arrester，更新 last_kept_position
                    last_kept_position = current_pos


    def modify_sw_distance(network, params):
        # 根据距离间隔保留 arrester
        distance_interval = value
        print(f"Keeping sw every {distance_interval} meters")
        # 按塔的 position 排序
        ohls_sorted = sorted(network.OHLs, key=lambda o: o.info.HeadTower_pos)  # 假设沿 x 轴排序
        last_kept_position = None

        for tower in ohls_sorted:
            print(tower.name)
            if not hasattr(tower, 'devices'):
                print(f"Warning: Invalid tower object in network.towers: {tower}")
                continue
            current_pos = tower.info.position
            if last_kept_position is None:
                last_kept_position = current_pos
            else:
                dist = distance(last_kept_position, current_pos)
                if dist < distance_interval:
                    # 删除此塔的所有 arrester
                    tower.devices.arrestors = []
                    tower.devices.arrestors_bran = []
                else:
                    # 保留此塔的 arrester，更新 last_kept_position
                    last_kept_position = current_pos


    def modify_sw_name(network, params):
        if params is not None:
            sw_name = params
            print(f"Removing OHL wire: {sw_name}")
            for ohl in network.OHLs:
                for wire in ohl.wires.air_wires:
                    if wire.name == sw_name:
                        ohl.wires.air_wires = [wire for wire in ohl.wires.air_wires if wire.name != sw_name]
                        ohl.wires_name = [wire.name for wire in ohl.wires.air_wires if wire.name != sw_name]
                        gnd = network.ground if network.global_ground == 1 else ohl.ground
                        ohl.reset_matrix()
                        network.OHL_building(ohl, network.max_length, gnd,network.fixed_frequency)
                        return network


    def modify_rod(network, params):
        if params.get("tower") is not None:
            tower_name = params["tower"]
            print(f"Modifying ROD for tower: {tower_name}")
            for tower in network.towers:
                if tower.info.name == tower_name:
                    for lump in tower.lump.resistor_inductors:
                        if lump.name == params.get("lump"):
                            if params.get("r") is not None:
                                lump.parameters["r"] = params["r"]
                            if params.get("l") is not None:
                                lump.parameters["l"] = params["l"]
                    gnd = network.ground if network.global_ground == 1 else tower.ground
                    if tower.info.con_mode == 1:
                        network.tower_building_variant_frequency(tower, gnd, network.varied_frequency, network.Nfit,
                                                                 network.dt)
                    else:
                        network.tower_building(tower, gnd)
                    break

    def modify_de(network, params):
        if params is not None:
            print(f"Modifying DE_max to {params}")
            for tower in network.towers:
                for ins in tower.devices.insulators:
                    for swh in ins.switch_disruptive_effect_models:
                        swh.parameters["DE_max"] = params

    # def modify_para_set(stroke1,selected_amplitudes):
    #     stroke1.parameters[0] = selected_amplitudes * 1000

    def modify_para_set(selected_parameter,closet_node):

        # 提取时间参数
        numbers = selected_parameter.split("/")[0], selected_parameter.split("/")[1].rstrip("us")
        front_time, tail_time = int(numbers[0]), int(numbers[1])

        # 计算雷击参数
        time = front_time * 1e-6 + 2 * tail_time * 1e-6
        stroke1 = Stroke('Heidler', duration=time, dt=1.0e-8, is_calculated=True, parameter_set=selected_parameter)

        network.T = time
        network.Nt = int(np.ceil(network.T / network.dt))
        stroke1.Nt = network.Nt
        stroke1.t_us = np.array(list(range(network.Nt))) * network.dt
        stroke1.calculate()
        strokes = [stroke1]
        channel = Channel(hit_pos=[50, 500, 0])
        light = Lightning(id=1, type='Direct', strokes=strokes, channel=channel)
        return light



    def modify_ground(network, params):
        if params is not None:
            print(f"Modifying ground parameters: {params}")
            if isinstance(params, dict):
                if "sig" in params:
                    network.sig = params["sig"]
                if "epr" in params:
                    network.epr = params["epr"]

    def calculate(network,result,swhs_node):
        tower_matrix = network.tower_individual_matrix()  # 合并tower矩阵
        line_matrix = network.line_individual_matrix()  # 合并cable和OHL矩阵

        branches, nodes = network.calculate_branches(network.max_length)
        result_tower, other = network.calculate_of_hybrid_mode(line_matrix,
                                                               tower_matrix,
                                                               network.sources,
                                                               network.Nt, network.dt,
                                                               network.GPU_calculation)
        df_result = pd.DataFrame(
            {name + "_" + str(value): abs(
                result_tower.loc[idx1] - result_tower.loc[idx2])
                for name, idx1, idx2 in swhs_node}
        ).T
        FO_result = other["SDEM"]
        result = pd.concat([result, df_result], axis=0)
        return result,FO_result



    # 构建图结构
    def build_graph(OHLs, Towers):
        # 图的邻接表：{tower_name: [(neighbor_name, distance)]}
        graph = defaultdict(list)
        # tower 名称到对象的映射
        tower_map = {tower.info.name: tower for tower in Towers}
        # tower 名称到位置的映射
        tower_positions = {tower.info.name: tower.info.position for tower in Towers}

        # 已访问的 tower 名称，用于推断尾 tower
        visited_towers = set()

        for ohl in OHLs:
            head_tower = ohl.info.HeadTower
            head_pos = ohl.info.HeadTower_pos

            # 找到可能的尾 tower（假设 Towers 中有连接信息或通过位置推断）
            tail_tower = None
            min_distance = float('inf')
            for tower in Towers:
                if tower.info.name != head_tower and tower.info.name not in visited_towers:
                    distance = calculate_distance(head_pos, tower.info.position)
                    if distance < min_distance:
                        min_distance = distance
                        tail_tower = tower.info.name

            if tail_tower:
                # 添加边
                distance = calculate_distance(head_pos, tower_positions[tail_tower])
                graph[head_tower].append((tail_tower, distance))
                graph[tail_tower].append((head_tower, distance))  # 无向图
                visited_towers.add(head_tower)
                visited_towers.add(tail_tower)

        return graph, tower_map, tower_positions

    # 计算距离函数
    def calculate_distance(pos1, pos2):
        # 假设为一维坐标
        #return abs(pos2 - pos1)
        # 如果是三维坐标
        return ((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2 + (pos2[2] - pos1[2])**2)**0.5

    # 使用 BFS 计算从起点到每个 tower 的距离
    def calculate_distances(graph, start_tower):
        distances = {start_tower: 0}
        queue = deque([start_tower])
        visited = set([start_tower])

        while queue:
            current = queue.popleft()
            for neighbor, distance in graph[current]:
                if neighbor not in visited:
                    distances[neighbor] = distances[current] + distance
                    queue.append(neighbor)
                    visited.add(neighbor)

        return distances

    # 判断是否为屏蔽线
    def is_shielding_wire(wire_name):
        first_part = wire_name.split('_')[0]
        return first_part[-1] == 'S'

    # 主逻辑
    def process_arrestor_distance(OHLs, Towers, interval):
        # 构建图和映射
        graph, tower_map, tower_positions = build_graph(OHLs, Towers)

        # 从第一个 ohl 的 HeadTower 开始
        start_tower = OHLs[0].info.HeadTower
        distances = calculate_distances(graph, start_tower)

        # 确定保留的 tower
        towers_to_keep = set()
        max_distance = max(distances.values(), default=0)
        for i in range(0, int(max_distance) + interval, interval):
            closest_tower_name = min(distances.items(), key=lambda x: abs(x[1] - i), default=(None, 0))[0]
            if closest_tower_name:
                towers_to_keep.add(closest_tower_name)

        # 处理 arrestors
        for tower_name, tower in tower_map.items():
            if tower_name not in towers_to_keep:
                if not tower.devices.transformers:  # 如果 transformers 为空
                    tower.devices.arrestors = []  # 删除 arrestors
                    tower.devices.arrestors_bran = []  # 删除 arrestors
                    tower.devices.arrestors_node1 = []  # 删除 arrestors
                    tower.devices.arrestors_node2 = []  # 删除 arrestors

                    tower.reset_matrix()
                    gnd = network.ground if network.global_ground == 1 else tower.ground
                    tower_building(tower, gnd)
                # 如果 transformers 有值，则保留 arrestors

    # 构建全局图结构并计算距离
    def build_graph_and_distances(OHLs):
        graph = defaultdict(list)
        node_positions = {}

        # 添加所有节点和边
        for ohl in OHLs:
            # HeadTower 和 TailTower 之间的边
            head, tail = ohl.info.HeadTower, ohl.info.TailTower  # 假设尾节点名称
            head_pos, tail_pos = ohl.info.HeadTower_pos, ohl.info.TailTower_pos
            distance = calculate_distance(head_pos, tail_pos)
            graph[head].append((tail, distance))
            graph[tail].append((head, distance))
            node_positions[head] = head_pos
            node_positions[tail] = tail_pos

            # wire 的连接
            for wire in ohl.wires.air_wires:
                start, end = wire.start_node.name, wire.end_node.name
                start_pos = (wire.start_node.x, wire.start_node.y, wire.start_node.z)
                end_pos = (wire.end_node.x, wire.end_node.y, wire.end_node.z)
                distance = calculate_distance(start_pos, end_pos)
                graph[start].append((end, distance))
                graph[end].append((start, distance))
                node_positions[start] = start_pos
                node_positions[end] = end_pos

        # 从第一个 HeadTower 开始计算距离
        start_node = OHLs[0].info.HeadTower
        distances = {start_node: 0}
        queue = deque([start_node])
        visited = set([start_node])

        while queue:
            current = queue.popleft()
            for neighbor, distance in graph[current]:
                if neighbor not in visited:
                    distances[neighbor] = distances[current] + distance
                    queue.append(neighbor)
                    visited.add(neighbor)

        return graph, distances, node_positions

    # 确定保留的屏蔽线
    def select_shielding_wires_to_keep(all_shielding_wires, interval):
        wires_to_keep = set()
        available_wires = all_shielding_wires.copy()  # 复制列表，避免修改原始数据

        max_distance = max([d for _, _, d in available_wires], default=0)
        for i in range(0, int(max_distance) + interval, interval):
            if not available_wires:  # 如果没有可选的屏蔽线了，退出
                break
            # 找到距离 i 最近的未保留屏蔽线
            closest = min(available_wires, key=lambda x: abs(x[2] - i))
            ohl, wire, distance = closest
            wires_to_keep.add(wire.name)
            # 从可用列表中移除已选择的屏蔽线
            available_wires.remove(closest)

        return wires_to_keep
    # 处理屏蔽线
    def process_shielding_wires(OHLs, interval):
        graph, distances, node_positions = build_graph_and_distances(OHLs)

        # 收集所有屏蔽线及其位置
        all_shielding_wires = []
        for ohl in OHLs:
            for wire in ohl.wires.air_wires:
                if is_shielding_wire(wire.name):
                    mid_distance = (distances.get(wire.start_node.name, 0) + distances.get(wire.end_node.name, 0)) / 2
                    all_shielding_wires.append((ohl, wire, mid_distance))

        if not all_shielding_wires:
            return

        # 确定保留的屏蔽线
        wires_to_keep = select_shielding_wires_to_keep(all_shielding_wires, interval)

        # 更新每个 ohl
        for ohl in OHLs:
            # 删除未保留的屏蔽线
            nodes_to_remove = set()
            wires_to_remove = set()
            ohl.wires.air_wires[:] = [wire for wire in ohl.wires.air_wires
                                      if not is_shielding_wire(wire.name) or wire.name in wires_to_keep]

            # 处理 nodes_name 和 wires_name
            ohl_name_prefix = ohl.name
            for name_list, removal_set in [(ohl.nodes_name, nodes_to_remove), (ohl.wires_name, wires_to_remove)]:
                new_list = []
                for name in name_list:
                    parts = name.split('_')
                    prefix = '_'.join(parts[:-2]) if len(parts) > 2 else name
                    if prefix == ohl_name_prefix:
                        removal_set.add(name)
                    else:
                        new_list.append(name)
                name_list[:] = new_list

            # 更新 capacitance_matrix
            if isinstance(ohl.capacitance_matrix, pd.DataFrame):
                ohl.capacitance_matrix = ohl.capacitance_matrix.drop(index=list(nodes_to_remove),
                                                                     columns=list(wires_to_remove),
                                                                     errors='ignore')


    """根据参数类型和运行模式执行敏感性分析，返回修改前和修改后结果"""
    os.makedirs(output_path, exist_ok=True)

    network.global_set(load_dict)
    constants = Constant()
    network.dt = network.max_length / constants.vc
    network.Nt = int(np.ceil(network.T / network.dt))

    network.initialize_network(load_dict, network.VF)
    tower_matrix = network.tower_individual_matrix()
    line_matrix = network.line_individual_matrix()
    network.H = {"Line": line_matrix, "Tower": tower_matrix}

    branches, nodes = network.calculate_branches(network.max_length)
    interval = 30  # 间隔 50 米
    process_shielding_wires(network.OHLs, interval)

    constants = Constant()
    share_dict = {}
    network.sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)

    # # 计算修改前的结果
    # result_tower , other = network.calculate_of_hybrid_mode(line_matrix, tower_matrix, network.sources, network.Nt, network.dt, network.GPU_calculation)
    # result_before = network.run_measure(result_tower)
    # pd.DataFrame(result_before if use_hybrid else result_before[0]).to_csv(
    #     f"{output_path}before_modified_{'hybrid' if use_hybrid else 'base'}_output.csv")
    # print(
    #     f"Result before modification saved to {output_path}before_modified_{'hybrid' if use_hybrid else 'base'}_output.csv")

    param_types = ["Stroke_para_set","Stroke_amplitude","Stroke_position", "Soil_sig","Soil_epr", "Arrester", "SW", "ROD", "DE", "ground"]
    #modifications = {key: sa_dict.get(key) for key in param_types if sa_dict.get(key) not in [None, [], {}]}
    # 连续变量/不连续变量
    # modifications = {
    #     key: (
    #         [i
    #          for i in range(sa_dict.get(key)[0],sa_dict.get(key)[1],sa_dict.get(key)[2]) ]
    #         if sa_dict.get("continue") == 1 and key != 'continue'
    #         else sa_dict.get(key)
    #     )
    #     for key in param_types
    #     if sa_dict.get(key) not in [None, [], {}] and key != 'continue'
    # }
    modifications = {
        key: (
            [i for i in range(sa_dict.get(key)[1], sa_dict.get(key)[2], sa_dict.get(key)[3])]
            if isinstance(sa_dict.get(key), (list, tuple)) and len(sa_dict.get(key)) >= 4 and sa_dict.get(key)[0] == 1
            else sa_dict.get(key)[1:] if isinstance(sa_dict.get(key), (list, tuple)) and sa_dict.get(key)[
                0] == 0 else sa_dict.get(key)
        )
        for key in param_types
        if sa_dict.get(key) not in [None, [], {}]
    }
    if mode == 1 or mode ==3:
        results_after = {}
        for param_type, params in modifications.items():
            print(f"Running single modification for {param_type}")

            empty = all(value is None or value == [] for value in params.values()) if isinstance(params, dict) \
                else params is None or params == []

            if not empty:
                result = {}
                filename = "unknown"
                if param_type == "Stroke_position" :
                    filename_current = f"{output_path}stroke_position_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                    swhs_node = {}
                    merged_df = pd.DataFrame()

                    for value in params:

                        load_dict = modify_stroke_position(load_dict, value)
                        network.sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)
                        target_tower = network.lightning.closet_node.target_tower
                        for tower in network.towers:
                            if tower.name in target_tower:
                                swhs_node.update({tower.name+"_"+swh.name: [swh.name, swh.node1[0], swh.node2[0]] for ins in
                                             tower.devices.insulators for swh in
                                             ins.switch_disruptive_effect_models})

                        result_tower, other = network.calculate_of_hybrid_mode(line_matrix, tower_matrix,
                                                                               network.sources,
                                                                               network.Nt, network.dt,
                                                                               network.GPU_calculation)
                        df_result = pd.DataFrame(
                            {v[0] + "_" + str(value["position"]): abs(result_tower.loc[v[1]] - result_tower.loc[v[2]])
                             for k, v in swhs_node.items()}).T
                        merged_df = pd.concat([merged_df, df_result], axis=0)  # 垂直拼接（行堆叠）
                    pd.DataFrame(merged_df).to_csv(filename_current)

                if param_type == "Stroke_para_set" :
                    target_tower = network.lightning.closet_node.target_tower
                    swhs_node = {}
                    for tower in network.towers:
                        if tower.name in target_tower:
                            swhs_node = {tower.name:[swh.name, swh.node1[0], swh.node2[0]]   for ins in
                                 tower.devices.insulators for swh in ins.switch_disruptive_effect_models  }
                    merged_df = pd.DataFrame()
                    filename_current = f"{output_path}stroke_para_set_modified_{'hybrid' if use_hybrid else 'base'}_output_current.csv"

                    for value in params:
                        lightning = modify_para_set(value, network.lightning.closet_node)
                        lightning.closet_node = network.lightning.closet_node
                        # 计算电压和电流
                        U_out = pd.DataFrame()
                        I_out = pd.DataFrame()
                        for j in range(len(lightning.strokes)):
                            new_U = InducedVoltage_calculate_direct(branches, lightning, j)
                            U_out = pd.concat([U_out, new_U], axis=1, ignore_index=True)
                            I_out = pd.concat([I_out,
                                               LightningCurrent_calculate_direct(lightning.closet_node, nodes, lightning,
                                                                                 stroke_sequence=j)],
                                              axis=1, ignore_index=True)

                        # 添加源并计算
                        sources = network.add_lump(U_out, I_out)
                        result_tower, other = network.calculate_of_hybrid_mode(line_matrix, tower_matrix,
                                                                               sources,
                                                                               network.Nt, network.dt,
                                                                               network.GPU_calculation)
                        df_result = pd.DataFrame(
                            {v[0]+ "_" + str(value): abs(result_tower.loc[v[1]] - result_tower.loc[v[2]])
                              for k,v in swhs_node.items()}).T
                        merged_df = pd.concat([merged_df, df_result], axis=0)  # 垂直拼接（行堆叠）
                    pd.DataFrame(merged_df).to_csv(filename_current)

                if param_type == "Stroke_amplitude":
                    target_tower = network.lightning.closet_node.target_tower
                    swhs_node = {}
                    for tower in network.towers:
                        if tower.name in target_tower:
                            swhs_node = {tower.name:[swh.name, swh.node1[0], swh.node2[0]]   for ins in
                                 tower.devices.insulators for swh in ins.switch_disruptive_effect_models  }
                    merged_df = pd.DataFrame()
                    filename_current = f"{output_path}stroke_amplitude_modified_{'hybrid' if use_hybrid else 'base'}_output_current.csv"

                    for value in params:
                        network.lightning.strokes[0].parameters[0] = value * 1000
                        network.lightning.strokes[0].calculate()
                        # 计算电压和电流
                        U_out = pd.DataFrame()
                        I_out = pd.DataFrame()
                        for j in range(len(network.lightning.strokes)):
                            new_U = InducedVoltage_calculate_direct(branches, network.lightning, j)
                            U_out = pd.concat([U_out, new_U], axis=1, ignore_index=True)
                            I_out = pd.concat([I_out,
                                               LightningCurrent_calculate_direct(network.lightning.closet_node, nodes, network.lightning,
                                                                                 stroke_sequence=j)],
                                              axis=1, ignore_index=True)

                        # 添加源并计算
                        sources = network.add_lump(U_out, I_out)
                        result_tower, other = network.calculate_of_hybrid_mode(line_matrix, tower_matrix,
                                                                               sources,
                                                                               network.Nt, network.dt,
                                                                               network.GPU_calculation)
                        df_result = pd.DataFrame(
                            {v[0] + "_" + str(value): abs(result_tower.loc[v[1]] - result_tower.loc[v[2]])
                             for k, v in swhs_node.items()}).T
                        merged_df = pd.concat([merged_df, df_result], axis=0)  # 垂直拼接（行堆叠）
                    pd.DataFrame(merged_df).to_csv(filename_current)

                elif param_type == "Soil_sig":
                    filename_voltage = f"{output_path}sig_modified_{'hybrid' if use_hybrid else 'base'}_volatage_output.csv"
                    merged_df = pd.DataFrame()
                    FO_all = pd.DataFrame()
                    for value in params:
                        swhs_node = modify_soil_sig(network, value)
                        merged_df,FO_result = calculate(network,merged_df,swhs_node)
                        FO_all
                    pd.DataFrame(merged_df).to_csv(filename_voltage)
                elif param_type == "Soil_epr":
                    filename_voltage = f"{output_path}epr_modified_{'hybrid' if use_hybrid else 'base'}_volatage_output.csv"

                    merged_df = pd.DataFrame()


                    for value in params:
                        swhs_node = modify_soil_epr(network, value)
                        merged_df = calculate(network,merged_df,swhs_node)
                    pd.DataFrame(merged_df).to_csv(filename_voltage)


                    filename = f"{output_path}soil_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                elif param_type == "DE" :
                    if isinstance(params, list):
                        merged_df = pd.DataFrame()
                        swhs_node = [[swh.name, swh.node1[0], swh.node2[0]] for tower in network.towers for ins in
                                     tower.devices.insulators for swh in ins.switch_disruptive_effect_models]
                        filename_current = f"{output_path}de_modified_{'hybrid' if use_hybrid else 'base'}_volatage_output.csv"
                        filename_FO = f"{output_path}de_modified_{'hybrid' if use_hybrid else 'base'}_FO_output.csv"
                        FO_all = pd.DataFrame()
                        for value in params:

                            # df_result_before = pd.DataFrame(
                            #     {name+"_before": abs(result_tower.loc[idx1] - result_tower.loc[idx2])
                            #      for name, idx1, idx2 in swhs_node}
                            # ).T
                            modify_de(network, value)

                            result_tower, other = network.calculate_of_hybrid_mode(line_matrix, tower_matrix,
                                                                                   network.sources,
                                                                                   network.Nt, network.dt,
                                                                                   network.GPU_calculation)


                            df_result = pd.DataFrame(
                                {name+"_"+str(value): abs(result_tower.loc[idx1] - result_tower.loc[idx2])
                                 for name, idx1, idx2 in swhs_node}
                            ).T

                            FO_result = pd.DataFrame({"FO_" + str(value): other["SDEM"]}).T
                            merged_df = pd.concat([merged_df, df_result], axis=0)  #垂直拼接（行堆叠）
                            FO_all = pd.concat([FO_all, FO_result], axis=0)  # 垂直拼接（行堆叠）
                        pd.DataFrame(merged_df).to_csv(filename_current)
                        pd.DataFrame(FO_all).to_csv(filename_FO)
                elif param_type == "ground":
                    modify_ground(network, params)
                    network.sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)
                    filename = f"{output_path}ground_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                else:
                    if param_type == "Arrester_distance":
                        swhs_node = [[swh.name, swh.node1[0], swh.node2[0]] for tower in network.towers for ins in
                                     tower.devices.insulators for swh in ins.switch_disruptive_effect_models]

                        filename = f"{output_path}{param_type.lower()}_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                        FO_all = pd.DataFrame()
                        merged_df = pd.DataFrame()
                        for value in params:
                            interval = value
                            process_arrestor_distance(network.OHLs, network.towers, interval)

                    if param_type == "Arrester_name":
                        modify_arrester_distance(network, params)
                        filename = f"{output_path}{param_type.lower()}_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                    elif param_type == "SW_distance" :
                        modify_sw_distance(network, params)
                        filename = f"{output_path}{param_type.lower()}_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                    elif param_type == "ROD":
                        merged_df = pd.DataFrame()
                        swhs_node = [[swh.name, swh.node1[0], swh.node2[0]] for tower in network.towers for ins in
                                     tower.devices.insulators for swh in ins.switch_disruptive_effect_models]
                        filename_voltage = f"{output_path}ROD_modified_{'hybrid' if use_hybrid else 'base'}_volatage_output.csv"
                        filename_FO = f"{output_path}ROD_modified_{'hybrid' if use_hybrid else 'base'}_FO_output.csv"
                        FO_all = pd.DataFrame()
                        target_tower = params["tower"]
                        lump_name = params["lump"]
                        merged_df = pd.DataFrame()
                        for tower in network.towers:
                            if tower.name == target_tower:

                                swhs_node = [[swh.name, swh.node1[0], swh.node2[0]] for ins in
                                 tower.devices.insulators for swh in ins.switch_disruptive_effect_models]
                                for rod in tower.lump.RODs:
                                    if rod.name == lump_name:
                                        for value in params['r']:
                                            tower.reset_matrix()
                                            rod.parameters['resistance'] = value
                                            gnd = network.ground if network.global_ground == 1 else tower.ground
                                            tower_building(tower, gnd)
                                            merged_df,FO_all = calculate(network, merged_df,FO_all,swhs_node)
                        pd.DataFrame(merged_df).to_csv(filename_voltage)



                        #modify_rod(network, params)
                        result = rerun_full(network, load_dict)
                        filename = f"{output_path}{param_type.lower()}_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"

        return None, results_after

    elif mode == 2:
        print("Running simultaneous modifications with combinations")
        results_after = {}

        # 获取所有参数的列表值
        param_values = {}
        for param_type, params in modifications.items():
            if param_type == "Stroke_position":
                param_values["Stroke_position"] = params if isinstance(params, dict) else [params]

            elif param_type == "Soil_sig":
                param_values["Soil_sig"] = params if isinstance(params, list) else [params]
            elif param_type == "Arrester":
                if params.get("name") is not None:
                    param_values["Arrester"] = params["name"] if isinstance(params["name"], list) else [params["name"]]
                if params.get("distance") is not None:
                    param_values["Arrester"] = params["distance"] if isinstance(params["distance"], list) else [params["distance"]]

            elif param_type == "SW":
                if params.get("name") is not None:
                    param_values["Arrester"] = params["name"] if isinstance(params["name"], list) else [params["name"]]
                if params.get("distance") is not None:
                    param_values["Arrester"] = params["distance"] if isinstance(params["distance"], list) else [params["distance"]]

            elif param_type == "ROD":
                rs = params.get("r", [None]) if isinstance(params.get("r"), list) else [params.get("r")]
                ls = params.get("l", [None]) if isinstance(params.get("l"), list) else [params.get("l")]
                param_values["ROD"] = params if isinstance(params, list) else [params]
            elif param_type == "DE":
                param_values["DE"] = params if isinstance(params, list) else [params]
            elif param_type == "ground":
                param_values["ground"] = params if isinstance(params, list) else [params]

        # 生成所有参数值的组合
        param_combinations = list(itertools.product(*param_values.values()))
        param_keys = list(param_values.keys())
        filename_voltage = f"{output_path}combination_iii_{'hybrid' if use_hybrid else 'base'}_volatage_output.csv"
        filename_FO = f"{output_path}combination_iii_{'hybrid' if use_hybrid else 'base'}_FO_output.csv"

        swhs_node = {}
        merged_df = pd.DataFrame()
        FO_all = {}
        for combo in param_combinations:
            combo_dict = dict(zip(param_keys, combo))
            combo_key = "_".join(
                f"{k}_{v if not isinstance(v, dict) else '_'.join(f'{kk}_{vv}' for kk, vv in v.items() if vv is not None)}"
                for k, v in combo_dict.items() if v is not None)
            FO = pd.DataFrame({combo_key: []})
            for param_type, value in combo_dict.items():
                if value is not None:
                    if param_type == "Stroke_position":
                        load_dict = modify_stroke_position(load_dict,value)
                    elif param_type == "Soil_sig":
                        swhs_node = modify_soil_sig(network,  value)
                    elif param_type == "Arrester":
                        if isinstance(value, int):
                            modify_arrester_distance(network,  value)
                    elif param_type == "SW":
                        modify_sw(network,  value)
                    elif param_type == "ROD":
                        filename_voltage = f"{output_path}ROD_modified_{'hybrid' if use_hybrid else 'base'}_volatage_output.csv"
                        filename_FO = f"{output_path}ROD_modified_{'hybrid' if use_hybrid else 'base'}_FO_output.csv"

                        target_tower = value["tower"]
                        lump_name = value["lump"]
                        merged_df = pd.DataFrame()
                        for tower in network.towers:
                            if tower.name == target_tower:

                                swhs_node = [[swh.name, swh.node1[0], swh.node2[0]] for ins in
                                             tower.devices.insulators for swh in ins.switch_disruptive_effect_models]
                                for rod in tower.lump.RODs:
                                    if rod.name == lump_name:
                                        for v in value['r']:
                                            tower.reset_matrix()
                                            rod.parameters['resistance'] = v
                                            gnd = network.ground if network.global_ground == 1 else tower.ground
                                            tower_building(tower, gnd)
                                            merged_df, FO = calculate(network, merged_df, FO, swhs_node)
                        pd.DataFrame(merged_df).to_csv(filename_voltage)

                    elif param_type == "DE":
                        modify_de(network,  value)
                    elif param_type == "ground":
                        modify_ground(network,  value)
                        modified_sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)

            merged_df,FO = calculate(network, merged_df,FO,swhs_node)
            FO_all[combo_key] = FO
        pd.DataFrame(merged_df).to_csv(filename_voltage)
        pd.DataFrame(FO_all).to_csv(filename_FO)
        print(f"Result saved to {filename_voltage}")

        return None, results_after

