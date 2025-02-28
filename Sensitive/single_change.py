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

    def modify_stroke(network, load_dict, params):
        lightning_data = load_dict["Source"]["Lightning"]
        sub_change = None
        if params.get("position") is not None:
            lightning_data["position"] = params["position"]
            sub_change = "position"
            lightning_data["area"] = params["area"]
            lightning_data["wire"] = params["wire"]
            if params.get("cir_id") is not None:
                lightning_data["cir_id"] = params["cir_id"]
            if params.get("phase") is not None:
                lightning_data["phase"] = params["phase"]
        if params.get("waveform"):
            lightning_data["waveform"] = params["waveform"]
            sub_change = "waveform"
        if params.get("paramenters"):
            lightning_data["parameters"] = params["paramenters"]
            sub_change = "paramenters"
        return network.source_initial(load_dict, nodes, branches, constants, share_dict),sub_change

    def modify_soil(network, params):
        if params:
            print(f"Modifying soil parameters: {params}")
            if len(params) >= 2:
                network.sig = params[0]
                network.epr = params[1]

    def modify_arrester(network, params):
        if params.get("name") is not None:
            arrester_name = params["name"]
            print(f"Removing arrester: {arrester_name}")
            for tower in network.towers:
                tower.devices.arrestors = [device for device in tower.devices.arrestors if device.name != arrester_name]
        for tower in network.towers:
            gnd = network.ground if network.global_ground == 1 else tower.ground
            if tower.info.con_mode == 1:
                network.tower_building_variant_frequency(tower, gnd, network.varied_frequency, network.Nfit, network.dt)
            else:
                network.tower_building(tower, gnd)

    def modify_sw(network, params):
        if params.get("name") is not None:
            sw_name = params["name"]
            print(f"Removing OHL wire: {sw_name}")
            for ohl in network.OHLs:
                ohl.wires.all_wires = {
                    key: wire for key, wire in ohl.wires.get_all_wires().items() if wire.name != sw_name
                }
        for ohl in network.OHLs:
            gnd = network.ground if network.global_ground == 1 else ohl.ground
            if ohl.info.con_mode == 1 or gnd.gnd_mode == 2:
                if network.Hybrid_method == 1:
                    network.OHL_building_hybrid_variant_frequency(ohl, network.max_length, gnd,
                                                                  network.varied_frequency, network.fixed_frequency,
                                                                  network.Nfit, network.dt)
                else:
                    network.OHL_building_variant_frequency(ohl, network.max_length, gnd, network.varied_frequency,
                                                           network.fixed_frequency, network.Nfit, network.dt)
            else:
                network.OHL_building(ohl, network.max_length, gnd)

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
            for switch in network.switch_disruptive_effect_models:
                switch.parameters["DE_max"] = params
            if not use_hybrid:
                network.H["switch_disruptive_effect_models"] = network.switch_disruptive_effect_models

    def modify_ground(network, params):
        if params is not None:
            print(f"Modifying ground parameters: {params}")
            if isinstance(params, dict):
                if "sig" in params:
                    network.sig = params["sig"]
                if "epr" in params:
                    network.epr = params["epr"]
    """根据参数类型和运行模式执行敏感性分析，返回修改前和修改后结果"""
    os.makedirs(output_path, exist_ok=True)

    network.global_set(load_dict)
    network.initialize_network(load_dict, network.VF)
    if use_hybrid:
        tower_matrix = network.tower_individual_matrix()
        line_matrix = network.line_individual_matrix()
        network.H = {"Line": line_matrix, "Tower": tower_matrix}
    else:
        network.combine_parameter_matrix()
    branches, nodes = network.calculate_branches(network.max_length)
    constants = Constant()
    share_dict = {}
    sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)

    # 计算修改前的结果
    result_before = rerun_solve(network, sources) if mode in [1, 2] else rerun_full(network, load_dict)
    pd.DataFrame(result_before if use_hybrid else result_before[0]).to_csv(
        f"{output_path}before_modified_{'hybrid' if use_hybrid else 'base'}_output.csv")
    print(
        f"Result before modification saved to {output_path}before_modified_{'hybrid' if use_hybrid else 'base'}_output.csv")

    param_types = ["Stroke", "Soil", "Arrester", "SW", "ROD", "DE", "ground"]
    modifications = {key: sa_dict.get(key) for key in param_types if sa_dict.get(key) not in [None, [], {}]}

    if mode == 1:
        results_after = {}
        for param_type, params in modifications.items():
            print(f"Running single modification for {param_type}")

            empty = all(value is None or value == [] for value in params.values()) if isinstance(params, dict) \
                else params is None or params == []

            if not empty:
                result = {}
                filename = "unknown"
                if param_type == "Stroke" :
                    modified_sources,sub_change = modify_stroke(network, load_dict, params)
                    result = rerun_solve(network, modified_sources)
                    filename = f"{output_path}stroke_{sub_change}_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                    param_type = param_type+"_"+sub_change
                elif param_type == "Soil":
                    modify_soil(network, params)
                    modified_sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)
                    result = rerun_solve(network, modified_sources)
                    filename = f"{output_path}soil_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                elif param_type == "DE" :
                    modify_de(network, params)
                    result = rerun_solve(network, sources)
                    filename = f"{output_path}de_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                elif param_type == "ground":
                    modify_ground(network, params)
                    modified_sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)
                    result = rerun_solve(network, modified_sources)
                    filename = f"{output_path}ground_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                else:
                    if param_type == "Arrester":
                        modify_arrester(network, params)
                        result = rerun_full(network, load_dict)
                        filename = f"{output_path}{param_type.lower()}_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                    elif param_type == "SW" :
                        modify_sw(network, params)
                        result = rerun_full(network, load_dict)
                        filename = f"{output_path}{param_type.lower()}_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                    elif param_type == "ROD":
                        modify_rod(network, params)
                        result = rerun_full(network, load_dict)
                        filename = f"{output_path}{param_type.lower()}_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
                results_after[param_type] = result
                pd.DataFrame(result if use_hybrid else result[0]).to_csv(filename)
                print(f"Result saved to {filename}")
        return result_before, results_after

    elif mode == 2:
        print("Running simultaneous modifications")
        modified_sources = sources
        for param_type, params in modifications.items():
            if param_type == "Stroke":
                modified_sources = modify_stroke(network, load_dict, params)
            elif param_type == "Soil":
                modify_soil(network, params)
                modified_sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)
            elif param_type == "Arrester":
                modify_arrester(network, params)
            elif param_type == "SW":
                modify_sw(network, params)
            elif param_type == "ROD":
                modify_rod(network, params)
            elif param_type == "DE":
                modify_de(network, params)
            elif param_type == "ground":
                modify_ground(network, params)
                modified_sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)

        if any(param_type in ["Arrester", "SW", "ROD"] for param_type in modifications):
            result_after = rerun_full(network, load_dict)
        else:
            result_after = rerun_solve(network, modified_sources)

        filename = f"{output_path}combined_modified_{'hybrid' if use_hybrid else 'base'}_output.csv"
        pd.DataFrame(result_after if use_hybrid else result_after[0]).to_csv(filename)
        print(f"Combined result saved to {filename}")
        return result_before, result_after

    elif mode == 3:
        print("Running sequential modifications")
        current_sources = sources
        results_after = {}
        for param_type, params in modifications.items():
            print(f"Modifying {param_type}")
            if param_type == "Stroke":
                current_sources = modify_stroke(network, load_dict, params)
                result = rerun_solve(network, current_sources)
            elif param_type == "Soil":
                modify_soil(network, params)
                current_sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)
                result = rerun_solve(network, current_sources)
            elif param_type == "DE":
                modify_de(network, params)
                result = rerun_solve(network, current_sources)
            elif param_type == "ground":
                modify_ground(network, params)
                current_sources = network.source_initial(load_dict, nodes, branches, constants, share_dict)
                result = rerun_solve(network, current_sources)
            else:
                if param_type == "Arrester":
                    modify_arrester(network, params)
                elif param_type == "SW":
                    modify_sw(network, params)
                elif param_type == "ROD":
                    modify_rod(network, params)
                result = rerun_full(network, load_dict)

            filename = f"{output_path}{param_type.lower()}_sequential_{'hybrid' if use_hybrid else 'base'}_output.csv"
            pd.DataFrame(result if use_hybrid else result[0]).to_csv(filename)
            print(f"Result after modifying {param_type} saved to {filename}")
            results_after[param_type] = result

        return result_before, results_after

    return result_before, None