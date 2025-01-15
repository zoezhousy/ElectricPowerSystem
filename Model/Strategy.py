from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from Driver.modeling.tower_modeling import tower_building
from Driver.initialization.initialization import initial_device, initial_lump, initialize_tower
import json
import pickle
from functools import reduce
import copy
from scipy.linalg import block_diag
from tqdm import tqdm
# import cupy as cp
from Utils.GPU import transfer_date_to_gpu, transfer_data_to_cpu

from Model import Network


class Strategy(ABC):

    def __init__(self):
        self.capacitance_matrix = None

    @abstractmethod
    def apply(self,netwotk,dt,Nt):
        C = np.array(netwotk.capacitance_matrix)  # 点点
        G = np.array(netwotk.conductance_matrix)
        L = np.array(netwotk.inductance_matrix)  # 线线
        R = np.array(netwotk.resistance_matrix)
        ima = np.array(netwotk.incidence_matrix_A)  # 线点
        imb = np.array(netwotk.incidence_matrix_B.T)  # 点线




class Change_light_pos(Strategy):
    def apply(self,network,lightning,area,wire,new_pos):
        print("change light position calculation is used")

        lightning.channel.hit_pos = new_pos
        cir_id = wire['cir_id']
        if wire['type'] == 'SW':
            bran = 'Y' + str(cir_id) + 'S'
            network.sources = network.source_calculate(lightning, area, bran, new_pos)
        elif wire['type'] == 'CIRO':
            bran = 'Y' + str(cir_id) + wire['phase']
            network.sources = network.source_calculate(lightning,area,bran, new_pos)
        else:
            network.sources = network.source_calculate(lightning, area, wire["name"], new_pos)
        network.calculate(network.dt,network.H,network.sources)

class Change_light_waveform(Strategy):
    def apply(self, network, lightning, new_waveform, pos_dict):
        print("change light waveform calculation is used")

        lightning.channel.hit_pos = new_waveform
        network.sources = network.source_calculate(lightning, pos_dict)
        network.calculate(network.dt, network.H, network.sources)

class Change_light_parameters(Strategy):
    def apply(self, network, lightning, new_parameters, pos_dict):
        print("change light parameters calculation is used")
        for stroke in lightning.strokes:
            stroke.parameters = new_parameters #
            stroke.current_waveform = []
            stroke.calculate()
        network.sources = network.source_calculate(lightning, pos_dict)
        network.calculate(network.dt, network.H, network.sources)

class Change_ROD(Strategy):
    def apply(self, network, load_dict,lumpname,r,l):
        print("change ROD calculation is used")
        for tower in load_dict["Tower"]:
            for lump in tower["Lump"]:
                if lump["name"] == lumpname:
                    lump["value1"] = r
                    lump["value2"] = l
        network.towers = []
        network.initial_tower(load_dict)
        network.combine_parameter_matrix()
        network.calculate(network.dt, network.H, network.sources)
        pd.DataFrame(network.run_measure()).to_csv("ROD_modified.csv")

class Change_ground(Strategy):
    def apply(self, load_dict, new_parameter):
        print("change ground calculation")
        load_dict['Global']["ground"] = new_parameter  # 修改值

        with open("modified", 'w') as file:
            json.dump(load_dict, file, indent=4)

        network = Network()
        network.run(load_dict)

class Change_Arrestor_pos(Strategy):
    def apply(self, network, load_dict, remove_tower_list):
        print("changing arrestor position calculation")
        # for tower in network.towers:
        #     for device in tower.devices:
        #         for arrestor in device.arrestors:
        #             if arrestor.name == name:
        #                 arrestor.capacitance_matrix = arrestor.capacitance_matrix.rename(columns=node_mapping,index=node_mapping)
        #                 arrestor.inductance_matrix = arrestor.inductance_matrix.rename(columns=wire_mapping,index=wire_mapping)
        #                 arrestor.incidence_matrix_A = arrestor.incidence_matrix_A.rename(columns=node_mapping,index=wire_mapping)
        #                 arrestor.inductance_matrix_B = arrestor.inductance_matrix_B.rename(columns=node_mapping,index=wire_mapping)
        #                 arrestor.resistance_matrix = arrestor.resistance_matrix.rename(columns=wire_mapping,index=wire_mapping)
        #                 arrestor.conductance_matrix = arrestor.conductance_matrix.rename(columns=node_mapping,index=node_mapping)
        #     tower.reset_matrix()
        #     gnd = network.ground if network.global_ground == 1 else tower.ground
        #     tower_building(tower, network.f0, network.max_length, gnd, network.varied_frequency)
        #
        #     network.combine_parameter_matrix()
        #     network.calculate(network.dt,network.H,network.sources)
        for tower in load_dict["Tower"]:
            if tower['name'] in remove_tower_list:
                tower["Device"] = [device for device in tower["Device"] if device['name'].split("_") != "Arrester"]
        network.run(load_dict)
        with open("modified_Arrester", 'w') as file:
            json.dump(load_dict, file, indent=4)

class Change_SW(Strategy):
    def apply(self, network, load_dict, name):

        for ohl in load_dict['OHL']:
            #if ohl.name == name:
                # if position:
                #     ohl["position"] = position # 修改值
                # if bran:
                #     ohl["bran"] = bran # 修改值
                # load_dict['OHL'][index] = ohl
            if ohl.name == name:
                ohl["Wire"] = [wire for wire in ohl["Wire"] if wire['type'] != "SW"]
        with open("modified_SW", 'w') as file:
            json.dump(load_dict, file, indent=4)

        network.run(load_dict)

class Change_DE_max(Strategy):
    def apply(self, network, DE_max):
        print("Changing DE max calculation is used")
        for index, lump in enumerate(network.switch_disruptive_effect_models):
            lump.parameters['DE_max'] = DE_max
            switch_disruptive_copy = copy.deepcopy(network.switch_disruptive_effect_models)
            switch_disruptive_copy[index] = lump
        network.H["switch_disruptive_effect_models"] = switch_disruptive_copy



class NonLinear(Strategy):
    def apply(self,Nt,dt,H,sources):
        print("Nonlinear calculation is used")
        branches, nodes = H["incidence_matrix_A"].shape
        source = np.array(sources)
        time_length = len(sources.columns.tolist())
        if Nt > time_length:
            Nt = time_length
        else:
            source = source[:, :Nt]
        out = np.zeros((branches + nodes, Nt))
        #stroke_len = time_length/stroke_num-1
        # source = np.array(sources)
        #switch_disruptive_effect = {}
        FO = []
        SAF = []
        E_cum = 0
        C = H["capacitance_matrix"].to_numpy()  # 点点
        G = H["conductance_matrix"].to_numpy()
        L = H["inductance_matrix"].to_numpy()  # 线线
        R = H["resistance_matrix"].to_numpy()
        ima = H["incidence_matrix_A"].to_numpy()  # 线点
        imb = H["incidence_matrix_B"].T.to_numpy()  # 点线

        for i in tqdm(range(Nt - 1)):
            #stroke_index = (i+1)//stroke_len

            Vnode = out[:nodes, i]
            Ibran = out[nodes:, i]
            Isource = source[:, i + 1]
            LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
            inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.hstack((- L @ Ibran / dt, C @ Vnode / dt))
            temp_result = inv_LEFT.dot(Isource + RIGHT)
            # temp_result = inv_LEFT.dot(RIGHT)
            out[:, i + 1] = np.copy(temp_result)
            temp_result = pd.DataFrame(temp_result,
                                       index=H["capacitance_matrix"].columns.tolist() + H["inductance_matrix"].columns.tolist())
            t = dt * (i + 1)
            H,FO = self.update_H(temp_result, t,H,dt)

        # for switch in H["switch_disruptive_effect_models"]:
        #     if switch.on_off==-1:
        #         FO.append(switch.name)
        solution = pd.DataFrame(out,index=H["capacitance_matrix"].columns.tolist() + H["inductance_matrix"].columns.tolist())


        return solution,FO

    def update_H(self, current_result, time, H, dt):
        FO = []
        for switch_v_list in [H["switch_disruptive_effect_models"], H["voltage_controled_switchs"]]:
            for switch_v in switch_v_list:
                v1 = current_result.loc[switch_v.node1[0], 0] if switch_v.node1[0] != 'ref' else 0
                v2 = current_result.loc[switch_v.node2[0], 0] if switch_v.node2[0] != 'ref' else 0
                resistance = switch_v.update_parameter(abs(v1 - v2), dt)
                if switch_v.on_off == -1:
                    FO.append(switch_v.name)
                H["resistance_matrix"].loc[switch_v.bran[0], switch_v.bran[0]] = resistance

        for switch_t in H["time_controled_switchs"]:
            resistance = switch_t.update_parameter(time)
            H["resistance_matrix"].loc[switch_t.bran[0], switch_t.bran[0]] = resistance

        for nolinear_resistor in H["nolinear_resistors"]:
            component_current = abs(current_result.loc[nolinear_resistor.bran[0], 0])
            resistance = nolinear_resistor.update_parameter(component_current)
            H["resistance_matrix"].loc[nolinear_resistor.bran[0], nolinear_resistor.bran[0]] = resistance
        return H,FO

class Linear(Strategy):
    def apply(self,Nt,dt,H,sources):
        print("linear calculation is used")
        C = np.array(H["capacitance_matrix"])  # 点点
        G = np.array(H["conductance_matrix"])
        L = np.array(H["inductance_matrix"])  # 线线
        R = np.array(H["resistance_matrix"])
        ima = np.array(H["incidence_matrix_A"])  # 线点
        imb = np.array(H["incidence_matrix_B"].T)  # 点线
        source = np.array(sources)
        nodes = len(H["capacitance_matrix"].columns.tolist())
        branches = len(H["inductance_matrix"].columns.tolist())
        time_length = len(sources.columns.tolist())
        if Nt > time_length:
            Nt = time_length
        else:
            source = source[:, :Nt]
        out = np.zeros((branches + nodes, Nt))
        branches, nodes = ima.shape
        LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
        inv_LEFT = np.linalg.inv(LEFT)
        for i in tqdm(range(Nt - 1)):
            Vnode = out[:nodes, i]
            Ibran = out[nodes:, i]
            Isource = source[:, i + 1]
            #LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
            #inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.hstack((- L @ Ibran / dt, C @ Vnode / dt))
            # temp_result = inv_LEFT.dot(RIGHT)
            temp_result = inv_LEFT @ (Isource + RIGHT)
            out[:, i + 1] = np.copy(temp_result)
        solution = pd.DataFrame(out,
                                        index=H["capacitance_matrix"].columns.tolist() + H["inductance_matrix"].columns.tolist())
        return solution

class MC_Linear(Strategy):
    def apply(self,Nt,dt,H,sources,RL_node):
        print("linear calculation is used")
        C = np.array(H["capacitance_matrix"])  # 点点
        G = np.array(H["conductance_matrix"])
        L = np.array(H["inductance_matrix"])  # 线线
        R = np.array(H["resistance_matrix"])
        ima = np.array(H["incidence_matrix_A"])  # 线点
        imb = np.array(H["incidence_matrix_B"].T)  # 点线
        source = np.array(sources)
        nodes = len(H["capacitance_matrix"].columns.tolist())
        branches = len(H["inductance_matrix"].columns.tolist())
        time_length = len(sources.columns.tolist())
        if Nt > time_length:
            Nt = time_length
        else:
            source = source[:, :Nt]
        out = np.zeros((branches + nodes, Nt))
        branches, nodes = ima.shape
        LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
        inv_LEFT = np.linalg.inv(LEFT)
        for i in tqdm(range(Nt - 1)):
            Vnode = out[:nodes, i]
            Ibran = out[nodes:, i]
            Isource = source[:, i + 1]
            RIGHT = np.hstack((- L @ Ibran / dt, C @ Vnode / dt))
            temp_result = inv_LEFT @ (Isource + RIGHT)

            out[:, i + 1] = np.copy(temp_result)
        solution = pd.DataFrame(out,index=H["capacitance_matrix"].columns.tolist() + H["inductance_matrix"].columns.tolist())
        # for x in tower_head_node:
        #     #if abs(np.max(out[H["capacitance_matrix"].columns.tolist().index(x[0]), :])) > 75000:
        #     if np.max(abs(out[H["capacitance_matrix"].columns.tolist().index(x), :])) > 75000:
        #         return True
        for x in RL_node:
            if (solution.loc[x[0]]-solution.loc[x[1]]).max() > 75000:
                return True
        return  False

class INS_NonLinear(Strategy):
    def apply(self,Nt,dt,H,sources):
        print("Nonlinear calculation is used")
        branches, nodes = H["incidence_matrix_A"].shape
        source = np.array(sources)
        time_length = len(sources.columns.tolist())
        if Nt > time_length:
            Nt = time_length
        else:
            source = source[:, :Nt]
        out = np.zeros((branches + nodes, Nt))
        #stroke_len = time_length/stroke_num-1
        # source = np.array(sources)
        #switch_disruptive_effect = {}
        FO = []
        SAF = []
        E_cum = 0
        for i in range(Nt - 1):
            #stroke_index = (i+1)//stroke_len
            C = H["capacitance_matrix"].to_numpy()  # 点点
            G = H["conductance_matrix"].to_numpy()
            L = H["inductance_matrix"].to_numpy()  # 线线
            R = H["resistance_matrix"].to_numpy()
            ima = H["incidence_matrix_A"].to_numpy()  # 线点
            imb = H["incidence_matrix_B"].T.to_numpy()  # 点线
            Vnode = out[:nodes, i]
            Ibran = out[nodes:, i]
            Isource = source[:, i + 1]
            LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
            inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.hstack((- L @ Ibran / dt, C @ Vnode / dt))

            temp_result = inv_LEFT.dot(Isource + RIGHT)
            # temp_result = inv_LEFT.dot(RIGHT)
            out[:, i + 1] = np.copy(temp_result)


            temp_result = pd.DataFrame(temp_result,
                                       index=H["capacitance_matrix"].columns.tolist() + H["inductance_matrix"].columns.tolist())


            t = dt * (i + 1)
            H,FO = self.update_H(temp_result, t,H,dt)

            #             switch.DE = 0
            if FO:
                return FO
        #solution = pd.DataFrame(out,index=H["capacitance_matrix"].columns.tolist() + H["inductance_matrix"].columns.tolist())
        return FO

    def update_H(self, current_result, time, H, dt):
        FO = False
        for switch_v_list in [H["switch_disruptive_effect_models"], H["voltage_controled_switchs"]]:
            for switch_v in switch_v_list:
                v1 = current_result.loc[switch_v.node1[0], 0] if switch_v.node1[0] != 'ref' else 0
                v2 = current_result.loc[switch_v.node2[0], 0] if switch_v.node2[0] != 'ref' else 0
                resistance = switch_v.update_parameter(abs(v1 - v2), dt)
                H["resistance_matrix"].loc[switch_v.bran[0], switch_v.bran[0]] = resistance
                if switch_v.on_off == -1:
                    FO = True

        for switch_t in H["time_controled_switchs"]:
            resistance = switch_t.update_parameter(time)
            H["resistance_matrix"].loc[switch_t.bran[0], switch_t.bran[0]] = resistance

        for nolinear_resistor in H["nolinear_resistors"]:
            component_current = abs(current_result.loc[nolinear_resistor.bran[0], 0])
            resistance = nolinear_resistor.update_parameter(component_current)
            H["resistance_matrix"].loc[nolinear_resistor.bran[0], nolinear_resistor.bran[0]] = resistance
        return H, FO


class SAF_NonLinear(Strategy):
    def apply(self,Nt,dt,H,sources,record):
        print("Nonlinear calculation is used")
        branches, nodes = H["incidence_matrix_A"].shape
        source = np.array(sources)
        time_length = len(sources.columns.tolist())
        if Nt > time_length:
            Nt = time_length
        else:
            source = source[:, :Nt]
        out = np.zeros((branches + nodes, Nt))
        #stroke_len = time_length/stroke_num-1
        # source = np.array(sources)
        #switch_disruptive_effect = {}
        FO = []
        SAF = []
        E_cum = 0
        for i in range(Nt - 1):
            #stroke_index = (i+1)//stroke_len
            C = H["capacitance_matrix"].to_numpy()  # 点点
            G = H["conductance_matrix"].to_numpy()
            L = H["inductance_matrix"].to_numpy()  # 线线
            R = H["resistance_matrix"].to_numpy()
            ima = H["incidence_matrix_A"].to_numpy()  # 线点
            imb = H["incidence_matrix_B"].T.to_numpy()  # 点线
            Vnode = out[:nodes, i]
            Ibran = out[nodes:, i]
            Isource = source[:, i + 1]
            LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
            inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.hstack((- L @ Ibran / dt, C @ Vnode / dt))

            temp_result = inv_LEFT.dot(Isource + RIGHT)
            # temp_result = inv_LEFT.dot(RIGHT)
            out[:, i + 1] = np.copy(temp_result)


            temp_result = pd.DataFrame(temp_result,
                                       index=H["capacitance_matrix"].columns.tolist() + H["inductance_matrix"].columns.tolist())

            for key,value in  record.items():
                if not E_cum[key]:
                    E_cum[key] = 0
                E =  temp_result[value[0]]*abs(temp_result[value[2]]-temp_result[value[1]]) *dt
                E = E_cum[key] + E
                E_cum[key] = E
                if E_cum[key]>64000:
                    SAF.append(key)

            t = dt * (i + 1)
            H = self.update_H(temp_result, t,H,dt)
            # if (i+1)%stroke_len==0:
            #     if stroke_index not in switch_disruptive_effect:
            #         switch_disruptive_effect[stroke_index] = {}
            #     for switch in H["switch_disruptive_effect_models"]:
            #         switch_disruptive_effect[stroke_index][switch.name] = switch.on_off
            #         if switch.on_off ==1:
            #             switch.DE = 0
        for switch in H["switch_disruptive_effect_models"]:
            if switch.on_off==-1:
                FO.append(switch.name)
        solution = pd.DataFrame(out,index=H["capacitance_matrix"].columns.tolist() + H["inductance_matrix"].columns.tolist())


        return solution,FO,SAF

    def update_H(self, current_result, time, H, dt):
        for switch_v_list in [H["switch_disruptive_effect_models"], H["voltage_controled_switchs"]]:
            for switch_v in switch_v_list:
                v1 = current_result.loc[switch_v.node1[0], 0] if switch_v.node1[0] != 'ref' else 0
                v2 = current_result.loc[switch_v.node2[0], 0] if switch_v.node2[0] != 'ref' else 0
                resistance = switch_v.update_parameter(abs(v1 - v2), dt)
                H["resistance_matrix"].loc[switch_v.bran[0], switch_v.bran[0]] = resistance

        for switch_t in H["time_controled_switchs"]:
            resistance = switch_t.update_parameter(time)
            H["resistance_matrix"].loc[switch_t.bran[0], switch_t.bran[0]] = resistance

        for nolinear_resistor in H["nolinear_resistors"]:
            component_current = abs(current_result.loc[nolinear_resistor.bran[0], 0])
            resistance = nolinear_resistor.update_parameter(component_current)
            H["resistance_matrix"].loc[nolinear_resistor.bran[0], nolinear_resistor.bran[0]] = resistance
        return H

class variant_frequency(Strategy):
    def apply(self, Nt, dt, H, sources):
        print("Variant_frequency calculation is used")

        branches, nodes = H["incidence_matrix_A"].shape
        source = np.array(sources)
        time_length = len(sources.columns.tolist())
        if Nt > time_length:
            Nt = time_length
        else:
            source = source[:, :Nt]
        out = np.zeros((branches + nodes, Nt))
        C = H["capacitance_matrix"].to_numpy()  # 点点
        G = H["conductance_matrix"].to_numpy()
        L = H["inductance_matrix"].to_numpy()  # 线线
        R = H["resistance_matrix"].to_numpy()
        ima = H["incidence_matrix_A"].to_numpy()  # 线点
        imb = H["incidence_matrix_B"].T.to_numpy()  # 点线
        result_index = H["capacitance_matrix"].columns.tolist() + H["inductance_matrix"].columns.tolist()
        for i in range(time_length - 1):
            Vnode = out[:nodes, i].reshape((-1, 1))
            Ibran = out[nodes:, i].reshape((-1, 1))
            pre_result = pd.DataFrame(np.vstack((Vnode, Ibran)), index=result_index)
            self.update_source_variant_frequency(pre_result, i)
            source = sources.to_numpy()[:, i].reshape((-1, 1))
            LEFT = np.block([[-ima, -R - L / dt], [G + C / dt, -imb]])
            # inv_LEFT = np.linalg.inv(LEFT)
            RIGHT = np.block([[(-L / dt).dot(Ibran)], [(C / dt).dot(Vnode)]])

            # temp_result = inv_LEFT.dot(source + RIGHT)
            temp_result = np.linalg.solve(LEFT, source + RIGHT)
            # temp_result = inv_LEFT.dot(RIGHT)
            out[:, i + 1] = np.copy(temp_result)[:, 0]
            # temp_result = pd.DataFrame(temp_result, index=result_index)
            # print(i)
            # network.update_source_variant_frequency(temp_result, i + 2)

        solution = pd.DataFrame(out, index=result_index)

    def update_source_variant_frequency(self, current_result, next_point, sources, towers, OHLs, cables):
        for tower in towers:
            I = current_result.loc[tower.wires_name, 0].to_numpy()
            phi_temp = []
            for i in range(tower.A.shape[-1]):
                phi_temp.append(tower.A[:, :, i].dot(I))
            phi = np.expand_dims(np.array(phi_temp), axis=2).transpose(1, 2, 0)
            tower.phi = phi + tower.B * tower.phi
            phi_hist = (tower.B * tower.phi).sum(-1)
            sources.loc[tower.wires_name, next_point] += phi_hist.reshape(-1)

        for model_list in [OHLs, cables]:
            for model in model_list:
                I = current_result.loc[model.wires_name, 0].to_numpy()
                n = int(I.shape[0] / model.A.shape[0])
                phi_temp = []
                for i_fit in range(model.A.shape[-1]):
                    A = np.copy(model.A[:, :, i_fit])
                    for i in range(n - 1):
                        A = block_diag(A, model.A[:, :, i_fit])
                    phi_temp.append(A.dot(I))
                phi = np.expand_dims(np.array(phi_temp), axis=2).transpose(1, 2, 0)
                B = np.tile(model.B, (int(model.phi.shape[0] / model.B.shape[0]), 1, 1))
                model.phi = phi + B * model.phi
                phi_hist = (B * model.phi).sum(-1)
                sources.loc[model.wires_name, next_point] = sources.loc[
                                                                model.wires_name, next_point].values + phi_hist.reshape(
                    -1)



class Hybrid_Strategy:

    def __init__(self):
        pass

    @abstractmethod
    def preparing_nonlinear_matrix(self, SDEM, VCS, TCS, NLR, index_t, columns_t):
        temp1 = pd.Series(range(columns_t.shape[0]), index=columns_t)
        temp2 = pd.Series(range(index_t.shape[0]), index=index_t)
        SDEM_index = np.zeros((SDEM.shape[0], 3), dtype=int)
        SDEM_index[:, 0] = temp2.loc[SDEM[:, 0]].to_list()
        SDEM_index[:, 1] = temp1.loc[SDEM[:, 1]].to_list()
        SDEM_index[:, 2] = temp1.loc[SDEM[:, 2]].to_list()
        SDEM_para = SDEM[:, 3:].astype(float)

        VCS_index = np.zeros((VCS.shape[0], 3), dtype=int)
        VCS_index[:, 0] = temp2.loc[VCS[:, 0]].to_list()
        VCS_index[:, 1] = temp1.loc[VCS[:, 1]].to_list()
        VCS_index[:, 2] = temp1.loc[VCS[:, 2]].to_list()
        VCS_para = VCS[:, 3].astype(float)

        TCS_index = np.zeros((TCS.shape[0],), dtype=int)
        TCS_index[:] = temp2.loc[TCS[:, 0]].to_list()
        TCS_para = TCS[:, 1:].astype(float)

        NLR_index = np.zeros((NLR.shape[0], 3), dtype=int)
        NLR_index[:, 0] = temp2.loc[NLR[:, 0]].to_list()
        NLR_index[:, 1] = temp1.loc[NLR[:, 1]].to_list()
        NLR_index[:, 2] = temp1.loc[NLR[:, 2]].to_list()
        return SDEM_index, SDEM_para, VCS_index, VCS_para, TCS_index, TCS_para, NLR_index, NLR[:, 3]

    def preparing_tower_matrix(self, tower_matrix, line_capacitance, sources, Nt, GPU):

        towers_calculation = []
        for tower in tower_matrix:
            ima_t = tower['incidence_matrix_A'].to_numpy()
            imb_t = tower['incidence_matrix_B'].to_numpy()
            R_t = tower['resistance_matrix'].to_numpy()
            L_t = tower['inductance_matrix'].to_numpy()
            G_t = tower['conductance_matrix'].to_numpy()
            C_t = tower['capacitance_matrix'].to_numpy()
            A_t = tower['A'].transpose(0, 2, 1)
            B_t = tower['B']
            phi_t = tower['phi']

            index_t = tower['incidence_matrix_A'].index
            columns_t = tower['incidence_matrix_A'].columns
            columns_o = line_capacitance.columns

            source_t_index = index_t.tolist()
            source_t_index.extend(columns_t.tolist())

            out_t_index = columns_t.tolist()
            out_t_index.extend(index_t.to_list())

            tower_nodes = set(columns_t.tolist())
            line_nodes = set(line_capacitance.index.tolist())
            tower_and_line_nodes = list(tower_nodes.intersection(line_nodes))
            tower_and_line_nodes.sort()
            temp = pd.Series(range(columns_t.shape[0]), index=columns_t)
            tower_cross_index = temp.loc[tower_and_line_nodes].to_list()
            source_t = sources.loc[source_t_index, :].to_numpy()
            temp = pd.Series(range(columns_o.shape[0]), index=columns_o)
            ohl_cross_index = temp.loc[tower_and_line_nodes].to_list()

            C0 = line_capacitance.loc[tower_and_line_nodes, tower_and_line_nodes]
            Nout_tower = len(tower_and_line_nodes)
            Cw = pd.DataFrame(np.zeros((Nout_tower, len(tower_nodes))), index=tower_and_line_nodes,
                              columns=columns_t.to_list())
            Cw.loc[tower_and_line_nodes, tower_and_line_nodes] = np.eye(Nout_tower)
            E = np.eye(Nout_tower)
            Nbran_tower, Nnode_tower = ima_t.shape
            zero = np.zeros((Nbran_tower, Nout_tower))

            out_t = np.zeros((Nbran_tower + Nnode_tower + Nout_tower, Nt))

            Cwn = Cw.to_numpy()
            C0n = C0.to_numpy() * 2
            # 构造非线性更新矩阵，便于利用矩阵形式进行非线性计算
            SDEM, VCS, TCS, NLR = tower['SDEM'], tower['VCS'], tower['TCS'], tower['NLR']
            SDEM_index, SDEM_para, VCS_index, VCS_para, TCS_index, TCS_para, NLR_index, NLR_para  = self.preparing_nonlinear_matrix(SDEM, VCS, TCS, NLR, index_t, columns_t)
            SDEM_bran = SDEM[:, 0].tolist()
            NLR_bran = NLR[:, 0].tolist()

            if GPU:
                A_t, B_t, phi_t = transfer_date_to_gpu(A_t, B_t, phi_t)
                ima_t, imb_t, R_t, G_t, L_t, C_t = transfer_date_to_gpu(ima_t, imb_t, R_t, G_t, L_t, C_t)
                source_t, C0n, Cwn, zero, out_t = transfer_date_to_gpu(source_t, C0n, Cwn, zero, out_t)
                SDEM_index, SDEM_para, VCS_index  = transfer_date_to_gpu(SDEM_index, SDEM_para, VCS_index)
                VCS_para, TCS_index, TCS_para = transfer_date_to_gpu(VCS_para, TCS_index, TCS_para)

            tower_cal = {'ima': ima_t, 'imb': imb_t, 'R': R_t, 'L': L_t, 'G': G_t, 'C': C_t, 'out': out_t,
                         'source': source_t, 'tower_cross_index': tower_cross_index, 'C0n': C0n, 'Cwn': Cwn, 'E': E,
                         'A': A_t, 'B': B_t, 'phi': phi_t, 'SDEM_index': SDEM_index,
                         'SDEM_para': SDEM_para, 'VCS_index': VCS_index, 'VCS_para': VCS_para, 'TCS_index': TCS_index,
                         'TCS_para': TCS_para, 'Nnode': Nnode_tower, 'Nbran': Nbran_tower,
                         'ohl_cross_index': ohl_cross_index, 'node_index': columns_t.tolist(),
                         'bran_index': index_t.tolist(), 'zero': zero, 'vf_bran': tower['vf_bran'],
                         'NLR_index': NLR_index, 'NLR_para': NLR_para, 'NLR_E': 0, 'SDEM_bran': SDEM_bran,
                         'NLR_bran': NLR_bran}

            towers_calculation.append(tower_cal)

        return towers_calculation

    def combine_results(self, T_cal, Nt, GPU):
        results_tower = pd.DataFrame()
        for itcal in T_cal:
            result = itcal['out'][:itcal['Nbran'] + itcal['Nnode'], :]
            if GPU:
                [result] = transfer_data_to_cpu(result)
            result = pd.DataFrame(result, index=itcal['node_index']+itcal['bran_index'], columns=range(Nt))
            results_tower = results_tower.add(result, fill_value=0).fillna(0)
        return results_tower


class hybrid_linear(Hybrid_Strategy):
    def __init__(self):
        super().__init__()

    def apply(self, line_matrix, tower_matrix, sources, Nt, dt, GPU,tower_head_node):
        print("Linear Hybrid calculation is used")

        im_o_df = line_matrix['incidence_matrix']
        im_o = im_o_df.to_numpy()
        L_o = line_matrix['inductance_matrix'].to_numpy()
        C_o = line_matrix['capacitance_matrix'].to_numpy()
        R_o = line_matrix['resistance_matrix'].to_numpy()
        G_o = line_matrix['conductance_matrix'].to_numpy()
        index_o = im_o_df.index.to_numpy()
        columns_o = im_o_df.columns.to_numpy()
        vs_o = sources.loc[index_o, :].to_numpy()
        is_o = sources.loc[columns_o, :].to_numpy()
        Nbran_ohl, Nnode_ohl = im_o_df.shape

        T_cal = self.preparing_tower_matrix(tower_matrix, line_matrix['capacitance_matrix'], sources, Nt, GPU)

        for itcal in T_cal:
            L_upper = np.hstack((-itcal['ima'], -itcal['R'] - itcal['L'] / dt, itcal['zero']))
            L_middle = np.hstack((itcal['G'] + itcal['C'] / dt, -itcal['imb'].T, -itcal['Cwn'].T))
            L_down = np.hstack((itcal['C0n'] @ itcal['Cwn'] / dt, itcal['zero'].T, itcal['E']))
            LEFT = np.vstack((L_upper, L_middle, L_down))
            inv_LEFT = np.linalg.inv(LEFT)
            itcal['inv_LEFT'] = inv_LEFT

        if GPU:
            import cupy as cp
            print("GPU calculation is used")
            L_o, R_o, C_o, G_o, vs_o, is_o, im_o = transfer_date_to_gpu(L_o, R_o, C_o, G_o, vs_o, is_o, im_o)

            inv = cp.linalg.inv
            zeros = cp.zeros
            hstack = cp.hstack
            copy = cp.copy
        else:
            inv = np.linalg.inv
            zeros = np.zeros
            hstack = np.hstack
            copy = np.copy

        inv_RL_o = inv(L_o / dt + R_o / 2)
        Vout_ohl = zeros((Nnode_ohl, Nt + 1))
        Iout_ohl = zeros((Nbran_ohl, Nt))
        LdeR = L_o / dt - R_o / 2
        inv_GC_o = inv(C_o / dt + G_o / 2)
        CdeG = C_o / dt - G_o / 2
        I_allcross = zeros((Nnode_ohl, ))
        for i in tqdm(range(Nt - 1)):
            # tower solution
            for itcal in T_cal:
                out_t = itcal['out']
                Nnode_tower = itcal['Nnode']
                Nbran_tower = itcal['Nbran']
                Vnode_t = out_t[:Nnode_tower, i]
                Ibran_t = out_t[Nnode_tower:Nbran_tower + Nnode_tower, i]
                Ivitual_t = out_t[Nbran_tower + Nnode_tower:, i]
                source_v_t_temp = itcal['source'][:, i + 1]

                RIGHT = hstack(
                    ((-itcal['L'] / dt).dot(Ibran_t), (itcal['C'] / dt).dot(Vnode_t), -Ivitual_t + itcal['C0n'] @ itcal['Cwn'] @ Vnode_t / dt))
                source_v_t_temp = hstack((source_v_t_temp, 2 * I_allcross[itcal['ohl_cross_index']]))
                temp_result = itcal['inv_LEFT'] @ (source_v_t_temp + RIGHT)

                itcal['out'][:, i + 1] = copy(temp_result)
                Vout_ohl[itcal['ohl_cross_index'], i + 1] = temp_result[itcal['tower_cross_index']]

            # ohl solution
            Iout_ohl[:, i + 1] = inv_RL_o @ (LdeR @ Iout_ohl[:, i] - im_o @ Vout_ohl[:, i + 1] - vs_o[:, i + 1])
            Vout_ohl[:, i + 2] = inv_GC_o @ (CdeG @ Vout_ohl[:, i + 1] + im_o.T @ Iout_ohl[:, i + 1] + is_o[:, i + 1])

            I_allcross = im_o.T @ Iout_ohl[:, i + 1]
        Vout_ohl = Vout_ohl[:, :-1]
        results_tower = self.combine_results(T_cal, Nt, GPU)
        if GPU:
            Vout_ohl = cp.asnumpy(Vout_ohl)
            Iout_ohl = cp.asnumpy(Iout_ohl)
        result_v_ohl = pd.DataFrame(Vout_ohl, index=columns_o, columns=np.arange(Nt))
        # result_i_ohl = pd.DataFrame(Iout_ohl, index=index_o, columns=np.arange(Nt) + 0.5)
        result_i_ohl = pd.DataFrame(Iout_ohl, index=index_o, columns=np.arange(Nt))
        results = pd.concat([results_tower, result_v_ohl])
        results.drop_duplicates(inplace=True)
        results = results.add(result_i_ohl, fill_value=0).fillna(0)
        broke = {"SDEM":[]}
        for x in tower_head_node:
            #if abs(np.max(out[H["capacitance_matrix"].columns.tolist().index(x[0]), :])) > 75000:
            if results.loc[x].abs().max() > 75000:
                broke["SDEM"].append(x)

        return results, broke


class hybrid_variant_frequency(Hybrid_Strategy):
    def __init__(self):
        super().__init__()

    def apply(self, line_matrix, tower_matrix, sources, Nt, dt, GPU):
        print("Variant frequent hybrid calculation is used")

        im_o_df = line_matrix['incidence_matrix']
        im_o = im_o_df.to_numpy()
        L_o = line_matrix['inductance_matrix'].to_numpy()
        C_o = line_matrix['capacitance_matrix'].to_numpy()
        R_o = line_matrix['resistance_matrix'].to_numpy()
        G_o = line_matrix['conductance_matrix'].to_numpy()
        index_o = im_o_df.index.to_numpy()
        columns_o = im_o_df.columns.to_numpy()
        vs_o = sources.loc[index_o, :].to_numpy()
        is_o = sources.loc[columns_o, :].to_numpy()
        Nbran_ohl, Nnode_ohl = im_o_df.shape

        T_cal = self.preparing_tower_matrix(tower_matrix, line_matrix['capacitance_matrix'], sources, Nt, GPU)

        temp = pd.Series(range(index_o.shape[0]), index=index_o)
        vf_bran_o = temp.loc[line_matrix['vf_bran']].to_list()
        A_o = line_matrix['A']
        B_o = line_matrix['B']
        phi_o = np.zeros_like(B_o)

        for itcal in T_cal:
            L_upper = np.hstack((-itcal['ima'], -itcal['R'] - itcal['L'] / dt, itcal['zero']))
            L_middle = np.hstack((itcal['G'] + itcal['C'] / dt, -itcal['imb'].T, -itcal['Cwn'].T))
            L_down = np.hstack((itcal['C0n'] @ itcal['Cwn'] / dt, itcal['zero'].T, itcal['E']))
            LEFT = np.vstack((L_upper, L_middle, L_down))
            inv_LEFT = np.linalg.inv(LEFT)
            itcal['inv_LEFT'] = inv_LEFT

        if GPU:
            import cupy as cp
            print("GPU calculation is used")
            L_o, R_o, C_o, G_o, vs_o, is_o, im_o = transfer_date_to_gpu(L_o, R_o, C_o, G_o, vs_o, is_o, im_o)
            A_o, B_o, phi_o = transfer_date_to_gpu(A_o, B_o, phi_o)

            inv = cp.linalg.inv
            zeros = cp.zeros
            hstack = cp.hstack
            copy = cp.copy
        else:
            inv = np.linalg.inv
            zeros = np.zeros
            hstack = np.hstack
            copy = np.copy

        I_allcross = zeros((Nnode_ohl, ))
        Vout_ohl = zeros((Nnode_ohl, Nt + 1))
        Iout_ohl = zeros((Nbran_ohl, Nt))
        LdeR = L_o / dt - R_o / 2
        R_o[vf_bran_o, :][:, vf_bran_o] += A_o.sum(-1, keepdims=False)
        inv_RL_o = inv(L_o / dt + R_o / 2)
        inv_GC_o = inv(C_o / dt + G_o / 2)
        CdeG = C_o / dt - G_o / 2

        A_o_nkn = A_o.transpose(0, 2, 1)

        for i in tqdm(range(Nt - 1)):
            # tower solution
            for itcal in T_cal:
                out_t = itcal['out']
                Nnode_tower = itcal['Nnode']
                Nbran_tower = itcal['Nbran']
                Vnode_t = out_t[:Nnode_tower, i]
                Ibran_t = out_t[Nnode_tower:Nbran_tower + Nnode_tower, i]
                Ivitual_t = out_t[Nbran_tower + Nnode_tower:, i]
                source_v_t_temp = itcal['source'][:, i + 1]
                source_v_t_temp[itcal['vf_bran']] += (itcal['B'] * itcal['phi']).sum(-1, keepdims=False)
                RIGHT = hstack(
                    ((-itcal['L'] / dt).dot(Ibran_t), (itcal['C'] / dt).dot(Vnode_t), -Ivitual_t + itcal['C0n'] @ itcal['Cwn'] @ Vnode_t / dt))
                source_v_t_temp = hstack((source_v_t_temp, 2 * I_allcross[itcal['ohl_cross_index']]))
                temp_result = itcal['inv_LEFT'] @ (source_v_t_temp + RIGHT)

                itcal['out'][:, i + 1] = copy(temp_result)
                Vout_ohl[itcal['ohl_cross_index'], i + 1] = temp_result[itcal['tower_cross_index']]

                I_bran_t_n = temp_result[Nnode_tower:Nbran_tower + Nnode_tower]
                itcal['phi'] = itcal['A'] @ I_bran_t_n[itcal['vf_bran']] + itcal['B'] * itcal['phi']

            v_source_o = vs_o[:, i + 1]
            v_source_o[vf_bran_o] += ((B_o + 1) * phi_o / 2).sum(-1, keepdims=False)
            Iout_ohl[:, i + 1] = inv_RL_o @ (LdeR @ Iout_ohl[:, i] - im_o @ Vout_ohl[:, i + 1] - v_source_o)
            Vout_ohl[:, i + 2] = inv_GC_o @ (CdeG @ Vout_ohl[:, i + 1] + im_o.T @ Iout_ohl[:, i + 1] + is_o[:, i + 1])

            phi_o = A_o_nkn @ Iout_ohl[vf_bran_o, i + 1] + B_o * phi_o

            I_allcross = im_o.T @ Iout_ohl[:, i + 1]

        Vout_ohl = Vout_ohl[:, :-1]
        results_tower = self.combine_results(T_cal, Nt, GPU)
        if GPU:
            Vout_ohl = cp.asnumpy(Vout_ohl)
            Iout_ohl = cp.asnumpy(Iout_ohl)
        result_v_ohl = pd.DataFrame(Vout_ohl, index=columns_o, columns=np.arange(Nt))
        # result_i_ohl = pd.DataFrame(Iout_ohl, index=index_o, columns=np.arange(Nt) + 0.5)
        result_i_ohl = pd.DataFrame(Iout_ohl, index=index_o, columns=np.arange(Nt))
        results = pd.concat([results_tower, result_v_ohl])
        results.drop_duplicates(inplace=True)
        results = results.add(result_i_ohl, fill_value=0).fillna(0)
        return results, {}


class hybrid_nonlinear(Hybrid_Strategy):
    def __init__(self):
        super().__init__()

    def apply(self, line_matrix, tower_matrix, sources, Nt, dt, GPU):
        print("Nonlinear hybrid calculation is used")
        GPU = 0
        #OHL parameter preparing
        im_o_df = line_matrix['incidence_matrix']
        im_o = im_o_df.to_numpy()
        L_o = line_matrix['inductance_matrix'].to_numpy()
        C_o = line_matrix['capacitance_matrix'].to_numpy()
        R_o = line_matrix['resistance_matrix'].to_numpy()
        G_o = line_matrix['conductance_matrix'].to_numpy()
        index_o = im_o_df.index.to_numpy()
        columns_o = im_o_df.columns.to_numpy()
        vs_o = sources.loc[index_o, :].to_numpy()
        is_o = sources.loc[columns_o, :].to_numpy()
        Nbran_ohl, Nnode_ohl = im_o_df.shape

        # Tower parameter preparing
        T_cal = self.preparing_tower_matrix(tower_matrix, line_matrix['capacitance_matrix'], sources, Nt, GPU)

        if GPU:
            import cupy as cp
            print("GPU calculation is used")
            L_o, R_o, C_o, G_o, vs_o, is_o, im_o = transfer_date_to_gpu(L_o, R_o, C_o, G_o, vs_o, is_o, im_o)
            inv = cp.linalg.inv
            zeros = cp.zeros
            hstack = cp.hstack
            copy = cp.copy
            vstack = cp.vstack
            delete = cp.delete
            argwhere = cp.argwhere
        else:
            inv = np.linalg.inv
            zeros = np.zeros
            hstack = np.hstack
            copy = np.copy
            vstack = np.vstack
            delete = np.delete
            argwhere = np.argwhere

        I_allcross = zeros((Nnode_ohl, ))
        Vout_ohl = zeros((Nnode_ohl, Nt + 1))
        Iout_ohl = zeros((Nbran_ohl, Nt))
        inv_RL_o = inv(L_o / dt + R_o / 2)
        LdeR = L_o / dt - R_o / 2
        inv_GC_o = inv(C_o / dt + G_o / 2)
        CdeG = C_o / dt - G_o / 2
        damaged_SDEM = []
        damaged_NLR = []
        for i in tqdm(range(Nt - 1)):
            # tower solution
            for itcal in T_cal:
                out_t = itcal['out']
                Nnode_tower = itcal['Nnode']
                Nbran_tower = itcal['Nbran']
                Vnode_t = out_t[:Nnode_tower, i]
                Ibran_t = out_t[Nnode_tower:Nbran_tower + Nnode_tower, i]
                Ivitual_t = out_t[Nbran_tower + Nnode_tower:, i]
                source_v_t_temp = itcal['source'][:, i + 1]
                L_upper = hstack((-itcal['ima'], -itcal['R'] - itcal['L'] / dt, itcal['zero']))
                L_middle = hstack((itcal['G'] + itcal['C'] / dt, -itcal['imb'].T, -itcal['Cwn'].T))
                L_down = hstack((itcal['C0n'] @ itcal['Cwn'] / dt, itcal['zero'].T, itcal['E']))
                LEFT = vstack((L_upper, L_middle, L_down))
                inv_LEFT = inv(LEFT)
                RIGHT = hstack(
                    ((-itcal['L'] / dt).dot(Ibran_t), (itcal['C'] / dt).dot(Vnode_t), -Ivitual_t + itcal['C0n'] @ itcal['Cwn'] @ Vnode_t / dt))
                source_v_t_temp = hstack((source_v_t_temp, 2 * I_allcross[itcal['ohl_cross_index']]))
                temp_result = inv_LEFT @ (source_v_t_temp + RIGHT)

                itcal['out'][:, i + 1] = copy(temp_result)
                Vout_ohl[itcal['ohl_cross_index'], i + 1] = temp_result[itcal['tower_cross_index']]

                #nonlinear update
                Vnode_t_ref = hstack((out_t[:Nnode_tower, i+1], [0]))
                Ibran_t_n = out_t[Nnode_tower:Nbran_tower + Nnode_tower, i+1]

                SDEM_index = itcal['SDEM_index']
                v_diff = abs(Vnode_t_ref[SDEM_index[:, 1]] - Vnode_t_ref[SDEM_index[:, 2]]) - itcal['SDEM_para'][:, 0]
                index_con1 = v_diff > 0
                itcal['SDEM_para'][index_con1, 1] += v_diff[index_con1] * dt * 1e6
                index_con2 = itcal['SDEM_para'][:, 1] >= itcal['SDEM_para'][:, 2]
                index_r = SDEM_index[index_con2, 0]
                itcal['R'][index_r, index_r] = 1e-6
                temp_dSDEM = [itcal['SDEM_bran'][dsem[0]] for dsem in argwhere(index_con2).tolist()]
                damaged_SDEM.extend(temp_dSDEM)
                itcal['SDEM_index'] = delete(SDEM_index, index_con2, axis=0)
                itcal['SDEM_para'] = delete(itcal['SDEM_para'], index_con2, axis=0)

                VCS_index = itcal['VCS_index']
                VCS_para = itcal['VCS_para']
                v_diff = abs(Vnode_t_ref[VCS_index[:, 1]] - Vnode_t_ref[VCS_index[:, 2]]) - VCS_para
                index_con = argwhere(v_diff > 0)
                index_r = VCS_index[index_con, 0]
                itcal['R'][index_r, index_r] = 1e-6
                itcal['VCS_index'] = delete(VCS_index, index_con, axis=0)
                itcal['VCS_para'] = delete(VCS_para, index_con, axis=0)

                TCS_index = itcal['TCS_index']
                TCS_para = itcal['TCS_para']
                t = dt * (i + 1)
                index_con1 = t >= TCS_para[:, 0] and t < TCS_index[:, 1]
                index_r1 = TCS_index[index_con1]
                itcal['R'][index_r1, index_r1] = TCS_para[index_con1, 2]
                index_con2 = argwhere(t >= TCS_para[:, 1])
                index_r2 = TCS_index[index_con2]
                itcal['R'][index_r2, index_r2] = TCS_para[index_con2, 3]
                itcal['TCS_index'] = delete(TCS_index, index_con2, axis=0)
                itcal['TCS_para'] = delete(TCS_para, index_con2, axis=0)

                for i_nlr in range(len(itcal['NLR_index'])):
                    index_r = itcal['NLR_index'][i_nlr, 0]
                    itcal['R'][index_r, index_r] = itcal['NLR_para'][i_nlr](max(abs(Ibran_t_n[index_r]), 1e-2))
                # 判断损坏代码
                v_diff = abs(Vnode_t_ref[itcal['NLR_index'][:, 1]] - Vnode_t_ref[itcal['NLR_index'][:, 2]])
                itcal['NLR_E'] += Ibran_t_n[itcal['NLR_index'][:, 0]] * v_diff * dt
                index_con = [itcal['NLR_bran'][dnlr[0]] for dnlr in argwhere(itcal['NLR_E'] >= 30000).tolist()]
                damaged_NLR.extend(index_con)
                # if index_con:
                #     break


            # ohl solution
            v_source_o = vs_o[:, i + 1]
            Iout_ohl[:, i + 1] = inv_RL_o @ (LdeR @ Iout_ohl[:, i] - im_o @ Vout_ohl[:, i + 1] - v_source_o)
            Vout_ohl[:, i + 2] = inv_GC_o @ (CdeG @ Vout_ohl[:, i + 1] + im_o.T @ Iout_ohl[:, i + 1] + is_o[:, i + 1])

            I_allcross = im_o.T @ Iout_ohl[:, i + 1]

        Vout_ohl = Vout_ohl[:, :-1]
        results_tower = self.combine_results(T_cal, Nt, GPU)

        if GPU:
            Vout_ohl = cp.asnumpy(Vout_ohl)
            Iout_ohl = cp.asnumpy(Iout_ohl)
        result_v_ohl = pd.DataFrame(Vout_ohl, index=columns_o, columns=np.arange(Nt))
        # result_i_ohl = pd.DataFrame(Iout_ohl, index=index_o, columns=np.arange(Nt) + 0.5)
        result_i_ohl = pd.DataFrame(Iout_ohl, index=index_o, columns=np.arange(Nt))
        results = pd.concat([results_tower, result_v_ohl])
        results.drop_duplicates(inplace=True)
        results = results.add(result_i_ohl, fill_value=0).fillna(0)
        others = {'SDEM': damaged_SDEM,'NLR':damaged_NLR}
        return results, others


class hybrid_nonliear_variant_frequency(Hybrid_Strategy):
    def __init__(self):
        super().__init__()

    def apply(self, line_matrix, tower_matrix, sources, Nt, dt, GPU):
        print("Nonlinear-variant_frequency hybrid calculation is used")

        #OHL parameter preparing
        im_o_df = line_matrix['incidence_matrix']
        im_o = im_o_df.to_numpy()
        L_o = line_matrix['inductance_matrix'].to_numpy()
        C_o = line_matrix['capacitance_matrix'].to_numpy()
        R_o = line_matrix['resistance_matrix'].to_numpy()
        G_o = line_matrix['conductance_matrix'].to_numpy()
        index_o = im_o_df.index.to_numpy()
        columns_o = im_o_df.columns.to_numpy()
        vs_o = sources.loc[index_o, :].to_numpy()
        is_o = sources.loc[columns_o, :].to_numpy()
        Nbran_ohl, Nnode_ohl = im_o_df.shape

        # Tower parameter preparing
        T_cal = self.preparing_tower_matrix(tower_matrix, line_matrix['capacitance_matrix'], sources, Nt, GPU)

        temp = pd.Series(range(index_o.shape[0]), index=index_o)
        vf_bran_o = temp.loc[line_matrix['vf_bran']].to_list()
        A_o = line_matrix['A']
        B_o = line_matrix['B']
        phi_o = np.zeros_like(B_o)

        if GPU:
            import cupy as cp
            print("GPU calculation is used")
            L_o, R_o, C_o, G_o, vs_o, is_o, im_o = transfer_date_to_gpu(L_o, R_o, C_o, G_o, vs_o, is_o, im_o)
            A_o, B_o, phi_o = transfer_date_to_gpu(A_o, B_o, phi_o)

            inv = cp.linalg.inv
            zeros = cp.zeros
            hstack = cp.hstack
            copy = cp.copy
            vstack = cp.vstack
            delete = cp.delete
            argwhere = cp.argwhere
        else:
            inv = np.linalg.inv
            zeros = np.zeros
            hstack = np.hstack
            copy = np.copy
            vstack = np.vstack
            delete = np.delete
            argwhere = np.argwhere

        I_allcross = zeros((Nnode_ohl, ))
        Vout_ohl = zeros((Nnode_ohl, Nt + 1))
        Iout_ohl = zeros((Nbran_ohl, Nt))
        LdeR = L_o / dt - R_o / 2
        R_o[vf_bran_o, :][:, vf_bran_o] += A_o.sum(-1, keepdims=False)
        inv_RL_o = inv(L_o / dt + R_o / 2)
        inv_GC_o = inv(C_o / dt + G_o / 2)
        CdeG = C_o / dt - G_o / 2

        A_o_nkn = A_o.transpose(0, 2, 1)

        damaged_SDEM = []
        damaged_NLR = []

        for i in tqdm(range(Nt - 1)):
            # tower solution
            for itcal in T_cal:
                out_t = itcal['out']
                Nnode_tower = itcal['Nnode']
                Nbran_tower = itcal['Nbran']
                Vnode_t = out_t[:Nnode_tower, i]
                Ibran_t = out_t[Nnode_tower:Nbran_tower + Nnode_tower, i]
                Ivitual_t = out_t[Nbran_tower + Nnode_tower:, i]
                source_v_t_temp = itcal['source'][:, i + 1]
                source_v_t_temp[itcal['vf_bran']] += (itcal['B'] * itcal['phi']).sum(-1, keepdims=False)
                L_upper = hstack((-itcal['ima'], -itcal['R'] - itcal['L'] / dt, itcal['zero']))
                L_middle = hstack((itcal['G'] + itcal['C'] / dt, -itcal['imb'].T, -itcal['Cwn'].T))
                L_down = hstack((itcal['C0n'] @ itcal['Cwn'] / dt, itcal['zero'].T, itcal['E']))
                LEFT = vstack((L_upper, L_middle, L_down))
                inv_LEFT = inv(LEFT)
                RIGHT = hstack(
                    ((-itcal['L'] / dt).dot(Ibran_t), (itcal['C'] / dt).dot(Vnode_t),
                     -Ivitual_t + itcal['C0n'] @ itcal['Cwn'] @ Vnode_t / dt))
                source_v_t_temp = hstack((source_v_t_temp, 2 * I_allcross[itcal['ohl_cross_index']]))
                temp_result = inv_LEFT @ (source_v_t_temp + RIGHT)

                itcal['out'][:, i + 1] = copy(temp_result)
                Vout_ohl[itcal['ohl_cross_index'], i + 1] = temp_result[itcal['tower_cross_index']]

                I_bran_t_n = temp_result[Nnode_tower:Nbran_tower + Nnode_tower]
                itcal['phi'] = itcal['A'] @ I_bran_t_n[itcal['vf_bran']] + itcal['B'] * itcal['phi']

                # nonlinear update
                Vnode_t_ref = hstack((out_t[:Nnode_tower, i + 1], [0]))
                Ibran_t_n = out_t[Nnode_tower:Nbran_tower + Nnode_tower, i + 1]

                SDEM_index = itcal['SDEM_index']
                v_diff = abs(Vnode_t_ref[SDEM_index[:, 1]] - Vnode_t_ref[SDEM_index[:, 2]]) - itcal['SDEM_para'][:, 0]
                index_con1 = v_diff > 0
                itcal['SDEM_para'][index_con1, 1] += v_diff[index_con1] * dt * 1e6
                index_con2 = itcal['SDEM_para'][:, 1] >= itcal['SDEM_para'][:, 2]
                index_r = SDEM_index[index_con2, 0]
                itcal['R'][index_r, index_r] = 1e-6
                temp_dSDEM = [itcal['SDEM_bran'][dsem] for dsem in argwhere(index_con2)]
                damaged_SDEM.extend(temp_dSDEM)
                itcal['SDEM_index'] = delete(SDEM_index, index_con2, axis=0)
                itcal['SDEM_para'] = delete(itcal['SDEM_para'], index_con2, axis=0)

                VCS_index = itcal['VCS_index']
                VCS_para = itcal['VCS_para']
                v_diff = abs(Vnode_t_ref[VCS_index[:, 1]] - Vnode_t_ref[VCS_index[:, 2]]) - VCS_para
                index_con = argwhere(v_diff > 0)
                index_r = VCS_index[index_con, 0]
                itcal['R'][index_r, index_r] = 1e-6
                itcal['VCS_index'] = delete(VCS_index, index_con, axis=0)
                itcal['VCS_para'] = delete(VCS_para, index_con, axis=0)

                TCS_index = itcal['TCS_index']
                TCS_para = itcal['TCS_para']
                t = dt * (i + 1)
                index_con1 = t >= TCS_para[:, 0] and t < TCS_index[:, 1]
                index_r1 = TCS_index[index_con1]
                itcal['R'][index_r1, index_r1] = TCS_para[index_con1, 2]
                index_con2 = argwhere(t >= TCS_para[:, 1])
                index_r2 = TCS_index[index_con2]
                itcal['R'][index_r2, index_r2] = TCS_para[index_con2, 3]
                itcal['TCS_index'] = delete(TCS_index, index_con2, axis=0)
                itcal['TCS_para'] = delete(TCS_para, index_con2, axis=0)

                for i_nlr in range(len(itcal['NLR_index'])):
                    index_r = itcal['NLR_index'][i_nlr, 0]
                    itcal['R'][index_r, index_r] = itcal['NLR_para'][i_nlr](Ibran_t_n[index_r])
                # 判断损坏代码
                # v_diff = abs(Vnode_t_ref[itcal['NLR_index'][:, 1]] - Vnode_t_ref[itcal['NLR_index'][:, 2]])
                # itcal['NLR_E'] += Ibran_t_n[itcal['NLR_index'][:, 0]] * v_diff * dt
                # index_con = [itcal['NLR_bran'][dnlr] for dnlr in argwhere(itcal['NLR_E'] >= 30000)]
                # damaged_NLR.extend(index_con)
                # if index_con:
                #     break

            # ohl solution
            v_source_o = vs_o[:, i + 1]
            v_source_o[vf_bran_o] += ((B_o + 1) * phi_o / 2).sum(-1, keepdims=False)
            Iout_ohl[:, i + 1] = inv_RL_o @ (LdeR @ Iout_ohl[:, i] - im_o @ Vout_ohl[:, i + 1] - v_source_o)
            Vout_ohl[:, i + 2] = inv_GC_o @ (CdeG @ Vout_ohl[:, i + 1] + im_o.T @ Iout_ohl[:, i + 1] + is_o[:, i + 1])

            I_allcross = im_o.T @ Iout_ohl[:, i + 1]
            phi_o = A_o_nkn @ Iout_ohl[vf_bran_o, i + 1] + B_o * phi_o

        Vout_ohl = Vout_ohl[:, :-1]
        results_tower = self.combine_results(T_cal, Nt, GPU)

        if GPU:
            Vout_ohl = cp.asnumpy(Vout_ohl)
            Iout_ohl = cp.asnumpy(Iout_ohl)
        result_v_ohl = pd.DataFrame(Vout_ohl, index=columns_o, columns=np.arange(Nt))
        # result_i_ohl = pd.DataFrame(Iout_ohl, index=index_o, columns=np.arange(Nt) + 0.5)
        result_i_ohl = pd.DataFrame(Iout_ohl, index=index_o, columns=np.arange(Nt))
        results = pd.concat([results_tower, result_v_ohl])
        results.drop_duplicates(inplace=True)
        results = results.add(result_i_ohl, fill_value=0).fillna(0)
        others = {'SDEM': damaged_SDEM}
        return results, others


class Measurement(Strategy):
    def apply(self, measurement, solution, dt):
        results = {}
        for key, value in measurement.items():
            data_type = value[0]  # 0:'branch',1: 'normal lump'
            measurement_type = value[1]  # 1:'current',2:'voltage'
            # 处理支路名或节点名
            if data_type == 0:
                branch_name = key
                current = solution.loc[branch_name].tolist() if branch_name in solution.index else None
                node1 = solution.loc[value[3]].tolist() if value[3] in solution.index else None
                node2 = solution.loc[value[4]].tolist() if value[4] in solution.index else None
                voltage = [abs(a - b) for a, b in zip(node1, node2)] if node1 and node2 else None
                p = [a * b for a, b in zip(current, voltage)] if current and voltage else None
                E = sum([i * dt for i in p]) if p else None
                if measurement_type == 1:
                    results["current"] = current
                elif measurement_type == 2:
                    results["voltage"] = voltage
                elif measurement_type == 3:
                    results["P"] = p
                elif measurement_type == 4:
                    results["E"] = E
                    results["P"] = p
                    results["voltage"] = voltage
                    results["current"] = current
                elif measurement_type == 11:
                    results["E"] = E

                results["index"] = [key, value[3], value[4]]
            elif data_type == 1:
                lump_name = key
                # 处理支路名可能是列表的情况
                branches = value[3]
                dict_result = {}
                for bran, n1, n2 in zip(value[2], value[3], value[4]):
                    current = solution.loc[bran].tolist() if bran in solution.index else None
                    node1 = solution.loc[n1].tolist() if n1 in solution.index else None
                    node2 = solution.loc[n2].tolist() if n2 in solution.index else None
                    voltage = [abs(a - b) for a, b in zip(node1, node2)] if (node1 and node2) else None
                    if n1 == "ref":
                        voltage = node2
                    if n2 == "ref":
                        voltage = node1
                    p = [a * b for a, b in zip(current, voltage)] if (current and voltage) else None
                    E = sum([i * dt for i in p]) if p else None

                    if measurement_type == 1:
                        dict_result["current"] = current
                    elif measurement_type == 2:
                        dict_result["voltage"] = voltage
                    elif measurement_type == 3:
                        dict_result["P"] = p
                    elif measurement_type == 4:
                        dict_result["current"] = current
                        dict_result["voltage"] = voltage
                        dict_result["P"] = p
                        dict_result["E"] = E
                    elif measurement_type == 11:
                        dict_result["E"] = E
                dict_result["index"] = [bran, n1, n2]
                results[lump_name] = dict_result
        return results


class Monteclarlo(Strategy):
    def apply(self, network):
        print("montecarlo")
