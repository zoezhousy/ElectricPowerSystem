import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import csv
import os
import numpy as np
import pandas as pd
from functools import reduce
from itertools import chain
import copy
import pyarrow as pa
import pyarrow.csv as pacsv
import matplotlib.pyplot as plt
# from Demos.BackupRead_BackupWrite import tempfile
from Risk_Evaluate.MC import run_MC
from Driver.initialization.initialization import initialize_OHL, initialize_tower, initial_lightning, initial_lump, \
    initialize_cable, initialize_ground
from Driver.modeling.OHL_modeling import OHL_building, OHL_building_variant_frequency, OHL_building_FDTD, \
    OHL_building_hybrid_variant_frequency
from Driver.modeling.cable_modeling import cable_building, cable_building_variant_frequency, cable_building_hybrid_variant_frequency
from Driver.modeling.tower_modeling import tower_building, tower_building_variant_frequency, tower_building_with_tube, \
    tower_building_variant_frequency_with_tube
from Function.Calculators.InducedVoltage_calculate import InducedVoltage_calculate, LightningCurrent_calculate, \
    H_MagneticField_calculate, ElectricField_calculate, ElectricField_above_lossy, InducedVoltage_calculate_indirect, \
    InducedVoltage_calculate_direct
# from Risk_Evaluate.MC import run_MC
from Model.Cable import Cable
from Model.Lightning import Lightning
from Model.Tower import Tower
from Model.Wires import OHLWire
from Utils.Math import distance, segment_branch
import Model.Strategy as Strategy
from Model.Contant import Constant
from Model.Lightning import Stroke, Lightning, Channel
import math
from multiprocessing import Process, Manager,Lock
from Risk_Evaluate.Huri_Method import Huri_Method
from Risk_Evaluate.Huri_Method_SAF import Huri_Method_SAF
from tqdm import tqdm
import cupy as cp
from tqdm import tqdm


class Network:
    def __init__(self, **kwargs):
        self.towers = kwargs.get('towers', [])
        self.cables = kwargs.get('cables', [])
        self.OHLs = kwargs.get('OHLs', [])
        self.lumps = kwargs.get('lumps', [])
        self.sources = pd.DataFrame()
        self.branches = {}
        self.starts = []
        self.ends = []
        self.H = {}
        self.solution = pd.DataFrame()
        self.measurement = {}
        self.incidence_matrix_A = pd.DataFrame()
        self.incidence_matrix_B = pd.DataFrame()
        self.resistance_matrix = pd.DataFrame()
        self.inductance_matrix = pd.DataFrame()
        self.capacitance_matrix = pd.DataFrame()
        self.conductance_matrix = pd.DataFrame()
        self.voltage_source_matrix = pd.DataFrame()
        self.current_source_matrix = pd.DataFrame()
        self.solution_type = {
            'hybrid': False,
            'nonlinear': False,
            'variant_frequency': False,
            'variant_step': False
        }
        self.Nfit = 9
        self.fixed_frequency = 2e4
        self.max_length = 200
        # self.varied_frequency = np.logspace(0, 9, 37)
        self.varied_frequency = np.array([])
        for i in range(6):
            temp = np.linspace(5e-2 * 10 ** i, 5e-1 * 10 ** i, 10)
            self.varied_frequency = np.hstack((self.varied_frequency, temp))
        self.global_ground = 0
        self.ground = None
        self.dt = None
        self.T = None
        self.VF = None
        self.switch_disruptive_effect_models = []
        self.voltage_controled_switchs = []
        self.time_controled_switchs = []
        self.nolinear_resistors = []
        self.lightning = None
        self.VF_dict = {}
        self.PoleXY = {}
        self.tower_head_node = {}

    # 记录电网元素之间的关系
    def tower_branches(self, branches):
        tower_nodes = []
        for tower in self.towers:
            for wire in list(tower.wires.get_all_wires().values()):
                startnode = {wire.start_node.name: [wire.start_node.x, wire.start_node.y, wire.start_node.z]}
                endnode = {wire.end_node.name: [wire.end_node.x, wire.end_node.y, wire.end_node.z]}
                tower_nodes.append(wire.start_node.name)
                tower_nodes.append(wire.end_node.name)
                branches[wire.name] = [startnode, endnode, tower.name]

        return branches, set(tower_nodes)

    def lump_nodes(self):
        lump_nodes = []
        for tower in self.towers:
            lump_nodes.extend(tower.lump.nodeList)
            for device_list in [tower.devices.insulators, tower.devices.arrestors, tower.devices.transformers]:
                for device in device_list:
                    lump_nodes.extend(device.nodeList)

        return set(lump_nodes)

    def OHL_branches(self, branches, maxlength):
        OHL_nodes = []
        for obj in self.OHLs:
            wires = list(obj.wires.get_all_wires().values())
            for wire in wires:
                position_obj_start = {wire.start_node.name: [wire.start_node.x,
                                                             wire.start_node.y,
                                                             wire.start_node.z]}
                position_obj_end = {wire.end_node.name: [wire.end_node.x,
                                                         wire.end_node.y,
                                                         wire.end_node.z]}
                Nt = int(np.ceil(distance(obj.info.HeadTower_pos, obj.info.TailTower_pos) / maxlength))
                OHL_nodes.append(wire.start_node.name)
                OHL_nodes.append(wire.end_node.name)
                branches[wire.name] = [position_obj_start, position_obj_end, obj.info.name, Nt]
        return branches, set(OHL_nodes)

    def cable_branches(self, branches, maxlength):
        cable_nodes = []
        for obj in self.cables:
            wires = list(obj.wires.get_all_wires().values())
            for wire in wires:
                position_obj_start = {wire.start_node.name: [wire.start_node.x,
                                                             wire.start_node.y,
                                                             wire.start_node.z]}
                position_obj_end = {wire.end_node.name: [wire.end_node.x,
                                                         wire.end_node.y,
                                                         wire.end_node.z]}
                Nt = int(np.ceil(distance(obj.info.HeadTower_pos, obj.info.TailTower_pos) / maxlength))
                cable_nodes.append(wire.start_node.name)
                cable_nodes.append(wire.end_node.name)
                branches[wire.name] = [position_obj_start, position_obj_end, obj.info.name, Nt]
        return branches, set(cable_nodes)

    def calculate_branches(self, maxlength):
        branches = {}
        branches, tb = self.tower_branches(branches)
        branches, ob = self.OHL_branches(branches, maxlength)
        branches,OHL_new_nodes = segment_branch(branches)
        branches, cb = self.cable_branches(branches, maxlength)
        nodes = tb.union(ob).union(cb).union(OHL_new_nodes)
        return branches, list(nodes)

    def tower_initial(self, load_dict):
        if 'Tower' in load_dict:
            self.towers = [initialize_tower(tower, max_length=self.max_length, dt=self.dt, T=self.T, VF=self.VF) for
                           tower in load_dict['Tower']]
            self.measurement = reduce(lambda acc, tower: {**acc, **tower.Measurement}, self.towers, {})

    def tower_building(self):

        for tower in self.towers:
            gnd = self.ground if self.global_ground == 1 else tower.ground
            self.PoleXY[tower.info.name] = tower.info.position[:2]
            self.tower_head_node[tower.info.name] = tower.info.Pole_Head_Node
            if tower.info.con_mode == 1:
                self.solution_type['variant_frequency'] = True
                print("tower apply variant frequency")
                if tower.tubeWire is None:
                    tower_building_variant_frequency(tower, gnd, self.varied_frequency, self.Nfit, self.dt)
                else:
                    tower_building_variant_frequency_with_tube(tower, self.fixed_frequency, gnd, self.varied_frequency, self.Nfit, self.dt)
            else:
                if tower.tubeWire is None:
                    tower_building(tower, gnd)
                else:
                    tower_building_with_tube(tower, self.fixed_frequency, gnd)

            #小矩阵不用
            self.switch_disruptive_effect_models.extend(tower.lump.switch_disruptive_effect_models)
            self.voltage_controled_switchs.extend(tower.lump.voltage_controled_switchs)
            self.time_controled_switchs.extend(tower.lump.time_controled_switchs)
            self.nolinear_resistors.extend(tower.lump.nolinear_resistors)
            for device_list in [tower.devices.insulators, tower.devices.arrestors, tower.devices.transformers]:
                for device in device_list:
                    self.switch_disruptive_effect_models.extend(device.switch_disruptive_effect_models)
                    self.voltage_controled_switchs.extend(device.voltage_controled_switchs)
                    self.time_controled_switchs.extend(device.time_controled_switchs)
                    self.nolinear_resistors.extend(device.nolinear_resistors)

            if self.switch_disruptive_effect_models or self.voltage_controled_switchs or self.time_controled_switchs or self.nolinear_resistors:
                self.solution_type['nonlinear'] = True

    # initialize internal network elements
    def OHL_initial(self, load_dict):
        if 'OHL' in load_dict:
            self.OHLs = [initialize_OHL(ohl, max_length=self.max_length) for ohl in load_dict['OHL']]

    def OHL_building(self):
        for ohl in self.OHLs:
            gnd = self.ground if self.global_ground == 1 else ohl.ground
            if ohl.info.con_mode == 1 or gnd.gnd_mode == 2:
                print("OHL apply variant frequency")
                self.solution_type['variant_frequency'] = True
                if self.Hybrid_method == 1:
                    print("OHL apply Hybrid model")
                    OHL_building_hybrid_variant_frequency(ohl, self.max_length, gnd, self.varied_frequency, self.fixed_frequency, self.Nfit, self.dt)
                else:
                    OHL_building_variant_frequency(ohl, self.max_length, gnd, self.varied_frequency, self.fixed_frequency, self.Nfit, self.dt)
            else:
                OHL_building(ohl, self.max_length, gnd, self.fixed_frequency)

    def cable_initial(self, load_dict, VF):
        if 'Cable' in load_dict:
            self.cables = [initialize_cable(cable, max_length=self.max_length, VF=VF) for cable in load_dict['Cable']]

    def cable_building(self):
        for cable in self.cables:
            gnd = self.ground if self.global_ground == 1 else cable.ground
            if cable.info.con_mode == 1 or gnd.gnd_mode == 2:
                print("Cable apply variant frequency")
                self.solution_type['variant_frequency'] = True
                if self.Hybrid_method == 1:
                    print("Cable apply Hybrid model")
                    cable_building_hybrid_variant_frequency(cable, gnd, self.varied_frequency, self.fixed_frequency, self.dt)
                else:
                    cable_building_variant_frequency(cable, gnd, self.varied_frequency, self.fixed_frequency, self.dt)  # 1,2
            else:
                cable_building(cable, gnd, self.fixed_frequency)

    def initialize_network(self, load_dict, VF):

        self.tower_initial(load_dict)
        self.tower_building()
        self.OHL_initial(load_dict)
        self.OHL_building()
        self.cable_initial(load_dict, VF)
        self.cable_building()

    def source_initial(self, load_dict, nodes, branches, constants, share_dict):

        if 'Source' in load_dict and "Lightning" in load_dict['Source']:
            light = load_dict["Source"]["Lightning"]
            lightning = initial_lightning(light, dt=self.dt)

            if light["area"].split("_")[0] == "OHL":
                bran = None
                cir_id = light["cir_id"]
                if int(cir_id) ==1001:
                    bran = 'Y' + str(cir_id) + 'S'
                elif int(cir_id) ==3001:
                    bran = 'Y' + str(cir_id) + light['phase_id']
                if bran is None:
                    raise ValueError("bran cannot be None")
                U_out, I_out, shared_dict = self.source_calculate(lightning, light["area"], bran,
                                                     light["position"], nodes, branches, constants,share_dict)
                sources = self.add_source(U_out, I_out)
                return sources,len(lightning.strokes)
            if light["area"].split("_")[0] == "tower":
                U_out, I_out, share_dict = self.source_calculate(lightning,
                                                                 light["area"], light["wire"], light["position"], nodes,
                                                                 branches, constants, share_dict)
                sources = self.add_source(U_out, I_out)
                return sources,len(lightning.strokes)
        else:
            U_out = pd.DataFrame()
            I_out = pd.DataFrame()
            for model_list in [self.towers, self.OHLs, self.cables]:
                for model in model_list:
                    U_out = U_out.add(model.voltage_source_matrix, fill_value=0).fillna(0)
                    I_out = I_out.add(model.current_source_matrix, fill_value=0).fillna(0)
            sources = pd.concat([U_out, I_out], axis=0)
            return sources, []


    # 输出U/I
    def source_calculate(self, lightning, area, wire, position, nodes, branches, constants, shared_dict):

        start = [list(l[0].values())[0] for l in list(branches.values())]
        end = [list(l[1].values())[0] for l in list(branches.values())]
        branches = list(branches.keys())  # 让branches只存储支路的列表，可节省内存
        pt_start = np.array(start)
        pt_end = np.array(end)

        U_out = pd.DataFrame()
        I_out = pd.DataFrame()
        if lightning.type == "Indirect":
            for i in range(len(lightning.strokes)):

                Er_lossy = 0
                Ez_lossy = 0
                erg = constants.epr
                sigma_g = constants.sigma
                if os.path.exists("Er_lossy.npy") and os.path.exists("Ez_lossy.npy"):
                    print("------------existing ---------------")
                    Er_lossy = np.load("Er_lossy.npy")
                    Ez_lossy = np.load("Ez_lossy.npy")

                #if (erg, sigma_g, 0) in shared_dict:
                    # Er_lossy = shared_dict[(erg, sigma_g, 0)]
                    # Ez_lossy = shared_dict[(erg, sigma_g, 1)]
                    # print("------------existing ---------------")
                else:
                    H_p = H_MagneticField_calculate(pt_start, pt_end, lightning.strokes[i],
                                                    lightning.channel,
                                                    constants.ep0, constants.vc)  # 计算磁场
                    Ez_T, Er_T = ElectricField_calculate(pt_start, pt_end, lightning.strokes[i],
                                                         lightning.channel,
                                                         constants.ep0, constants.vc)  # 计算电场
                    # 计算有损地面的电场
                    Er_lossy = ElectricField_above_lossy(-H_p, Er_T, constants, shared_dict, constants.sigma)
                    np.save("Er_lossy.npy", Er_lossy)
                    #shared_dict[(erg, sigma_g, 0)] = Er_lossy
                    Ez_lossy = Ez_T
                    np.save("Ez_lossy.npy", Ez_lossy)
                    #shared_dict[(erg, sigma_g, 1)] = Ez_lossy
                new_U = InducedVoltage_calculate_indirect(pt_start, pt_end, branches, lightning,
                                                          stroke_sequence=i, Er_lossy=Er_lossy, Ez_lossy=Ez_lossy)
                U_out = pd.concat([U_out, new_U], axis=1, ignore_index=True)
                I_out = pd.concat(
                    [I_out,
                     LightningCurrent_calculate(area, wire, position, self, nodes, lightning, stroke_sequence=i)],
                    axis=1, ignore_index=True)
        if lightning.type == "Direct":
            for i in range(len(lightning.strokes)):
                new_U = InducedVoltage_calculate_direct(branches, lightning, i)
                U_out = pd.concat([U_out, new_U], axis=1, ignore_index=True)
                I_out = pd.concat([I_out,
                     LightningCurrent_calculate(area, wire, position, self, nodes, lightning, stroke_sequence=i)],
                    axis=1, ignore_index=True)
        # Source_Matrix = pd.concat([I_out, U_out], axis=0)
        return U_out, I_out, shared_dict
    def add_source(self, U_out, I_out):
        for model_list in [self.towers, self.OHLs, self.cables]:
            for model in model_list:
                U_out = U_out.add(model.voltage_source_matrix, fill_value=0).fillna(0)
                I_out = I_out.add(model.current_source_matrix, fill_value=0).fillna(0)
        return pd.concat([U_out, I_out], axis=0)
    # U/I矩阵 加上Lump的U/I，输出source
    def add_lump(self, U_out, I_out):

        lumps = [tower.lump for tower in self.towers]
        devices = [tower.devices for tower in self.towers]
        for lump in lumps:
            U_out = U_out.add(lump.voltage_source_matrix, fill_value=0).fillna(0)
            I_out = I_out.add(lump.current_source_matrix, fill_value=0).fillna(0)
        for lumps in list(map(lambda device: device.arrestors + device.insulators + device.transformers, devices)):
            for lump in lumps:
                U_out = U_out.add(lump.voltage_source_matrix, fill_value=0).fillna(0)
                I_out = I_out.add(lump.current_source_matrix, fill_value=0).fillna(0)
        return pd.concat([U_out, I_out], axis=0)

    # R,L,G,C矩阵合并
    def tower_matrix(self):
        for tower in self.towers:
            self.incidence_matrix_A = self.incidence_matrix_A.add(tower.incidence_matrix_A, fill_value=0).fillna(0)
            self.incidence_matrix_B = self.incidence_matrix_B.add(tower.incidence_matrix_B, fill_value=0).fillna(0)
            self.resistance_matrix = self.resistance_matrix.add(tower.resistance_matrix, fill_value=0).fillna(0)
            self.inductance_matrix = self.inductance_matrix.add(tower.inductance_matrix, fill_value=0).fillna(0)
            self.capacitance_matrix = self.capacitance_matrix.add(tower.capacitance_matrix, fill_value=0).fillna(0)
            self.conductance_matrix = self.conductance_matrix.add(tower.conductance_matrix, fill_value=0).fillna(0)

    def OHL_matrix(self):
        for ohl in self.OHLs:
            self.incidence_matrix_A = self.incidence_matrix_A.add(ohl.incidence_matrix, fill_value=0).fillna(0)
            self.incidence_matrix_B = self.incidence_matrix_B.add(ohl.incidence_matrix, fill_value=0).fillna(0)
            self.resistance_matrix = self.resistance_matrix.add(ohl.resistance_matrix, fill_value=0).fillna(0)
            self.inductance_matrix = self.inductance_matrix.add(ohl.inductance_matrix, fill_value=0).fillna(0)
            self.capacitance_matrix = self.capacitance_matrix.add(ohl.capacitance_matrix, fill_value=0).fillna(0)
            self.conductance_matrix = self.conductance_matrix.add(ohl.conductance_matrix, fill_value=0).fillna(0)

    def cable_matrix(self):
        for cable in self.cables:
            self.incidence_matrix_A = self.incidence_matrix_A.add(cable.incidence_matrix, fill_value=0).fillna(0)
            self.incidence_matrix_B = self.incidence_matrix_B.add(cable.incidence_matrix, fill_value=0).fillna(0)
            self.resistance_matrix = self.resistance_matrix.add(cable.resistance_matrix, fill_value=0).fillna(0)
            self.inductance_matrix = self.inductance_matrix.add(cable.inductance_matrix, fill_value=0).fillna(0)
            self.capacitance_matrix = self.capacitance_matrix.add(cable.capacitance_matrix, fill_value=0).fillna(0)
            self.conductance_matrix = self.conductance_matrix.add(cable.conductance_matrix, fill_value=0).fillna(0)

    def tower_individual_matrix(self):
        tower_matrix = []
        for tower in self.towers:
            if tower.info.con_mode == 1:
                A = tower.A
                B = tower.B
                phi = tower.phi
                vf_bran = tower.incidence_matrix_A.index.tolist()
            else:
                A = np.zeros((0, 0, self.Nfit))
                B = np.zeros((0, self.Nfit))
                phi = np.zeros((0, self.Nfit))
                vf_bran = []

            nonlinear_dict = self.prepare_nonlinear_update_matrix(tower.lump)
            for device_list in [tower.devices.insulators, tower.devices.arrestors, tower.devices.transformers]:
                for device in device_list:
                    temp_dict = self.prepare_nonlinear_update_matrix(device)
                    nonlinear_dict['SDEM'] = np.vstack((nonlinear_dict['SDEM'], temp_dict['SDEM']))
                    nonlinear_dict['VCS'] = np.vstack((nonlinear_dict['VCS'], temp_dict['VCS']))
                    nonlinear_dict['TCS'] = np.vstack((nonlinear_dict['TCS'], temp_dict['TCS']))
                    nonlinear_dict['NLR'] = np.vstack((nonlinear_dict['NLR'], temp_dict['NLR']))

            matrix_dict = {'incidence_matrix_A': tower.incidence_matrix_A,
                           'incidence_matrix_B': tower.incidence_matrix_B,
                           'resistance_matrix': tower.resistance_matrix, 'inductance_matrix': tower.inductance_matrix,
                           'capacitance_matrix': tower.capacitance_matrix,
                           'conductance_matrix': tower.conductance_matrix, 'A': A,
                           'B': B, 'phi': phi, 'SDEM': nonlinear_dict['SDEM'], 'VCS': nonlinear_dict['VCS'],
                           'TCS': nonlinear_dict['TCS'], 'NLR': nonlinear_dict['NLR'], 'vf_bran': vf_bran}
            tower_matrix.append(matrix_dict)
        return tower_matrix

    def line_individual_matrix(self):
        incidence_matrix = pd.DataFrame()
        resistance_matrix = pd.DataFrame()
        inductance_matrix = pd.DataFrame()
        capacitance_matrix = pd.DataFrame()
        conductance_matrix = pd.DataFrame()
        A = np.zeros((0, 0, self.Nfit))
        B = np.zeros((0, self.Nfit))
        vf_bran = []
        for model_list in [self.OHLs + self.cables]:
            for model in model_list:
                incidence_matrix = incidence_matrix.add(model.incidence_matrix, fill_value=0).fillna(0)
                resistance_matrix = resistance_matrix.add(model.resistance_matrix, fill_value=0).fillna(0)
                inductance_matrix = inductance_matrix.add(model.inductance_matrix, fill_value=0).fillna(0)
                capacitance_matrix = capacitance_matrix.add(model.capacitance_matrix, fill_value=0).fillna(0)
                conductance_matrix = conductance_matrix.add(model.conductance_matrix, fill_value=0).fillna(0)
                gnd = self.ground if self.global_ground == 1 else model.ground
                if model.info.con_mode == 1 or gnd.gnd_mode == 2:
                    A = block_diag_3dim(A, model.A)
                    B = np.vstack((B, model.B))
                    vf_bran.extend(model.incidence_matrix.index.tolist())

        matrix_dict = {'incidence_matrix': incidence_matrix, 'resistance_matrix': resistance_matrix,
                       'inductance_matrix': inductance_matrix, 'capacitance_matrix': capacitance_matrix,
                       'conductance_matrix': conductance_matrix, 'A': A, 'B': B, 'vf_bran': vf_bran}
        return matrix_dict

    def combine_parameter_matrix(self):

        # 按照towers，cables，ohls顺序合并参数矩阵
        self.tower_matrix()
        self.OHL_matrix()
        self.cable_matrix()
        self.build_H()

    def build_H(self):
        self.H["incidence_matrix_A"] = copy.deepcopy(self.incidence_matrix_A)
        self.H["incidence_matrix_B"] = copy.deepcopy(self.incidence_matrix_B)
        self.H["resistance_matrix"] = copy.deepcopy(self.resistance_matrix)
        self.H["inductance_matrix"] = copy.deepcopy(self.inductance_matrix)
        self.H["capacitance_matrix"] = copy.deepcopy(self.capacitance_matrix)
        self.H["conductance_matrix"] = copy.deepcopy(self.conductance_matrix)
        self.H["switch_disruptive_effect_models"] = copy.deepcopy(self.switch_disruptive_effect_models)
        self.H["voltage_controled_switchs"] = copy.deepcopy(self.voltage_controled_switchs)
        self.H["time_controled_switchs"] = copy.deepcopy(self.time_controled_switchs)
        self.H["nolinear_resistors"] = copy.deepcopy(self.nolinear_resistors)
        return self.H

    def reset_matrix(self):
        self.incidence_matrix_A = self.H["incidence_matrix_A"]
        self.incidence_matrix_B = self.H["incidence_matrix_B"]
        self.resistance_matrix = self.H["resistance_matrix"]
        self.inductance_matrix = self.H["inductance_matrix"]
        self.capacitance_matrix = self.H["capacitance_matrix"]
        self.conductance_matrix = self.H["conductance_matrix"]

    def update_solution_type(self):
        solution_type_id = ''
        for values in self.solution_type.values():
            solution_type_id += str(int(bool(values)))

        return solution_type_id

    def prepare_nonlinear_update_matrix(self, lumps):
        SDEM_update_matrix = np.empty((0, 6))
        for SDEM in lumps.switch_disruptive_effect_models:
            temp = []
            temp.append(SDEM.bran[0])
            temp.append(SDEM.node1[0]) if SDEM.node1[0] != 'ref' else -1
            temp.append(SDEM.node2[0]) if SDEM.node2[0] != 'ref' else -1
            temp.append(SDEM.parameters['v_initial'])
            temp.append(0)
            temp.append(SDEM.parameters['DE_max'])
            temp = np.array(temp)
            SDEM_update_matrix = np.vstack((SDEM_update_matrix, temp))

        VCS_update_matrix = np.empty((0, 4))
        for VCS in lumps.voltage_controled_switchs:
            temp = []
            temp.append(VCS.bran[0])
            temp.append(VCS.node1[0]) if VCS.node1[0] != 'ref' else -1
            temp.append(VCS.node2[0]) if VCS.node2[0] != 'ref' else -1
            temp.append(VCS.parameters['voltage'])
            temp = np.array(temp)
            VCS_update_matrix = np.vstack((VCS_update_matrix, temp))

        TCS_update_matrix = np.empty((0, 5))
        for TCS in lumps.time_controled_switchs:
            temp = []
            temp.append(TCS.bran[0])
            temp.append(min(TCS.parameters['close_time'], TCS.parameters['open_time']))
            temp.append(max(TCS.parameters['close_time'], TCS.parameters['open_time']))
            temp.append(TCS.parameters['resistance']) if TCS.parameters['type_of_data'] != 1 else 1e-6
            temp.append(TCS.parameters['resistance']) if TCS.parameters['type_of_data'] == 1 else 1e-6
            temp = np.array(temp)
            TCS_update_matrix = np.vstack((TCS_update_matrix, temp))

        NLR_update_matrix = np.empty((0, 4))
        for NLR in lumps.nolinear_resistors:
            temp = []
            temp.append(NLR.bran[0])
            temp.append(NLR.node1[0]) if NLR.node1[0] != 'ref' else -1
            temp.append(NLR.node2[0]) if NLR.node2[0] != 'ref' else -1
            temp.append(NLR.parameters['ri_characteristic']) if NLR.default == 1 else lambda x: NLR.parameters['resistance']
            temp = np.array(temp)
            NLR_update_matrix = np.vstack((NLR_update_matrix, temp))
        return {'SDEM': SDEM_update_matrix, 'VCS': VCS_update_matrix, 'TCS': TCS_update_matrix, 'NLR': NLR_update_matrix}


    #执行不同的算法：线性/非线性
    def SAF_calculate(self,T,dt,H,sources):
        ins_FO = {}
        SAF = []
        for tower in self.towers:
            for ins in tower.devices.insulators:
                for switch in ins.switch_disruptive_effect_models:
                    ins_FO[switch.name] = {'FO': 0,"Tower":tower.name,"Wire":switch.bran}
        ins_FO["FO"] = False
        if not self.switch_disruptive_effect_models and not self.voltage_controled_switchs and not self.time_controled_switchs and not self.nolinear_resistors:
            strategy = Strategy.MC_Linear()
            ins_FO =strategy.apply(T,dt,H,sources,tower_head_node)
        else:
            strategy = Strategy.SAF_NonLinear()
            solution,switch_disruptive_effect_models,SAF = strategy.apply(T,dt,H,sources)
            for switch in switch_disruptive_effect_models:
                if switch in ins_FO.keys():
                    ins_FO[switch]["FO"] = 1
                    ins_FO["FO"] = True
        return solution,ins_FO,SAF
    def INS_calculate(self,T,dt,H,sources):
        ins_FO = {}
        tower_list = ["tower_8","tower_9","tower_10","tower_11"]
        tower_head_node =[self.tower_head_node[tower] for tower in tower_list if tower in self.tower_head_node]
        # for tower in self.towers:
        #     #tower_head_node.append(tower.info.Pole_Head_Node)
        #     for ins in tower.devices.insulators:
        #         for switch in ins.switch_disruptive_effect_models:
        #             ins_FO[switch.name] = {'FO': 0,"Tower":tower.name,"Wire":switch.bran}
        # ins_FO["FO"] = False
        if not self.switch_disruptive_effect_models and not self.voltage_controled_switchs and not self.time_controled_switchs and not self.nolinear_resistors:
            strategy = Strategy.MC_Linear()
            ins_FO =strategy.apply(T,dt,H,sources,tower_head_node)
        else:
            strategy = Strategy.INS_NonLinear()
            ins_FO = strategy.apply(T,dt,H,sources)
        return ins_FO
    def calculate(self,T,dt,H,sources):
        ins_FO = {}
        SAF = []
        for tower in self.towers:
            for ins in tower.devices.insulators:
                for switch in ins.switch_disruptive_effect_models:
                    ins_FO[switch.name] = {'FO': 0,"Tower":tower.name,"Wire":switch.bran}
        ins_FO["FO"] = False
        if not self.switch_disruptive_effect_models and not self.voltage_controled_switchs and not self.time_controled_switchs and not self.nolinear_resistors:
            strategy = Strategy.Linear()
            solution =strategy.apply(T,dt,H,sources)
        else:
            strategy = Strategy.NonLinear()
            solution,switch_disruptive_effect_models = strategy.apply(T,dt,H,sources)
            for switch in switch_disruptive_effect_models:
                if switch in ins_FO.keys():
                    ins_FO[switch]["FO"] = 1
                    ins_FO["FO"] = True
        return solution,ins_FO

    def calculate_of_hybrid_mode(self, line_matrix, tower_matrix, sources, Nt, dt, GPU):
        """
        solution_type_id: 'hybrid', 'nonlinear', 'variable_frequency', 'variant_step'
        """
        solution_type_id = self.update_solution_type()
        match solution_type_id:
            case '1000':
                strategy = Strategy.hybrid_linear()
                tower_list = ["tower_8", "tower_9", "tower_10", "tower_11"]
                tower_head_node = [self.tower_head_node[tower] for tower in tower_list if tower in self.tower_head_node]
                return strategy.apply(line_matrix, tower_matrix, sources, Nt, dt, GPU,tower_head_node)
            case '1001':
                raise Exception('The variant_step module is not accessible.')
            case '1010':
                strategy = Strategy.hybrid_variant_frequency()
            case '1011':
                raise Exception('The variable_frequency-variant_step module is not accessible.')
            case '1100':
                strategy = Strategy.hybrid_nonlinear()
            case '1101':
                raise Exception('The nonlinear-variant_step module is not accessible.')
            case '1110':
                strategy = Strategy.hybrid_nonliear_variant_frequency()
            case '1111':
                raise Exception('The nonlinear-variant_frequency-variant_step module is not accessible.')
            case _:
                raise Exception('The model was build in hybrid mode, but hybrid calculation module is not used.')

        return strategy.apply(line_matrix, tower_matrix, sources, Nt, dt, GPU)
    # 设置全局参数
    def global_set(self, load_dict):
        self.frq = np.concatenate([
            np.arange(1, 91, 10),
            np.arange(100, 1000, 100),
            np.arange(1000, 10000, 1000),
            np.arange(10000, 100000, 10000)
        ])
        self.VF = {'odc': 10,
                   'frq': self.frq}

        # 是否有定义
        if 'Global' in load_dict:
            self.dt = load_dict['Global']['delta_time']
            self.T = load_dict['Global']['time']
            f0 = load_dict['Global']['constant_frequency']
            self.fixed_frequency = np.array([f0]).reshape(-1)
            self.max_length = load_dict['Global']['max_length']
            self.global_ground = load_dict['Global']['global_ground']
            self.ground = initialize_ground(load_dict['Global']['ground']) if self.global_ground else None
            self.GPU_calculation = load_dict['Global']['GPU_calculation']
            self.Nt = int(np.ceil(self.T / self.dt))
            self.Hybrid_method = load_dict['Global']['Hybrid_method']

    def nodes_of_hybrid_mode(self):
        tower_branches = {}
        ohl_branches = {}
        cable_branches = {}

        tower_branches, tower_nodes = self.tower_branches(tower_branches)
        tower_nodes.discard('ref')
        ohl_branches, ohl_nodes = self.OHL_branches(ohl_branches, self.max_length)
        ohl_nodes.discard('ref')
        cable_branches, cable_nodes = self.cable_branches(cable_branches, self.max_length)
        cable_nodes.discard('ref')
        lump_nodes = self.lump_nodes()
        lump_nodes.discard('ref')
        tower_or_lump_nodes = tower_nodes.union(lump_nodes)
        line_nodes = ohl_nodes.union(cable_nodes)
        tower_and_line_nodes = list(tower_or_lump_nodes.intersection(line_nodes))
        tower_and_line_nodes.sort()
        return tower_branches, tower_nodes, tower_or_lump_nodes, tower_and_line_nodes
    # 基础模块 分布运行
    def run_hybrid(self, load_dict):
        print("Hybrid model is used")
        self.global_set(load_dict)
        constants = Constant()
        self.dt = self.max_length / constants.vc
        self.Nt = int(np.ceil(self.T / self.dt))

        self.initialize_network(load_dict, self.VF)

        tower_matrix = self.tower_individual_matrix()  # 合并tower矩阵
        line_matrix = self.line_individual_matrix() # 合并cable和OHL矩阵

        # tower_branches, tower_nodes, tower_or_lump_nodes, tower_and_line_nodes = self.nodes_of_hybrid_mode()
        branches, nodes = self.calculate_branches(self.max_length)
        # nodes = self.capacitance_matrix.columns.tolist()

        # 3. 初始化源，计算结果
        share_dict = {}
        constants = Constant()
        sources,stroke_num = self.source_initial(load_dict, nodes,branches,constants,share_dict)

        self.H = {"Line": line_matrix,"Tower": tower_matrix}
        result_tower, other = self.calculate_of_hybrid_mode(line_matrix, tower_matrix, sources, self.Nt, self.dt, self.GPU_calculation)

        result_tower = pa.Table.from_pandas(result_tower.T)
        # result_v_ohl = pa.Table.from_pandas(result_v_ohl.T)
        # result_i_ohl = pa.Table.from_pandas(result_i_ohl.T)
        sources_pa = pa.Table.from_pandas(sources.T)
        pacsv.write_csv(result_tower, "Data/Output/result_tower_output.csv")
        # pacsv.write_csv(result_v_ohl, "Data/Output/result_v_ohl_output.csv")
        # pacsv.write_csv(result_i_ohl, "Data/Output/result_i_ohl_output.csv")
        pacsv.write_csv(sources_pa, "Data/Output/result_lightning.csv")

        # result_tower.to_csv("Data/Output/result_tower_output.csv")
        # result_v_ohl.to_csv("Data/Output/result_v_ohl_output.csv")
        # result_i_ohl.to_csv("Data/Output/result_i_ohl_output.csv")
    def run_base(self, load_dict, *basestrategy):
        # 0. 手动预设值
        self.global_set(load_dict)
        hybrid = load_dict["Global"]["Hybrid_method"]
        if hybrid == 1:
            self.solution_type['hybrid'] = True
            self.run_hybrid(load_dict)

        else:
            self.run(load_dict)
    # 基础模块合并运行
    def run(self, load_dict, *basestrategy):
        # 0. 手动预设值
        # self.dt = 1e-6
        # self.T = 1e-5
        # self.Nt = int(np.ceil(self.T / self.dt))

        # 1. 初始化电网，根据电网信息计算源
        self.initialize_network(load_dict, self.VF)
        self.combine_parameter_matrix()

        # 2. 保存支路节点信息(for source calculate)
        # 合并计算的时候是这样设置
        branches, nodes = self.calculate_branches(self.max_length)

        # nodes = self.capacitance_matrix.columns.tolist()

        # 3. 初始化源，计算结果
        share_dict = {}
        start_time = time.time()  # 记录开始时间
        constants = Constant()
        sources,stroke_num = self.source_initial(load_dict, nodes,branches,constants,share_dict)
        end = time.time()  # 记录开始时间
        print(f"Total running time: {end - start_time} seconds")  # 打印运行时长
        solution = self.calculate(self.Nt, self.dt, self.H, sources)
        end2 = time.time()  # 记录开始时间
        pd.DataFrame(solution[0]).to_csv("Data/Output/combine_output.csv")
        print(f"Total running time: {end2 - end} seconds")  # 打印运行时长
        print(solution)

    def sensitive_analysis(self, load_dict):

        if load_dict["Sensitivity_analysis"]["Stroke"]["position"]:
            wire = load_dict["Sensitivity_analysis"]["Stroke"]["area"]
            area = load_dict["Sensitivity_analysis"]["Stroke"]["area"]
            if area.split("_")[0] == "tower":
                for obj in load_dict["Tower"]:
                    for w in obj["Wire"]:
                        if load_dict["Sensitivity_analysis"]["Stroke"]["wire"]:
                            if w["name"] ==wire:
                                Strategy.Change_light_pos().apply(self,self.lightning,
                                                  load_dict["Sensitivity_analysis"]["Stroke"]["area"],
                                                  w,
                                                load_dict["Sensitivity_analysis"]["Stroke"]["position"])
            if area.split("_")[0] == "OHL":
                for obj in load_dict["OHL"]:
                    for w in obj["Wire"]:
                        if load_dict["Sensitivity_analysis"]["Stroke"]["cir_id"]:
                            if w["cir_id"] ==load_dict["Sensitivity_analysis"]["Stroke"]["cir_id"]\
                                    and w["phase"] ==load_dict["Sensitivity_analysis"]["Stroke"]["phase"]:
                                Strategy.Change_light_pos().apply(self, self.lightning,
                                                                  load_dict["Sensitivity_analysis"]["Stroke"]["area"],
                                                                  w,
                                                                  load_dict["Sensitivity_analysis"]["Stroke"]["position"])
                                break
                    break
        if load_dict["Sensitivity_analysis"]["Stroke"]["waveform"]:
            Strategy.Change_light_waveform().apply(self,load_dict["Source"]["Lightning"],
                                                   load_dict["Sensitivity_analysis"]["Stroke"]["waveform"],
                                                   load_dict)

        if load_dict["Sensitivity_analysis"]["Stroke"]["paramenters"]:
            Strategy.Change_light_waveform().apply(self, load_dict["Source"]["Lightning"],
                                                   load_dict["Sensitivity_analysis"]["Stroke"]["paramenters"],
                                                   load_dict)

        if load_dict["Sensitivity_analysis"]["Arrester"]["name"]:
            name = load_dict["Sensitivity_analysis"]["Arrester"]["name"]
            Strategy.Change_Arrestor().apply(self, load_dict,name)

        if load_dict["Sensitivity_analysis"]["SW"]["name"]:
            name = load_dict["Sensitivity_analysis"]["SW"]["name"]
            Strategy.Change_SW().apply(self,load_dict,name)

        if load_dict["Sensitivity_analysis"]["DE"]:
            Strategy.Change_DE_max().apply(self,load_dict["Sensitivity_analysis"]["DE"])
            self.calculate(self.Nt, self.H, self.sources)
            pd.DataFrame(self.run_measure()).to_csv("DE_modified.csv")


        if load_dict["Sensitivity_analysis"]["ROD"]:
            for tower in load_dict["Tower"]:
                if tower["name"] ==load_dict["Sensitivity_analysis"]["ROD"]["tower"]:
                    Strategy.Change_ROD().apply(self,load_dict,
                                        load_dict["Sensitivity_analysis"]["ROD"]["lump"],
                                        load_dict["Sensitivity_analysis"]["ROD"]["r"],
                                        load_dict["Sensitivity_analysis"]["ROD"]["l"])

        print("you are starting sensitive analysis")
    def run_measure(self):
        # lumpname/branname: label(bran0/lump1),probe,branname,n1,n2,(towername)
        if self.measurement:
            return Strategy.Measurement().apply(measurement=self.measurement, solution=self.solution,dt=self.dt)
    def run_MC(self,load_dict,Distance):
    ### 用大矩阵----------
        # 0. 手动预设值
        # self.global_set(load_dict)
        # self.dt = 1e-8
        # #self.Nt = 1000
        # self.T = 2e-5
        # self.Nt = int(np.ceil(self.T / self.dt))
        # # 1. 初始化电网，根据电网信息计算源
        # self.initialize_network(load_dict,self.VF)
        # self.combine_parameter_matrix()
        # branches = self.calculate_branches(self.max_length)
        # nodes = self.capacitance_matrix.columns.tolist()

#### 用小矩阵-----------
        print("Hybrid model is used")

        self.solution_type['hybrid'] = True
        self.global_set(load_dict)
        constants = Constant()
        self.dt = self.max_length / constants.vc
        self.Nt = int(np.ceil(self.T / self.dt))

        self.initialize_network(load_dict, self.VF)

        tower_matrix = self.tower_individual_matrix()  # 合并tower矩阵
        line_matrix = self.line_individual_matrix() # 合并cable和OHL矩阵

        # tower_branches, tower_nodes, tower_or_lump_nodes, tower_and_line_nodes = self.nodes_of_hybrid_mode()
        tower_branches = {}
        tower_branches, tower_nodes = self.tower_branches(tower_branches)
        tower_nodes.discard('ref')

        self.H = {"Line": line_matrix,"Tower": tower_matrix}

        # 2. 保存支路节点信息
        branches,nodes = self.calculate_branches(self.max_length)

        #branches = segment_branch(branches)
        #nodes =list(tower_nodes | set(line_matrix["capacitance_matrix"].columns.tolist()))



        # 3. 生成多个雷电
        if load_dict["MC"]:
            print("running Monte Carlo to generate lightnings")
            df27,parameterst,stroke_result,PoleXY = run_MC(self,load_dict,Distance)

            MC_result_list = []
            summary = {"nonFO_indirect":0,"nonFO_direct":0,"FO_indirect":0,"FO_direct":0,"Huri":0,"RunTime":0}
            with Manager() as manager:
                shared_dict = manager.dict()
                index = 0
                MC_result = []
                #对每个flash
                for i in df27.groupby("flash"):
                    stroke_list = []
                    #初始化每个stroke
                    for j in range(i[1].shape[0]):
                        stroke_type = "Heidler"
                        duration = self.T
                        dt = self.dt
                        stroke = Stroke(stroke_type, duration=duration, dt=dt, is_calculated=True, parameter_set=None,
                                    parameters=[parameterst[index].tolist()[2]*1e3,parameterst[index].tolist()[4],
                                    parameterst[index].tolist()[6],parameterst[index].tolist()[5], parameterst[index].tolist()[3]])
                        stroke.calculate()
                        index += 1
                        stroke_list.append(stroke)
                    flash_type = stroke_result[0][index - 1]
                    area = int(stroke_result[1][index - 1])
                    area_id = 0 if math.isnan(stroke_result[2][index - 1]) else str(int(stroke_result[2][index - 1]))
                    cir_id = 0 if math.isnan(float(stroke_result[3][index - 1])) else int(
                        float(stroke_result[3][index - 1]))
                    phase_id = 0 if math.isnan(float(stroke_result[4][index - 1])) else int(
                        float(stroke_result[4][index - 1]))
                    position_xy = stroke_result[8][index - 1]
                    position = None
                    wire = None
                    if phase_id == 0:
                        phase_lgt = 'S'
                    elif phase_id == 1:
                        phase_lgt = 'A'
                    elif phase_id == 2:
                        phase_lgt = 'B'
                    else:
                        phase_lgt = 'C'

                    if area == 0:
                        area = "Ground"
                        position = position_xy.append(0)
                    elif area == 1:
                        area = "tower_" + area_id
                        for tower in load_dict["Tower"]:
                            if tower["Info"]["name"] == area:
                                z = tower["Info"]["pole_height"]
                                position = position_xy.append(z)
                                for w in tower["Wire"]:
                                    if w["pos_1"][2] == z or w["pos_2"][2] == z:
                                        wire = w["bran"]
                                if wire is None:
                                    wire = tower["Wire"][0]
                    elif area == 2:
                        area = "OHL_" + area_id
                        for ohl in load_dict["OHL"]:
                            if ohl["Info"]["name"] == area:
                                for w in ohl["Wire"]:
                                    cir_id_ohl = w['cir_id']
                                    phase_id_ohl = w['phase']
                                    if cir_id_ohl == cir_id:
                                        z = w["node1_pos"][2]
                                        position = position_xy.append(z)
                                        if w['type'] == 'SW':
                                            wire = 'Y' + str(cir_id) + 'S'
                                        elif w['type'] == 'CIRO':
                                            wire = 'Y' + str(cir_id) + w['phase']

                    lightning = Lightning(id=1, type=flash_type, strokes=stroke_list, channel=Channel(position_xy))
                    MC_result.append((lightning, area, wire, position_xy))
                #MC_result_list.append(MC_result)
                record_SAF = load_dict["MC"]["Record_SAF"]
                start_time = time.time()
                  # 创建一个共享字典
                if record_SAF==1:
                    FO_direct,FO_indirect,SAF_direct,SAF_indirect  = self.Huri_method_SAF(MC_result, nodes, branches,shared_dict)
                else:
                    FO_direct,FO_indirect,huri = self.Huri_method_INS(MC_result, nodes, branches,shared_dict)
                    end_time = time.time()  # 记录结束时间
                    duration = end_time - start_time  # 计算运行时长
                    true_count_direct = len(list(filter(lambda x: x, FO_direct)))
                    false_count_direct = len(FO_direct) -true_count_direct
                    true_count_indirect = len(list(filter(lambda x: x, FO_indirect)))
                    false_count_indirect = len(FO_indirect) - true_count_indirect
                    summary["FO_direct"] = summary["FO_direct"]+true_count_direct
                    summary["FO_indirect"] = summary["FO_indirect"] + true_count_indirect
                    summary["nonFO_direct"] = summary["nonFO_direct"]+false_count_direct
                    summary["nonFO_indirect"] = summary["nonFO_indirect"] + false_count_indirect
                    summary["Huri"] = huri
                    summary["RunTime"] = duration
                    print("FO_direct: ", summary["FO_direct"])
                    print("FO_indirect: ", summary["FO_indirect"])
                    print("nonFO_direct: ", summary["FO_direct"])
                    print("nonFO_indirect: ", summary["FO_indirect"])
                    print("Huri: ",summary["Huri"])
                    print("Running time: ",summary["RunTime"])  # 打印运行时长
                    df = pd.DataFrame(summary)
                    # 保存DataFrame到CSV文件
                    df.to_csv(f'Data/input/case3_nonlinear/summary_values_ins.csv', index=False)


    def Pre_run_MC(self, load_dict):
        ### 用大矩阵----------
        # 0. 手动预设值
        self.global_set(load_dict)
        self.dt = 1e-8
        #self.Nt = 1000
        self.T = 2e-5
        self.Nt = int(np.ceil(self.T / self.dt))
        # 1. 初始化电网，根据电网信息计算源
        self.initialize_network(load_dict,self.VF)
        self.combine_parameter_matrix()
        branches,nodes = self.calculate_branches(self.max_length)
        nodes.remove('ref')


        #### 用小矩阵-----------
        # print("Hybrid model is used")
        #
        # self.solution_type['hybrid'] = True
        # self.global_set(load_dict)
        # constants = Constant()
        # self.dt = self.max_length / constants.vc
        # self.Nt = int(np.ceil(self.T / self.dt))
        #
        # self.initialize_network(load_dict, self.VF)
        #
        # tower_matrix = self.tower_individual_matrix()  # 合并tower矩阵
        # line_matrix = self.line_individual_matrix()  # 合并cable和OHL矩阵
        #
        # # tower_branches, tower_nodes, tower_or_lump_nodes, tower_and_line_nodes = self.nodes_of_hybrid_mode()
        # tower_branches = {}
        # tower_branches, tower_nodes = self.tower_branches(tower_branches)
        # tower_nodes.discard('ref')
        #
        # self.H = {"Line": line_matrix, "Tower": tower_matrix}
        #
        # # 2. 保存支路节点信息
        # branches,nodes = self.calculate_branches(self.max_length)
       # nodes = list(tower_nodes | set(line_matrix["capacitance_matrix"].columns.tolist()))

        # 3. 生成多个雷电
        if load_dict["MC"]:
            print("running Monte Carlo to generate lightnings")
            df27_list, parameterst_list, stroke_result_list, PoleXY = run_MC(self, load_dict,None)
            dataset = []
            MC_result_list = []
            summary = {"FOR": [],"FO": [], "Huri": [], "RunTime": []}
            FO_num = 0
            with Manager() as manager:
                shared_dict = manager.dict()
                for a in range(len(df27_list)):
                    index = 0
                    MC_result = []
                    df27 = df27_list[a]
                    parameterst = parameterst_list[a]
                    stroke_result = stroke_result_list[a]
                    # 对每个flash
                    for i in df27.groupby("flash"):
                        stroke_list = []
                        # 初始化每个stroke
                        for j in range(i[1].shape[0]):
                            stroke_type = "Heidler"
                            duration = self.T
                            dt = self.dt
                            stroke = Stroke(stroke_type, duration=duration, dt=dt, is_calculated=True,
                                            parameter_set=None,
                                            parameters=[parameterst[index].tolist()[2]*1e3 ,
                                                        parameterst[index].tolist()[3],
                                                        parameterst[index].tolist()[5], parameterst[index].tolist()[4]])
                            stroke.calculate()
                            index += 1
                            stroke_list.append(stroke)
                        flash_type = stroke_result[0][index - 1]
                        area = int(stroke_result[1][index - 1])
                        area_id = 0 if math.isnan(stroke_result[2][index - 1]) else str(
                            int(stroke_result[2][index - 1]))
                        cir_id = 0 if math.isnan(float(stroke_result[3][index - 1])) else int(
                            float(stroke_result[3][index - 1]))
                        phase_id = 0 if math.isnan(float(stroke_result[4][index - 1])) else int(
                            float(stroke_result[4][index - 1]))
                        position_xy = stroke_result[8][index - 1]
                        position = None
                        wire = None
                        if area == 0:
                            area = "Ground"
                            position = position_xy.append(0)
                        elif area == 1:
                            area = "tower_" + area_id
                            for tower in load_dict["Tower"]:
                                if tower["Info"]["name"] == area:
                                    z = tower["Info"]["pole_height"]
                                    position = position_xy.append(z)
                                    for w in tower["Wire"]:
                                        if w["pos_1"][2] == z or w["pos_2"][2] == z:
                                            wire = w["bran"]
                                    if wire is None:
                                        wire = tower["Wire"][0]
                        elif area == 2:
                            area = "OHL_" + area_id
                            for ohl in load_dict["OHL"]:
                                if ohl["Info"]["name"] == area:
                                    for w in ohl["Wire"]:
                                        cir_id_ohl = w['cir_id']
                                        phase_id_ohl = w['phase_id']
                                        if cir_id_ohl == cir_id:
                                            z = w["node1_pos"][2]
                                            position = position_xy.append(z)
                                            if w['type'] == 'SW':
                                                wire = 'Y' + str(cir_id) + 'S'
                                            elif w['type'] == 'CIRO':
                                                wire = 'Y' + str(cir_id) + w['phase']
                        lightning = Lightning(id=1, type=flash_type, strokes=stroke_list, channel=Channel(position_xy))
                        MC_result.append((lightning, area, wire, position_xy))
                    start_time = time.time()
                    # 创建一个共享字典
                    FO, huri,dataset = self.Find_Dmax(MC_result, nodes, branches, shared_dict,dataset)
                    end_time = time.time()  # 记录结束时间
                    duration = end_time - start_time  # 计算运行时长
                    true_count = len(list(filter(lambda x: x, FO)))
                    FO_num = FO_num + true_count
                    FOR = 100*FO_num/df27_list[-1].iloc[-1, 0]*2

                    summary["FO"].append(FO_num)
                    summary["FOR"].append(FOR)
                    summary["Huri"].append(huri)
                    summary["RunTime"].append(duration)

                self.save_result(summary,path='Data/input/case2_linear/')

            positions = [i+1 for i, (a, b) in enumerate(zip(summary["FO"], summary["FO"][1:])) if b - a < 1]
            Distance_max = len(df27_list)*100
            if positions[0]:
                Distance_max = (positions[0]+1) * 100
            print("maximum distance is: ", Distance_max)
            print("total flash count: ", df27_list[-1].iloc[-1, 0])
            return Distance_max
    def save_result(self,summary,path):
        print("FO: ", summary["FO"])
        print("Huri: ", summary["Huri"])
        print("Running time: ", summary["RunTime"])  # 打印运行时长
        for key, values in summary.items():
            plt.figure()  # 创建一个新的图形
            plt.plot(values)
            plt.title(f'Plot for {key}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            # 保存每个图表为图片文件
            plt.savefig(f'{path}{key}_plot.png')
            plt.close()  # 关闭图形，避免内存泄漏

            # 将数据保存到CSV文件
            # 创建一个DataFrame，其中包含索引和对应的值
            df = pd.DataFrame({'Index': range(len(values)), 'Value': values})
            # 保存DataFrame到CSV文件
            df.to_csv(f'Data/input/case2_linear/{key}_values.csv', index=False)
        # 所有图片保存后，显示它们
        for key, values in summary.items():
            plt.figure()
            plt.plot(values)
            plt.title(f'Plot for {key}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.show()
            # self.run_multiprocessed(MC_result, nodes, branches,shared_dict,PoleXY)

    def run_multiprocessed(self, MC_results, nodes, branches,shared_dict,PoleXY):
        # 创建一个Manager对象，用于创建共享字典
        # with Manager() as manager:
        #     shared_dict = manager.dict()  # 创建一个共享字典
        #lock = Lock()
        # 创建进程列表
        start_time = time.time()  # 记录开始时间
        processes = []
        for index,MC in enumerate(MC_results):
            # 创建Process对象，传递当前实例和MC_result的元素
            p = Process(target=process_item, args=(MC, nodes, branches, self,index,shared_dict))
            processes.append(p)
            p.start()  # 启动进程

            ins = process_calculate(MC, nodes, branches, self,index,shared_dict,1)


        # 等待所有进程完成
        for p in processes:
            p.join()
        end_time = time.time()  # 记录结束时间
        duration = end_time - start_time  # 计算运行时长
        print(f"Total running time: {duration} seconds")  # 打印运行时长

        # 保存运行时长到文件
        with open("runtime_limit_time.txt", "w") as f:
            f.write(f"Total running time: {duration} seconds\n")

        return duration

    def Huri_method_SAF(self,MC_result, nodes, branches,shared_dict):
        icurr = []
        dataset_ins = []
        dataset_saf = []
        FO_direct = []
        FO_indirect = []
        SAF_direct = []
        SAF_indirect = []
        R = 100
        constants = Constant()
        constants.ep0 = 8.85e-12
        for MC in MC_result:
            if len(dataset_saf) > 0 and len(dataset_ins):
                flash_position = MC[0].channel.hit_pos[:2]
                for i, stroke in enumerate(MC[0].strokes):
                    icur = np.append(stroke.parameters, flash_position)
                    result_ins = Huri_Method(R, dataset_saf, icur)
                    result_saf = Huri_Method_SAF(R, dataset_saf, icur)
                    if result_saf == 1 and result_ins == 1:
                        SAF_indirect.append(False)
                        FO_indirect.append(False)
                        print("using Huri skip calculation")
                    else:
                        U_out, I_out, shared_dict = self.source_calculate(MC[0], MC[1], MC[2], MC[3], nodes, branches,
                                                                          constants, shared_dict)
                        sources = self.add_lump(U_out, I_out)
                        solution, ins, saf = self.SAF_calculate(self.Nt, self.dt, self.H, sources)
                        SAF_direct,SAF_indirect = self.Huri_saf(MC, nodes, branches, constants, shared_dict, SAF_direct,SAF_indirect, dataset_saf, ins, saf, R)
                        FO_direct,FO_indirect = self.Huri_ins(MC, nodes, branches, constants, shared_dict,FO_direct,FO_indirect,dataset_ins, ins, saf,R)
            else:
                U_out, I_out, shared_dict = self.source_calculate(MC[0], MC[1], MC[2], MC[3], nodes, branches,
                                                                  constants, shared_dict)
                sources = self.add_lump(U_out, I_out)
                solution, ins, saf = self.SAF_calculate(self.Nt, self.dt, self.H, sources)
                SAF_direct,SAF_indirect = self.Huri_saf(MC, nodes, branches, constants, shared_dict, SAF_direct,SAF_indirect, dataset_saf, ins, saf, R)
                FO_direct,FO_indirect = self.Huri_ins(MC, nodes, branches, constants, shared_dict,FO_direct,FO_indirect,dataset_ins, ins, saf,R)
        return FO_direct,FO_indirect,SAF_direct,SAF_indirect

    def Huri_ins(self,MC, nodes, branches, constants, shared_dict,FO_direct,FO_indirect,dataset_ins, ins, saf,R):
        # 1. 直接雷
        if MC[0].type == "Direct":
            U_out, I_out, shared_dict = self.source_calculate(MC[0], MC[1], MC[2], MC[3], nodes, branches,
                                                              constants, shared_dict)
            sources = self.add_lump(U_out, I_out)
            solution, ins, saf = self.SAF_calculate(self.Nt, self.dt, self.H, sources)

            if ins:
                FO_direct.append(saf)  # 1.1 直接雷闪络
            else:
                FO_direct.append(False)  # 1.2 直接雷不闪络
            return FO_direct,FO_indirect

        # 2. 间接雷
        flash_position = MC[0].channel.hit_pos[:2]
        # 2.1 间接雷没有dataset
        if len(dataset_ins) == 0:
            if ins:
                FO_indirect.append(ins)
            else:
                distances = []
                for coord in list(self.PoleXY.values()):
                    distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
                    distances.append((coord, distance))
                distances.sort(key=lambda item: item[1])
                closest_two = distances[:2]
                PoleApp = [closest_two[0][0][0], closest_two[0][0][1], closest_two[1][0][0], closest_two[1][0][1]
                    , closest_two[0][1], closest_two[1][1]]
                dataset_ins.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))
                FO_indirect.append(False)
            return FO_direct,FO_indirect

        # 2.2 间接雷有dataset
        if len(dataset_ins) > 0:
            for i, stroke in enumerate(MC[0].strokes):
                icur = np.append(stroke.parameters, flash_position)
                result = Huri_Method(R, dataset_ins, icur)
                if result == 1:
                    FO_indirect.append(False)
                    print("using Huri skip calculation")
                    return FO_direct,FO_indirect  # 间接雷用huri跳过
        # 间接雷不跳过
        if ins: #闪络
            FO_indirect.append(ins)  # 不跳过闪络
        else: #不闪落
            distances = []
            for coord in list(self.PoleXY.values()):
                distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
                distances.append((coord, distance))
            distances.sort(key=lambda item: item[1])
            closest_two = distances[:2]
            PoleApp = [closest_two[0][0][0], closest_two[0][0][1], closest_two[1][0][0], closest_two[1][0][1]
                , closest_two[0][1], closest_two[1][1]]
            dataset_ins.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))
            FO_indirect.append(False)  # 不跳过不闪络
        return FO_direct,FO_indirect

    def Huri_saf(self, MC, nodes, branches, constants, shared_dict, SAF_direct,SAF_indirect, dataset_saf, ins, saf, R):
        # 1. 直接雷
        if MC[0].type == "Direct":
            U_out, I_out, shared_dict = self.source_calculate(MC[0], MC[1], MC[2], MC[3], nodes, branches,
                                                              constants, shared_dict)
            sources = self.add_lump(U_out, I_out)
            solution, ins, saf = self.SAF_calculate(self.Nt, self.dt, self.H, sources)

            if saf:
                SAF_direct.append(saf)  # 1.1 直接雷闪络
            else:
                SAF_direct.append(False)  # 1.2 直接雷不闪络
            return SAF_direct,SAF_indirect

        # 2. 间接雷
        flash_position = MC[0].channel.hit_pos[:2]
        # 2.1 间接雷没有dataset
        if len(dataset_saf) == 0:
            U_out, I_out, shared_dict = self.source_calculate(MC[0], MC[1], MC[2], MC[3], nodes, branches,
                                                              constants, shared_dict)
            sources = self.add_lump(U_out, I_out)
            solution, ins, saf = self.SAF_calculate(self.Nt, self.dt, self.H, sources)
            if saf:
                SAF_indirect.append(saf)
            else:
                distances = []
                for coord in list(self.PoleXY.values()):
                    distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
                    distances.append((coord, distance))
                distances.sort(key=lambda item: item[1])
                closest_two = distances[:2]
                PoleApp = [closest_two[0][0][0], closest_two[0][0][1], closest_two[1][0][0], closest_two[1][0][1]
                    , closest_two[0][1], closest_two[1][1]]
                dataset_saf.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))
                SAF_indirect.append(False)
            return SAF_direct,SAF_indirect

        # 2.2 间接雷有dataset
        if len(dataset_saf) > 0:
            for i, stroke in enumerate(MC[0].strokes):
                icur = np.append(stroke.parameters, flash_position)
                result = Huri_Method(R, dataset_saf, icur)
                if result == 1:
                    SAF_indirect.append(False)
                    print("using Huri skip calculation")
                    return SAF_direct,SAF_indirect  # 间接雷用huri跳过
        # 间接雷不跳过
        U_out, I_out, shared_dict = self.source_calculate(MC[0], MC[1], MC[2], MC[3], nodes, branches,
                                                          constants, shared_dict)
        sources = self.add_lump(U_out, I_out)
        solution, ins,saf = self.SAF_calculate(self.Nt, self.dt, self.H, sources)
        if saf:
            SAF_indirect.append(ins)  # 不跳过闪络
        else:
            distances = []
            for coord in list(self.PoleXY.values()):
                distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
                distances.append((coord, distance))
            distances.sort(key=lambda item: item[1])
            closest_two = distances[:2]
            PoleApp = [closest_two[0][0][0], closest_two[0][0][1], closest_two[1][0][0], closest_two[1][0][1]
                , closest_two[0][1], closest_two[1][1]]
            dataset_saf.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))
            SAF_indirect.append(False)  # 不跳过不闪络
        return SAF_direct,SAF_indirect

    def Huri_method_INS(self,MC_result, nodes, branches,shared_dict):
        icurr = []
        dataset = []
        FO_indirect = []
        FO_direct = []
        R = 100
        constants = Constant()
        constants.ep0 = 8.85e-12
        index = 0
        calculate_ins = 1
        huri = 0
        for MC in MC_result:
            if MC[0].type == "Direct":
                #solution,ins = process_calculate(MC, nodes, branches, self,index,shared_dict,calculate_ins)
                ins = process_calculate(MC, nodes, branches, self, index, shared_dict, calculate_ins)
                index = index+1
                #if ins["FO"]:
                if ins:
                    FO_direct.append(ins)
                else:
                    FO_direct.append(False)
                continue
            flash_position = MC[0].channel.hit_pos[:2]
            if len(dataset) == 0:
                #solution, ins = process_calculate(MC, nodes, branches, self, index, shared_dict,calculate_ins)
                ins = process_calculate(MC, nodes, branches, self, index, shared_dict, calculate_ins)
                index = index + 1
                #if ins["FO"]:
                if ins:
                    FO_indirect.append(ins)
                    continue
                else:
                    distances = []
                    for coord in list(self.PoleXY.values()):
                        distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
                        distances.append((coord, distance))
                    distances.sort(key=lambda item: item[1])
                    closest_two = distances[:2]
                    PoleApp = [closest_two[0][0][0], closest_two[0][0][1], closest_two[1][0][0], closest_two[1][0][1]
                        , closest_two[0][1], closest_two[1][1]]
                    dataset.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))
                    FO_indirect.append(False)
                    continue
            if len(dataset)>0:
                #for i,stroke in enumerate(MC[0].strokes):
                stroke = MC[0].strokes[0]
                icur = np.append(stroke.parameters,flash_position)
                result = Huri_Method(R,dataset,icur)
                if result ==1:
                    FO_indirect.append(False)
                    huri = huri+1
                    distances = []
                    for coord in list(self.PoleXY.values()):
                        distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
                        distances.append((coord, distance))
                    distances.sort(key=lambda item: item[1])
                    closest_two = distances[:2]
                    PoleApp = [closest_two[0][0][0], closest_two[0][0][1], closest_two[1][0][0], closest_two[1][0][1]
                        , closest_two[0][1], closest_two[1][1]]
                    dataset.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))

                    print("using Huri skip calculation")
                    continue
            #solution, ins = process_item(MC, nodes, branches, self, index, shared_dict,calculate_ins)
            ins = process_calculate(MC, nodes, branches, self, index, shared_dict, calculate_ins)
            index = index + 1
           # if ins["FO"]:
            if ins:
                FO_indirect.append(ins)

            else:
                distances = []
                for coord in list(self.PoleXY.values()):
                    distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
                    distances.append((coord, distance))
                distances.sort(key=lambda item: item[1])
                closest_two = distances[:2]
                PoleApp = [closest_two[0][0][0],closest_two[0][0][1],closest_two[1][0][0],closest_two[1][0][1]
                                                ,closest_two[0][1],closest_two[1][1]]
                dataset.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))
                FO_indirect.append(False)
        return FO_direct,FO_indirect,huri

    def Find_Dmax(self,MC_result, nodes, branches,shared_dict,dataset):
        icurr = []
       # dataset = []
        FO = []
        R = 100
        constants = Constant()
        constants.ep0 = 8.85e-12
        index = 0
        calculate_ins = 1
        huri = 0
        for MC in MC_result:
            if MC[0].type == "Direct":
                #solution,ins = process_item(MC, nodes, branches, self,index,shared_dict,calculate_ins)
                ins = process_item(MC, nodes, branches, self, index, shared_dict, calculate_ins)
                index = index+1
                #if ins["FO"]:
                if ins:
                    FO.append(ins)
                else:
                    FO.append(False)
                continue
            flash_position = MC[0].channel.hit_pos[:2]
            if len(dataset) == 0:
                #solution, ins = process_item(MC, nodes, branches, self, index, shared_dict,calculate_ins)
                ins = process_item(MC, nodes, branches, self, index, shared_dict, calculate_ins)
                index = index + 1
                #if ins["FO"]:
                if ins:
                    FO.append(ins)
                    continue
                else:
                    distances = []
                    for coord in list(self.PoleXY.values()):
                        distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
                        distances.append((coord, distance))
                    distances.sort(key=lambda item: item[1])
                    closest_two = distances[:2]
                    PoleApp = [closest_two[0][0][0], closest_two[0][0][1], closest_two[1][0][0], closest_two[1][0][1]
                        , closest_two[0][1], closest_two[1][1]]
                    dataset.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))
                    FO.append(False)
                    continue
            if len(dataset)>0:
                #for i,stroke in enumerate(MC[0].strokes):
                stroke = MC[0].strokes[0]
                icur = np.append(stroke.parameters,flash_position)
                result = Huri_Method(R,dataset,icur)
                if result ==1:
                    FO.append(False)
                    huri = huri+1
                    distances = []
                    for coord in list(self.PoleXY.values()):
                        distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
                        distances.append((coord, distance))
                    distances.sort(key=lambda item: item[1])
                    closest_two = distances[:2]
                    PoleApp = [closest_two[0][0][0], closest_two[0][0][1], closest_two[1][0][0], closest_two[1][0][1]
                        , closest_two[0][1], closest_two[1][1]]
                    dataset.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))

                    print("using Huri skip calculation")
                    continue
            #solution, ins = process_item(MC, nodes, branches, self, index, shared_dict,calculate_ins)
            ins = process_item(MC, nodes, branches, self, index, shared_dict, calculate_ins)
            index = index + 1
           # if ins["FO"]:
            if ins:
                FO.append(ins)

            else:
                distances = []
                for coord in list(self.PoleXY.values()):
                    distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
                    distances.append((coord, distance))
                distances.sort(key=lambda item: item[1])
                closest_two = distances[:2]
                PoleApp = [closest_two[0][0][0],closest_two[0][0][1],closest_two[1][0][0],closest_two[1][0][1]
                                                ,closest_two[0][1],closest_two[1][1]]
                dataset.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))
                FO.append(False)
        return FO,huri,dataset

def process_calculate(MC, nodes, branches, self_ref,index,shared_dict,calculate_ins):
    constants = Constant()
    constants.ep0 = 8.85e-12
    U_out, I_out,shared_dict = self_ref.source_calculate(MC[0], MC[1], MC[2], MC[3], nodes, branches,constants,shared_dict)
    sources = self_ref.add_lump(U_out, I_out)
    line_matrix = self_ref.H["Line"]
    tower_matrix = self_ref.H["Tower"]
    result_tower, ins_bran = self_ref.calculate_of_hybrid_mode(line_matrix, tower_matrix, sources, self_ref.Nt, self_ref.dt, self_ref.GPU_calculation)
    print("calculate"+str(index))

    ins = False
    if len(ins_bran["SDEM"])>0:
        ins = True
        # 指定CSV文件名
        filename = "Data/output/MC_ins.csv"
        # 使用'a'模式打开文件，准备追加内容
        with open(filename, 'a', newline='') as csvfile:
            # 创建一个csv写入器
            writer = csv.writer(csvfile)
            writer.writerow(ins_bran["SDEM"])

    return ins

def process_item(MC, nodes, branches, self_ref,index,shared_dict,calculate_ins):
    constants = Constant()
    constants.ep0 = 8.85e-12
    U_out, I_out,shared_dict = self_ref.source_calculate(MC[0], MC[1], MC[2], MC[3], nodes, branches,constants,shared_dict)
    sources = self_ref.add_lump(U_out, I_out)
    # line_matrix = self_ref.H["Line"]
    # tower_matrix = self_ref.H["Tower"]
    # result_tower, ins_bran = self_ref.calculate_of_hybrid_mode(line_matrix, tower_matrix, sources, self_ref.Nt, self_ref.dt, self_ref.GPU_calculation)
    # print("calculate"+str(index))
    #
    # ins = False
    # if len(ins_bran["SDEM"])>0:
    #     ins = True
    #     # 指定CSV文件名
    #     filename = "Data/output/MC_ins.csv"
    #     # 使用'a'模式打开文件，准备追加内容
    #     with open(filename, 'a', newline='') as csvfile:
    #         # 创建一个csv写入器
    #         writer = csv.writer(csvfile)
    #         writer.writerow(ins_bran["SDEM"])
    #
    # return ins

  ## 用大矩阵------------

    H = self_ref.build_H()
    if calculate_ins ==1:
        ins = self_ref.INS_calculate(self_ref.Nt,self_ref.dt, H, sources)
        print("calculate"+str(index))
        return ins

