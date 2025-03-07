import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import csv
import os
import numpy as np
import pandas as pd
from functools import reduce
from Model.Node import Node
from itertools import chain
import copy
import pyarrow as pa
import pyarrow.csv as pacsv
import matplotlib.pyplot as plt
from Sensitive.single_change import run_sensitivity_analysis
# from Demos.BackupRead_BackupWrite import tempfile
from Risk_Evaluate.MC import run_MC
from Driver.initialization.initialization import initialize_OHL, initialize_tower, initial_lightning, initial_lump, \
    initialize_cable, initialize_ground
from Driver.modeling.OHL_modeling import OHL_building, OHL_building_variant_frequency, OHL_building_FDTD, \
    OHL_building_hybrid_variant_frequency
from Driver.modeling.cable_modeling import cable_building, cable_building_variant_frequency, cable_building_hybrid_variant_frequency
from Driver.modeling.tower_modeling import tower_building, tower_building_variant_frequency, tower_building_with_tube, \
    tower_building_variant_frequency_with_tube
from Function.Calculators.InducedVoltage_calculate import InducedVoltage_calculate, \
    H_MagneticField_calculate, ElectricField_calculate, ElectricField_above_lossy, InducedVoltage_calculate_indirect, \
    InducedVoltage_calculate_direct, LightningCurrent_calculate_direct, LightningCurrent_calculate_indirect
# from Risk_Evaluate.MC import run_MC
from Model.Cable import Cable
from Model.Lightning import Lightning
from Model.Tower import Tower
from Model.Wires import OHLWire, Wires
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
from Utils.Matrix import block_diag_3dim
from collections import Counter
import random

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
        self.OHL_node = []
        self.sig=None
        self.epr = None
        self.RL_node = []
        self.FO = {}
        self.broken = {}
        self.arrestor_bran2Tower = {}
        self.FO_Tower = []
        self.Tower2ins_bran = {}
        self.Distance = 0
        self.MC_flash = []
        self.weak_point = {}
        self.sensitive_target = {}
        self.global_ground= {}

    # 记录电网元素之间的关系
    def tower_branches(self, branches):
        tower_nodes = []
        for tower in self.towers:
            for wire in list(tower.wires.get_all_wires().values()):
                startnode = {wire.start_node.name: [wire.start_node.x+tower.info.position[0], wire.start_node.y+tower.info.position[1],
                                                    wire.start_node.z+tower.info.position[2]]}
                endnode = {wire.end_node.name: [wire.end_node.x+tower.info.position[0], wire.end_node.y+tower.info.position[1],
                                                wire.end_node.z+tower.info.position[2]]}
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
        self.OHL_node = list(set(OHL_nodes))
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
        self.arrestor_bran2Tower = {bran: [tower.name, device.name] for tower in self.towers for device in tower.devices.arrestors for bran in
                                    device.branList}
        # self.broken = {device.name:0 for tower in self.towers for device in
        #                             tower.devices.arrestors}
        # self.FO_Tower = {tower.name: 0 for tower in self.towers}
        for tower in self.towers:
            self.weak_point[tower.name] = 0
            gnd = self.ground if self.global_ground == 1 else tower.ground
            self.PoleXY[tower.info.name] = tower.info.position[:2]
            self.tower_head_node[tower.info.name] = tower.info.Pole_Head_Node
            self.RL_node = self.RL_node+[[RL.node1[0],RL.node2[0]]for RL in tower.lump.resistor_inductors]

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
                    self.Tower2ins_bran[tower.name] = [bran for swh in device.switch_disruptive_effect_models for bran in swh.bran ]
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

            self.lightning = lightning
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
                sources = self.add_lump(U_out, I_out)
                return sources
            if light["area"].split("_")[0] == "tower":
                U_out, I_out, share_dict = self.source_calculate(lightning,
                                                                 light["area"], light["wire"], light["position"], nodes,
                                                                 branches, constants, share_dict)
                sources = self.add_lump(U_out, I_out)
                return sources
            if light["area"].split("_")[0] == "Ground":
                U_out, I_out, share_dict = self.source_calculate(lightning,
                                                                 light["area"], light["wire"], light["position"], nodes,
                                                                 branches, constants, share_dict)
                sources = self.add_lump(U_out, I_out)
                return sources


        else:
            U_out = pd.DataFrame()
            I_out = pd.DataFrame()
            sources = self.add_lump(U_out, I_out)
            return sources

    #计算直击雷
    def source_Direct(self, lightning, area, wire, position, nodes, branches, constants, shared_dict):
        start = [list(l[0].values())[0] for l in list(branches.values())]
        end = [list(l[1].values())[0] for l in list(branches.values())]
        branches = list(branches.keys())  # 让branches只存储支路的列表，可节省内存
        pt_start = np.array(start)
        pt_end = np.array(end)

        U_out = pd.DataFrame()
        I_out = pd.DataFrame()
        closet_node = self.closest_node(area, wire, position)
        for i in range(len(lightning.strokes)):
            new_U = InducedVoltage_calculate_direct(branches, lightning, i)
            U_out = pd.concat([U_out, new_U], axis=1, ignore_index=True)
            I_out = pd.concat([I_out,
                               LightningCurrent_calculate_direct(closet_node,nodes, lightning,
                                                          stroke_sequence=i)],
                              axis=1, ignore_index=True)
        return U_out, I_out, shared_dict
    def source_Indirect(self, lightning, nodes, branches, constants, shared_dict):
        start = [list(l[0].values())[0] for l in list(branches.values())]
        end = [list(l[1].values())[0] for l in list(branches.values())]
        branches = list(branches.keys())  # 让branches只存储支路的列表，可节省内存
        pt_start = np.array(start)
        pt_end = np.array(end)

        U_out = pd.DataFrame()
        I_out = pd.DataFrame()
        for i in range(len(lightning.strokes)):

            Er_lossy = 0
            Ez_lossy = 0
            epr = self.epr
            sig = self.sig
            GPU = self.GPU_calculation
            # if (erg, sigma_g, 0) in shared_dict:
            Ez_T, Er_T = ElectricField_calculate(pt_start, pt_end, lightning.strokes[i],
                                                 lightning.channel,
                                                 constants.ep0, constants.vc, GPU)  # 计算电场
            if self.gnd_mode == 0:
                Er_lossy = Er_T
                Ez_lossy = Ez_T
                print("------------existing ---------------")
            else:
                H_p = H_MagneticField_calculate(pt_start, pt_end, lightning.strokes[i],
                                                lightning.channel,
                                                constants.ep0, constants.vc, GPU)  # 计算磁场

                # 计算有损地面的电场
                Er_lossy = ElectricField_above_lossy(-H_p, Er_T, constants, shared_dict, self.dt, epr, sig, sigma0=None)
                Ez_lossy = Ez_T
            new_U = InducedVoltage_calculate_indirect(pt_start, pt_end, branches, lightning,
                                                      stroke_sequence=i, Er_lossy=Er_lossy, Ez_lossy=Ez_lossy)
            U_out = pd.concat([U_out, new_U], axis=1, ignore_index=True)
            I_out = pd.concat(
                [I_out,
                 LightningCurrent_calculate_indirect(nodes, lightning, stroke_sequence=i)],
                axis=1, ignore_index=True)
        return U_out, I_out, shared_dict

    # 输出U/I
    def source_calculate(self, lightning, area, wire, position, nodes, branches,constants, shared_dict):

        start = [list(l[0].values())[0] for l in list(branches.values())]
        end = [list(l[1].values())[0] for l in list(branches.values())]
        branches = list(branches.keys())  # 让branches只存储支路的列表，可节省内存
        pt_start = np.array(start)
        pt_end = np.array(end)

        U_out = pd.DataFrame()
        I_out = pd.DataFrame()
        if lightning.type == "Indirect":
            closet_node = self.closest_node( area, wire, position)
            lightning.closet_node = closet_node
            for i in range(len(lightning.strokes)):

                Er_lossy = 0
                Ez_lossy = 0
                epr = self.epr
                sig = self.sig
                GPU = self.GPU_calculation
                #if (erg, sigma_g, 0) in shared_dict:
                Ez_T, Er_T = ElectricField_calculate(pt_start, pt_end, lightning.strokes[i],
                                                     lightning.channel,
                                                     constants.ep0, constants.vc, GPU)  # 计算电场
                if self.gnd_mode ==0:
                    Er_lossy = Er_T
                    Ez_lossy = Ez_T
                    print("------------existing ---------------")
                else:
                    H_p = H_MagneticField_calculate(pt_start, pt_end, lightning.strokes[i],
                                                    lightning.channel,
                                                    constants.ep0, constants.vc, GPU)  # 计算磁场

                    # 计算有损地面的电场
                    Er_lossy = ElectricField_above_lossy(-H_p, Er_T, constants, shared_dict,self.dt,epr,sig, sigma0=None)
                    Ez_lossy = Ez_T
                new_U = InducedVoltage_calculate_indirect(pt_start, pt_end, branches, lightning,
                                                          stroke_sequence=i, Er_lossy=Er_lossy, Ez_lossy=Ez_lossy)
                U_out = pd.concat([U_out, new_U], axis=1, ignore_index=True)
                I_out = pd.concat(
                    [I_out,
                     LightningCurrent_calculate_indirect(nodes, lightning, stroke_sequence=i)],
                    axis=1, ignore_index=True)
        elif lightning.type == "Direct":
            closet_node = self.closest_node(area, wire, position)
            lightning.closet_node = closet_node
            for i in range(len(lightning.strokes)):
                new_U = InducedVoltage_calculate_direct(branches, lightning, i)
                U_out = pd.concat([U_out, new_U], axis=1, ignore_index=True)
                I_out = pd.concat([I_out,
                     LightningCurrent_calculate_direct(closet_node, nodes, lightning, stroke_sequence=i)],
                    axis=1, ignore_index=True)
        else:
            print("Lightning type is not correct, please input correct type: Indirect, Direct")
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

            arrestor_bran = np.array(tower.devices.arrestors_bran)
            arrestor_node1 = np.array(tower.devices.arrestors_node1)
            arrestor_node2 = np.array(tower.devices.arrestors_node2)

            matrix_dict = {'incidence_matrix_A': tower.incidence_matrix_A,
                           'incidence_matrix_B': tower.incidence_matrix_B,
                           'resistance_matrix': tower.resistance_matrix, 'inductance_matrix': tower.inductance_matrix,
                           'capacitance_matrix': tower.capacitance_matrix,
                           'conductance_matrix': tower.conductance_matrix, 'A': A,
                           'B': B, 'phi': phi, 'SDEM': nonlinear_dict['SDEM'], 'VCS': nonlinear_dict['VCS'],
                           'TCS': nonlinear_dict['TCS'], 'NLR': nonlinear_dict['NLR'], 'vf_bran': vf_bran,
                           "arrestor_bran": arrestor_bran, "arrestor_node1": arrestor_node1,
                           "arrestor_node2": arrestor_node2}
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
            temp.append(SDEM.node1[0]) if SDEM.node1[0] != 'ref' else temp.append(-1)
            temp.append(SDEM.node2[0]) if SDEM.node2[0] != 'ref' else temp.append(-1)
            temp.append(SDEM.parameters['v_initial'])
            temp.append(0)
            temp.append(SDEM.parameters['DE_max'])
            temp = np.array(temp)
            SDEM_update_matrix = np.vstack((SDEM_update_matrix, temp))

        VCS_update_matrix = np.empty((0, 4))
        for VCS in lumps.voltage_controled_switchs:
            temp = []
            temp.append(VCS.bran[0])
            temp.append(VCS.node1[0]) if VCS.node1[0] != 'ref' else temp.append(-1)
            temp.append(VCS.node2[0]) if VCS.node2[0] != 'ref' else temp.append(-1)
            temp.append(VCS.parameters['voltage'])
            temp = np.array(temp)
            VCS_update_matrix = np.vstack((VCS_update_matrix, temp))

        TCS_update_matrix = np.empty((0, 5))
        for TCS in lumps.time_controled_switchs:
            temp = []
            temp.append(TCS.bran[0])
            temp.append(min(TCS.parameters['close_time'], TCS.parameters['open_time']))
            temp.append(max(TCS.parameters['close_time'], TCS.parameters['open_time']))
            temp.append(TCS.parameters['resistance']) if TCS.parameters['type_of_data'] != 1 else temp.append(1e-6)
            temp.append(TCS.parameters['resistance']) if TCS.parameters['type_of_data'] == 1 else temp.append(1e-6)
            temp = np.array(temp)
            TCS_update_matrix = np.vstack((TCS_update_matrix, temp))

        NLR_update_matrix = np.empty((0, 4))
        for NLR in lumps.nolinear_resistors:
            temp = []
            temp.append(NLR.bran[0])
            temp.append(NLR.node1[0]) if NLR.node1[0] != 'ref' else temp.append(-1)
            temp.append(NLR.node2[0]) if NLR.node2[0] != 'ref' else temp.append(-1)
            temp.append(NLR.parameters['ri_characteristic']) if NLR.default == 1 else temp.append(lambda x: NLR.parameters['resistance'])
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
            ins_FO =strategy.apply(T,dt,H,sources,self.RL_node)
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
        #tower_list = ["tower_8","tower_9","tower_10","tower_11"]
        #tower_head_node =[self.tower_head_node[tower] for tower in tower_list if tower in self.tower_head_node]

        # for tower in self.towers:
        #     #tower_head_node.append(tower.info.Pole_Head_Node)
        #     for ins in tower.devices.insulators:
        #         for switch in ins.switch_disruptive_effect_models:
        #             ins_FO[switch.name] = {'FO': 0,"Tower":tower.name,"Wire":switch.bran}
        # ins_FO["FO"] = False
        if not self.switch_disruptive_effect_models and not self.voltage_controled_switchs and not self.time_controled_switchs and not self.nolinear_resistors:
            strategy = Strategy.MC_Linear()
            ins_FO =strategy.apply(T,dt,H,sources,self.RL_node)
        else:
            strategy = Strategy.INS_NonLinear()
            ins_FO = strategy.apply(T,dt,H,sources)
        return ins_FO
    def calculate(self,T,dt,H,sources):

        # SAF = []

        # ins_FO["FO"] = False
        if not self.switch_disruptive_effect_models and not self.voltage_controled_switchs and not self.time_controled_switchs and not self.nolinear_resistors:
            strategy = Strategy.Linear()
            solution =strategy.apply(T,dt,H,sources)
        else:
            strategy = Strategy.NonLinear()
            solution,switch_disruptive_effect_models = strategy.apply(T,dt,H,sources)
            swh = {}
            for tower in self.towers:
                for ins in tower.devices.insulators:
                    for switch in ins.switch_disruptive_effect_models:
                       # ins_FO[switch.name] = {'FO': 0, "Tower": tower.name, "Wire": switch.bran}
                       swh[switch.name] = {"Tower": tower.name, "Wire": switch.bran}
            for switch in switch_disruptive_effect_models:
                if switch in swh.keys():
                    self.FO[switch]= swh[switch]
                    #ins_FO["FO"] = True
        return solution

    def calculate_of_hybrid_mode(self, line_matrix, tower_matrix, sources, Nt, dt, GPU):
        """
        solution_type_id: 'hybrid', 'nonlinear', 'variable_frequency', 'variant_step'
        """
        GPU=0
        solution_type_id = self.update_solution_type()
        match solution_type_id:
            case '1000':
                strategy = Strategy.hybrid_linear()
                # tower_list = ["tower_8", "tower_9", "tower_10", "tower_11"]
                # tower_head_node = [self.tower_head_node[tower] for tower in tower_list if tower in self.tower_head_node]
                return strategy.apply(line_matrix, tower_matrix, sources, Nt, dt, GPU,self.RL_node)
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
            self.sig = load_dict['Global']['ground']['sig']
            self.epr = load_dict['Global']['ground']['epr']
            self.gnd_mode = load_dict['Global']['ground']['gnd_mode']

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

        branches, nodes = self.calculate_branches(self.max_length)

        # 3. 初始化源，计算结果
        share_dict = {}
        constants = Constant()
        self.sources = self.source_initial(load_dict, nodes,branches,constants,share_dict)

        self.H = {"Line": line_matrix,"Tower": tower_matrix}
        result_tower, other = self.calculate_of_hybrid_mode(line_matrix, tower_matrix, self.sources, self.Nt, self.dt, self.GPU_calculation)
        #记录哪个塔损坏次数更多
        fo_tower = [tower for fo_bran in other['SDEM']
                         for tower, tower_bran in self.Tower2ins_bran.items() if fo_bran in tower_bran ]
        self.weak_point = {t: self.weak_point.get(t) + (1 if t in fo_tower else 0) for t in list(self.weak_point.keys())}

        broken_arrestor_list = [self.arrestor_bran2Tower[bran][1] for bran in other["NLR"] if bran in self.arrestor_bran2Tower.keys()]
        self.broken = list(set(broken_arrestor_list))
        measure_result = self.run_measure(result_tower)
        result_tower_pa = pa.Table.from_pandas(result_tower.T)
        sources_pa = pa.Table.from_pandas(self.sources.T)
        target_tower = self.lightning.closet_node.target_tower
        swhs_node = {}
        FO_node = {}
        for tower in self.towers:
            if tower.name in target_tower:
                swhs_node.update({tower.name + "_" + swh.name: [swh.name, swh.node1[0], swh.node2[0],swh.bran[0]] for ins in
                     tower.devices.insulators for swh in
                     ins.switch_disruptive_effect_models})
            FO_node.update({tower.name + "_" + swh.name: [swh.name, swh.node1[0], swh.node2[0],swh.bran[0]] for ins in
                 tower.devices.insulators for swh in ins.switch_disruptive_effect_models if swh.bran[0] in other["SDEM"]})
        #添加离source近的塔的结果

        targeted_result = pd.DataFrame(
            {v[0] : abs(result_tower.loc[v[1]] - result_tower.loc[v[2]])
             for k, v in swhs_node.items()}).T
        FO_result = pd.DataFrame(
            {v[0] : abs(result_tower.loc[v[1]] - result_tower.loc[v[2]])
             for k, v in FO_node.items()}).T

        return targeted_result,result_tower_pa,sources_pa,FO_result
        # result_tower.to_csv("Data/Output/result_tower_output.csv")
        # result_v_ohl.to_csv("Data/Output/result_v_ohl_output.csv")
        # result_i_ohl.to_csv("Data/Output/result_i_ohl_output.csv")
    def run_base(self, load_dict, *basestrategy):
        # 0. 手动预设值
        self.global_set(load_dict)
        hybrid = load_dict["Global"]["Hybrid_method"]
        if hybrid == 1:
            self.solution_type['hybrid'] = True
            measure_result = self.run_hybrid(load_dict)

        else:
            measure_result = self.run(load_dict)
        return measure_result
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
        sources = self.source_initial(load_dict, nodes,branches,constants,share_dict)
        end = time.time()  # 记录结束时间
        print(f"Source simulation running time: {end - start_time} seconds")  # 打印运行时长
        solution = self.calculate(self.Nt, self.dt, self.H, sources)
        end2 = time.time()  # 记录开始时间
        pd.DataFrame(solution[0]).to_csv("Data/Output/combine_output.csv")
        measure_result = self.run_measure(solution)

        print(f"Calculation running time: {end2 - end} seconds")  # 打印运行时长
        print(solution)
        return measure_result

    # 运行一次的灵敏度分析
    def sensitive_analysis(self, load_dict,path):
        # 检查是否使用 hybrid 模式
        use_hybrid = load_dict.get("Global", {}).get("Hybrid_method", 0) == 1

        if use_hybrid:
            self.solution_type['hybrid'] = True
            # initial_result = self.run_hybrid(load_dict)
            # initial_filename = "Data/Output/initial_hybrid_output.csv"
            # pd.DataFrame(initial_result).to_csv(initial_filename)
            # print(f"Initial hybrid result saved to {initial_filename}")
        # else:
        #     initial_result = self.run_base(load_dict)
        #     initial_filename = "Data/Output/initial_base_output.csv"
        #     pd.DataFrame(initial_result[0]).to_csv(initial_filename)
        #     print(f"Initial base result saved to {initial_filename}")
        #
        # if "Sensitivity_analysis" not in load_dict:
        #     print("No sensitivity analysis parameters provided.")
        #     return initial_result, None

        sa_dict = load_dict["Sensitivity_analysis"]
        mode = sa_dict.get("mode", 1)

        result_before, result_after = run_sensitivity_analysis(self, load_dict, sa_dict, use_hybrid, mode,path)
        return result_before, result_after

    # 运行多次的灵敏度分析
    def sensitive_MC(self,load_dict):
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

        constants = Constant()
        constants.ep0 = 8.85e-12
        index = 0


        if load_dict["MC"]:
            # MC_result = self.MC_generate_flash(load_dict)
            # self.MC_flash = MC_result
            # record_SAF = load_dict["MC"]["Record_SAF"]
            # summary = self.MC_calculate(record_SAF,MC_result, nodes, branches)

            # 定义幅值和对应的概率
            amplitudes = [30, 60, 120]  # 单位：kA
            probabilities = [0.5, 0.35, 0.15]

            parameter = ['8/20us','1/50us']
            probabilities2 = [0.5,0.5]


            MC_list = self.MC_generate_flash(load_dict)
            MC_list = [MC for MC in MC_list if MC[0].type=="Direct"]
            Arrestor_name = [device.name for tower in self.towers for device in  tower.devices.arrestors]

            stroke_node = list(set(self.OHL_node) | set(self.tower_head_node.values()))


            all_stroke_node = {}

            for MC in MC_list:
                # 根据波尾时间调整计算时间
                # self.T = (MC[0].strokes[0].parameters[1] + MC[0].strokes[0].parameters[2] * 2) * 1e-6
                # self.Nt = int(np.ceil(self.T / self.dt))
                # MC[0].strokes[0].duration = (MC[0].strokes[0].parameters[1] + MC[0].strokes[0].parameters[2] * 2) * 1e-6
                # MC[0].strokes[0].Nt = self.Nt
                # MC[0].strokes[0].t_us = np.array(list(range(self.Nt))) * self.dt
                # MC[0].strokes[0].calculate()
                closet_node = self.closest_node(MC[1],MC[2],MC[3])
                if closet_node not in all_stroke_node:
                    all_stroke_node[closet_node] = 0
                    # 累计统计
                all_stroke_node[closet_node] += 1
            # 转换为DataFrame显示结果
            result = pd.DataFrame(list(all_stroke_node.items()),
                                  columns=["Node", "Strike Count"])
            result["Probability"] = result["Strike Count"] / len(MC_list)

            print("雷击统计结果：")
            print(result)


            # 生成所有列名的组合
            column_names = [
                f"{node.name}_{amp}_{param.replace('/', '_')}"  # 将斜杠替换为下划线，便于后续处理
                for node in all_stroke_node.keys()
                for amp in amplitudes
                for param in parameter
            ]
            FO_matrix = pd.DataFrame(0,
                                  index=[tower.name for tower in self.towers],
                                  columns=column_names)
            SAF_matrix = pd.DataFrame(0,
                                     index=Arrestor_name,
                                     columns=column_names)
            # 创建name到对象的映射（因为字典的key是对象）
            node_dict = {node.name: node for node in all_stroke_node.keys()}
            # 定义处理每个列的函数
            def process_col(args):
                i, col, self_copy, node_dict, branches, nodes = args
                # 解析列名
                node_name, selected_amplitudes, selected_parameter = col.split('_', 2)
                selected_parameter = selected_parameter.replace('_', '/')
                selected_amplitudes = int(selected_amplitudes)
                closet_node = node_dict[node_name]

                # 提取时间参数
                numbers = selected_parameter.split("/")[0], selected_parameter.split("/")[1].rstrip("us")
                front_time, tail_time = int(numbers[0]), int(numbers[1])

                # 计算雷击参数
                time = front_time * 1e-6 + 2 * tail_time * 1e-6
                stroke1 = Stroke('Heidler', duration=time, dt=1.0e-8, is_calculated=True, parameter_set=selected_parameter)
                stroke1.parameters[0] = selected_amplitudes * 1000
                self_copy.T = time
                self_copy.Nt = int(np.ceil(self_copy.T / self_copy.dt))
                stroke1.Nt = self_copy.Nt
                stroke1.t_us = np.array(list(range(self_copy.Nt))) * self_copy.dt
                stroke1.calculate()
                strokes = [stroke1]
                channel = Channel(hit_pos=[50, 500, 0])
                lightning = Lightning(id=1, type='Direct', strokes=strokes, channel=channel)

                # 计算电压和电流
                U_out = pd.DataFrame()
                I_out = pd.DataFrame()
                for j in range(len(lightning.strokes)):
                    new_U = InducedVoltage_calculate_direct(branches, lightning, j)
                    U_out = pd.concat([U_out, new_U], axis=1, ignore_index=True)
                    I_out = pd.concat([I_out,
                                    LightningCurrent_calculate_direct(closet_node, nodes, lightning, stroke_sequence=j)],
                                    axis=1, ignore_index=True)

                # 添加源并计算
                sources = self_copy.add_lump(U_out, I_out)
                ins, saf = process_calculate(sources, self_copy, i, calculate_saf=1)

                # 返回结果
                return col, self_copy.FO_Tower, self_copy.broken

            # 主程序部分
            from pathos.multiprocessing import ProcessPool
            from tqdm import tqdm
            # import pandas as pd
            # import numpy as np

            # 假设 self 是包含必要属性的实例，column_names、FO_matrix、SAF_matrix 已定义
            # 示例：FO_matrix = pd.DataFrame(...), SAF_matrix = pd.DataFrame(...)

            # 创建进程池
            num_cores = os.cpu_count()
            pool = ProcessPool(nodes=num_cores)  # 使用 4 个进程，可根据 CPU 核心数调整

            # 准备参数列表
            args_list = [(i, col, self, node_dict, branches, nodes) for i, col in enumerate(column_names)]

            # 并行执行并获取结果迭代器
            results = pool.imap(process_col, args_list)

            # 收集结果并更新矩阵，同时显示进度
            for col, FO_rows, SAF_rows in tqdm(results, total=len(column_names)):
                for tower_name in FO_rows:
                    FO_matrix.loc[tower_name, col] = 1
                for broke_arrestor in SAF_rows:
                    SAF_matrix.loc[broke_arrestor, col] = 1

            # 关闭进程池
            pool.close()
            pool.join()

            # 输出结果
            print(FO_matrix)
            return FO_matrix, SAF_matrix


    def closest_node(self,p1,p2,position):
        area = p1.split("_")[0]
        # 1. 找到用户指定点所在的wire
        selected_wire = None
        target_tower = None
        nodes = set()
        if area == "tower":
            selected_tower = [tower for tower in self.towers if tower.info.name == p1]
            target_tower = selected_tower[0].name
            selected_wire = [wire for wire in list(selected_tower[0].wires.get_all_wires().values()) if
                             wire.name.split("_")[0] == p2.split("_")[0]]

        elif area == "OHL":
            selected_ohl = [ohl for ohl in self.OHLs if ohl.name == p1]
            target_tower = [selected_ohl[0].info.HeadTower,selected_ohl[0].info.TailTower]
            selected_wire = [wire for wire in list(selected_ohl[0].wires.get_all_wires().values()) if
                             wire.name.split("_")[0] == p2.split("_")[0]]
            #all_node = [wire for wire in list(selected_ohl[0].wires.get_all_nodes())]
            #nodes = set(all_node)
        elif area == "cable":
            selected_cable = [cable for cable in self.cables if cable.name == p1]
            selected_wire = [wire for wire in list(selected_cable[0].wires.get_all_wires().values()) if
                             wire.name.split("_")[0] == p2]

        elif area == "Ground":
            dis = {tower.info.name:distance(tower.info.position,position) for tower in self.towers}
            close_tower = [k for k, v in sorted(dis.items(), key=lambda x: x[1])][:int(len(dis) * 0.1)]
            node = Node(None,0,0,0)
            node.target_tower = close_tower
            return node

        # 2. 找到用户指定点距离该wire上最近的node

        for wire in selected_wire:
            nodes.add(wire.start_node)
            nodes.add(wire.end_node)

        closest_point = None
        min_distance = float('inf')

        for node in nodes:
            node_pos = [node.x, node.y, node.z]
            dist = distance(node_pos, position)
            if dist < min_distance:
                min_distance = dist
                closest_point = node
        closest_point.target_tower = target_tower
        return closest_point

    def run_measure(self,solution):
        # lumpname/branname: label(bran0/lump1),probe,branname,n1,n2,(towername)
        if self.measurement:
            return Strategy.Measurement().apply(measurement=self.measurement, solution=solution,dt=self.dt)
    def run_MC(self,load_dict):
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
            MC_result = self.MC_generate_flash(load_dict)
            self.MC_flash = MC_result
            record_SAF = load_dict["MC"]["Record_SAF"]
            summary = self.MC_calculate(record_SAF,MC_result, nodes, branches)
            return summary
        print("MC is not exist")

        # if load_dict["MC"]:
        #     print("running Monte Carlo to generate lightnings")
        #     df27,parameterst,stroke_result,PoleXY = run_MC(self,load_dict,Distance)
        #
        #     MC_result_list = []
        #     summary = {"nonFO_indirect":0,"nonFO_direct":0,"FO_indirect":0,"FO_direct":0,"Huri":0,"RunTime":0,'FOR_direct':0,'FOR_indirect':0}
        #     shared_dict = None
        #     index = 0
        #     MC_result = []
        #     #对每个flash
        #     for i in df27.groupby("flash"):
        #         stroke_list = []
        #         #初始化每个stroke
        #         for j in range(i[1].shape[0]):
        #             stroke_type = "Heidler"
        #             duration = self.T
        #             dt = self.dt
        #             stroke = Stroke(stroke_type, duration=duration, dt=dt, is_calculated=True, parameter_set=None,
        #                         parameters=[parameterst[index].tolist()[2]*1e3,parameterst[index].tolist()[3],
        #                         parameterst[index].tolist()[5],parameterst[index].tolist()[4]])
        #             stroke.calculate()
        #             index += 1
        #             stroke_list.append(stroke)
        #         flash_type = stroke_result[0][index - 1]
        #         area = int(stroke_result[1][index - 1])
        #         area_id = 0 if math.isnan(stroke_result[2][index - 1]) else str(int(stroke_result[2][index - 1]))
        #         cir_id = 0 if math.isnan(float(stroke_result[3][index - 1])) else int(
        #             float(stroke_result[3][index - 1]))
        #         phase_id = 0 if math.isnan(float(stroke_result[4][index - 1])) else int(
        #             float(stroke_result[4][index - 1]))
        #         position_xy = stroke_result[8][index - 1]
        #         position = None
        #         wire = None
        #         if phase_id == 0:
        #             phase_lgt = 'S'
        #         elif phase_id == 1:
        #             phase_lgt = 'A'
        #         elif phase_id == 2:
        #             phase_lgt = 'B'
        #         else:
        #             phase_lgt = 'C'
        #
        #         if area == 0:
        #             area = "Ground"
        #             position = position_xy.append(0)
        #         elif area == 1:
        #             area = "tower_" + area_id
        #             for tower in load_dict["Tower"]:
        #                 if tower["Info"]["name"] == area:
        #                     z = tower["Info"]["pole_height"]
        #                     position = position_xy.append(z)
        #                     for w in tower["Wire"]:
        #                         if w["pos_1"][2] == z or w["pos_2"][2] == z:
        #                             wire = w["bran"]
        #                     if wire is None:
        #                         wire = tower["Wire"][0]
        #         elif area == 2:
        #             area = "OHL_" + area_id
        #             for ohl in load_dict["OHL"]:
        #                 if ohl["Info"]["name"] == area:
        #                     for w in ohl["Wire"]:
        #                         cir_id_ohl = w['cir_id']
        #                         phase_id_ohl = w['phase']
        #                         if cir_id_ohl == cir_id:
        #                             z = w["node1_pos"][2]
        #                             position = position_xy.append(z)
        #                             if w['type'] == 'SW':
        #                                 wire = 'Y' + str(cir_id) + 'S'
        #                             elif w['type'] == 'CIRO':
        #                                 wire = 'Y' + str(cir_id) + w['phase']
        #
        #         lightning = Lightning(id=1, type=flash_type, strokes=stroke_list, channel=Channel(position_xy))
        #         MC_result.append((lightning, area, wire, position_xy))
        #     #MC_result_list.append(MC_result)
        #     record_SAF = load_dict["MC"]["Record_SAF"]
        #     start_time = time.time()
        #       # 创建一个共享字典
        #     if record_SAF==1:
        #         FO_direct,FO_indirect,SAF_direct,SAF_indirect  = self.Huri_method_SAF(MC_result, nodes, branches,shared_dict)
        #     else:
        #         FO_direct,FO_indirect,huri = self.Huri_method_INS(MC_result, nodes, branches,shared_dict)
        #         end_time = time.time()  # 记录结束时间
        #         duration = end_time - start_time  # 计算运行时长
        #         true_count_direct = len(list(filter(lambda x: x, FO_direct)))
        #         false_count_direct = len(FO_direct) -true_count_direct
        #         true_count_indirect = len(list(filter(lambda x: x, FO_indirect)))
        #         false_count_indirect = len(FO_indirect) - true_count_indirect
        #         summary["FO_direct"] = true_count_direct
        #         summary["FO_indirect"] = true_count_indirect
        #         summary["nonFO_direct"] = false_count_direct
        #         summary["nonFO_indirect"] = false_count_indirect
        #         summary["FOR_direct"] = 100*true_count_direct/len(FO_direct)*2 if len(FO_direct)*2>0 else 0
        #         summary["FOR_indirect"] = 100 * true_count_indirect / len(FO_indirect) * 2
        #         summary["Huri"] = huri
        #         summary["RunTime"] = duration
        #         print("FO_direct: ", summary["FO_direct"])
        #         print("FO_indirect: ", summary["FO_indirect"])
        #         print("nonFO_direct: ", summary["nonFO_direct"])
        #         print("nonFO_indirect: ", summary["nonFO_indirect"])
        #         print("Huri: ",summary["Huri"])
        #         print("Running time: ",summary["RunTime"])  # 打印运行时长
        #         #df = pd.DataFrame(summary)
        #         # 保存DataFrame到CSV文件
        #
        #
        #         return summary

    def MC_generate_flash(self,load_dict):
        print("running Monte Carlo to generate lightnings")
        df27,parameterst,stroke_result,PoleXY = run_MC(self,load_dict,self.Distance)


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
                            parameters=[parameterst[index].tolist()[2]*1e3,parameterst[index].tolist()[3],
                            parameterst[index].tolist()[5],parameterst[index].tolist()[4]])
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
                                if w['phase'] == phase_lgt:
                                    wire = 'Y' + str(cir_id) + str(phase_lgt)
                                # if w['type'] == 'SW':
                                #     wire = 'Y' + str(cir_id) + 'S'
                                # elif w['type'] == 'CIRO':
                                #     wire = 'Y' + str(cir_id) + w['phase']

            # if flash_type=="Indirect":
            #     continue
            lightning = Lightning(id=1, type=flash_type, strokes=stroke_list, channel=Channel(position_xy))
            MC_result.append((lightning, area, wire, position_xy))
        return MC_result
        #MC_result_list.append(MC_result)
    # def MC_calculate(self,record_SAF,MC_result, nodes, branches):
    #     shared_dict = None
    #     MC_result_list = []
    #     summary = {"nonFO_indirect": 0, "nonFO_direct": 0, "FO_indirect": 0, "FO_direct": 0, "Huri": 0, "RunTime": 0,
    #                'FOR_direct': 0, 'FOR_indirect': 0,"nonSAF_indirect": 0, "nonSAF_direct": 0,
    #                "SAF_indirect": 0, "SAF_direct": 0, "Huri_SAF": 0}
    #
    #     start_time = time.time()
    #     icurr = []
    #     dataset_ins = []
    #     dataset_saf = []
    #     FO_direct = []
    #     FO_indirect = []
    #     SAF_direct = []
    #     SAF_indirect = []
    #     dataset = []
    #
    #     R = 100
    #     constants = Constant()
    #     constants.ep0 = 8.85e-12
    #     index = 0
    #     calculate_saf = 1
    #     huri = 0
    #       # 创建一个共享字典
    #     if record_SAF==1:
    #         for MC in MC_result:
    #             FO_direct,FO_indirect,SAF_direct,SAF_indirect,huri,index  = self.Huri_method_SAF(MC, nodes, branches,shared_dict,
    #                                                                                    dataset_ins,dataset_saf,FO_direct,FO_indirect,
    #                                                                                    SAF_direct,SAF_indirect,constants,index,huri)
    #         SAF_true_count_direct = len(list(filter(lambda x: x, SAF_direct)))
    #         SAF_false_count_direct = len(FO_direct) - SAF_true_count_direct
    #         SAF_true_count_indirect = len(list(filter(lambda x: x, SAF_indirect)))
    #         SAF_false_count_indirect = len(FO_indirect) - SAF_true_count_indirect
    #         summary["SAF_direct"] = SAF_true_count_direct
    #         summary["SAF_indirect"] = SAF_true_count_indirect
    #         summary["nonSAF_direct"] = SAF_false_count_direct
    #         summary["nonSAF_indirect"] = SAF_false_count_indirect
    #
    #     else:
    #         for MC in MC_result:
    #             FO_direct,FO_indirect,huri,index  = self.Huri_method_INS(MC, nodes, branches,shared_dict,dataset,
    #                                                               FO_indirect,FO_direct,constants,index,huri)
    #     end_time = time.time()  # 记录结束时间
    #     duration = end_time - start_time  # 计算运行时长
    #     true_count_direct = len(list(filter(lambda x: x, FO_direct)))
    #     false_count_direct = len(FO_direct) -true_count_direct
    #     true_count_indirect = len(list(filter(lambda x: x, FO_indirect)))
    #     false_count_indirect = len(FO_indirect) - true_count_indirect
    #     summary["FO_direct"] = true_count_direct
    #     summary["FO_indirect"] = true_count_indirect
    #     summary["nonFO_direct"] = false_count_direct
    #     summary["nonFO_indirect"] = false_count_indirect
    #     summary["FOR_direct"] = 100*true_count_direct/len(FO_direct)*2 if len(FO_direct)*2>0 else 0
    #     summary["FOR_indirect"] = 100 * true_count_indirect / len(FO_indirect) * 2 if len(FO_indirect) !=0 else 0
    #     summary["Huri"] = huri
    #     summary["RunTime"] = duration
    #     print("FO_direct: ", summary["FO_direct"])
    #     print("FO_indirect: ", summary["FO_indirect"])
    #     print("nonFO_direct: ", summary["nonFO_direct"])
    #     print("nonFO_indirect: ", summary["nonFO_indirect"])
    #     print("Huri: ",summary["Huri"])
    #     print("SAF_direct: ", summary["SAF_direct"])
    #     print("SAF_indirect: ", summary["SAF_indirect"])
    #     print("nonSAF_direct: ", summary["nonSAF_direct"])
    #     print("nonSAF_indirect: ", summary["nonSAF_indirect"])
    #     print("Huri: ",summary["Huri"])
    #     print("Running time: ",summary["RunTime"])  # 打印运行时长
    #         #df = pd.DataFrame(summary)
    #         # 保存DataFrame到CSV文件
    #
    #     return summary

    def MC_calculate(self, record_SAF, MC_result, nodes, branches):
        from pathos.multiprocessing import ProcessPool
        from tqdm import tqdm
        import os
        import time

        # 初始化summary字典
        summary = {
            "nonFO_indirect": 0, "nonFO_direct": 0, "FO_indirect": 0, "FO_direct": 0,
            "Huri": 0, "RunTime": 0, "FOR_direct": 0, "FOR_indirect": 0,
            "nonSAF_indirect": 0, "nonSAF_direct": 0, "SAF_indirect": 0, "SAF_direct": 0,
            "Huri_SAF": 0
        }

        # 常量和初始变量
        constants = Constant()
        constants.ep0 = 8.85e-12

        # 单MC处理函数
        def process_single_MC(args):
            MC, self_copy, nodes_copy, branches_copy, record_SAF, constants = args
            FO_direct = []
            FO_indirect = []
            SAF_direct = []
            SAF_indirect = []
            dataset_ins = []
            dataset_saf = []
            dataset = []
            huri = 0
            index = 0

            if record_SAF == 1:
                FO_direct_out, FO_indirect_out, SAF_direct_out, SAF_indirect_out, huri, index = self_copy.Huri_method_SAF(
                    MC, nodes_copy, branches_copy, None, dataset_ins, dataset_saf,
                    FO_direct, FO_indirect, SAF_direct, SAF_indirect, constants, index, huri
                )
                return {
                    "FO_direct": FO_direct_out, "FO_indirect": FO_indirect_out,
                    "SAF_direct": SAF_direct_out, "SAF_indirect": SAF_indirect_out,
                    "huri": huri
                }
            else:
                FO_direct_out, FO_indirect_out, huri, index = self_copy.Huri_method_INS(
                    MC, nodes_copy, branches_copy, None, dataset, FO_indirect, FO_direct,
                    constants, index, huri
                )
                return {
                    "FO_direct": FO_direct_out, "FO_indirect": FO_indirect_out,
                    "SAF_direct": [], "SAF_indirect": [], "huri": huri
                }

        # 主程序
        start_time = time.time()

        # 创建进程池
        num_cores = min(os.cpu_count(), 4)  # 限制最大核心数，可调整
        pool = ProcessPool(nodes=num_cores)

        # 准备参数列表，每个MC使用self的深拷贝
        args_list = [(MC, copy.deepcopy(self), nodes, branches, record_SAF, constants)
                     for MC in MC_result]

        # 并行执行
        results = pool.imap(process_single_MC, args_list)

        # 收集结果
        FO_direct_all = []
        FO_indirect_all = []
        SAF_direct_all = []
        SAF_indirect_all = []
        huri_total = 0

        for result in tqdm(results, total=len(MC_result), desc="Processing MCs"):
            FO_direct_all.extend(result["FO_direct"])
            FO_indirect_all.extend(result["FO_indirect"])
            SAF_direct_all.extend(result["SAF_direct"])
            SAF_indirect_all.extend(result["SAF_indirect"])
            huri_total += result["huri"]

        # 关闭进程池
        pool.close()
        pool.join()

        # 计算统计数据
        end_time = time.time()
        duration = end_time - start_time

        true_count_direct = len(list(filter(lambda x: x, FO_direct_all)))
        false_count_direct = len(FO_direct_all) - true_count_direct
        true_count_indirect = len(list(filter(lambda x: x, FO_indirect_all)))
        false_count_indirect = len(FO_indirect_all) - true_count_indirect

        summary["FO_direct"] = true_count_direct
        summary["FO_indirect"] = true_count_indirect
        summary["nonFO_direct"] = false_count_direct
        summary["nonFO_indirect"] = false_count_indirect
        summary["FOR_direct"] = 100 * true_count_direct / len(FO_direct_all) * 2 if len(FO_direct_all) * 2 > 0 else 0
        summary["FOR_indirect"] = 100 * true_count_indirect / len(FO_indirect_all) * 2 if len(
            FO_indirect_all) != 0 else 0
        summary["Huri"] = huri_total
        summary["RunTime"] = duration

        if record_SAF == 1:
            SAF_true_count_direct = len(list(filter(lambda x: x, SAF_direct_all)))
            SAF_false_count_direct = len(FO_direct_all) - SAF_true_count_direct
            SAF_true_count_indirect = len(list(filter(lambda x: x, SAF_indirect_all)))
            SAF_false_count_indirect = len(FO_indirect_all) - SAF_true_count_indirect

            summary["SAF_direct"] = SAF_true_count_direct
            summary["SAF_indirect"] = SAF_true_count_indirect
            summary["nonSAF_direct"] = SAF_false_count_direct
            summary["nonSAF_indirect"] = SAF_false_count_indirect
            summary["Huri_SAF"] = huri_total

        # 打印结果
        print("FO_direct: ", summary["FO_direct"])
        print("FO_indirect: ", summary["FO_indirect"])
        print("nonFO_direct: ", summary["nonFO_direct"])
        print("nonFO_indirect: ", summary["nonFO_indirect"])
        print("FOR_direct: ", summary["FOR_direct"])
        print("FOR_indirect: ", summary["FOR_indirect"])
        print("Huri: ", summary["Huri"])
        if record_SAF == 1:
            print("SAF_direct: ", summary["SAF_direct"])
            print("SAF_indirect: ", summary["SAF_indirect"])
            print("nonSAF_direct: ", summary["nonSAF_direct"])
            print("nonSAF_indirect: ", summary["nonSAF_indirect"])
            print("Huri_SAF: ", summary["Huri_SAF"])
        print("Running time: ", summary["RunTime"])

        return summary

    def Pre_run_MC(self, load_dict):
        ### 用大矩阵----------
        # 0. 手动预设值
        self.global_set(load_dict)
        #self.dt = 1e-8
        #self.Nt = 1000
        #self.T = 2e-5
        self.Nt = int(np.ceil(self.T / self.dt))
        # 1. 初始化电网，根据电网信息计算源
        self.initialize_network(load_dict,self.VF)
        self.combine_parameter_matrix()
        branches,nodes = self.calculate_branches(self.max_length)
        if 'ref' in nodes:
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
            #summary = {"FOR": [],"FO": [], "Huri": [], "RunTime": []}
            summary = {"FOR": [], "FO": []}
            FO_num = 0
            shared_dict = None
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
                # 尋找最大包圍綫
                FO, huri,dataset = self.Find_Dmax(MC_result, nodes, branches, shared_dict,dataset)
                end_time = time.time()  # 记录结束时间
                duration = end_time - start_time  # 计算运行时长
                true_count = len(list(filter(lambda x: x, FO))) #閃絡次數，FO中有True和False
                FO_num = FO_num + true_count #累計所有的閃絡次數
                FOR = 100*FO_num/df27_list[-1].iloc[-1, 0]*2

                summary["FO"].append(FO_num)
                summary["FOR"].append(FOR)
                #summary["Huri"].append(huri)
                #summary["RunTime"].append(duration)

            self.save_result(summary,path='Data/input/case2_linear/')

            positions = [i+1 for i, (a, b) in enumerate(zip(summary["FO"], summary["FO"][1:])) if b - a < 1] #第幾個區域開始閃絡值不變
            Distance_max = len(df27_list)*100
            if positions[0]:
                Distance_max = positions[0] * 100
            print("maximum distance is: ", Distance_max)
            print("total flash count: ", df27_list[-1].iloc[-1, 0])
            self.Distance = Distance_max
            return Distance_max
    def save_result(self,summary,path):
        name = 'Heidler_7500_loss_IP200_epr4_dt3000'
        #name = 'Heidler_3750_perfectV2'
        print("FO: ", summary["FO"])
        print("FOR: ", summary["FOR"])
        #("Huri: ", summary["Huri"])
        #print("Running time: ", summary["RunTime"])  # 打印运行时长
        for key, values in summary.items():
            plt.figure()  # 创建一个新的图形
            plt.plot(values)
            plt.title(f'Plot for {key}_{name}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            # 保存每个图表为图片文件
            plt.savefig(f'{path}{key}_{name}.png')
            plt.close()  # 关闭图形，避免内存泄漏

            # 将数据保存到CSV文件
            # 创建一个DataFrame，其中包含索引和对应的值
            df = pd.DataFrame({'Index': range(len(values)), 'Value': values})
            # 保存DataFrame到CSV文件
            df.to_csv(f'Data/input/case4_linear/{key}_{name}.csv', index=False)
        df = pd.DataFrame(summary).T
        df.index = ['FOR_'+name,'FO_'+name]  # 设置索引为实验名称
        # 将DataFrame写入CSV文件，使用追加模式，不包含列名
        df.to_csv('experiments.csv', mode='a', header=False)
        # 所有图片保存后，显示它们
        for key, values in summary.items():
            plt.figure()
            plt.plot(values)
            plt.title(f'Plot for {key}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.show()
            # self.run_multiprocessed(MC_result, nodes, branches,shared_dict,PoleXY)
    def show_result(self,dict):
        for key, values in dict.items():
            plt.figure()  # 创建一个新的图形
            plt.plot(values)
            plt.title(f'Plot for {key}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            # 保存每个图表为图片文件
            #plt.savefig(f'{path}{key}.png')
            plt.close()  # 关闭图形，避免内存泄漏
        for key, values in dict.items():
            plt.figure()
            plt.plot(values)
            plt.title(f'Plot for {key}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.show()
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

    # def Huri_method_SAF(self,MC_result, nodes, branches,shared_dict):
    #     icurr = []
    #     dataset_ins = []
    #     dataset_saf = []
    #     FO_direct = []
    #     FO_indirect = []
    #     SAF_direct = []
    #     SAF_indirect = []
    #     R = 100
    #     constants = Constant()
    #     constants.ep0 = 8.85e-12
    #     for MC in MC_result:
    #         if len(dataset_saf) > 0 and len(dataset_ins):
    #             flash_position = MC[0].channel.hit_pos[:2]
    #             for i, stroke in enumerate(MC[0].strokes):
    #                 icur = np.append(stroke.parameters, flash_position)
    #                 result_ins = Huri_Method(R, dataset_saf, icur)
    #                 result_saf = Huri_Method_SAF(R, dataset_saf, icur)
    #                 if result_saf == 1 and result_ins == 1:
    #                     SAF_indirect.append(False)
    #                     FO_indirect.append(False)
    #                     print("using Huri skip calculation")
    #                 else:
    #                     U_out, I_out, shared_dict = self.source_calculate(MC[0], MC[1], MC[2], MC[3], nodes, branches,
    #                                                                       constants, shared_dict)
    #                     sources = self.add_lump(U_out, I_out)
    #                     solution, ins, saf = self.SAF_calculate(self.Nt, self.dt, self.H, sources)
    #                     SAF_direct,SAF_indirect = self.Huri_saf(MC, nodes, branches, constants, shared_dict, SAF_direct,SAF_indirect, dataset_saf, ins, saf, R)
    #                     FO_direct,FO_indirect = self.Huri_ins(MC, nodes, branches, constants, shared_dict,FO_direct,FO_indirect,dataset_ins, ins, saf,R)
    #         else:
    #             U_out, I_out, shared_dict = self.source_calculate(MC[0], MC[1], MC[2], MC[3], nodes, branches,
    #                                                               constants, shared_dict)
    #             sources = self.add_lump(U_out, I_out)
    #             solution, ins, saf = self.SAF_calculate(self.Nt, self.dt, self.H, sources)
    #             SAF_direct,SAF_indirect = self.Huri_saf(MC, nodes, branches, constants, shared_dict, SAF_direct,SAF_indirect, dataset_saf, ins, saf, R)
    #             FO_direct,FO_indirect = self.Huri_ins(MC, nodes, branches, constants, shared_dict,FO_direct,FO_indirect,dataset_ins, ins, saf,R)
    #     return FO_direct,FO_indirect,SAF_direct,SAF_indirect

    def Huri_ins(self,MC, nodes, branches, constants, shared_dict,FO_direct,FO_indirect,dataset_ins, ins, saf,R):
        # 1. 直接雷
        if MC[0].type == "Direct":
            U_out, I_out, shared_dict = self.source_Direct(MC[0], MC[1], MC[2], MC[3], nodes, branches, constants, shared_dict)
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

    def Huri_method_SAF(self,MC, nodes, branches,shared_dict,dataset_ins,dataset_saf,FO_direct,FO_indirect,SAF_direct,
                        SAF_indirect,constants,index,huri):
        R = 100
        calculate_saf = 1



        if MC[0].type == "Direct":
            U_out, I_out, shared_dict = self.source_Direct(MC[0], MC[1], MC[2], MC[3], nodes, branches,
                                                                  constants, shared_dict)
            sources = self.add_lump(U_out, I_out)
            #solution,ins = process_calculate(MC, nodes, branches, self,index,shared_dict,calculate_ins)
            ins,saf = process_calculate(sources, self, index, calculate_saf)
            index = index+1
            #if ins["FO"]:
            if saf:
                SAF_direct.append(ins)
            else:
                SAF_direct.append(False)
            if ins:
                FO_direct.append(ins)
            else:
                FO_direct.append(False)
            return FO_direct, FO_indirect, SAF_direct, SAF_indirect, huri, index
        flash_position = MC[0].channel.hit_pos[:2]

        if len(dataset_saf) == 0:
            U_out, I_out, shared_dict = self.source_Indirect(MC[0], nodes, branches,constants, shared_dict)
            sources = self.add_lump(U_out, I_out)
            #solution, ins = process_calculate(MC, nodes, branches, self, index, shared_dict,calculate_ins)
            ins,saf = process_calculate(sources, self, index, calculate_saf)
            index = index + 1
            #if ins["FO"]:
            if saf:
                SAF_indirect.append(saf)
                return FO_direct,FO_indirect,SAF_direct,SAF_indirect,huri,index
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
                FO_indirect.append(False)
                return FO_direct, FO_indirect, SAF_direct, SAF_indirect, huri, index
        if len(dataset_saf)>0:
            #for i,stroke in enumerate(MC[0].strokes):
            stroke = MC[0].strokes[0]
            icur = np.append(stroke.parameters,flash_position)
            result = Huri_Method(R,dataset_ins,icur)
            result_saf = Huri_Method(R, dataset_saf, icur)
            if result ==1 and result_saf==1:
                SAF_indirect.append(False)
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
                dataset_ins.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))
                dataset_saf.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))

                print("using Huri skip calculation")
                return FO_direct, FO_indirect, SAF_direct, SAF_indirect, huri, index
        #solution, ins = process_item(MC, nodes, branches, self, index, shared_dict,calculate_ins)
        U_out, I_out, shared_dict = self.source_Indirect(MC[0], nodes, branches, constants, shared_dict)
        sources = self.add_lump(U_out, I_out)
        ins,saf = process_calculate(sources, self, index, calculate_saf)
        index = index + 1
       # if ins["FO"]:
        if ins:
            FO_indirect.append(saf)

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
        if saf:
            SAF_indirect.append(saf)

        else:
            distances = []
            for coord in list(self.PoleXY.values()):
                distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
                distances.append((coord, distance))
            distances.sort(key=lambda item: item[1])
            closest_two = distances[:2]
            PoleApp = [closest_two[0][0][0],closest_two[0][0][1],closest_two[1][0][0],closest_two[1][0][1]
                                            ,closest_two[0][1],closest_two[1][1]]
            dataset_saf.append(np.append(np.append(MC[0].strokes[0].parameters, flash_position), PoleApp))
            SAF_indirect.append(False)
        return FO_direct,FO_indirect,SAF_direct,SAF_indirect,huri,index
    def Huri_method_INS(self,MC, nodes, branches,shared_dict,dataset,FO_indirect,FO_direct,constants,index,huri):
        R = 100
        calculate_ins = 1

        self.T = (MC[0].strokes[0].parameters[1])*1e-6
        self.Nt = int(np.ceil(self.T / self.dt))
        MC[0].strokes[0].duration = (MC[0].strokes[0].parameters[1])*1e-6
        MC[0].strokes[0].Nt = self.Nt
        MC[0].strokes[0].t_us = np.array(list(range(self.Nt))) * self.dt
        MC[0].strokes[0].calculate()
        if MC[0].type == "Direct":
            U_out, I_out, shared_dict = self.source_Direct(MC[0], MC[1], MC[2], MC[3], nodes, branches,
                                                           constants, shared_dict)
            sources = self.add_lump(U_out, I_out)
            # solution,ins = process_calculate(MC, nodes, branches, self,index,shared_dict,calculate_ins)
            ins, saf = process_calculate(sources, self, index, calculate_ins)
            index = index+1
            #if ins["FO"]:
            if ins:
                FO_direct.append(ins)
            else:
                FO_direct.append(False)
            return FO_direct,FO_indirect,huri,index
        flash_position = MC[0].channel.hit_pos[:2]
        if len(dataset) == 0:
            U_out, I_out, shared_dict = self.source_Indirect(MC[0], nodes, branches, constants, shared_dict)
            sources = self.add_lump(U_out, I_out)
            #solution, ins = process_calculate(MC, nodes, branches, self, index, shared_dict,calculate_ins)
            ins,saf = process_calculate(sources, self, index, calculate_ins)
            index = index + 1
            #if ins["FO"]:
            if ins:
                FO_indirect.append(ins)
                return FO_direct,FO_indirect,huri,index
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
                return FO_direct,FO_indirect,huri,index
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
                return FO_direct,FO_indirect,huri,index
        U_out, I_out, shared_dict = self.source_Indirect(MC[0], nodes, branches, constants, shared_dict)
        sources = self.add_lump(U_out, I_out)
        # solution, ins = process_calculate(MC, nodes, branches, self, index, shared_dict,calculate_ins)
        ins,saf = process_calculate(sources, self, index, calculate_ins)
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
        return FO_direct,FO_indirect,huri,index

    @staticmethod
    def handle_no_ins(mc, flash_position, poleXY, dataset):
        distances = []
        for coord in poleXY.values():
            distance = math.sqrt((flash_position[0] - coord[0]) ** 2 + (flash_position[1] - coord[1]) ** 2)
            distances.append((coord, distance))
        distances.sort(key=lambda item: item[1])
        closest_two = distances[:2]
        PoleApp = [
            closest_two[0][0][0], closest_two[0][0][1],
            closest_two[1][0][0], closest_two[1][0][1],
            closest_two[0][1], closest_two[1][1]
        ]
        dataset.append(np.append(np.append(mc[0].strokes[0].parameters, flash_position), PoleApp))

    @staticmethod
    def process_mc(mc, nodes, branches, self_ref, index, shared_dict,
                   calculate_ins, dataset, R):
        """
        A static method: used as the target in `ProcessPoolExecutor.map()`.
        """
        flash_position = mc[0].channel.hit_pos[:2]

        # Direct stroke
        if mc[0].type == "Direct":
            ins = process_item(mc, nodes, branches, self_ref, index, shared_dict, calculate_ins)
            return ins, None

        # If dataset is empty, handle that first
        if len(dataset) == 0:
            ins = process_item(mc, nodes, branches, self_ref, index, shared_dict, calculate_ins)
            if ins:
                return ins, None
            else:
                # call handle_no_ins
                Network.handle_no_ins(mc, flash_position, self_ref.PoleXY, dataset)
                return False, None

        # Otherwise, Huri_Method
        stroke = mc[0].strokes[0]
        icur = np.append(stroke.parameters, flash_position)
        result = Huri_Method(R, dataset, icur)
        if result == 1:
            return False, None
        else:
            ins = process_item(mc, nodes, branches, self_ref, index, shared_dict, calculate_ins)
            return ins, None

    def Find_Dmax(self, MC_result, nodes, branches, shared_dict, dataset):
        """
        use `Network.process_mc` as the function for executor.map
        """
        FO = []
        R = 100
        index = 0
        calculate_ins = 1
        huri = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            # map each MC in MC_result, and pass all other parameters to `process_mc`.
            results = list(executor.map(
                Network.process_mc,
                MC_result,
                [nodes] * len(MC_result),
                [branches] * len(MC_result),
                [self] * len(MC_result),
                [index] * len(MC_result),
                [shared_dict] * len(MC_result),
                [calculate_ins] * len(MC_result),
                [dataset] * len(MC_result),
                [R] * len(MC_result)
            ))

        # `results`: list (ins, _) from `process_mc`
        for ins, _ in results:
            if ins:
                FO.append(ins)
            else:
                FO.append(False)

        return FO, huri, dataset

def process_calculate(sources,self_ref,index,calculate_saf):
    constants = Constant()
    constants.ep0 = 8.85e-12

    line_matrix = self_ref.H["Line"]
    tower_matrix = self_ref.H["Tower"]
    result_tower, bran = self_ref.calculate_of_hybrid_mode(line_matrix, tower_matrix, sources, self_ref.Nt, self_ref.dt, self_ref.GPU_calculation)
    #print("calculate"+str(index))
    FO_Tower = []
    SAF_Arrestor = []
    ins = False
    saf = False
    # 记录哪个塔损坏次数更多
    fo_tower = [tower for fo_bran in bran['SDEM']
                for tower, tower_bran in self_ref.Tower2ins_bran.items() if fo_bran in tower_bran]
    self_ref.weak_point = {t: self_ref.weak_point.get(t) + (1 if t in fo_tower else 0) for t in list(self_ref.weak_point.keys())}

    if len(bran["SDEM"])>0:
        ins = True
        # 指定CSV文件名
        filename = "Data/output/MC_ins.csv"
        # 使用'a'模式打开文件，准备追加内容
        with open(filename, 'a', newline='') as csvfile:
            # 创建一个csv写入器
            writer = csv.writer(csvfile)
            writer.writerow(bran["SDEM"])

        FO_Tower = ["tower_"+bran.split("_")[-1] for bran in bran["SDEM"]]
        self_ref.FO_Tower = FO_Tower
    if len(bran["NLR"])>0:
        broken_arrestor_list = [self_ref.arrestor_bran2Tower[bran][1] for bran in bran["NLR"] if bran in self_ref.arrestor_bran2Tower.keys()]
        #broken_Tower_list = [self_ref.arrestor_bran2Tower[bran][1] for bran in bran["NLR"] if
        #                        bran in self_ref.arrestor_bran2Tower.keys()]
        #SAF_Tower = list(set(broken_Tower_list))
        self_ref.broken = broken_arrestor_list
        saf = True
        return ins,saf
    return ins,saf

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

