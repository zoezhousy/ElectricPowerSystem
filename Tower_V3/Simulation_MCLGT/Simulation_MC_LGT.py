import numpy as np
import os
import sys
import shutil
import pandas as pd
import ast

from Tower_V3.PARA_MCLG.MC_Init import MC_Init
from Tower_V3.PARA_MCLG.MC_Init_S import MC_Init_S
from Tower_V3.PARA_MCLG.MC_Data_Gene import MC_Data_Gene
from Tower_V3.PARA_MCLG.MC_Data_Gene_S import MC_Data_Gene_S
from Tower_V3.Simulation_MCLGT.LGTMC_Solu import LGTMC_Solu


class Simulation_MC_LGT():
    def __init__(self, FDIR):
        self.FDIR = FDIR  # Initialize the Simulation_LGT class with the given file directory paths

    def Simulation_MC_LGT(self,Tower, Span, Cable, GLB, LGT):
        # GLB['SSdy']['flag'] = [1, 0, 0, 0, 0, 0, 0] 没有GLB.SSdy
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注释了！！！！！！！！！！！！！
        [LatDis, WaveModel, MC_lgtn, DSave,AR] = MC_Init() # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        filepath = GLB['FDIR']['dataMCLG']
        if os.path.exists(filepath):
            shutil.rmtree(filepath)

        os.mkdir(filepath)


        [EdgeXY, Icurr, Summary, StrokePosi, Iwave, FSdist, PoleXY] \
            = MC_Data_Gene.MC_Data_Gene(self, GLB, Span, Tower, MC_lgtn, DSave,LatDis, WaveModel,AR)

        # [LatDis_S, WaveModel_S, MC_lgtn_S, DSave_S] = MC_Init_S()
        # filepath_S = GLB['FDIR']['dataMCLG2']
        # if os.path.exists(filepath_S):
        #     shutil.rmtree(filepath_S)
        #
        # os.mkdir(filepath_S)
        #
        # [EdgeXY_S, Icurr_S, Summary_S, StrokePosi_S, Iwave_S, FSdist_S, PoleXY_S] \
        #     = MC_Data_Gene_S.MC_Data_Gene_S(self, GLB, Span, Tower, MC_lgtn_S, DSave_S, LatDis_S, WaveModel_S)

        excel_path = filepath + '/' + 'Pole_XY.xlsx'
        PoleXY = pd.read_excel(excel_path,).to_numpy()
        excel_path = filepath + '/' + 'Flash_Stroke Dist.xlsx'
        FSdist = pd.read_excel(excel_path).to_numpy()
        excel_path = filepath + '/' + 'Current Waveform_CIGRE.xlsx'
        Iwave = pd.read_excel(excel_path).to_numpy()
        excel_path = filepath + '/' + 'Current Parameter.xlsx'
        Icurr = pd.read_excel(excel_path).to_numpy()
        excel_path = filepath + '/' + 'Stroke Position.xlsx'
        StrokePosi2 = pd.read_excel(excel_path).to_numpy()
        StrokePosi = np.empty((StrokePosi2.shape[0], StrokePosi2.shape[1] + 2), dtype=object)
        StrokePosi[:, 0:7] = StrokePosi2[:, 0:7]
        for ist in range(StrokePosi2.shape[0]):
            St1a = ast.literal_eval(StrokePosi2[ist, 7])
            StrokePosi[ist, 7:9] = np.array(St1a)
            if (StrokePosi2[ist, 8])=='[nan, nan]':
                StrokePosi[ist, 9:11] = np.nan*2
            else:
                St1b = ast.literal_eval(StrokePosi2[ist, 8])
                StrokePosi[ist, 9:11] = np.array(St1b)

        MCLGT = {}
        MCLGT['Num'] = len(FSdist)

        MCLGT['flag'] = 1
        MCLGT['huri'] = []
        MCLGT['radi'] = 60
        MCLGT['tsel'] = 0

        FO_Type = 1
        GLB['FOtype'] = FO_Type

        output = LGTMC_Solu(Tower, Span, Cable, GLB, LGT, MCLGT,
        Icurr, Iwave, StrokePosi, FSdist, PoleXY)

        return output
