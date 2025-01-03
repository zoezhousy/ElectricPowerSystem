import numpy as np

def LGTMC_Solu(Tower, Span, Cable, GLB, LGT, MCLGT, Icurr, Iwave, StrPosi, FSdist, PoleXY):
    # Perform MC analysis to get FO indices
    Flash_Num = MCLGT['Num']
    output = None  # Initialize output

    for id in range(Flash_Num):
        # Getting LGT position and current data
        GLB, LGT = Sim_MCLGT_Init(Icurr, Iwave, StrPosi, FSdist, GLB, LGT, id)
        flash = LGT['Soc']['flash']  # for every flash
        StrNum = flash['head'][1]

        if FSdist[id][2] == 1 and (MCLGT['tsel'] == -1 or MCLGT['tsel'] == 1):
            # Direct lightning for MCLGT.tsel =-1 and 1
            output = LGT1_Solu(Tower, Span, Cable, GLB, LGT)  # 0=FO,1 =Non-FO
        elif FSdist[id][2] == 0 and (MCLGT['tsel'] == -1 or MCLGT['tsel'] == 0):
            # Indirect lightning for MCLGT.tsel =-1 and 0
            # Performing pre-judgement of simulation via heuristic approach
            PoleApp = []  # initiate every flash
            StrPos = LGT['Soc']['pos'][:, 1:3]  # x and y of a flash
            if len(MCLGT['huri']) > 0:
                for jd in range(StrNum):
                    icur = np.append(flash['para' ][jd, 2:6], StrPos[:, jd])
                    result_huri = Huri_Method(MCLGT, icur)
                    if result_huri == 1:  # non-flashover
                        if jd == 0 or not PoleApp:
                            dist = np.sqrt((icur[4] - PoleXY[:, 1])**2 + (icur[5] - PoleXY[:, 2])**2)
                            # Find the two adjacent poles with minimum distance
                            dmin = np.sort(dist)
                            dex = np.argsort(dist)
                            PoleApp = np.append(PoleXY[dex[0:2], 1:3].flatten(), dmin[0:2])
                        MCLGT['huri'] = np.append(MCLGT['huri'], [icur, PoleApp])
                        flash['flag'][0, jd] = 0
                LGT['Soc']['flash'] = flash
                GLB['Soc']['flash'] = flash

            # Performing single-LGT event simulation
            output = LGT1_Solu(Tower, Span, Cable, GLB, LGT)  # 0=FO,1=Non-FO

            # Updating the heuristic dataset for stroke without FO
            tmp1 = output[0]['FO'] * flash['flag']
            dex1 = np.where(tmp1 > 0)[0]
            if not PoleApp.size:
                dist = np.sqrt((StrPos[:, 0] - PoleXY[:, 1])**2 + (StrPos[:, 1] - PoleXY[:, 2])**2)
                # Find the two adjacent poles with minimum distance
                dmin = np.sort(dist)
                dex = np.argsort(dist)
                PoleApp = np.append(PoleXY[dex[0:2], 1:3].flatten(), dmin[0:2])
            tmp2 = np.tile([StrPos, PoleApp], (len(dex1), 1))
            MCLGT['huri'] = np.append(MCLGT['huri'], [flash['para'][dex1, 2:6], tmp2])

    return output

# 注意：Sim_MCLGT_Init, LGT1_Solu, 和 Huri_Method 需要被定义或转换为Python函数。