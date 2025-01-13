import matplotlib.pyplot as plt
import pandas as pd
from sympy import false

name = "Heidler_perfect_1000_IP200"
summary = {"nonFO_indirect":789,"nonFO_direct":0,"FO_indirect":191,"FO_direct":20,"Huri":172,"RunTime":9773.815575361252,'FOR_direct':0,'FOR_indirect':38.2}
df = pd.DataFrame(summary,index=[name])
df.to_csv(f'Data/input/case3_nonlinear/summary_values_ins.csv', mode='a',header=false)