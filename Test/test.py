import pickle

# 指定 pickle 文件的路径
file_path = 'network.pkl'

# 打开文件并加载数据
with open(file_path, 'rb') as file:
    data = pickle.load(file)

C = data.H['capacitance_matrix'].to_numpy()
G = data.H["conductance_matrix"].to_numpy()
L = data.H["inductance_matrix"].to_numpy()  # 线线
R = data.H["resistance_matrix"].to_numpy()
ima = data.H["incidence_matrix_A"].to_numpy()  # 线点
imb = data.H["incidence_matrix_B"].T.to_numpy()  # 点线

bran_num = len(data.H["incidence_matrix_A"].index)
node_num = len(data.H["incidence_matrix_A"].columns)
# 打印加载的数据
print(data)