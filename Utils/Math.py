import numpy as np
import math
import cupy as cp


def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 +
                     (pos1[1] - pos2[1]) ** 2 +
                     (pos1[2] - pos2[2]) ** 2)

def Bessel_K2(z, n1, n2):
    """
    【函数功能】修正的第二类Bessel函数相除Kn-1(z)/Kn(z)近似表达
    【入参】
    z(float): Bessel函数变量
    n1(int): 分子的Bessel函数阶数
    n2(int): 分母的Bessel函数阶数

    【出参】
    K2(float): Kn-1(z)/Kn(z)近似表达
    """
    K2 = (1 + ((4 * n1 ** 2 - 1) / (8 * z)) + ((4 * n1 ** 2 - 1) * (4 * n1 ** 2 - 9) / (2 * (8 * z) ** 2)) + (
            (4 * n1 ** 2 - 1) * (4 * n1 ** 2 - 9) * (4 * n1 ** 2 - 25) / (6 * (8 * z) ** 3))) / (
                 1 + ((4 * n2 ** 2 - 1) / (8 * z)) + (
                 (4 * n2 ** 2 - 1) * (4 * n2 ** 2 - 9) / (2 * (8 * z) ** 2)) + (
                         (4 * n2 ** 2 - 1) * (4 * n2 ** 2 - 9) * (4 * n2 ** 2 - 25) / (6 * (8 * z) ** 3)))
    return K2


def Bessel_IK(z1, n1, z2, n2):
    """
    【函数功能】修正的第一类Bessel函数和第二类Bessel函数相乘，In1(z1)*Kn2(z2)近似表达
    【入参】
    z1(float): 第一类Bessel函数变量
    n1(int): 第一类BBessel函数阶数
    z2(float): 第二类Bessel函数变量
    n2(int): 第二类Bessel函数阶数

    【出参】
    IK(float): In1(z1)*Kn2(z2)近似表达
    """
    IK = np.exp(z1 - z2) / 2 / np.sqrt(z1 * z2) * (
            1 - ((4 * n1 ** 2 - 1) / (8 * z1)) + ((4 * n2 ** 2 - 1) / (8 * z2)) - ((4 * n1 ** 2 - 1) / (8 * z1)) * (
            (4 * n2 ** 2 - 1) / (8 * z2)))
    return IK


def calculate_distances(points1, points2):
    """
    计算两个 n x 3 矩阵中对应行之间的距离.
    
    参数:
    points1 (np.ndarray): 第一个 n x 3 矩阵,每行表示一个点的 x, y, z 坐标
    points2 (np.ndarray): 第二个 n x 3 矩阵,每行表示一个点的 x, y, z 坐标
    
    返回:
    np.ndarray: 一个 n x 1 矩阵,表示两个矩阵中对应行之间的距离
    """
    if points1.shape != points2.shape:
        raise ValueError("两个输入矩阵必须有相同的形状!")
    
    distances = np.zeros((points1.shape[0], 1))
    for i in range(points1.shape[0]):
        distances[i] = np.sum((points1[i] - points2[i])**2)
    
    return distances


def calculate_direction_cosines(start_points, end_points, lengths):
    """
    计算 x、y 和 z 方向上的余弦值矩阵。

    参数:
    start_points (numpy.ndarray): n*3 矩阵,表示 n 条线段的起点坐标(x, y, z)
    end_points (numpy.ndarray): n*3 矩阵,表示 n 条线段的终点坐标(x, y, z)
    lengths (numpy.ndarray): n*1 矩阵,表示 n 条线段的长度

    返回:
    x_cosines, y_cosines, z_cosines(numpy.ndarray, numpy.ndarray, numpy.ndarray): x、y 和 z 方向上的余弦值矩阵
    """
    # 计算 x 方向上的余弦值
    x_cosines = (end_points[:, 0] - start_points[:, 0]).reshape(lengths.shape[0], 1) / lengths

    # 计算 y 方向上的余弦值
    y_cosines = (end_points[:, 1] - start_points[:, 1]).reshape(lengths.shape[0], 1) / lengths

    # 计算 z 方向上的余弦值
    z_cosines = (end_points[:, 2] - start_points[:, 2]).reshape(lengths.shape[0], 1) / lengths

    return x_cosines, y_cosines, z_cosines


def segment_branch(network_branches):
    """
    【功能】对于传进来的branches，按照读取的Nt进行分段
    """
    branches = network_branches.copy()  # branches的副本，用于新增或删减支路
    OHL_new_nodes = []
    for key, value in network_branches.items():
        # key是支路， value是起点，终点，分段数
        keys_tobe_delete = []
        if 'OHL' in value[2]  and value[3] != 1:
            start_node_coord = list(value[0].values())[0]
            end_node_coord = list(value[1].values())[0]
            # 生成新节点
            x = np.linspace(start_node_coord[0], end_node_coord[0], value[3] + 1, dtype=float)
            y = np.linspace(start_node_coord[1], end_node_coord[1], value[3] + 1, dtype=float)
            z = np.linspace(start_node_coord[2], end_node_coord[2], value[3] + 1, dtype=float)
            new_nodes_name = [f"{key}_MiddleNode_{i}".format(i) for i in range(1, value[3])]
            new_nodes_name.insert(0, list(value[0].keys())[0])
            new_nodes_name.append(list(value[1].keys())[0])
            OHL_new_nodes.extend(new_nodes_name)
            for i in range(len(new_nodes_name) - 1):
                start_node_after_seg_dict = {new_nodes_name[i]: [x[i], y[i], z[i]]}
                end_node_after_seg_dict = {new_nodes_name[i+1]: [x[i+1], y[i+1], z[i+1]]}

                branches[f"{key}_Splited_{i+1}"] = [start_node_after_seg_dict, end_node_after_seg_dict, value[2], value[3]]
            del branches[key]
    return branches,set(OHL_new_nodes)


def calculate_distances_between_lineseg_and_channelseg(points_a, points_b):
    """
    计算points_a中每个点到points_b中每个点的距离矩阵
    """
    # 计算差值
    differences_xyz = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    differences_xy = points_a[:, np.newaxis, :2] - points_b[np.newaxis, :, :2]
    differences_z = points_a[:, np.newaxis, 2] - points_b[np.newaxis, :, 2]

    distances_xyz = np.linalg.norm(differences_xyz, axis=2)
    distances_xy = np.linalg.norm(differences_xy, axis=2)
    distances_z = differences_z
    distances_z.reshape(-1, 1)
    return distances_xyz, distances_xy, distances_z


def calculate_distances_between_lineseg_and_channelseg_GPU(points_a, points_b):
    """
    计算points_a中每个点到points_b中每个点的距离矩阵
    """

    # 计算差值
    differences_xyz = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    differences_xy = points_a[:, np.newaxis, :2] - points_b[np.newaxis, :, :2]
    differences_z = points_a[:, np.newaxis, 2] - points_b[np.newaxis, :, 2]

    distances_xyz = cp.linalg.norm(differences_xyz, axis=2)
    distances_xy = cp.linalg.norm(differences_xy, axis=2)
    distances_z = differences_z
    distances_z.reshape(-1, 1)
    
    return distances_xyz, distances_xy, distances_z


def get_t_delay_index1(arr):
    # 筛选出大于等于0的元素
    positive_part = arr[arr > 0]
    # 创建一个与输入数组相同长度的零数组
    index1 = np.zeros_like(arr)
    # 将大于0的元素放在前面
    index1[:len(positive_part)] = 1
    return index1


def get_t_delay_index2(arr):
    # 筛选出大于等于0的元素
    positive_part = arr[arr > 0]
    # 创建一个与输入数组相同长度的零数组
    index2 = np.zeros_like(arr)
    # 将大于0的元素放在后面
    if positive_part.size > 0:
        index2[-len(positive_part):] = 1
    return index2

def get_chunks(D, T):
    """
    返回某一维度的分块索引范围列表。
    假设维度为 D，阈值为 T。
    例如：D=450, T=200 时，返回:
    [(1,200), (201,400), (401,450)]
    """
    if D <= T:
        return [[0, D]]
    else:
        # 分块数
        num_chunks = math.ceil(D / T)
        chunks = []
        start = 0
        for i in range(num_chunks):
            end = min(start + T, D)
            chunks.append([start, end])
            start = end
        return chunks

def split_3d_matrix(shape, thresholds):
    """
    对一个 3D 矩阵按照阈值进行分块，并返回分块信息。
    shape: (D1, D2, D3)
    thresholds: (T1, T2, T3)
    返回:
      - total_blocks: 分块数
      - blocks_info: 每个分块在原矩阵中 (dim1, dim2, dim3) 三个维度的索引范围
    """
    D1, D2, D3 = shape
    T1, T2, T3 = thresholds
    
    chunks_d1 = get_chunks(D1, T1)
    chunks_d2 = get_chunks(D2, T2)
    chunks_d3 = get_chunks(D3, T3)
    
    # 计算三维分块的笛卡尔积
    blocks_info = []
    for cd1 in chunks_d1:
        for cd2 in chunks_d2:
            for cd3 in chunks_d3:
                blocks_info.append([cd1, cd2, cd3])
                
    total_blocks = len(blocks_info)
    
    return total_blocks, blocks_info

def calculate_electric_field_down_r_and_z(pt_start, pt_end, stroke, channel, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, GPU, air_or_img):
    """
    功能：
    计算雷击影响下，不同时刻，自由空间或镜像电场

    参数说明：
    pt_start (np.array, (n, 3)): 导体段的起点坐标，每行代表坐标(x, y, z)
    pt_end (np.array, (n, 3)): 导体段的终点坐标，每行代表坐标(x, y, z)
    stroke (Stroke对象)
    channel (Channel对象)
    z_channel (list): 每个雷电通道段终点的z坐标列表，从小到大排列
    i_sr (np.array, (1, stroke.Nt): 雷电流时间序列
    t_sr (np.array, (1, stroke.Nt): 时刻点
    i_sr_int (np.array, (1, stroke.Nt): 雷电流时间序列的积分
    i_sr_div (np.array, (1, stroke.Nt): 雷电流时间序列的微分
    ep0 (float): 常数，真空介电常数
    vc(float): 常数，光速
    air_or_img (bool): 指示计算自由空间电厂还是镜像电场

    返回：
    Ez (np.array, (stroke.Nt,  n): n为导体段的个数，z方向的电场，每行代表一个时刻点，每列代表导体段
    Er (np.array, (stroke.Nt,  n): n为导体段的个数，r方向，即xoy平面方向的电场，每行代表一个时刻点，每列代表导体段
    """

    # 初始化径向和垂直方向的电场
    a00 = pt_start.shape[0]  # 观察点或导体分段的总数
    b00 = z_channel.shape[0]
    Nt = stroke.Nt
    dt = stroke.dt
    # channel参数赋值
    H = channel.H
    lamda = channel.lamda
    vcof = channel.vcof

    line_mid_points = (pt_start + pt_end) / 2
    channel_mid_points = np.column_stack( (np.full(z_channel.shape[0], channel.hit_pos[0]), np.full(z_channel.shape[0], channel.hit_pos[1]), z_channel))

    if GPU == 1:
        print('GPU is used to accelerate the calculation of electric field.')

        # z_channel = cp.array(z_channel, dtype=cp.float32)
        # vc = cp.array(vc, dtype=cp.float32)
        # vcof = cp.array(vcof, dtype=cp.float32)
        # i_sr = cp.array(i_sr, dtype=cp.float32)
        # t_sr = cp.array(t_sr, dtype=cp.float32)
        # i_sr_int = cp.array(i_sr_int, dtype=cp.float32)
        # i_sr_div = cp.array(i_sr_div, dtype=cp.float32)
        # line_mid_points = cp.array(line_mid_points, dtype=cp.float32)
        # channel_mid_points = cp.array(channel_mid_points, dtype=cp.float32)

        z_channel = cp.array(z_channel)
        vc = cp.array(vc)
        vcof = cp.array(vcof)
        i_sr = cp.array(i_sr)
        t_sr = cp.array(t_sr)
        i_sr_int = cp.array(i_sr_int)
        i_sr_div = cp.array(i_sr_div)
        line_mid_points = cp.array(line_mid_points)
        channel_mid_points = cp.array(channel_mid_points)
        threshold = (100, 1e10, 1e10)  # 矩阵分块的阈值

        # 空间距离和时间延迟的计算可以通过矩阵运算来实现
        Rxyz, Rxy, Rz = calculate_distances_between_lineseg_and_channelseg_GPU(line_mid_points, channel_mid_points)
        
        # 修改函数名
        array = cp.array
        new_abs = cp.abs
        new_linalg = cp.linalg
        floor = cp.floor
        sort = cp.sort
        flip = cp.flip
        expand_dims = cp.expand_dims
        tile = cp.tile
        zeros = cp.zeros
        sumarize = cp.sum
        pi = cp.pi
        exp = cp.exp
        float32 = cp.float32
    else:
        threshold = (300, 200, 20000)  # 无GPU加速时，根据内存大小设定矩阵分块的阈值
        # 空间距离和时间延迟的计算可以通过矩阵运算来实现
        Rxyz, Rxy, Rz = calculate_distances_between_lineseg_and_channelseg(line_mid_points, channel_mid_points)

        array = np.array
        new_abs = np.abs
        new_linalg = np.linalg
        floor = np.floor
        sort = np.sort
        flip = np.flip
        expand_dims = np.expand_dims
        tile = np.tile
        zeros = np.zeros
        sumarize = np.sum
        pi = np.pi
        exp = np.exp
        float32 = np.float32

    # 增加维度，便于广播
    t_delay = (new_abs(z_channel) / vc / vcof + Rxyz / vc)
    # t_delay_expand = np.tile(t_delay[:, :, np.newaxis], (1, 1, stroke.Nt))  # 变成 (a00, b00, 1)
    # t_sr_expand = np.tile(t_sr, (a00, b00, 1))  # 变成 (1, 1, 2000)
    t_sr_expand = t_sr.reshape(1, 1, -1)  # (1, 1, Nt)
    t_delay_expand = t_delay.reshape(*t_delay.shape, 1)  # (a00, b00, 1) 

    # 计算时间延迟，获取时间索引
    # n_td_tmp = np.floor((t_sr_expand - t_delay_expand) / stroke.dt).astype(int)
    n_td_tmp = floor((t_sr_expand * 1e-6 - t_delay_expand) / dt).astype(int)

    # index_head = np.apply_along_axis(get_t_delay_index1, 2, n_td_tmp)
    # index_tail = np.apply_along_axis(get_t_delay_index2, 2, n_td_tmp)
    mask = (n_td_tmp >= 0) & (n_td_tmp < Nt)
    reverse_mask = flip(sort(mask, axis=2), axis=2)
    del n_td_tmp, t_sr_expand # 清内存
    # index_head = np.where(index_head == 1)

    # index_tail = np.where(index_tail == 1)
    # id_t = n_td_tmp > 0


    # 根据自由空间和镜像，选择对应的系数计算方式
    if air_or_img == 0:
        # 根据不同的传播模型，选择系数
        if channel.channel_model == 'TL':
            cof_isr = 1 / (4 * pi * ep0)
        elif channel.channel_model == 'MTLL':
            cof_isr = 1 / (4 * pi * ep0) * (1 - z_channel / H)
        else:
            cof_isr = 1 / (4 * pi * ep0) * exp(-z_channel / lamda)
    else:
        if channel.channel_model == 'TL':
            cof_isr = 1 / (4 * pi * ep0)
        elif channel.channel_model == 'MTLL':
            cof_isr = 1 / (4 * pi * ep0) * (1 + z_channel / H)
        else:
            cof_isr = 1 / (4 * pi * ep0) * exp(-new_abs(z_channel) / lamda)

    total_blocks, blocks_info = split_3d_matrix((a00, b00, Nt), threshold)
    blocks_info = array(blocks_info)
    print(f"分块数量{total_blocks}")
    Ez = np.zeros((a00, Nt), dtype=float32)
    Er = np.zeros((a00, Nt), dtype=float32)
    for i in range(total_blocks):
        
        x1 = blocks_info[i, 0, 0] # 第一个维度的起始索引
        y1 = blocks_info[i, 0, 1] # 第一个维度的终止索引
        x2 = blocks_info[i, 1, 0]
        y2 = blocks_info[i, 1, 1]
        x3 = blocks_info[i, 2, 0]
        y3 = blocks_info[i, 2, 1]
        a00_block = int(y1 - x1)
        b00_block = int(y2 - x2)
        Nt_block = int(y3 - x3)
        dEz_1_cof = cof_isr * (2 * Rz[x1:y1, x2:y2] ** 2 - Rxy[x1:y1, x2:y2] ** 2) / Rxyz[x1:y1, x2:y2] ** 5
        dEz_2_cof = cof_isr * (2 * Rz[x1:y1, x2:y2] ** 2 - Rxy[x1:y1, x2:y2] ** 2) / Rxyz[x1:y1, x2:y2] ** 4 / vc
        dEz_3_cof = cof_isr * (Rxy[x1:y1, x2:y2] ** 2) / Rxyz[x1:y1, x2:y2] ** 3 / vc ** 2

        # 扩充维度为三维矩阵
        dEz_1_cof = expand_dims(dEz_1_cof, axis=-1)
        dEz_2_cof = expand_dims(dEz_2_cof, axis=-1)
        dEz_3_cof = expand_dims(dEz_3_cof, axis=-1)

        dEr_1_cof = cof_isr * (3 * Rz[x1:y1, x2:y2] * Rxy[x1:y1, x2:y2]) / Rxyz[x1:y1, x2:y2] ** 5
        dEr_2_cof = cof_isr * (3 * Rz[x1:y1, x2:y2] * Rxy[x1:y1, x2:y2]) / Rxyz[x1:y1, x2:y2] ** 4 / vc
        dEr_3_cof = cof_isr * (Rz[x1:y1, x2:y2] * Rxy[x1:y1, x2:y2]) / Rxyz[x1:y1, x2:y2] ** 3 / vc ** 2

        dEr_1_cof = expand_dims(dEr_1_cof, axis=-1)
        dEr_2_cof = expand_dims(dEr_2_cof, axis=-1)
        dEr_3_cof = expand_dims(dEr_3_cof, axis=-1)

        # del Rxyz, Rz, Rxy

        # 使用广播和布尔索引来累加对应时刻的贡献
        # i_sr_expand = tile(i_sr, (a00, b00, 1))
        # i_sr_int_expand = tile(i_sr_int, (a00, b00, 1))
        # i_sr_div_expand = tile(i_sr_div, (a00, b00, 1))
        i_sr_expand = i_sr.reshape(1, 1, -1)[:, :, x3:y3]  # (1, 1, Nt)
        i_sr_int_expand = i_sr_int.reshape(1, 1, -1)[:, :, x3:y3]
        i_sr_div_expand = i_sr_div.reshape(1, 1, -1)[:, :, x3:y3]
        mask = mask[x1:y1, x2:y2, x3:y3]
        reverse_mask = reverse_mask[x1:y1, x2:y2, x3:y3]

        dEz1 = zeros((a00_block, b00_block, Nt_block), dtype=float32)
        dEz2 = zeros((a00_block, b00_block, Nt_block), dtype=float32)
        dEz3 = zeros((a00_block, b00_block, Nt_block), dtype=float32)

        # dEz1[index_tail] = (dEz_1_cof * i_sr_int_expand)[index_head]
        # dEz2[index_tail] = (dEz_2_cof * i_sr_expand)[index_head]
        # dEz3[index_tail] = (dEz_3_cof * i_sr_div_expand)[index_head]
        dEz1[mask] = (dEz_1_cof * i_sr_int_expand)[reverse_mask]
        dEz2[mask] = (dEz_2_cof * i_sr_expand)[reverse_mask]
        dEz3[mask] = (dEz_3_cof * i_sr_div_expand)[reverse_mask]

        dEr1 = zeros((a00_block, b00_block, Nt_block), dtype=float32)
        dEr2 = zeros((a00_block, b00_block, Nt_block), dtype=float32)
        dEr3 = zeros((a00_block, b00_block, Nt_block), dtype=float32)

        # dEr1[index_tail] = (dEr_1_cof * i_sr_int_expand)[index_head]
        # dEr2[index_tail] = (dEr_2_cof * i_sr_expand)[index_head]
        # dEr3[index_tail] = (dEr_3_cof * i_sr_div_expand)[index_head]
        dEr1[mask] = (dEr_1_cof * i_sr_int_expand)[reverse_mask]
        dEr2[mask] = (dEr_2_cof * i_sr_expand)[reverse_mask]
        dEr3[mask] = (dEr_3_cof * i_sr_div_expand)[reverse_mask]
        if GPU == 1:
            Ez[int(x1):int(y1), int(x3):int(y3)] = cp.asnumpy(sumarize(dEz1 + dEz2 - dEz3, axis=1))
            Er[int(x1):int(y1), int(x3):int(y3)] = cp.asnumpy(sumarize(dEr1 + dEr2 + dEr3, axis=1))
            print(f"分块{i}最大电场{np.max(Er)}")
            # del dEz1, dEz2, dEz3, dEr1, dEr2, dEr3, i_sr_expand, i_sr_int_expand, i_sr_div_expand
            # cp._default_memory_pool.free_all_blocks()
        else:
            Ez[x1:y1, x3:y3] = sumarize(dEz1 + dEz2 - dEz3, axis=1)
            Er[x1:y1, x3:y3] = sumarize(dEr1 + dEr2 + dEr3, axis=1)
            del dEz1, dEz2, dEz3, dEr1, dEr2, dEr3, i_sr_expand, i_sr_int_expand, i_sr_div_expand

    # Ez = sumarize(dEz1 + dEz2 - dEz3, axis=1)
    # Er = sumarize(dEr1 + dEr2 + dEr3, axis=1)
    # 转置电场矩阵
    # if GPU == 1:
    #     Ez = cp.asnumpy(Ez.T)
    #     Er = cp.asnumpy(Er.T)
    # else:
    Ez = Ez.T
    Er = Er.T
    return Ez, Er

def calculate_H_magnetic_field_down_r(pt_start, pt_end, stroke, channel, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, GPU, air_or_img):
    """
    功能：
    计算雷击影响下，不同时刻，r方向（z方向为0）的自由空间或镜像磁场？

    参数说明：
    pt_start (np.array, (n, 3)): 导体段的起点坐标，每行代表坐标(x, y, z)
    pt_end (np.array, (n, 3)): 导体段的终点坐标，每行代表坐标(x, y, z)
    stroke (Stroke对象)
    channel (Channel对象)
    z_channel (list): 每个雷电通道段终点的z坐标列表，从小到大排列
    i_sr (np.array, (1, stroke.Nt): 雷电流时间序列
    t_sr (np.array, (1, stroke.Nt): 时刻点
    i_sr_int (np.array, (1, stroke.Nt): 雷电流时间序列的积分
    i_sr_div (np.array, (1, stroke.Nt): 雷电流时间序列的微分
    ep0 (float): 常数，真空介电常数
    vc(float): 常数，光速
    air_or_img (bool): 指示计算自由空间电厂还是镜像电场

    返回：
    Ez (np.array, (stroke.Nt,  n): n为导体段的个数，每行代表一个时刻点，每列代表导体段
    Er (np.array, (stroke.Nt,  n): n为导体段的个数，每行代表一个时刻点，每列代表导体段
    """

    a00 = pt_start.shape[0]  # 观察点或导体分段的总数
    b00 = z_channel.shape[0]
    # stroke参数赋值
    Nt = stroke.Nt
    dt = stroke.dt
    # channel参数赋值
    H = channel.H
    lamda = channel.lamda
    vcof = channel.vcof

    line_mid_points = (pt_start + pt_end) / 2
    channel_mid_points = np.column_stack( (np.full(z_channel.shape[0], channel.hit_pos[0]), np.full(z_channel.shape[0], channel.hit_pos[1]), z_channel))
    line_mid_points[:, 2] = 0 # 计算磁场时，不考虑z方向

    if GPU == 1:
        print('GPU is used to accelerate the calculation of electric field.')

        z_channel = cp.array(z_channel)
        vc = cp.array(vc)
        vcof = cp.array(vcof)
        i_sr = cp.array(i_sr)
        t_sr = cp.array(t_sr)
        i_sr_int = cp.array(i_sr_int)
        i_sr_div = cp.array(i_sr_div)
        line_mid_points = cp.array(line_mid_points)
        channel_mid_points = cp.array(channel_mid_points)

        # 空间距离和时间延迟的计算可以通过矩阵运算来实现
        Rxyz, Rxy, Rz = calculate_distances_between_lineseg_and_channelseg_GPU(line_mid_points, channel_mid_points)
        
        # 修改函数名
        new_abs = cp.abs
        # new_linalg = cp.linalg
        floor = cp.floor
        sort = cp.sort
        flip = cp.flip
        expand_dims = cp.expand_dims
        tile = cp.tile
        zeros = cp.zeros
        sumarize = cp.sum
        pi = cp.pi
        exp = cp.exp
        float32 = cp.float32
    else:
        # 空间距离和时间延迟的计算可以通过矩阵运算来实现
        Rxyz, Rxy, Rz = calculate_distances_between_lineseg_and_channelseg(line_mid_points, channel_mid_points)

        new_abs = np.abs
        # new_linalg = np.linalg
        floor = np.floor
        sort = np.sort
        flip = np.flip
        expand_dims = np.expand_dims
        tile = np.tile
        zeros = np.zeros
        sumarize = np.sum
        pi = np.pi
        exp = np.exp
        float32 = np.float32

    # 增加维度，便于广播
    t_delay = (new_abs(z_channel) / vc / vcof + Rxyz / vc)
    # t_delay_expand = np.tile(t_delay[:, :, np.newaxis], (1, 1, stroke.Nt))  # 变成 (a00, b00, 1)
    # t_sr_expand = np.tile(t_sr, (a00, b00, 1))  # 变成 (1, 1, 2000)
    t_sr_expand = t_sr.reshape(1, 1, -1)  # (1, 1, Nt)
    t_delay_expand = t_delay.reshape(*t_delay.shape, 1)  # (a00, b00, 1) 

    # 计算时间延迟，获取时间索引
    # n_td_tmp = np.floor((t_sr_expand - t_delay_expand) / stroke.dt).astype(int)
    n_td_tmp = np.floor((t_sr_expand * 1e-6 - t_delay_expand) / dt).astype(int)
    # index_head = np.apply_along_axis(get_t_delay_index1, 2, n_td_tmp)
    # index_tail = np.apply_along_axis(get_t_delay_index2, 2, n_td_tmp)
    mask = (n_td_tmp >= 0) & (n_td_tmp < Nt)
    reverse_mask = flip(sort(mask, axis=2), axis=2)
    del t_delay_expand, n_td_tmp # 清除变量，释放内存
    # index_head = np.where(index_head == 1)
    # index_tail = np.where(index_tail == 1)
    # id_t = n_td_tmp > 0


    # 根据自由空间和镜像，选择对应的系数计算方式
    if air_or_img == 0:
        # 根据不同的传播模型，选择系数
        if channel.channel_model == 'TL':
            cof_isr = 1 / (4 * pi * ep0)
        elif channel.channel_model == 'MTLL':
            cof_isr = 1 / (4 * pi * ep0) * (1 - z_channel / H)
        else:
            cof_isr = 1 / (4 * pi * ep0) * exp(-z_channel / lamda)
    else:
        if channel.channel_model == 'TL':
            cof_isr = 1 / (4 * pi * ep0)
        elif channel.channel_model == 'MTLL':
            cof_isr = 1 / (4 * pi * ep0) * (1 + z_channel / H)
        else:
            cof_isr = 1 / (4 * pi * ep0) * np.exp(-new_abs(z_channel) / lamda)
    # if hasattr(stroke, "GPU"):
    #     print('GPU is used to accelerate the calculation of magnetic field.')

    #     cof_isr = cp.array(cof_isr)
    #     i_sr = cp.array(i_sr)
    #     i_sr_int = cp.array(i_sr_int)
    #     i_sr_div = cp.array(i_sr_div)
    #     index_head = cp.array(index_head)
    #     index_tail = cp.array(index_tail)
    #     Rxyz = cp.array(Rxyz)
    #     Rz = cp.array(Rz)
    #     Rxy = cp.array(Rxy)

    #     expand_dims = cp.expand_dims
    #     tile = cp.tile
    #     zeros = cp.zeros
    #     sumarize = cp.sum
    # else:
    #     expand_dims = np.expand_dims
    #     tile = np.tile
    #     zeros = np.zeros
    #     sumarize = np.sum

    dEr_1_cof = 0 * cof_isr * (3 * Rz * Rxy) / Rxyz ** 5
    dEr_1_cof = expand_dims(dEr_1_cof, axis=-1)
    i_sr_int_expand = i_sr_int.reshape(1, 1, -1)  # 使用广播和布尔索引来累加对应时刻的贡献
    dEr1 = zeros((a00, b00, Nt), dtype=float32)
    dEr1[mask] = (dEr_1_cof * i_sr_int_expand)[reverse_mask]
    del dEr_1_cof, i_sr_int_expand

    dEr_2_cof = cof_isr * (Rxy) / Rxyz ** 3
    dEr_2_cof = expand_dims(dEr_2_cof, axis=-1)
    i_sr_expand = i_sr.reshape(1, 1, -1)
    dEr2 = zeros((a00, b00, Nt), dtype=float32)
    dEr2[mask] = (dEr_2_cof * i_sr_expand)[reverse_mask]
    del dEr_2_cof, i_sr_expand

    dEr_3_cof = cof_isr * (Rxy) / Rxyz ** 2 / vc
    dEr_3_cof = expand_dims(dEr_3_cof, axis=-1)
    i_sr_div_expand = i_sr_div.reshape(1, 1, -1)
    dEr3 = zeros((a00, b00, Nt), dtype=float32)
    dEr3[mask] = (dEr_3_cof * i_sr_div_expand)[reverse_mask]
    del dEr_3_cof, i_sr_div_expand

    Er = sumarize(dEr1 + dEr2 + dEr3, axis=1)
    # 转置电场矩阵
    if GPU == 1:
        Er = cp.asnumpy(Er.T)
    else:
        Er = Er.T
    return Er


# def calculate_electric_field_down_r_and_z(pt_start, pt_end, stroke, channel, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, air_or_img):
#     # 初始化径向和垂直方向的电场
#     a00 = pt_start.shape[0]  # 观察点或导体分段的总数
#     b00 = z_channel.shape[0]
#
#     # 雷击点位置
#     hit_pos = channel.hit_pos[0:2]
#
#     Rx = (pt_start[:, 0] + pt_end[:, 0]) / 2 - hit_pos[0]  # 雷击点在x方向到每个导线段中点的距离
#     Ry = (pt_start[:, 1] + pt_end[:, 1]) / 2 - hit_pos[1]  # 雷击点在y方向到每个导线段中点的距离
#     Rxy = np.sqrt(Rx ** 2 + Ry ** 2)  # 水平距离
#
#     # 初始化径向和垂直方向的电场
#     Er = np.zeros((stroke.Nt, a00))
#     Ez = np.zeros((stroke.Nt, a00))
#
#     for ik in range(a00):
#         x1, y1, z1 = pt_start[ik, :]
#         x2, y2, z2 = pt_end[ik, :]
#
#         dEz1, dEz2, dEz3 = np.zeros((stroke.Nt, channel.N_channel_segment)), np.zeros((stroke.Nt, channel.N_channel_segment)), np.zeros((stroke.Nt, channel.N_channel_segment))
#         dEr1, dEr2, dEr3 = np.zeros((stroke.Nt, channel.N_channel_segment)), np.zeros((stroke.Nt, channel.N_channel_segment)), np.zeros((stroke.Nt, channel.N_channel_segment))
#         Rxyz = np.zeros(channel.N_channel_segment)
#         Rz = np.zeros(channel.N_channel_segment)
#
#         # 自由空间的电场计算
#         for ig in range(channel.N_channel_segment):
#             Rxyz[ig] = np.sqrt(Rxy[ik] ** 2 + ((z1 + z2) / 2 - z_channel[ig]) ** 2)  # 第ig个通道段与导体段的空间距离
#
#             n_td_tmp = np.floor((t_sr * 1e-6 - (abs(z_channel[ig]) / vc / channel.vcof + Rxyz[ig] / vc)) / stroke.dt).astype(int)
#             index_head = np.apply_along_axis(get_t_delay_index1, 1, n_td_tmp)
#             index_tail = np.apply_along_axis(get_t_delay_index2, 1, n_td_tmp)
#             index_head = np.where(index_head == 1)
#             index_tail = np.where(index_tail == 1)
#             id_t = n_td_tmp > 0
#
#             Rz[ig] = (z1 + z2) / 2 - z_channel[ig]
#
#             if air_or_img == 0:
#                 # 根据不同的传播模型，选择系数
#                 if channel.channel_model == 'TL':
#                     cof_isr = 1 / (4 * np.pi * ep0)
#                 elif channel.channel_model == 'MTLL':
#                     cof_isr = 1 / (4 * np.pi * ep0) * (1 - z_channel[ig] / channel.H)
#                 else:
#                     cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-z_channel[ig] / channel.lamda)
#             else:
#                 if channel.channel_model == 'TL':
#                     cof_isr = 1 / (4 * np.pi * ep0)
#                 elif channel.channel_model == 'MTLL':
#                     cof_isr = 1 / (4 * np.pi * ep0) * (1 + z_channel[ig] / channel.H)
#                 else:
#                     cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-np.abs(z_channel[ig]) / channel.lamda)
#
#             dEz_1_cof = cof_isr * (2 * Rz[ig] ** 2 - Rxy[ik] ** 2) / Rxyz[ig] ** 5
#             dEz_2_cof = cof_isr * (2 * Rz[ig] ** 2 - Rxy[ik] ** 2) / Rxyz[ig] ** 4 / vc
#             dEz_3_cof = cof_isr * Rxy[ik] ** 2 / Rxyz[ig] ** 3 / vc ** 2
#
#             dEr_1_cof = cof_isr * 3 * Rz[ig] * Rxy[ik] / Rxyz[ig] ** 5
#             dEr_2_cof = cof_isr * 3 * Rz[ig] * Rxy[ik] / Rxyz[ig] ** 4 / vc
#             dEr_3_cof = cof_isr * Rz[ig] * Rxy[ik] / Rxyz[ig] ** 3 / vc ** 2
#
#             dEz1[index_tail[1], ig] = (dEz_1_cof * i_sr_int)[index_head]
#             dEz2[index_tail[1], ig] = (dEz_2_cof * i_sr)[index_head]
#             dEz3[index_tail[1], ig] = (dEz_3_cof * i_sr_div)[index_head]
#
#             dEr1[index_tail[1], ig] = (dEr_1_cof * i_sr_int)[index_head]
#             dEr2[index_tail[1], ig] = (dEr_2_cof * i_sr)[index_head]
#             dEr3[index_tail[1], ig] = (dEr_3_cof * i_sr_div)[index_head]
#
#         Ez[:, ik] = np.sum(dEz1 + dEz2 - dEz3, axis=1)
#         Er[:, ik] = np.sum(dEr1 + dEr2 + dEr3, axis=1)
#
#     # # 转置电场矩阵
#     # Ez = Ez.T
#     # Er = Er.T
#
#     return Ez, Er
#
#
# def calculate_H_magnetic_field_down_r(pt_start, pt_end, stroke, channel, z_channel, i_sr, t_sr, i_sr_int, i_sr_div, ep0, vc, air_or_img):
#     # 初始化径向和垂直方向的电场
#     a00 = pt_start.shape[0]  # 观察点或导体分段的总数
#     b00 = z_channel.shape[0]
#
#     # 雷击点位置
#     hit_pos = channel.hit_pos[0:2]
#
#
#     Rx = (pt_start[:, 0] + pt_end[:, 0]) / 2 - hit_pos[0]  # 雷击点在x方向到每个导线段中点的距离
#     Ry = (pt_start[:, 1] + pt_end[:, 1]) / 2 - hit_pos[1]  # 雷击点在y方向到每个导线段中点的距离
#     Rxy = np.sqrt(Rx ** 2 + Ry ** 2)  # 水平距离
#
#     # 初始化径向和垂直方向的电场
#     Er = np.zeros((stroke.Nt, a00))
#     Rxyz = np.zeros(channel.N_channel_segment)
#     Rz = np.zeros(channel.N_channel_segment)
#
#     for ik in range(a00):
#         x1, y1, z1 = pt_start[ik, :]
#         x2, y2, z2 = pt_end[ik, :]
#
#         dEr1, dEr2, dEr3 = np.zeros((stroke.Nt, channel.N_channel_segment)), np.zeros((stroke.Nt, channel.N_channel_segment)), np.zeros((stroke.Nt, channel.N_channel_segment))
#
#         Rxyz = np.sqrt(Rxy[ik] ** 2 + (z1 + z2) / 2 - z_channel) ** 2  # 第ig个通道段与导体段的空间距离
#
#         # 自由空间的电场计算
#         for ig in range(channel.N_channel_segment):
#             Rxyz[ig] = np.sqrt(Rxy[ik] ** 2 + ((z1 + z2) / 2 - z_channel[ig]) ** 2)  # 第ig个通道段与导体段的空间距离
#             n_td_tmp = np.floor((t_sr * 1e-6 - (abs(z_channel[ig]) / vc / channel.vcof + Rxyz[ig] / vc)) / stroke.dt).astype(int)
#             index_head = np.apply_along_axis(get_t_delay_index1, 1, n_td_tmp)
#             index_tail = np.apply_along_axis(get_t_delay_index2, 1, n_td_tmp)
#             index_head = np.where(index_head == 1)
#             index_tail = np.where(index_tail == 1)
#
#             Rz[ig] = (z1 + z2) / 2 - z_channel[ig]
#
#             if air_or_img == 0:
#                 # 根据不同的传播模型，选择系数
#                 if channel.channel_model == 'TL':
#                     cof_isr = 1 / (4 * np.pi * ep0)
#                 elif channel.channel_model == 'MTLL':
#                     cof_isr = 1 / (4 * np.pi * ep0) * (1 - z_channel[ig] / channel.H)
#                 else:
#                     cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-z_channel[ig] / channel.lamda)
#             else:
#                 if channel.channel_model == 'TL':
#                     cof_isr = 1 / (4 * np.pi * ep0)
#                 elif channel.channel_model == 'MTLL':
#                     cof_isr = 1 / (4 * np.pi * ep0) * (1 + z_channel[ig] / channel.H)
#                 else:
#                     cof_isr = 1 / (4 * np.pi * ep0) * np.exp(-np.abs(z_channel[ig]) / channel.lamda)
#
#             dEr_1_cof = 0 * cof_isr * 3 * Rz[ig] * Rxy[ik] / Rxyz[ig] ** 5
#             dEr_2_cof = cof_isr * Rxy[ik] * Rxy[ik] / Rxyz[ig] ** 3
#             dEr_3_cof = cof_isr * Rxy[ik] * Rxy[ik] / Rxyz[ig] ** 2 / vc
#
#             dEr1[index_tail[1], ig] = dEr_1_cof * i_sr_int[index_head]
#             dEr2[index_tail[1], ig] = dEr_2_cof * i_sr[index_head]
#             dEr3[index_tail[1], ig] = dEr_3_cof * i_sr_div[index_head]
#
#         Er[:, ik] = np.sum(dEr1 + dEr2 + dEr3, axis=1)
#     # # 转置电场矩阵
#     # Er = Er.T
#     return Er
