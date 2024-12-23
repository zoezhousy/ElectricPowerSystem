from sre_constants import error

import numpy as np
import scipy as sp
# from cupyx.scipy.linalg import block_diag
import time

#from numba.core.cgutils import printf


def expand_matrix(matrix, i, end, m):
    """
    从矩阵中取出第i行和第i列,并将其复制m次扩充成(n+m)*(n+m)的矩阵
    
    参数:
    matrix (numpy.ndarray): 输入矩阵
    i (int): 要取出的行和列索引
    end (int): 要取出的行列的截止位置和赋值的截止位置(保证在循环过程中不会取出多余的部分)
    m (int): 复制的次数
    
    返回:
    expanded_matrix (numpy.ndarray): 扩充后的矩阵
    """
    n = matrix.shape[0]  # 获取原始矩阵的大小
    
    # 创建一个全零矩阵,大小为(n+m)*(n+m)
    expanded_matrix = np.zeros((n+m, n+m), dtype=matrix.dtype)
    
    # 将原始矩阵复制到新矩阵的左上角
    expanded_matrix[:n, :n] = matrix
    
    # 将第i行复制到新矩阵的最后m行
    expanded_matrix[n:, :end] = np.tile(matrix[i, :end], (m, 1))
    
    # 将第i列复制到新矩阵的最后m列
    expanded_matrix[:end, n:] = np.tile(matrix[:end, i][:, np.newaxis], (1, m))
    
    return expanded_matrix


def copy_and_expand_matrix(original_matrix, m):
    """
    将一个n*n的矩阵扩展为一个mn*mn的矩阵,其中主对角线上的m*m矩阵块为0,
    其余的m*m矩阵块按照原先n*n的矩阵对应位置的数值复制填充。

    参数:
    original_matrix (numpy.ndarray): 输入n*n的矩阵
    m (int): 每个子矩阵块的大小

    返回:
    expanded_matrix (numpy.ndarray): 扩展后的mn*mn矩阵
    """
    n = original_matrix.shape[0]  # 原始矩阵的大小
    expanded_size = n * m  # 扩展后矩阵的大小

    # 创建一个全零的扩展矩阵
    expanded_matrix = np.zeros((expanded_size, expanded_size), dtype=original_matrix.dtype)

    # 遍历原始矩阵的每个元素
    for i in range(n):
        for j in range(n):
            # 计算子矩阵块的起始位置
            row_start = i * m
            col_start = j * m

            # 将当前元素复制为m*m的子矩阵块
            expanded_matrix[row_start:row_start+m, col_start:col_start+m] = np.full((m, m), original_matrix[i, j])

            # 将主对角线上的m*m矩阵块设置为0
            if i == j:
                expanded_matrix[row_start:row_start+m, col_start:col_start+m] = 0

    return expanded_matrix


def update_matrix(matrix, indices, submatrix):
    """
    将 n*n 的 NumPy 矩阵中,指定的m行m列的索引对应的区域,替换为 m*m 的子矩阵相应的数据。

    参数:
    matrix (numpy.ndarray): 输入的 n*n 矩阵
    indices (list): 长度为 m 的列表,表示要替换的行列索引
    submatrix (numpy.ndarray): 要替换的 m*m 子矩阵

    返回:
    updated_matrix (numpy.ndarray): 更新后的 n*n 矩阵
    """
    n = matrix.shape[0]

    updated_matrix = matrix.copy()  # 创建输入矩阵的副本
    i = 0
    for index in indices:
        updated_matrix[index, indices] += submatrix[i, :]
        i += 1

    return updated_matrix


def update_and_sum_matrix(matrix):
    """
    对给定的 n*n 方阵执行矩阵操作:
    1. 对第二行到末尾行,将其第二个元素到末尾的元素相加的相反数作为当前行的第一个元素的值
    2. 对第二列到末尾列,将其第二个元素到末尾的元素相加的相反数作为当前行的第一个元素的值
    3. 对第一行和第一列进行同样的加和操作, 将和的-0.5倍作为左上角元素的值
    """
    n = matrix.shape[0]
    new_matrix = matrix.copy()

    # 处理第二行到最后一行
    for i in range(1, n):
        new_matrix[i, 0] = -np.sum(new_matrix[i, 1:])

    # 处理第二列到最后一列
    for j in range(1, n):
        new_matrix[0, j] = -np.sum(new_matrix[1:, j])

    # 处理第一行和第一列
    new_matrix[0, 0] = -0.5 * (np.sum(new_matrix[0, 1:]) + np.sum(new_matrix[1:, 0]))

    return new_matrix

def block_diag_3dim(*arrays):
    """
    Create a block diagonal matrix from provided arrays, only for 3 dimensions matrix.
    The third dimension of each matrix have same length
    """
    # start_time = time.time()
    matrix_num = len(arrays)

    length_3dim = arrays[0].shape[2]

    matrix_shape = np.zeros((matrix_num, 2), dtype=int)
    for i in range(matrix_num):
        matrix_shape[i, :] = arrays[i].shape[0:2]

    blocked_matrix = np.zeros((int(matrix_shape[:, 0].sum()), int(matrix_shape[:, 1].sum()), length_3dim))
    n = 0
    m = 0
    for i in range(matrix_num):
        blocked_matrix[n:n + matrix_shape[i, 0], m:m + matrix_shape[i, 1], :] = arrays[i]
        n += matrix_shape[i, 0]
        m += matrix_shape[i, 1]

    # blocked_matrix = np.copy(arrays[0])
    # for i in range(matrix_num-1):
    #     zeros_upper = np.zeros((blocked_matrix.shape[0], arrays[i + 1].shape[1], length_3dim))
    #     zeros_down = np.zeros((arrays[i + 1].shape[0], blocked_matrix.shape[1], length_3dim))
    #     temp_upper = np.hstack((blocked_matrix, zeros_upper))
    #     temp_down = np.hstack((zeros_down, arrays[i + 1]))
    #     blocked_matrix = np.vstack((temp_upper, temp_down))
    # end_time = time.time()
    # print("耗时：{:f}秒".format(end_time - start_time))
    return blocked_matrix

def repet_block_diag(matrix, times: int):
    if len(matrix.shape) == 2:
        block_diag = sp.linalg.block_diag
    elif len(matrix.shape) == 3:
        block_diag = block_diag_3dim
    else:
        raise Exception('The input matrix should has 2 or 3 dimensions')

    expression = 'block_diag(matrix'
    for i in range(times):
        expression += ', matrix'
    expression += ')'

    return eval(expression)
