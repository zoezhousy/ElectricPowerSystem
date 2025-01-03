import cupy as cp

def transfer_date_to_gpu(*arrays):
    transfered_data = []
    for object in arrays:
        transfered_data.append(cp.array(object))

    return transfered_data


def transfer_data_to_cpu(*arrays):
    transfered_data = []
    for object in arrays:
        transfered_data.append(cp.asnumpy(object))

    return transfered_data