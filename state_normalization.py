import numpy as np
from param import *

def normalize_node_inputs(node_inputs):
    """
    node_inputs: [N, 5]
    """
    MAX_CPU_CYCLE = 289.92081379572573
    MAX_DATA_SIZE = 4096000
    MAX_DDL = 5.0

    norm = np.zeros_like(node_inputs, dtype=np.float32)

    # cpu cycles
    norm[:, 0] = np.minimum(node_inputs[:, 0] / MAX_CPU_CYCLE, 1.0)

    # data size
    norm[:, 1] = np.minimum(node_inputs[:, 1] / MAX_DATA_SIZE, 1.0)

    # computation time / ddl
    norm[:, 3] = node_inputs[:, 3] / (node_inputs[:, 4] + 1e-8)
    norm[:, 3] = np.clip(norm[:, 3], 0.0, 1.0)

    # ddl
    norm[:, 4] = node_inputs[:, 4] / MAX_DDL

    return norm


def normalize_job_inputs(job_inputs):
    """
    job_inputs: [J, 4]
    """
    MAX_CAP = 1.2e10

    norm = np.zeros_like(job_inputs, dtype=np.float32)

    # mec id（先原样保留）
    norm[:, 0] = job_inputs[:, 0]

    # job ddl
    for i in range(0, len(args.mec_capacity)):
        norm[:, i + 1] = job_inputs[:, i + 1] / MAX_CAP

    return norm

