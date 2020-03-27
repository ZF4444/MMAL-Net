import numpy as np

def calculate_iou(coor1, coor2):
    """
    :param coor1:dtype = np.array, shape = [:,4]
    :param coor2:
    :return:
    """
    # 排除与选中的anchor box iou大于阈值的anchor boxes
    start_max = np.maximum(coor1[:, 0:2], coor2[:, 0:2])  # [338,2]
    end_min = np.minimum(coor1[:, 2:4], coor2[:, 2:4])  # [338,2]
    lengths = end_min - start_max + 1  # [338,2]

    intersection = lengths[:, 0] * lengths[:, 1]
    intersection[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0

    union = ((coor1[:, 2] - coor1[:, 0] + 1) * (coor1[:, 3] - coor1[:, 1] + 1)
             + (coor2[:, 2] - coor2[:, 0] + 1) * (coor2[:, 3] - coor2[:, 1] + 1)
             - intersection)

    iou = intersection / union  # (338,)

    return iou