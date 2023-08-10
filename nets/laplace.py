import torch
import numpy as np
import cv2
import torch.nn as nn


relu = nn.ReLU(inplace=True)

def laplace(x):
    laplace_filter = np.array([
        [1, 1, 1],
        [1, -8.3, 1],
        [1, 1, 1],
    ])
    x_temp = x.detach().cpu().numpy()
    batch_size, image_num, _, _ = x_temp.shape
    for index in range(batch_size):
        for index2 in range(image_num):
            x_temp[index][index2] = cv2.GaussianBlur(x_temp[index][index2], (3, 3), 0)
            x_temp[index][index2] = cv2.filter2D(x_temp[index][index2], -1, laplace_filter)
            if sum(np.sum(x_temp[index][index2], axis=1)[2:5]) > 0:
                x_temp[index][index2] = -x_temp[index][index2]
            x_temp[index][index2] = (x_temp[index][index2] + abs(x_temp[index][index2])) / 2
    x_temp_tensor = torch.from_numpy(x_temp.copy())
    x_temp_tensor = x_temp_tensor.cuda()
    x_temp_tensor = relu(x_temp_tensor)
    return x_temp_tensor