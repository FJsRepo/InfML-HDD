import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

relu = nn.ReLU(inplace=True)

def get_stage_images(stage, x, threshold, num_classes=5):
    temp_x = x
    temp_x_numpy = temp_x.detach().cpu().numpy()
    batch_size = temp_x_numpy.shape[0]
    tires = temp_x_numpy.shape[1]
    left_temp = []
    right_temp = []
    left_points = np.zeros(batch_size)
    right_points = np.zeros(batch_size)
    for index1 in range(batch_size):
        for index2 in range(tires):
            if temp_x_numpy[index1, index2, :, :].max() != 0:
                temp_img = temp_x_numpy[index1, index2, :, :]
                img_h, img_w = temp_img.shape  # img_h 72   img_w 96
                temp_img = temp_img * (255 / (temp_img.max()))
                # NoBinary_img = temp_img
                _, temp_img = cv2.threshold(temp_img, 30, 255, cv2.THRESH_BINARY)
                temp_img = temp_img.astype(np.uint8)
                for index3 in range(3):
                    temp_img[index3, :] = temp_img[3 + index3, :]
                    temp_img[:, index3] = temp_img[:, 3 + index3]
                for index4 in range(3):
                    temp_img[img_h - 1 - index4, :] = 0
                    temp_img[:, img_w - 1 - index4] = 0

                row_average = (np.sum(temp_img, axis=1)) / img_w
                # plt.imshow(temp_img, cmap = 'gray')
                # plt.show()
                # plt.savefig('/home/wacht/0-SSL-DeepLearning/1/images_Binary/tire{:05d}_{}.png'.format(index2, stage), dpi=1000)
                # plt.close()
                # self.Draw_Row_Average(row_average, index2, stage, threshold)
                row_number = []
                for index4 in range(img_h):
                    if row_average[index4] > threshold:
                        row_number.append(index4)
                if len(row_number) >= 2:
                    if (sum(temp_img[row_number[0], :int(img_w / 2)]) > sum(temp_img[row_number[0], int(img_w / 2):])):
                        left_temp.append(row_number[0] / img_h)
                        right_temp.append(row_number[1] / img_h)
                    else:
                        left_temp.append(row_number[1] / img_h)
                        right_temp.append(row_number[0] / img_h)
                elif len(row_number) == 1:
                    left_temp.append(row_number[0] / img_h)
                    right_temp.append(row_number[0] / img_h)

        left_temp.sort()
        right_temp.sort()

        if len(left_temp) != 0:
            left_points[index1] = left_temp[3 * (len(left_temp) // 4)]
        else:
            left_points[index1] = 0
        if len(right_temp) != 0:
            right_points[index1] = right_temp[3 * (len(right_temp) // 4)]
        else:
            right_points[index1] = 0
        if len(left_temp) == 0 and len(right_temp) != 0:
            left_points[index1] = right_points[index1]
        elif len(left_temp) != 0 and len(right_temp) == 0:
            right_points[index1] = left_points[index1]

        left_temp = []
        right_temp = []
    endpoints = np.zeros((batch_size, num_classes))
    for i in range(batch_size):
        endpoints[i, 1] = left_points[i]
        endpoints[i, 2] = right_points[i]
    endpoints = torch.tensor(endpoints, dtype=torch.float32)
    endpoints = endpoints.cuda()
    endpoints = relu(endpoints)
    return endpoints

def Draw_Row_Average(self, row, index2, stage, threshold):
    X = np.arange(0, len(row), 1)  # 0-287
    Y = row[X]
    Y_1 = np.full(len(row), threshold)
    plt.plot(X, Y)
    plt.plot(X, Y_1, color='red', linewidth=2)
    plt.xlabel('Row Sequence')
    plt.ylabel('Average Row Grayscale')
    # plt.ylim(-60, 60)
    plt.ylim(0, 255)
    # plt.show()
    plt.savefig('/home/wacht/0-SSL-DeepLearning/1/images/tire{:05d}_{}_RowAverage.png'.format(index2, stage), dpi=1000)
    plt.close()